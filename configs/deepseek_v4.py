def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


# ── Attention 维度 hooks ────────────────────────────────────

def _head_dim(model_params):
    """DeepSeek-V4 使用 qk_rope_head_dim 作为实际 head_dim。"""
    return getattr(model_params, "qk_rope_head_dim")


def get_attention_dim(model_params):
    """返回 (qk_head_size, v_head_size)。"""
    hd = _head_dim(model_params)
    return hd, hd


def get_kv_cache_dim_per_token(model_params):
    """MQA: head_dim * num_kv_heads。"""
    return _head_dim(model_params) * get_num_key_value_heads(model_params)


def get_kv_proj_names(model_params):
    return ["k_proj", "v_proj"]


# ── act / add layers ────────────────────────────────────────

def get_act_layers(model_params):
    num_experts_per_tok = getattr(model_params, "num_experts_per_tok", 1)
    moe_intermediate_size = getattr(model_params, "moe_intermediate_size")
    n_shared_experts = getattr(model_params, "n_shared_experts", 0)
    shared_size = getattr(model_params, "intermediate_size")
    return {
        "moe_mlp_act": num_experts_per_tok * moe_intermediate_size,
        "shared_mlp_act": n_shared_experts * shared_size,
    }


def get_add_layers(model_params):
    h = get_hidden_size(model_params)
    return {
        "attn_add": h,
        "mlp_add": h,
        "moe_combine": h,
    }


# ── lm_head ─────────────────────────────────────────────────

def post_process(model_params, args):
    h = get_hidden_size(model_params)
    v = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append({
            "name": "lm_head",
            "stage": stage,
            "OPs": args["batchsize"] * h * v * 1,
            "load_weight": h * v * args["w_byte"],
            "load_act": h * args["a_byte"],
            "store_act": v * args["a_byte"],
        })
    return layers


# ── 线性层定义 (MQA + Q-LoRA + Indexer + MoE) ──────────────

def get_linear_layers(model_params, tp_size: int):
    h = get_hidden_size(model_params)
    n_heads = get_num_attention_heads(model_params)
    n_kv_heads = get_num_key_value_heads(model_params)
    head_dim = _head_dim(model_params)
    q_lora = getattr(model_params, "q_lora_rank")

    attn_out = n_heads * head_dim
    kv_dim = head_dim * n_kv_heads

    # MoE
    num_experts_per_tok = getattr(model_params, "num_experts_per_tok", 1)
    moe_i = getattr(model_params, "moe_intermediate_size")
    n_shared = getattr(model_params, "n_shared_experts", 0)
    shared_i = getattr(model_params, "intermediate_size")
    n_routed = getattr(model_params, "n_routed_experts", 0)

    routed_eff = num_experts_per_tok * moe_i
    shared_eff = n_shared * shared_i

    if tp_size > 1:
        assert attn_out % tp_size == 0
        assert kv_dim % tp_size == 0
        assert routed_eff % tp_size == 0
        assert shared_eff % tp_size == 0

    layers = {
        # Attention (MQA + Q-LoRA)
        "q_a_proj": [h, q_lora],
        "q_b_proj": [q_lora, attn_out // tp_size],
        "k_proj": [h, kv_dim],
        "v_proj": [h, kv_dim],
        "out_proj": [attn_out // tp_size, h],
        # MoE
        "router": [h, n_routed],
        "moe_gate_proj": [h, routed_eff // tp_size],
        "moe_up_proj": [h, routed_eff // tp_size],
        "moe_down_proj": [routed_eff // tp_size, h],
        "shared_gate_proj": [h, shared_eff // tp_size],
        "shared_up_proj": [h, shared_eff // tp_size],
        "shared_down_proj": [shared_eff // tp_size, h],
    }

    # Sparse Indexer
    idx_n = getattr(model_params, "index_n_heads", 0)
    idx_d = getattr(model_params, "index_head_dim", 0)
    if idx_n > 0 and idx_d > 0:
        layers["indexer_q_proj"] = [q_lora, idx_n * idx_d]
        layers["indexer_k_proj"] = [h, idx_d]

    return layers


# ── 动态计算图 ───────────────────────────────────────────────

def _build_graph(attn_node, has_indexer):
    graph = {
        "input": [],
        "attn_norm": ["input"],
        "q_a_proj": ["attn_norm"],
        "q_b_proj": ["q_a_proj"],
        "k_proj": ["attn_norm"],
        "v_proj": ["attn_norm"],
        attn_node: ["q_b_proj", "k_proj"],
        "out_proj": [attn_node],
        "attn_add": ["input", "out_proj"],
        "mlp_norm": ["attn_add"],
        "router": ["mlp_norm"],
        "moe_gate_proj": ["mlp_norm", "router"],
        "moe_up_proj": ["mlp_norm", "router"],
        "moe_mlp_act": ["moe_up_proj", "moe_gate_proj"],
        "moe_down_proj": ["moe_mlp_act"],
        "shared_gate_proj": ["mlp_norm"],
        "shared_up_proj": ["mlp_norm"],
        "shared_mlp_act": ["shared_up_proj", "shared_gate_proj"],
        "shared_down_proj": ["shared_mlp_act"],
        "moe_combine": ["moe_down_proj", "shared_down_proj"],
        "mlp_add": ["attn_add", "moe_combine"],
        "output": ["mlp_add"],
    }

    use_fa = (attn_node == "fused_attention")
    if not use_fa:
        graph["softmax"] = ["qk_matmul"]
        graph["sv_matmul"] = ["softmax", "v_proj"]
        graph["out_proj"] = ["sv_matmul"]

    if has_indexer:
        graph["indexer_q_proj"] = ["q_a_proj"]
        graph["indexer_k_proj"] = ["attn_norm"]

    return graph


def get_graph(model_params, use_flashattention=False):
    idx_n = getattr(model_params, "index_n_heads", 0)
    has_indexer = idx_n > 0
    attn_node = "fused_attention" if use_flashattention else "qk_matmul"
    return _build_graph(attn_node, has_indexer)


# 兼容性变量
transformer_layer_graph = _build_graph("qk_matmul", True)
flashattention_transformer_layer_graph = _build_graph("fused_attention", True)
