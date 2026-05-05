def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_norm_layers(model_params):
    return ["attn_norm", "q_a_norm", "kv_a_norm", "mlp_norm"]


# ── MLA 专用 hooks ──────────────────────────────────────────

def get_attention_dim(model_params):
    """返回 (qk_head_size, v_head_size)，供 model_analyzer 计算 attention OPs。"""
    qk = getattr(model_params, "qk_nope_head_dim") + getattr(model_params, "qk_rope_head_dim")
    v = getattr(model_params, "v_head_dim")
    return qk, v


def get_kv_cache_dim_per_token(model_params):
    """MLA 的 KV cache 只存 kv_lora + RoPE K。"""
    return getattr(model_params, "kv_lora_rank") + getattr(model_params, "qk_rope_head_dim")


def get_kv_proj_names(model_params):
    """写入 KV cache 的层名（MLA 是 kv_a_proj_with_mqa）。"""
    return ["kv_a_proj_with_mqa"]


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


# ── 线性层定义 (MLA + Indexer + MoE) ─────────────────────────

def get_linear_layers(model_params, tp_size: int):
    h = get_hidden_size(model_params)
    n_heads = get_num_attention_heads(model_params)
    q_lora = getattr(model_params, "q_lora_rank")
    kv_lora = getattr(model_params, "kv_lora_rank")
    qk_nope = getattr(model_params, "qk_nope_head_dim")
    qk_rope = getattr(model_params, "qk_rope_head_dim")
    v_dim = getattr(model_params, "v_head_dim")
    idx_n = getattr(model_params, "index_n_heads")
    idx_d = getattr(model_params, "index_head_dim")

    routed = getattr(model_params, "num_experts_per_tok", 1) * getattr(model_params, "moe_intermediate_size")
    shared = getattr(model_params, "n_shared_experts", 0) * getattr(model_params, "intermediate_size")
    n_routed_experts = getattr(model_params, "n_routed_experts", 0)

    # MLA
    q_out_dim = n_heads * (qk_nope + qk_rope)
    kv_out_dim = n_heads * (qk_nope + v_dim)
    o_in_dim = n_heads * v_dim

    # Indexer
    idx_q_out = idx_n * idx_d

    if tp_size > 1:
        assert q_out_dim % tp_size == 0
        assert kv_out_dim % tp_size == 0
        assert o_in_dim % tp_size == 0
        assert idx_q_out % tp_size == 0

    return {
        # MLA
        "q_a_proj": [h, q_lora],
        "q_b_proj": [q_lora, q_out_dim // tp_size],
        "kv_a_proj_with_mqa": [h, kv_lora + qk_rope],
        "kv_b_proj": [kv_lora, kv_out_dim // tp_size],
        "out_proj": [o_in_dim // tp_size, h],
        # Sparse Indexer
        "indexer_q_proj": [q_lora, idx_q_out // tp_size],
        "indexer_k_proj": [h, idx_d],
        "indexer_weights_proj": [h, idx_n],
        # MoE
        "router": [h, n_routed_experts],
        "moe_gate_proj": [h, routed // tp_size],
        "moe_up_proj": [h, routed // tp_size],
        "moe_down_proj": [routed // tp_size, h],
        "shared_gate_proj": [h, shared // tp_size],
        "shared_up_proj": [h, shared // tp_size],
        "shared_down_proj": [shared // tp_size, h],
    }


# ── 计算图 ───────────────────────────────────────────────────

transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_a_proj": ["attn_norm"],
    "q_a_norm": ["q_a_proj"],
    "indexer_q_proj": ["q_a_norm"],
    "indexer_k_proj": ["attn_norm"],
    "q_b_proj": ["q_a_norm"],
    "kv_a_proj_with_mqa": ["attn_norm"],
    "kv_a_norm": ["kv_a_proj_with_mqa"],
    "kv_b_proj": ["kv_a_norm"],
    "qk_matmul": ["q_b_proj", "kv_b_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "kv_b_proj"],
    "out_proj": ["sv_matmul"],
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

flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_a_proj": ["attn_norm"],
    "q_a_norm": ["q_a_proj"],
    "indexer_q_proj": ["q_a_norm"],
    "indexer_k_proj": ["attn_norm"],
    "q_b_proj": ["q_a_norm"],
    "kv_a_proj_with_mqa": ["attn_norm"],
    "kv_a_norm": ["kv_a_proj_with_mqa"],
    "kv_b_proj": ["kv_a_norm"],
    "fused_attention": ["q_b_proj", "kv_b_proj"],
    "out_proj": ["fused_attention"],
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
