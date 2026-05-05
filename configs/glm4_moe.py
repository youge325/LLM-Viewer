def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    """共享专家或 dense 层的 intermediate_size。"""
    return getattr(model_params, "intermediate_size")


def get_moe_intermediate_size(model_params):
    return getattr(model_params, "moe_intermediate_size", get_intermediate_size(model_params))


def get_num_experts_per_tok(model_params):
    return getattr(model_params, "num_experts_per_tok", 1)


def get_n_routed_experts(model_params):
    return getattr(model_params, "n_routed_experts", 0)


def get_n_shared_experts(model_params):
    return getattr(model_params, "n_shared_experts", 0)


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


def get_act_layers(model_params):
    """MoE 模型有两个独立的 SwiGLU 激活点：
    - moe_mlp_act：在路由专家的 gate_proj/up_proj 之后
    - shared_mlp_act：在共享专家的 gate_proj/up_proj 之后
    """
    num_experts_per_tok = get_num_experts_per_tok(model_params)
    moe_intermediate_size = get_moe_intermediate_size(model_params)
    n_shared_experts = get_n_shared_experts(model_params)
    intermediate_size = get_intermediate_size(model_params)
    return {
        "moe_mlp_act": num_experts_per_tok * moe_intermediate_size,
        "shared_mlp_act": n_shared_experts * intermediate_size,
    }


def get_add_layers(model_params):
    """除标准 attn_add/mlp_add 外，新增 moe_combine 表示路由+共享专家结果合并。"""
    hidden_size = get_hidden_size(model_params)
    return {
        "attn_add": hidden_size,
        "mlp_add": hidden_size,
        "moe_combine": hidden_size,
    }


def post_process(model_params, args):
    hiddensize = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append(
            {
                "name": "lm_head",
                "stage": stage,
                "OPs": args["batchsize"] * hiddensize * vocab_size * 1,
                "load_weight": hiddensize * vocab_size * args["w_byte"],
                "load_act": hiddensize * args["a_byte"],
                "store_act": vocab_size * args["a_byte"],
            }
        )
    return layers


def get_linear_layers(model_params, tp_size: int):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    moe_intermediate_size = get_moe_intermediate_size(model_params)
    num_experts_per_tok = get_num_experts_per_tok(model_params)
    n_routed_experts = get_n_routed_experts(model_params)
    n_shared_experts = get_n_shared_experts(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)

    routed_size = num_experts_per_tok * moe_intermediate_size
    shared_size = n_shared_experts * intermediate_size

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert routed_size % tp_size == 0
        assert shared_size % tp_size == 0
        assert key_value_heads % tp_size == 0

    return {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [
            hidden_size,
            hidden_size * key_value_heads // attention_heads // tp_size,
        ],
        "v_proj": [
            hidden_size,
            hidden_size * key_value_heads // attention_heads // tp_size,
        ],
        "out_proj": [hidden_size // tp_size, hidden_size],
        "router": [hidden_size, n_routed_experts],
        "moe_gate_proj": [hidden_size, routed_size // tp_size],
        "moe_up_proj": [hidden_size, routed_size // tp_size],
        "moe_down_proj": [routed_size // tp_size, hidden_size],
        "shared_gate_proj": [hidden_size, shared_size // tp_size],
        "shared_up_proj": [hidden_size, shared_size // tp_size],
        "shared_down_proj": [shared_size // tp_size, hidden_size],
    }


transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
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
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_proj", "k_proj", "v_proj"],
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
