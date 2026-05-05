def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    """MoE 模型的 effective intermediate size。
    每 token 激活 num_experts_per_tok 个路由专家和 n_shared_experts 个共享专家。
    前 first_k_dense_replace 层为 dense 层（占比小，按 MoE 近似误差 < 2%）。
    """
    num_experts_per_tok = getattr(model_params, "num_experts_per_tok", 1)
    moe_intermediate_size = getattr(
        model_params, "moe_intermediate_size", getattr(model_params, "intermediate_size")
    )
    n_shared_experts = getattr(model_params, "n_shared_experts", 0)
    shared_intermediate_size = getattr(
        model_params, "intermediate_size", moe_intermediate_size
    )

    routed_effective = num_experts_per_tok * moe_intermediate_size
    shared_effective = n_shared_experts * shared_intermediate_size
    return routed_effective + shared_effective


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


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
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
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
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj": [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }


from configs.Llama import (
    flashattention_transformer_layer_graph,
    transformer_layer_graph,
)
