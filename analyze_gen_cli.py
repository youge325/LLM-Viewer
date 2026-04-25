from model_analyzer import ModelAnalyzer
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="模型 ID")
parser.add_argument(
    "hardware",
    type=str,
    help="硬件名称，例如 nvidia_V100 或 nvidia_A6000",
)
parser.add_argument("--config_file", type=str, default=None, help="配置文件")
parser.add_argument("--batchsize", type=int, default=1, help="批大小")
parser.add_argument("--seqlen", type=int, default=1024, help="序列长度")
parser.add_argument("--promptlen", type=int, default=128, help="提示序列长度")
parser.add_argument("--w_bit", type=int, default=16, help="权重位宽")
parser.add_argument("--a_bit", type=int, default=16, help="临时激活位宽")
parser.add_argument("--kv_bit", type=int, default=16, help="KV Cache 位宽")
parser.add_argument("--use_flashattention", action="store_true", help="使用 FlashAttention")
parser.add_argument(
    "--tp-size",
    type=int,
    default=1,
    help="用于张量并行的设备数量"
)
args = parser.parse_args()

analyzer=ModelAnalyzer(args.model_id,args.hardware,args.config_file)
ret = analyzer.analyze_generate_task(
    args.promptlen,
    args.seqlen,
    args.batchsize,
    args.w_bit,
    args.a_bit,
    args.kv_bit,
    args.use_flashattention,
    tp_size=args.tp_size
)
elapse = ret["inference_time"]
prefill_elapse = ret["prefill_time"]
print(f"{args.hardware}: 首 token 延迟 {prefill_elapse:.2f}, 总延迟 {elapse:.2f}, 吞吐 {args.seqlen * args.batchsize / elapse:.2f} Token/sec")
