# RoofLine 模型使用文档

本文档系统说明 LLM-Viewer 中 RoofLine 模型的四种使用方式：CLI 单层分析、CLI 完整生成任务、Web UI、以及二次开发直接调用 `roofline_analyze`。所有内容均与仓库当前源码一一对应，引用处给出文件与行号，方便对照查证。

---

## 1. RoofLine 模型简介

### 1.1 核心函数

RoofLine 模型在仓库中只有一个核心函数，定义于 [roofline_model.py:1-19](../roofline_model.py#L1-L19)：

```python
def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        performance = y_max
    return arithmetic_intensity, performance, bound
```

### 1.2 参数与返回值

| 名称 | 含义 | 单位 |
|------|------|------|
| `bandwidth` | 硬件内存带宽 | Bytes/s |
| `max_OPS` | 硬件理论峰值算力 | OPs/s |
| `OPs` | 当前算子的总运算量 | OPs |
| `memory_access` | 当前算子的总内存访问量 | Bytes |
| 返回 `arithmetic_intensity` | 算术强度 = OPs / memory_access | OPs/Byte |
| 返回 `performance` | 该算子在当前硬件上能达到的实际算力 | OPs/s |
| 返回 `bound` | `"memory"` 或 `"compute"` | — |

### 1.3 判定逻辑

- **拐点（turning point）** = `max_OPS / bandwidth`，单位为 OPs/Byte。
- 若 `arithmetic_intensity < turning_point`：算子受**内存带宽**限制，`performance = arithmetic_intensity * bandwidth`。
- 否则：算子受**计算峰值**限制，`performance = max_OPS`。

### 1.4 整体工作流

LLM-Viewer 在 RoofLine 之上构建了完整的 LLM 推理性能分析流水线（来自 [README.md](../README.md) 的 7 步流程）：

1. 输入 LLM，提取每层关键信息（计算量、张量形状、数据依赖）。
2. 输入硬件参数，构造 RoofLine 模型。
3. 配置推理参数（batch size、prompt 长度、生成长度）。
4. 配置优化参数（量化位宽、FlashAttention、解码方法等）。
5. 由 `ModelAnalyzer` 结合 RoofLine 与层信息逐层分析，并跟踪峰值内存。
6. 汇总各层得到整网性能与内存报告。
7. 通过 Web UI 交互式查看与调参。

---

## 2. 安装与依赖

### 2.1 Python 依赖

```bash
pip install transformers flask flask_cors easydict
```

PyTorch 也需安装（[analyze_cli.py:2](../analyze_cli.py#L2) 中 `import torch.nn as nn`）：

```bash
pip install torch
```

### 2.2 前端依赖（仅 Web UI 模式）

```bash
cd frontend
npm install
```

---

## 3. CLI 模式 1：单次逐层分析（`analyze_cli.py`）

### 3.1 命令模板

```bash
python3 analyze_cli.py <model_id> <hardware> [选项]
```

### 3.2 参数表

参数定义见 [analyze_cli.py:8-38](../analyze_cli.py#L8-L38)。

| 参数 | 必填 | 默认 | 含义 |
|------|------|------|------|
| `model_id` | 是 | — | HuggingFace 模型 ID 或本地模型名（例：`meta-llama/Llama-2-7b-hf`、`DiT-XL/2`） |
| `hardware` | 是 | — | 硬件名，见第 7 章清单（例：`nvidia_A6000`） |
| `--source` | 否 | `huggingface` | 模型来源；非 `huggingface` 时使用 `model_params/<source>.py` 中的本地参数 |
| `--config_file` | 否 | 自动匹配 | 显式指定算子图配置（MoE 模型必填，见第 8 章） |
| `--batchsize` | 否 | `1` | 批大小 |
| `--seqlen` | 否 | `1024` | 序列长度 |
| `--w_bit` | 否 | `16` | 权重位宽（用于量化分析） |
| `--a_bit` | 否 | `16` | 激活位宽 |
| `--kv_bit` | 否 | `16` | KV Cache 位宽 |
| `--use_flashattention` | 否 | 关闭 | 启用 FlashAttention/FlashDecoding |
| `--tp-size` | 否 | `1` | 张量并行设备数 |

> 注：当 `w_bit`、`a_bit`、`kv_bit` 三者同时 ≤ 8 时，`ModelAnalyzer` 会自动切换到硬件的 INT8 峰值算力（[model_analyzer.py:534-537](../model_analyzer.py#L534-L537)）。

### 3.3 典型示例

```bash
# 最小示例：FP16 OPT-125M 在 A6000 上做单次推理分析
python3 analyze_cli.py facebook/opt-125m nvidia_A6000

# Llama-2-7B 单 batch、2K 序列
python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000 --batchsize 1 --seqlen 2048

# Llama-2-13B 大 batch、长上下文
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 16 --seqlen 2048

# Llama-2-13B 8K 长序列、启用 FlashAttention
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 8192 --use_flashattention

# MoE 模型必须显式指定 config_file
python3 analyze_cli.py zai-org/GLM-4.5 nvidia_H100 --config_file configs/glm4_moe.py --batchsize 1 --seqlen 4096

# DiT 系列：使用本地 model_params/DiT.py
python3 analyze_cli.py DiT-XL/2 nvidia_A6000 --batchsize 1 --seqlen 256 --source DiT
```

### 3.4 输出文件

CSV 落盘逻辑见 [model_analyzer.py:96-126](../model_analyzer.py#L96-L126)。同一次运行会生成两份 CSV：

```
output/<org>/<model>_decode.csv
output/<org>/<model>_prefill.csv
```

例如：`python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000` → `output/meta-llama/Llama-2-7b-hf_decode.csv` 与 `..._prefill.csv`。

每次运行以**追加（append）**方式写入，运行多次会在同一文件内堆叠多段；CSV 字段含义见第 6 章。

---

## 4. CLI 模式 2：完整生成任务（`analyze_gen_cli.py`）

该入口模拟一次完整的"prompt → 逐 token 生成"过程，输出端到端延迟与吞吐。

### 4.1 命令模板

```bash
python3 analyze_gen_cli.py <model_id> <hardware> [选项]
```

### 4.2 参数表

参数定义见 [analyze_gen_cli.py:8-29](../analyze_gen_cli.py#L8-L29)。与 `analyze_cli.py` 相同，仅多一个 `--promptlen`，且不接受 `--source`：

| 参数 | 必填 | 默认 | 含义 |
|------|------|------|------|
| `model_id` | 是 | — | 模型 ID |
| `hardware` | 是 | — | 硬件名 |
| `--config_file` | 否 | 自动匹配 | 算子图配置 |
| `--promptlen` | 否 | `128` | 输入 prompt 长度（参与 prefill） |
| `--seqlen` | 否 | `1024` | 总生成 token 数（含 decode 步数） |
| `--batchsize`, `--w_bit`, `--a_bit`, `--kv_bit`, `--use_flashattention`, `--tp-size` | 否 | 同上 | 同 `analyze_cli.py` |

### 4.3 输出格式

打印格式（[analyze_gen_cli.py:42-44](../analyze_gen_cli.py#L42-L44)）：

```
nvidia_A6000: 首 token 延迟 0.18, 总延迟 12.34, 吞吐 82.99 Token/sec
```

含义：

| 字段 | 含义 | 计算来源 |
|------|------|----------|
| 首 token 延迟（s） | prefill 阶段耗时 | `prefill_time` |
| 总延迟（s） | prefill + 所有 decode 步累加 | `inference_time` |
| 吞吐（Token/sec） | `seqlen * batchsize / inference_time` | 同上 |

### 4.4 内部循环逻辑

来自 [model_analyzer.py:505-530](../model_analyzer.py#L505-L530)：

1. 用 `prompt_len` 跑一次 `analyze()`，取得 prefill 时间。
2. 从 `i = prompt_len` 到 `prompt_len + gen_len` 循环，每步以当前累计长度 `i` 调用一次 `analyze()`，累加 decode 时间。
3. 由于 KV Cache 随生成步增长，逐步分析能反映 attention/KV 内存访问随长度的变化。

> 注意：每次生成 token 都重新构建一遍 RoofLine 分析，因此 `--seqlen` 较大时执行会变慢；这只影响**分析过程**的耗时，不是模型本身的推理时延。

---

## 5. Web UI 模式

### 5.1 后端启动

代码位于 [backend_app.py:40-47](../backend_app.py#L40-L47)：

```bash
python3 backend_app.py [--port 5050] [--local] [--debug]
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--port` | `5050` | 监听端口 |
| `--local` | 关闭 | 仅监听 `127.0.0.1`；不开启则监听 `0.0.0.0` |
| `--debug` | 关闭 | Flask 调试模式 |

### 5.2 前端启动

```bash
cd frontend
npm run dev
```

前端基于 Vue 3 + Vite，会自动打开浏览器并连接到本地后端。

### 5.3 公网托管

无需自建，直接访问 [http://llm-viewer.com](http://llm-viewer.com)。点击网络图中的任意节点可查看该层的 RoofLine 分析结果。

### 5.4 HTTP 接口

后端暴露两个接口（[backend_app.py:17-38](../backend_app.py#L17-L38)）：

| 方法 | 路径 | 入参 | 返回 |
|------|------|------|------|
| `POST` | `/get_graph` | `{"model_id": ..., "hardware": ..., "inference_config": {...}}` | `{"nodes": [...], "edges": [...], "total_results": {...}, "hardware_info": {...}}` |
| `GET` | `/get_avaliable` | — | `{"avaliable_hardwares": [...], "avaliable_model_ids": [...]}` |

可用 `curl` 快速验证后端连通性：

```bash
curl http://127.0.0.1:5050/get_avaliable
```

---

## 6. 输出字段对照表

CSV 表头与每行数据格式见 [model_analyzer.py:117-126](../model_analyzer.py#L117-L126)，共 12 列：

| 列名 | 单位 | 含义 |
|------|------|------|
| `layer_name` | — | 层（算子）名称 |
| `OPs` | OPs | 当前算子的总运算量 |
| `Access` | Bytes | 当前算子的总内存访问量（= load_weight + load_act + store_act + load_kv_cache + store_kv_cache） |
| `arithmetic_intensity` | OPs/Byte | 算术强度 |
| `performance` | OPs/s | RoofLine 给出的可达算力 |
| `bound` | — | `memory` 或 `compute` |
| `load_weight` | Bytes | 从内存加载权重的字节数 |
| `load_act` | Bytes | 从内存加载激活的字节数 |
| `store_act` | Bytes | 写回激活的字节数 |
| `load_kv_cache` | Bytes | 读 KV Cache 的字节数 |
| `store_kv_cache` | Bytes | 写 KV Cache 的字节数 |
| `inference_time` | s | 该层耗时 = `OPs / performance` |

数值显示通过 [utils.py](../utils.py) 中的 `str_number` / `str_number_time` 做了 K/M/G/T 单位简写。

CSV 顶部还会写入一行运行参数，便于区分多次追加：

```
=== meta-llama/Llama-2-7b-hf nvidia_A6000 w_bit=16 a_bit=16 kv_bit=16 batchsize=1 seqlen=2048 tp_size=1 ===
```

---

## 7. 支持的硬件清单

完整定义见 [hardwares/hardware_params.py:3-35](../hardwares/hardware_params.py#L3-L35)。

| 硬件名 | bandwidth | FP16 算力 | INT8 算力 | onchip_buffer |
|--------|-----------|-----------|-----------|---------------|
| `nvidia_V100` | 900 GB/s | 112 TOPS | 62 TOPS | 20 MB |
| `nvidia_A6000` | 768 GB/s | 154.8 TOPS | 309.7 TOPS | 21 MB |
| `nvidia_A6000_Ada` | 960 GB/s | 364.2 TOPS | 728.5 TOPS | 36 MB |
| `nvidia_A100` | 1555 GB/s | 312 TOPS | 624 TOPS | 27 MB |
| `nvidia_A100_40G` | 1555 GB/s | 312 TOPS | 624 TOPS | 27 MB |
| `nvidia_A100_80G` | 2039 GB/s | 312 TOPS | 624 TOPS | 27 MB |
| `nvidia_A800_80G_SXM` | 2039 GB/s | 312 TOPS | 624 TOPS | 27 MB |
| `nvidia_A40` | 696 GB/s | 149.7 TOPS | 299.3 TOPS | 21 MB |
| `nvidia_H100` | 3072 GB/s | 989.5 TOPS | 1979 TOPS | 33 MB |
| `nvidia_H100_SXM` | 3072 GB/s | 989.5 TOPS | 1979 TOPS | 33 MB |
| `nvidia_H100_PCIe` | 2048 GB/s | 756.5 TOPS | 1513 TOPS | 29 MB |
| `nvidia_L40` | 864 GB/s | 181 TOPS | 362 TOPS | 36 MB |
| `intel_13900k` | 89.6 GB/s | ~1.38 TOPS | — | 36 MB |

> 说明：
> - `H100` 与 `H100_SXM` / `H100_PCIe` 的 FP16/INT8 数值已按"开启稀疏后除以 2"还原为致密算力（源码中的 `1979e12 / 2`）。
> - GPU 的 `onchip_buffer` 取自 Register File 大小，用于 FlashAttention 等需要片上 SRAM 的算子建模。
> - V100 在 Tensor Core 上不支持 INT8，文件中给出的 INT8 性能仅供参考。

---

## 8. 支持的模型清单

完整定义见 [backend_settings.py:3-27](../backend_settings.py#L3-L27)。可在 `--source huggingface`（默认）下直接通过模型 ID 拉取。

### 8.1 LLaMA / OPT / GPT-J / ChatGLM（自动匹配 config）

| 模型 ID | 配置文件 |
|---------|----------|
| `meta-llama/Llama-2-7b-hf` | `configs/Llama.py` |
| `meta-llama/Llama-2-13b-hf` | `configs/Llama.py` |
| `meta-llama/Llama-2-70b-hf` | `configs/Llama.py` |
| `EleutherAI/gpt-j-6B` | `configs/gpt-j-6B.py` |
| `THUDM/chatglm3-6b` | `configs/chatglm3.py` |
| `facebook/opt-125m` | `configs/opt.py` |
| `facebook/opt-1.3b` | `configs/opt.py` |
| `facebook/opt-2.7b` | `configs/opt.py` |
| `facebook/opt-6.7b` | `configs/opt.py` |
| `facebook/opt-30b` | `configs/opt.py` |
| `facebook/opt-66b` | `configs/opt.py` |

> 这一组模型 [model_analyzer.py:31-35](../model_analyzer.py#L31-L35) 通过文件名匹配自动选择配置，**无需** `--config_file`。

### 8.2 GLM-4.x / 5.x（MoE，必须显式 `--config_file`）

| 模型 ID | 必填配置 |
|---------|----------|
| `zai-org/GLM-4.5` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.5-Air` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.6` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.7` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-5` | `--config_file configs/moe_dsa.py` |
| `zai-org/GLM-5.1` | `--config_file configs/moe_dsa.py` |

### 8.3 DeepSeek 系列（MoE，必须显式 `--config_file`）

| 模型 ID | 必填配置 |
|---------|----------|
| `deepseek-ai/DeepSeek-V3.1` | `--config_file configs/moe_dsa.py` |
| `deepseek-ai/DeepSeek-R1` | `--config_file configs/moe_dsa.py` |
| `deepseek-ai/DeepSeek-V4-Pro` | `--config_file configs/deepseek_v4.py` |
| `deepseek-ai/DeepSeek-V4-Flash` | `--config_file configs/deepseek_v4.py` |

### 8.4 DiT 系列（本地模型）

[backend_settings.py:25-26](../backend_settings.py#L25-L26) 中两条 DiT 条目目前**已注释**（不会出现在 Web UI 下拉框）：

```python
# "DiT-XL/2": {"source": "DiT"},
# "DiT-XL/4": {"source": "DiT"},
```

但本地参数文件 [model_params/DiT.py](../model_params/DiT.py) 与算子图 [configs/DiT.py](../configs/DiT.py) 仍可用，CLI 调用时通过 `--source DiT` 触发：

```bash
python3 analyze_cli.py DiT-XL/2 nvidia_A6000 --batchsize 1 --seqlen 256 --source DiT
```

---

## 9. 二次开发：直接调用 `roofline_analyze`

若不需要分析整张 LLM 网络，只想对单个算子或自定义场景做 RoofLine 估算，可直接调用核心函数。

### 9.1 最小示例

```python
from roofline_model import roofline_analyze

# 场景：A100-80G 上一次 4096x4096 的 FP16 GEMM，batch=1
ai, perf, bound = roofline_analyze(
    bandwidth=2039e9,         # A100-80G 带宽 2039 GB/s
    max_OPS=312e12,           # FP16 312 TOPS
    OPs=2 * 4096 * 4096 * 4096,   # GEMM 的乘加运算量
    memory_access=4096 * 4096 * 2 * 3,  # 加载 A、B，写回 C，每元素 2 字节
)
print(f"算术强度 = {ai:.2f} OPs/Byte")
print(f"可达算力 = {perf/1e12:.2f} TOPS")
print(f"瓶颈类型 = {bound}")
```

### 9.2 复合场景：扫不同 batch size

```python
import numpy as np
from roofline_model import roofline_analyze

bandwidth, max_OPS = 2039e9, 312e12
hidden, vocab = 4096, 32000

for bs in [1, 4, 16, 64]:
    OPs = 2 * bs * hidden * vocab
    mem = hidden * vocab * 2 + bs * hidden * 2 + bs * vocab * 2
    ai, perf, bound = roofline_analyze(bandwidth, max_OPS, OPs, mem)
    print(f"bs={bs:3d}  AI={ai:7.2f}  perf={perf/1e12:6.2f} TOPS  bound={bound}")
```

### 9.3 配合 Notebook 复现论文图表

仓库 `examples/` 目录下提供了 6 个 Jupyter Notebook，可直接运行：

| Notebook | 用途 |
|----------|------|
| [examples/plot_roofline.ipynb](../examples/plot_roofline.ipynb) | 绘制经典 RoofLine 曲线，叠加各算子点 |
| [examples/plot_hardware.ipynb](../examples/plot_hardware.ipynb) | 对比不同硬件下同一模型的瓶颈分布 |
| [examples/plot_serving.ipynb](../examples/plot_serving.ipynb) | 服务化场景（吞吐 vs batch size、延迟 vs 序列长度） |
| [examples/plot_inference_time.ipynb](../examples/plot_inference_time.ipynb) | 逐层推理时延堆叠图 |
| [examples/plot_memory.ipynb](../examples/plot_memory.ipynb) | 峰值内存占用分析 |
| [examples/plot_flashattention.ipynb](../examples/plot_flashattention.ipynb) | FlashAttention 开/关对比 |

### 9.4 重要提醒

- RoofLine 模型给出的是**理论上限**，与真实硬件的实际吞吐存在差距，主要用于**相对关系**分析（A 模型 vs B 模型、A 硬件 vs B 硬件、不同 batch/seqlen 的趋势对比）。
- 计算量与内存访问量的统计准确度依赖各模型的 `configs/*.py` 算子图建模；MoE 等复杂结构在专用 config（`moe_dsa.py`、`glm4_moe.py`、`deepseek_v4.py`）中维护。
- 若要扩展新硬件，只需在 [hardwares/hardware_params.py](../hardwares/hardware_params.py) 新增一项 `{"bandwidth": ..., "FP16": ..., "INT8": ..., "onchip_buffer": ...}`；要扩展新模型架构，则在 `configs/` 下新建一个对应的 `.py` 算子图文件。
