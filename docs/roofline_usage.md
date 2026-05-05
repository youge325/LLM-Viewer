# RoofLine 模型使用文档

本文档系统讲解 LLM-Viewer 中 RoofLine 模型的**原理**与**解读方法**：核心函数语义、推理三阶段（prefill / decode / chat）对每个节点算术强度的影响、CSV 输出字段含义、所支持的硬件与模型清单、以及二次开发直接调用 `roofline_analyze` 的方式。所有内容均与仓库当前源码一一对应，引用处给出文件与行号，方便对照查证。

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

## 2. 推理阶段对算术强度的影响

LLM-Viewer 把一次 LLM 推理拆分为三个阶段：**prefill（预填充）**、**decode（解码）**、**chat（对话）**。在 Web UI 中切换阶段，每个节点的算术强度（即 RoofLine 图上代表该节点的虚线位置）会重新计算。但**不同类型的算子受阶段切换的影响差异巨大**——这一节给出每类算子的敏感度。

### 2.1 三个阶段的本质区别

源码：算子定义在 [model_analyzer.py:228-450](../model_analyzer.py#L228-L450)，chat 合成逻辑在 [get_model_graph.py:120-144](../get_model_graph.py#L120-L144)。

| 阶段 | Q 长度 | KV Cache 行为 | 适用场景 |
|---|---|---|---|
| **Prefill** | `seqlen` | 写入新 K/V，不读历史 | 处理用户 prompt |
| **Decode** | `1` | 读取完整历史 K/V，仅追加 1 个 token | 自回归逐 token 生成 |
| **Chat** | 混合 | 混合 | 模拟"prefill + 多步 decode"的端到端对话 |

### 2.2 四类算子的阶段敏感度

按算术强度（AI = OPs / memory_access）对阶段切换的敏感度由强到弱排列。

#### 公式中变量约定

下面所有表格的公式中出现的简写，含义如下（对应 [model_analyzer.py:199-224](../model_analyzer.py#L199-L224) 中的实际变量名）：

| 简写 | 源码变量 | 含义 |
|---|---|---|
| `ic` | `ic` | **input channels**：当前 Linear 层的输入维度（如 Llama-2-7B 的 `q_proj` 层输入 = 4096） |
| `oc` | `oc` | **output channels**：当前 Linear 层的输出维度（如 `q_proj` 层输出 = 4096） |
| `bs` | `batchsize` | 批大小，CLI 参数 `--batchsize` |
| `seqlen` | `seqlen` | 序列长度，CLI 参数 `--seqlen`（prefill 中是 prompt 长度，decode 中是已生成的 token 数） |
| `hidden` | `hidden_size` | Transformer 隐藏层维度（与 `ic`、`oc` 在大多数主干 Linear 上相等） |
| `num_heads` | `num_attention_heads` | Attention head 数 |
| `qk_head_size` | `qk_head_size` | 每个 head 中 Q/K 的维度 = `hidden / num_heads` |
| `v_head_size` | `v_head_size` | 每个 head 中 V 的维度（GQA/MQA 情况下可能与 `qk_head_size` 不同） |
| `w_byte` | `self.w_bit / 8` | 权重每元素字节数（FP16 → 2，INT8 → 1，INT4 → 0.5） |
| `a_byte` | `self.a_bit / 8` | 激活每元素字节数 |
| `kv_byte` | `self.kv_bit / 8` | KV Cache 每元素字节数 |

> 公式末尾常见的 `× 2` 因子来自"一次乘加（multiply-add）算作 2 个 OP"的约定；LayerNorm 中的 `× 7` 来自 softmax 那段注释里 `max → sub → exp → sum → div → mul → add` 共 7 个逐元素操作。

#### ① Linear 层（QKV / Output / MLP 投影）— 影响巨大

源码：[model_analyzer.py:228-248](../model_analyzer.py#L228-L248)

| 阶段 | OPs | 关键访存（权重） | 算术强度近似 |
|---|---|---|---|
| Prefill | `ic × oc × bs × seqlen × 2` | `ic × oc × w_byte` | **`2 × bs × seqlen / w_byte`** |
| Decode | `ic × oc × bs × 2` | `ic × oc × w_byte`（完全相同） | **`2 × bs / w_byte`** |

**关键洞察**：权重在两个阶段都需完整加载（不变），但 OPs 在 prefill 下是 decode 的 **seqlen 倍** → AI 也大约是 seqlen 倍。

> 例：FP16、batchsize=1、seqlen=2048 时
> - Prefill 的 Linear AI ≈ 2048 OPs/Byte（**compute-bound**）
> - Decode 的 Linear AI ≈ 1 OPs/Byte（**严重 memory-bound**）

→ 这就是为什么单 batch 的 LLM 解码总是被显存带宽卡住。

#### ② Attention 算子（qk_matmul / sv_matmul / softmax）— 影响形态差异

源码：decode 在 [model_analyzer.py:252-316](../model_analyzer.py#L252-L316)，prefill 在 [model_analyzer.py:362-417](../model_analyzer.py#L362-L417)。

| 阶段 | qk_matmul OPs | KV Cache 访存 | AI 趋势 |
|---|---|---|---|
| Prefill | `seqlen² × ...`（O(N²)） | 写入，不读旧 KV | AI 随 seqlen 增大 |
| Decode | `1 × seqlen × ...`（O(N)） | 读全部历史 KV（O(N)） | AI **不随 seqlen 变化**，约 `1/kv_byte` |

**关键洞察**：Prefill 的 attention 算量是 O(seqlen²)、访存也是 O(seqlen²)，AI 随 seqlen 上升；Decode 的 attention 算量是 O(seqlen)、KV Cache 访存也是 O(seqlen)，AI 是常数（只依赖 kv_byte）。

→ 这就是 FlashAttention 在长上下文下能大幅提速的原因（提高 prefill 时的 AI）。

#### ③ 逐元素算子（Norm / Add / 激活函数）— 几乎不影响

源码：decode 在 [model_analyzer.py:318-360](../model_analyzer.py#L318-L360)，prefill 在 [model_analyzer.py:418-450](../model_analyzer.py#L418-L450)。

| 阶段 | LayerNorm OPs | LayerNorm 访存 | AI |
|---|---|---|---|
| Prefill | `bs × hidden × seqlen × 7` | `bs × hidden × seqlen × 2 × a_byte` | **`7 / (2 × a_byte)`** |
| Decode | `bs × hidden × 1 × 7` | `bs × hidden × 1 × 2 × a_byte` | **`7 / (2 × a_byte)`** |

**关键洞察**：OPs 和 memory_access 都正比于 token 数，**比值完全相同**。所以 LayerNorm、残差 Add、SwiGLU/GELU 这些算子在 prefill 和 decode 下 AI 一致——它们**永远是 memory-bound**。

#### ④ 仅在某个阶段出现的访存

| 数据流 | Prefill | Decode |
|---|---|---|
| `load_kv_cache` | 0（K/V 还没存） | `seqlen × ...`（必须读完整历史） |
| `store_kv_cache` | `seqlen × ...`（写入新 K/V） | `1 × ...`（仅追加 1 个 token） |

这就是为什么 KV Cache 的存在让 decode 阶段更"内存饥渴"。

### 2.3 Chat 阶段如何合成

Chat 不是独立的计算模式，而是在 [get_model_graph.py:120-144](../get_model_graph.py#L120-L144) 中按下式合成：

```python
total_results["chat"] = total_results["prefill"]
n_divide = min(10, gen_length)
for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
    gen_result = analyzer.analyze(seqlen=lengthi, ...)
    total_results["chat"][k] += v * gen_length / n_divide
```

数学上等价于：

$$\text{chat} \approx \text{prefill} + \sum_{i=1}^{\text{gen\_length}} \text{decode}(\text{seqlen}=i)$$

由于 decode 在每个长度下 KV Cache 访存都不同，直接逐步累加 gen_length 次太慢，所以用 10 个采样点近似积分。

**对每类算子的影响**：

- **类型①（Linear）**：chat 的 OPs 与 memory_access 都是"prefill 一次 + decode gen_length 次"之和。当 `gen_length >> prompt_length` 时，chat 的 AI 趋近 decode 的 AI；反之趋近 prefill 的 AI。
- **类型②（Attention）**：chat 的累加涉及不同 seqlen 下的 decode，通过 10 个采样点近似全长度积分。
- **类型③（逐元素）**：因为 prefill 和 decode 的 AI 本来就相同，chat 的 AI **也保持不变**。

### 2.4 速记表

| 算子类别 | Prefill AI | Decode AI | Chat AI | 切换阶段时的 AI 移动 |
|---|---|---|---|---|
| **Linear（权重主导）** | 高（∝ seqlen） | 低（≈ 1/w_byte） | 加权平均 | **大幅左右移动** |
| **Attention** | 中–高（∝ seqlen） | 低（≈ 1/kv_byte，常数） | 加权平均 | 形态变化明显 |
| **LayerNorm / Add / GELU** | 低（≈ 7/(2 × a_byte)） | 低（与 prefill 相同） | 与 prefill 相同 | **几乎纹丝不动** |

→ 在 Web UI 切换"预填充 / 解码 / 对话"时，**Linear 与 Attention 节点的虚线会大幅左右移动**，而 LayerNorm 类节点的虚线几乎纹丝不动。这也是诊断模型瓶颈的关键直觉：当某个非逐元素节点在解码阶段明显左移到 memory-bound 区，它就是该阶段的优化重点。

---

## 3. 输出字段对照表

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

## 4. 支持的硬件清单

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

## 5. 支持的模型清单

完整定义见 [backend_settings.py:3-27](../backend_settings.py#L3-L27)。可在 `--source huggingface`（默认）下直接通过模型 ID 拉取。

### 5.1 LLaMA / OPT / GPT-J / ChatGLM（自动匹配 config）

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

### 5.2 GLM-4.x / 5.x（MoE，必须显式 `--config_file`）

| 模型 ID | 必填配置 |
|---------|----------|
| `zai-org/GLM-4.5` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.5-Air` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.6` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-4.7` | `--config_file configs/glm4_moe.py` |
| `zai-org/GLM-5` | `--config_file configs/moe_dsa.py` |
| `zai-org/GLM-5.1` | `--config_file configs/moe_dsa.py` |

### 5.3 DeepSeek 系列（MoE，必须显式 `--config_file`）

| 模型 ID | 必填配置 |
|---------|----------|
| `deepseek-ai/DeepSeek-V3.1` | `--config_file configs/moe_dsa.py` |
| `deepseek-ai/DeepSeek-R1` | `--config_file configs/moe_dsa.py` |
| `deepseek-ai/DeepSeek-V4-Pro` | `--config_file configs/deepseek_v4.py` |
| `deepseek-ai/DeepSeek-V4-Flash` | `--config_file configs/deepseek_v4.py` |

### 5.4 DiT 系列（本地模型）

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

## 6. 二次开发：直接调用 `roofline_analyze`

若不需要分析整张 LLM 网络，只想对单个算子或自定义场景做 RoofLine 估算，可直接调用核心函数。

### 6.1 最小示例

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

### 6.2 复合场景：扫不同 batch size

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

### 6.3 配合 Notebook 复现论文图表

仓库 `examples/` 目录下提供了 6 个 Jupyter Notebook，可直接运行：

| Notebook | 用途 |
|----------|------|
| [examples/plot_roofline.ipynb](../examples/plot_roofline.ipynb) | 绘制经典 RoofLine 曲线，叠加各算子点 |
| [examples/plot_hardware.ipynb](../examples/plot_hardware.ipynb) | 对比不同硬件下同一模型的瓶颈分布 |
| [examples/plot_serving.ipynb](../examples/plot_serving.ipynb) | 服务化场景（吞吐 vs batch size、延迟 vs 序列长度） |
| [examples/plot_inference_time.ipynb](../examples/plot_inference_time.ipynb) | 逐层推理时延堆叠图 |
| [examples/plot_memory.ipynb](../examples/plot_memory.ipynb) | 峰值内存占用分析 |
| [examples/plot_flashattention.ipynb](../examples/plot_flashattention.ipynb) | FlashAttention 开/关对比 |

### 6.4 重要提醒

- RoofLine 模型给出的是**理论上限**，与真实硬件的实际吞吐存在差距，主要用于**相对关系**分析（A 模型 vs B 模型、A 硬件 vs B 硬件、不同 batch/seqlen 的趋势对比）。
- 计算量与内存访问量的统计准确度依赖各模型的 `configs/*.py` 算子图建模；MoE 等复杂结构在专用 config（`moe_dsa.py`、`glm4_moe.py`、`deepseek_v4.py`）中维护。
- 若要扩展新硬件，只需在 [hardwares/hardware_params.py](../hardwares/hardware_params.py) 新增一项 `{"bandwidth": ..., "FP16": ..., "INT8": ..., "onchip_buffer": ...}`；要扩展新模型架构，则在 `configs/` 下新建一个对应的 `.py` 算子图文件。
