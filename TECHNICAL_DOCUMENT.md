# LLM-Viewer 技术文档

> 本文档基于论文 *LLM Inference Unveiled: Survey and Roofline Model Insights* (arXiv:2402.16363) 及项目源码编写，详细阐述 LLM-Viewer 的算法原理与具体实现。

---

## 1. 项目概述

LLM-Viewer 是一个面向大语言模型（LLM）推理的**网络级性能分析工具**。它结合 **Roofline 模型**对模型在特定硬件上的瓶颈进行定量分析，覆盖以下维度：

- **每层计算量（OPs）**
- **内存访问量（Memory Access）**
- **算术强度（Arithmetic Intensity）**
- **理论最大性能与瓶颈类型（Compute-bound / Memory-bound）**
- **峰值内存占用（Memory Footprint）**

支持预填充（Prefill）与解码（Decode）两阶段的独立分析，并可配置量化位宽、FlashAttention、张量并行等优化策略。

---

## 2. Roofline 模型原理

Roofline 模型是 LLM-Viewer 的核心理论框架，用于评估神经网络层在特定硬件上的理论性能上限。

### 2.1 基本概念

神经网络层在硬件上的执行涉及三个步骤：
1. 从主存（DDR/HBM）向芯片缓存传输数据；
2. 由处理单元执行计算；
3. 将结果写回主存。

因此，性能同时受**内存带宽**和**计算能力**约束。

### 2.2 关键指标

| 符号 | 含义 | 单位 |
|------|------|------|
| $B$ | 峰值内存带宽 | bytes/s |
| $P_{\max}$ | 峰值计算性能 | OPs/s |
| $W$ | 该层的总计算量 | OPs |
| $M$ | 该层的总内存访问量 | bytes |
| $I$ | **算术强度** $I = W / M$ | OPs/byte |

### 2.3 Roofline 曲线

以算术强度 $I$ 为横轴、性能（OPS/s）为纵轴：

- **水平线**：$y = P_{\max}$，表示硬件峰值算力（Compute Roof）。
- **对角线**：$y = B \cdot I$，表示内存带宽上限（Memory Roof）。
- **转折点**：$I_{\text{turn}} = P_{\max} / B$。

### 2.4 瓶颈判定算法

对于每一层，给定计算量 $W$ 和内存访问量 $M$：

**算法 `roofline_analyze`**（见 `roofline_model.py`）：

```python
def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth: B (bytes/s)
    # max_OPS: P_max (OPs/s)
    # OPs: W (OPs)
    # memory_access: M (bytes)

    turning_point = max_OPS / bandwidth          # I_turn = P_max / B
    arithmetic_intensity = OPs / memory_access    # I = W / M

    if arithmetic_intensity < turning_point:
        bound = "memory"                          # 内存受限
        performance = arithmetic_intensity * bandwidth  # P = B * I
    else:
        bound = "compute"                         # 计算受限
        performance = max_OPS                     # P = P_max

    return arithmetic_intensity, performance, bound
```

**推理时间估算**：
$$T = \frac{W}{P}$$

---

## 3. LLM 推理的两阶段分析

### 3.1 模型结构

主流 LLM 采用 Transformer 解码器架构，每层包含：
- **MHA（Masked Multi-Head Attention）**
- **MLP（Multi-Layer Perceptron）**

### 3.2 预填充阶段（Prefill Stage）

输入提示序列 $\mathbf{X}_{\text{pre}} \in \mathbb{R}^{n \times d}$，其中 $n$ 为序列长度，$d$ 为隐藏维度。

计算 Query、Key、Value：
$$
\begin{aligned}
\mathbf{Q}_{\text{pre}} &= \mathbf{X}_{\text{pre}} \cdot \mathbf{W}_q \\
\mathbf{K}_{\text{pre}} &= \mathbf{X}_{\text{pre}} \cdot \mathbf{W}_k \\
\mathbf{V}_{\text{pre}} &= \mathbf{X}_{\text{pre}} \cdot \mathbf{W}_v
\end{aligned}
$$

注意力输出：
$$
\mathbf{O}_{\text{pre}} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{pre}} \cdot \mathbf{K}_{\text{pre}}^T}{\sqrt{d}}\right) \cdot \mathbf{V}_{\text{pre}} \cdot \mathbf{W}_o + \mathbf{X}_{\text{pre}}
$$

此阶段将 $\mathbf{K}_{\text{pre}}$ 和 $\mathbf{V}_{\text{pre}}$ 存入 **KV Cache**。

### 3.3 解码阶段（Decode Stage）

逐个生成 token，输入 $\mathbf{X}_{\text{dec}} \in \mathbb{R}^{1 \times d}$。

计算并拼接 KV：
$$
\begin{aligned}
\mathbf{Q}_{\text{dec}} &= \mathbf{X}_{\text{dec}} \cdot \mathbf{W}_q \\
\mathbf{K}_{\text{cat}} &= [\mathbf{K}_{\text{cache}}, \; \mathbf{X}_{\text{dec}} \cdot \mathbf{W}_k] \\
\mathbf{V}_{\text{cat}} &= [\mathbf{V}_{\text{cache}}, \; \mathbf{X}_{\text{dec}} \cdot \mathbf{W}_v]
\end{aligned}
$$

注意力输出：
$$
\mathbf{O}_{\text{dec}} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{dec}} \cdot \mathbf{K}_{\text{cat}}^T}{\sqrt{d}}\right) \cdot \mathbf{V}_{\text{cat}} \cdot \mathbf{W}_o + \mathbf{X}_{\text{dec}}
$$

---

## 4. 各层计算与内存访问算法

`ModelAnalyzer.analyze()`（`model_analyzer.py`）对每层进行以下量化分析。

### 4.1 符号约定

| 符号 | 含义 |
|------|------|
| $d$ | hidden_size |
| $d_{\text{ffn}}$ | intermediate_size |
| $h$ | num_attention_heads |
| $h_{\text{kv}}$ | num_key_value_heads (GQA) |
| $d_h$ | head_size = $d / h$ |
| $b$ | batch_size |
| $n$ | sequence_length (Prefill) / KV Cache 长度 (Decode) |
| $w_b$ | 权重量化位宽 / 8（字节）|
| $a_b$ | 激活量化位宽 / 8（字节）|
| $kv_b$ | KV Cache 量化位宽 / 8（字节）|
| $L$ | num_hidden_layers |

### 4.2 线性层（Linear Layers）

线性层包括：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj。

对于权重矩阵 $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$：

**预填充阶段**：
- **计算量**：$W_{\text{linear}}^{\text{pre}} = 2 \cdot b \cdot n \cdot d_{\text{in}} \cdot d_{\text{out}}$
- **加载权重**：$M_{\text{weight}} = d_{\text{in}} \cdot d_{\text{out}} \cdot w_b$
- **加载激活**：$M_{\text{act\_in}} = b \cdot n \cdot d_{\text{in}} \cdot a_b$
- **存储激活**：$M_{\text{act\_out}} = b \cdot n \cdot d_{\text{out}} \cdot a_b$（k_proj/v_proj 为 0，因为直接存入 KV Cache）
- **存储 KV Cache**（仅 k_proj, v_proj）：$M_{\text{kv\_store}} = b \cdot n \cdot d_{\text{out}} \cdot kv_b$

**解码阶段**：
- **计算量**：$W_{\text{linear}}^{\text{dec}} = 2 \cdot b \cdot 1 \cdot d_{\text{in}} \cdot d_{\text{out}}$
- **加载权重**：同预填充（权重复用）
- **加载激活**：$M_{\text{act\_in}} = b \cdot 1 \cdot d_{\text{in}} \cdot a_b$
- **存储 KV Cache**（仅 k_proj, v_proj）：$M_{\text{kv\_store}} = b \cdot 1 \cdot d_{\text{out}} \cdot kv_b$

**代码实现**（`model_analyzer.py` 第 200–224 行）：

```python
for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
    is_kv_proj = name in ["k_proj", "v_proj"]
    is_normal_proj = not is_kv_proj

    # Decode
    self._analyze_to_results(
        "decode", name,
        OPs=ic * oc * batchsize * 2,
        load_weight=ic * oc * w_byte,
        load_act=ic * batchsize * a_byte,
        store_act=0 if is_kv_proj else oc * batchsize * a_byte,
        store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
    )

    # Prefill
    self._analyze_to_results(
        "prefill", name,
        OPs=ic * oc * batchsize * seqlen * 2,
        load_weight=ic * oc * w_byte,
        load_act=ic * batchsize * seqlen * a_byte,
        store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
        store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
    )
```

### 4.3 注意力矩阵乘法

#### 4.3.1 QK MatMul

计算 $\mathbf{Q} \cdot \mathbf{K}^T$：

**预填充**：$W_{\text{qk}}^{\text{pre}} = 2 \cdot b \cdot h \cdot n \cdot n \cdot d_h$
- 加载 Q: $b \cdot n \cdot d \cdot a_b$
- 加载 K: $b \cdot n \cdot d_{\text{kv}} \cdot kv_b$（$d_{\text{kv}} = d_h \cdot h_{\text{kv}}$）
- 存储注意力分数: $b \cdot h \cdot n \cdot n \cdot a_b$

**解码**：$W_{\text{qk}}^{\text{dec}} = 2 \cdot b \cdot h \cdot 1 \cdot n \cdot d_h$
- 加载 Q: $b \cdot 1 \cdot d \cdot a_b$
- 加载 K Cache: $b \cdot n \cdot d_{\text{kv}} \cdot kv_b$
- 存储注意力分数: $b \cdot h \cdot 1 \cdot n \cdot a_b$

#### 4.3.2 SV MatMul

计算 $\text{Softmax}(\dots) \cdot \mathbf{V}$：

**预填充**：$W_{\text{sv}}^{\text{pre}} = 2 \cdot b \cdot h \cdot n \cdot n \cdot d_h$
- 加载注意力分数: $b \cdot h \cdot n \cdot n \cdot a_b$
- 加载 V: $b \cdot n \cdot d_{\text{kv}} \cdot kv_b$
- 存储输出: $b \cdot n \cdot d \cdot a_b$

**解码**：$W_{\text{sv}}^{\text{dec}} = 2 \cdot b \cdot h \cdot 1 \cdot n \cdot d_h$
- 加载注意力分数: $b \cdot h \cdot 1 \cdot n \cdot a_b$
- 加载 V Cache: $b \cdot n \cdot d_{\text{kv}} \cdot kv_b$
- 存储输出: $b \cdot 1 \cdot d \cdot a_b$

#### 4.3.3 Softmax

包含 5 步操作（max, sub, exp, sum, div）：

**预填充**：$W_{\text{softmax}}^{\text{pre}} = 5 \cdot b \cdot h \cdot n \cdot n$
**解码**：$W_{\text{softmax}}^{\text{dec}} = 5 \cdot b \cdot h \cdot 1 \cdot n$

内存访问为加载和存储注意力矩阵。

### 4.4 FlashAttention / FlashDecoding

FlashAttention 将 QK MatMul、Softmax、SV MatMul 融合为单一算子 `fused_attention`，避免存储中间注意力矩阵。

**内存访问模型**（基于 FlashAttention-2 分块策略）：

```python
# onchip_buffer: 硬件片上缓存大小 (bytes)
block_size_r = min(ceil(onchip_buffer / (kv_byte * head_size)), head_size)
n_blocks_r = ceil(seqlen / block_size_r)   # Prefill
# n_blocks_r = ceil(1 / block_size_r)      # Decode

load_kv_cache = n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2
```

关键点：
- **无需存储**中间注意力矩阵，显著减少内存访问；
- 通过分块（tiling）在片上完成 softmax 和矩阵乘法；
- 预填充阶段收益相对较小（部分层已是 compute-bound）；
- **解码阶段收益显著**（原为 memory-bound，减少内存访问直接降低时延）。

### 4.5 归一化层（LayerNorm / RMSNorm）

每层 Transformer 包含 attn_norm 和 mlp_norm，共 7 个逐元素操作（sum, sub, pow, sum, div, mul, add）：

**预填充**：$W_{\text{norm}}^{\text{pre}} = 7 \cdot b \cdot n \cdot d$
**解码**：$W_{\text{norm}}^{\text{dec}} = 7 \cdot b \cdot 1 \cdot d$

内存访问：加载输入 + 存储输出（各 $b \cdot n \cdot d \cdot a_b$）。

### 4.6 残差连接与激活函数

| 算子 | 预填充 OPs | 解码 OPs | 说明 |
|------|-----------|----------|------|
| `attn_add`, `mlp_add` | $b \cdot n \cdot d$ | $b \cdot 1 \cdot d$ | 逐元素加法 |
| `mlp_act` | $2 \cdot b \cdot n \cdot d$ | $2 \cdot b \cdot 1 \cdot d$ | SiLU / GELU 等激活（如 SwiGLU 需两份输入）|

---

## 5. 网络级汇总算法

### 5.1 单阶段汇总

对每层结果乘以层数 $L$：

```python
for stage in ["decode", "prefill"]:
    for layer_name, result in self.results[stage].items():
        for data_name in ALL_DATA_NAMES:
            total_results[stage][data_name] += result[data_name] * num_hidden_layers
```

### 5.2 峰值内存占用

峰值内存 = 权重内存 + KV Cache 内存 + 临时激活内存

```python
# 权重 + KV Cache（在预填充阶段全部产生）
weight_kv_footprint = total_results["prefill"]["load_weight"] \
                    + total_results["prefill"]["store_kv_cache"]

# 解码阶段临时激活（逐层产生，按层累加峰值）
decode_tmp_act = sum(result["store_act"] for result in self.results["decode"].values())
total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint

# 预填充阶段临时激活（序列维度大，通常远大于解码）
prefill_tmp_act = sum(result["store_act"] for result in self.results["prefill"].values())
total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
```

### 5.3 完整生成任务时延

对于提示长度 $n_p$、生成长度 $n_g$ 的任务：

```python
prefill_time = analyze(prompt_len, batchsize)["prefill"]["inference_time"]
total_time = prefill_time

for i in range(prompt_len, prompt_len + gen_len):
    total_time += analyze(i, batchsize)["decode"]["inference_time"]
```

即：一次预填充 + $n_g$ 次解码（KV Cache 长度逐次增加）。

---

## 6. 量化对 Roofline 模型的影响

### 6.1 计算维度

硬件对不同位宽的支持不同。例如 Nvidia A6000：
- FP16: 155 TOPS
- INT8: 310 TOPS

若 $w_b \le 8$ 且 $a_b \le 8$ 且 $kv_b \le 8$，则使用 INT8 峰值算力，否则使用 FP16 峰值算力（见 `get_hardware_info`）。

```python
def get_hardware_info(self):
    bandwidth = hardware_params[self.hardware]["bandwidth"]
    if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
        max_OPS = hardware_params[self.hardware]["INT8"]
    else:
        max_OPS = hardware_params[self.hardware]["FP16"]
    onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
    return bandwidth, max_OPS, onchip_buffer
```

### 6.2 内存消耗维度

| 张量类型 | 量化影响 |
|----------|----------|
| 权重 (Weight) | $W_b$ 位量化直接按比例减少 |
| 临时激活 (Tmp Act) | 生命周期短，总量小，影响有限 |
| KV Cache | 随序列长度和 batch size 线性增长，长序列下占主导，量化收益显著 |

### 6.3 内存访问维度

量化减少内存访问量，提高算术强度 $I$，可能导致：

1. **仍处 Memory-bound**：$I$ 提升，$P = B \cdot I$ 提升，时延下降（常见于小 batch 解码阶段）。
2. **从 Memory-bound 进入 Compute-bound**：性能大幅提升。
3. **原本即 Compute-bound**：量化无显著收益（如大 batch 预填充）。

---

## 7. 系统优化分析

### 7.1 算子融合（Operator Fusion）

将相邻算子合并，消除中间结果的存储与加载，直接减少内存访问。

Roofline 视角：
- **Memory-bound 区域**：融合提升算术强度，显著降低时延；
- **Compute-bound 区域**：内存访问占比低，收益有限。

典型融合模式：
- Linear + SiLU / GELU
- FlashAttention：QK MatMul + Softmax + SV MatMul → fused_attention
- DeepSpeed-Fusion：QKV GeMM + LayerNorm 等

### 7.2 张量并行（Tensor Parallelism）

通过 `tp_size` 将线性层切分到多设备：

```python
def get_linear_layers(model_params, tp_size):
    if tp_size > 1:
        hidden_size //= tp_size
        intermediate_size //= tp_size
        key_value_heads //= tp_size
    # ...
```

效果：
- 每层权重和计算量按 $1/\text{tp\_size}$ 减少；
- 设备间引入通信开销（本文档暂未建模 all-reduce）。

---

## 8. 硬件参数定义

`hardwares/hardware_params.py` 中定义了主流硬件的 Roofline 参数：

```python
hardware_params = {
    "nvidia_A6000": {
        "bandwidth": 768e9,      # 768 GB/s
        "FP16": 154.8e12,        # 154.8 TFLOPS
        "INT8": 309.7e12,        # 309.7 TOPS
        "onchip_buffer": 21504e3 # 21.5 MB (Register File)
    },
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 989.5e12,
        "INT8": 1979e12,
        "onchip_buffer": 33792e3
    },
    # ...
}
```

**注意**：GPU 的 `onchip_buffer` 使用 **Register File Size** 估算 FlashAttention 的分块大小。

---

## 9. 模型配置扩展

新模型通过 `configs/<model>.py` 定义结构解析函数，例如 `configs/Llama.py`：

```python
def get_linear_layers(model_params, tp_size):
    # 返回 {"q_proj": [in_features, out_features], ...}
    ...

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
    # ... MLP 部分
}
```

---

## 10. CLI 与 API 使用

### 10.1 命令行

```bash
python analyze_cli.py <model_id> <hardware> \
    --batchsize 1 \
    --seqlen 2048 \
    --w_bit 16 \
    --a_bit 16 \
    --kv_bit 16 \
    --use_flashattention \
    --tp-size 1
```

### 10.2 Web API

后端基于 Flask 提供 `/get_graph` 接口：

```json
POST /get_graph
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "hardware": "nvidia_A6000",
  "inference_config": {
    "w_quant": "FP16",
    "a_quant": "FP16",
    "kv_quant": "FP16",
    "seq_length": 2048,
    "batch_size": 1,
    "use_flashattention": false,
    "gen_length": 128,
    "tp_size": 1,
    "stage": "prefill"
  }
}
```

返回每层节点的 OPs、Access、性能瓶颈等详细信息。

---

## 11. 引用

若使用 LLM-Viewer，请引用：

```bibtex
@misc{yuan2024llm,
  title={LLM Inference Unveiled: Survey and Roofline Model Insights},
  author={Zhihang Yuan and Yuzhang Shang and Yang Zhou and Zhen Dong and Chenhao Xue and Bingzhe Wu and Zhikai Li and Qingyi Gu and Yong Jae Lee and Yan Yan and Beidi Chen and Guangyu Sun and Kurt Keutzer},
  year={2024},
  eprint={2402.16363},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
