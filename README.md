# LLM-Viewer

<img src="figs/eye.png" alt="LLM-Viewer" width="50"/>

LLM-Viewer 是一个用于可视化大语言模型（LLM）并分析其在不同硬件平台上性能的工具。它支持网络级分析，综合考虑峰值内存占用与总推理时延等因素。通过 LLM-Viewer，你可以更直观地理解 LLM 推理过程与性能优化方向。

你可以在浏览器中使用 LLM-Viewer，也可以通过命令行（CLI）使用。Web 版本提供了更友好的交互界面，便于配置和可视化，访问地址为 [LLM-Viewer Web](http://llm-viewer.com)。

欢迎阅读我们的论文 [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363.pdf)。
论文中我们基于 LLM-Viewer 对高效 LLM 推理的最新进展进行了系统分析。

该项目仍在持续更新。TODO 列表：
- 展示张量形状。
- 为非 Transformer 层补充前处理与后处理分析。
- 展示完整网络。
- 扩展硬件平台兼容性，并支持手动配置硬件参数。
- 增加对更多 LLM 的支持，并支持手动配置模型图。

## 工作流程

![LLM-Viewer Workflow](figs/workflow.svg)

如图所示，整体流程包括以下步骤：

1. 输入 LLM，提取每层关键信息，包括计算量、输入/输出张量形状和数据依赖关系。
2. 输入硬件参数，生成考虑硬件计算能力与内存带宽的 Roofline 模型。
3. 配置推理参数，如 batch size、prompt token 长度和生成 token 长度。
4. 配置优化参数，如量化位宽、是否启用 FlashAttention、解码方法及其他系统优化技术。
5. 使用 LLM-Viewer Analyzer 结合 Roofline 模型和层信息分析每层性能，并跟踪每层内存使用情况，基于数据依赖计算峰值内存。汇总所有层后可得到整个 LLM 网络性能。
6. 生成报告，包含每层和网络的峰值性能、性能瓶颈和内存占用。可进一步分析 batch size-性能、序列长度-性能等曲线，理解不同配置对性能的影响。
7. 通过 LLM-Viewer Web 界面直观查看网络结构与分析结果，便于交互式调参与逐层数据查看。

## Web 使用

在浏览器中使用 LLM-Viewer，请访问 [LLM-Viewer Web](http://llm-viewer.com)。
你可以点击节点查看该层的详细分析信息。

## CLI 使用

从 GitHub 克隆 LLM-Viewer 仓库：
```bash
git clone https://github.com/hahnyuan/LLM-Viewer.git
```

安装依赖：
```bash
pip install transformers flask flask_cors easydict
```

使用命令行分析 LLM 的示例命令如下：

```bash
python3 analyze_cli.py facebook/opt-125m nvidia_A6000
python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000 --batchsize 1 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 16 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 8192

# DiT models
python3 analyze_cli.py DiT-XL/2 nvidia_A6000 --batchsize 1 --seqlen 256 --source DiT
```

注意：Roofline 模型估算的是硬件理论可达到的性能上限。
该工具旨在帮助读者更清晰地理解影响 LLM 推理性能的关键因素，结果主要用于相对关系分析。

## 引用

如果你在研究中使用了 LLM-Viewer，请引用我们的论文：

```
@misc{yuan2024llm,
      title={LLM Inference Unveiled: Survey and Roofline Model Insights},
      author={Zhihang Yuan and Yuzhang Shang and Yang Zhou and Zhen Dong and Chenhao Xue and Bingzhe Wu and Zhikai Li and Qingyi Gu and Yong Jae Lee and Yan Yan and Beidi Chen and Guangyu Sun and Kurt Keutzer},
      year={2024},
      eprint={2402.16363},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```