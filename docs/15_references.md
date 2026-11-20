# 15. 参考资料

以下链接均为公开可访问的官方文档、源码或模型卡；版本以 2026-11-20 为准。

## 15.1 vLLM 官方

1. Batch Invariance 文档（latest）
   <https://docs.vllm.ai/en/latest/features/batch_invariance>
   - 本文引用：BI 为 beta；支持 CC≥8.0；tested models 含
     `Qwen/Qwen3-30B-A3B`；实现细节第 5 条说明 tuned matmul 表只覆盖
     Ada/Hopper/Blackwell；TP all-reduce 的 16-byte 限制。

2. Batch invariance tracking issue #27433
   <https://github.com/vllm-project/vllm/issues/27433>

3. BI 源码（`batch_invariant.py`）
   <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py>

4. Automatic Prefix Caching 文档
   <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching>

5. vLLM 0.23.0 release
   <https://github.com/vllm-project/vllm/releases/tag/v0.23.0>

6. vLLM RL 示例中的 BI 用法
   <https://docs.vllm.ai/en/latest/examples/rl/batch_invariance>

## 15.2 模型

7. Qwen3-30B-A3B 模型卡
   <https://huggingface.co/Qwen/Qwen3-30B-A3B>

8. Qwen3 Technical Report
   <https://arxiv.org/abs/2505.09388>

9. Qwen3-0.6B 模型卡（本文的稠密对照）
   <https://huggingface.co/Qwen/Qwen3-0.6B>

## 15.3 运行时与硬件

10. PyTorch `torch.compile` 文档
    <https://docs.pytorch.org/docs/stable/torch.compiler.html>

11. FlashAttention 仓库
    <https://github.com/Dao-AILab/flash-attention>

12. NCCL 用户指南
    <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>

13. NVIDIA RTX A6000 产品页
    <https://www.nvidia.com/en-us/design-visualization/rtx-a6000/>

14. NVIDIA CUDA GPU 计算能力表
    <https://developer.nvidia.com/cuda-gpus>

## 15.4 本文内部资料

| 内容 | 路径 |
|---|---|
| v1 单文件报告（已被本目录取代） | `../BI_validation_Qwen3-30B-A3B_4xA6000_vLLM-0.23.0.md` |
| v1 离线表 A–D | `../results/analysis_tables.md` |
| v1 在线表 E–H | `../results/http/analysis_tables.md` |
| v2 表 I–L | `../results/extra/analysis_tables.md` |
| v2 表 M–P | `../results/http/concurrency/analysis_tables.md` |
| 原始离线 JSON | `../results/*.json` |
| 原始在线 JSON | `../results/http/*.json` |
| 扩展 JSON | `../results/extra/*.json` |
| 并发/长度 JSON | `../results/http/concurrency/*.json` |
| 第 8 章机制检查 JSON | `../results/tier1/*.json` |
| 第 9 章算子与打点 JSON | `../results/tier2/*.json` |
| 第 10 章组成、回放与 TP=2 JSON | `../results/tier3/*.json` |
| 第 10 章边界、回放与形状矩阵 JSON | `../results/tier4/*.json` |
| SGLang 原始结果 | `../sglang/results/*.json` |

## 15.5 引用说明

- 文档中的版本号、限制和实现细节以官方页面为准；如果上游更新，以新页面为准。
- 本文的实测数字只对应第 2 章的软件/硬件组合，不应直接外推到其他版本或 GPU。
- 外部链接仅用于说明官方语义，不构成对本文实验结果的背书。
