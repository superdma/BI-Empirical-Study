# 2. 实验环境与理论依据

## 2.1 硬件

| 项目 | 值 |
|---|---|
| GPU | 4 × NVIDIA RTX A6000 48 GB |
| 架构 | Ampere，SM86，CC 8.6 |
| 驱动 | 550.120 |
| 互联 | PCIe；本机无 InfiniBand |
| 主机内存 | 约 484 GiB 可用 |

A6000 是 SM86，不是 Hopper/Blackwell。它没有 TMA、没有 FP8 tensor core，也
没有 NVLink 交换机；这些都会影响 vLLM 选择哪条 kernel 路径。

## 2.2 软件栈

| 组件 | 版本 |
|---|---|
| vLLM | 0.23.0+cu129 |
| PyTorch | 2.11.0+cu129 |
| CUDA | 12.9 |
| NCCL | 2.28.9 |
| Python | 3.12 |
| 环境 | `$HOME/miniconda3/envs/vllm_update` |

## 2.3 模型

- `Qwen/Qwen3-30B-A3B`
- BF16，MoE，总参数约 30B，每 token 激活约 3B
- TP=4
- `max_model_len=8192`
- 权重路径：`$HOME/models/Qwen3-30B-A3B`

MoE 结构影响本次结果：默认非量化 MoE 后端是 **TRITON**
（启动日志：`Using TRITON Unquantized MoE backend ...`），其 `fused_moe_kernel`
会根据每个 expert 的 token 数选择不同的 tile 配置，并可能在推理过程中触发
Triton JIT（日志明确警告 `Triton kernel JIT compilation during inference:
fused_moe_kernel`）。

## 2.4 必需的环境变量

```bash
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### 为什么必须 `NCCL_IB_DISABLE=1`

本机没有 InfiniBand，但 NCCL 2.28.9 仍会加载 IB 插件，并在
`ncclNetPluginInit` 处 segfault。所有 TP=4 运行都必须关闭 IB。

### 为什么 `VLLM_USE_FLASHINFER_SAMPLER=0`

`vllm_update` 环境没有安装 `ninja`，FlashInfer 的 top-p/top-k sampler 无法 JIT。
本次全部使用 greedy（temperature=0）或固定 seed 的 sampling；关闭 FlashInfer
sampler 不影响 logprob 比较，但会影响 sampling 路径的实现。严格复现时应保持
该变量固定。

## 2.5 为什么“理论上”应该可行

### 2.5.1 官方 tested models 含 Qwen3-30B-A3B

vLLM 的 batch invariance 文档列出了经过测试的模型，Qwen3 系列在其中。
这说明该组合不是未经验证的边角配置。

### 2.5.2 SM86 的矩阵乘走 Triton persistent matmul

在 BI 模式下，vLLM 对 `aten::mm` 等算子做覆盖；SM86 上走 Triton persistent
matmul 路径，归约顺序固定。这是“同一进程内、batch 变化不影响线性层”的基础。

### 2.5.3 TP=4 的 all-reduce 是确定的

启动日志显示 NCCL 使用确定的 all-reduce 算法；TP=4 的切分与归约顺序固定。
同一进程内重复运行，all-reduce 结果可复现。

### 2.5.4 Attention / MoE 后端声明支持 BI

启动日志显示：

- attention backend：`FLASH_ATTN`；
- MoE backend：`TRITON`；
- BI 覆盖日志：`aten::mm` override 位于 `batch_invariant.py:905`。

“声明支持”不等于“所有形状都位级一致”。v2 的结果表明，**attention/MoE 的
形状相关 kernel 选择**正是 eager 模式失效的地方：BI 覆盖了 matmul，但没有覆盖
所有 fused kernel 的 tile/split 选择。

### 2.5.5 官方实现细节：A6000 不在“tuned matmul”名单里

vLLM 当前文档（`latest/features/batch_invariance`）写明：

- BI 支持 compute capability **8.0 及以上**的 NVIDIA GPU；
- tested models 包含 `Qwen/Qwen3-30B-A3B`；
- 实现细节第 5 条：

  > On CUDA devices with tuned matmul table entries for the model's bf16
  > unquantized forward linear layers (**Ada, Hopper, Blackwell**), runs
  > without `torch.compile` using breakable CUDA graphs so tile configs follow
  > the runtime batch size; set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to opt out.

A6000 是 **Ampere（SM86）**，不在 Ada/Hopper/Blackwell 的 tuned-table 名单里。
这与本次观察到的分裂一致：

- A6000 名义上满足 CC≥8.0，vLLM 也会初始化 BI；
- 但它没有官方 tuned matmul 表，实际走的是另一条路径；
- eager 下 fused kernel 的形状选择没有被完全固定，于是出现 32-prompt 的
  1/32、bs256 的失败；
- compiled 路径在本次所有形状上一致，但它是“实测通过”，不是“官方 tuned
  路径”。

这一条文档说明：**A6000 不是 BI 的“验证过的黄金平台”，
只是“满足最低 CC 要求、能开启”的平台。** 本次失败集中在共享的线性层/
matmul 路径（换 attention、MoE、稠密模型都复现），也与“没有 tuned matmul
表、kernel 按形状动态选 tile”相符。官方调优覆盖的平台是 Ada/Hopper/Blackwell（例如 L40S、H100、
B200）。

参考：<https://docs.vllm.ai/en/latest/features/batch_invariance>

## 2.6 本次没有覆盖的环境

- 没有 SM120（RTX 5090 / RTX PRO 6000 Blackwell）；
- 没有 A100/H100；
- 没有 InfiniBand / 多机；
- 没有 TP=8；
- 没有 32k 以上上下文；
- 没有 FP8/INT8 量化；
- 没有 CUDA graph 之外的图捕获后端。

这些都在 [09_limitations.md](09_limitations.md) 中列出。
