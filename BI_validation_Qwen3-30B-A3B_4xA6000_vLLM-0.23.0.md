# Qwen3-30B-A3B × 4×RTX A6000 × vLLM 0.23.0
## 批次不变性（Batch Invariance, BI）实测报告

> **v1 归档说明（2026-11-17）**
>
> 本文是 v1 单文件报告，其中的“短 prompt 位级一致”只对一条 5-token target
> 成立。v2 把比较扩展到 32 条 prompt 后发现：eager 模式 31/32 不一致，
> compiled 模式 32/32 一致。请以文档集 [`docs/00_index.md`](docs/00_index.md)
> 为准；本文保留作为历史记录和 target-only 数据来源。

- **测试日期**：2026-11-17（离线引擎 + 在线 HTTP 服务）
- **测试主机**：`oscar-Super-Server`（Ubuntu 22.04.4 LTS，4×NVIDIA RTX A6000 48GB）
- **测试对象**：`Qwen/Qwen3-30B-A3B`，BF16，Tensor Parallel = 4
- **在线验证**：`vllm serve` + OpenAI-compatible `/v1/completions`，H1–H6 六套配置（见 5.5、6.3）
- **软件栈**：vLLM 0.23.0+cu129，PyTorch 2.11.0+cu129，CUDA 12.9
- **报告结论一句话**：
  **这套硬件 + 模型 + vLLM 版本可以做到 BI，离线引擎和在线 HTTP 服务都验证通过。但必须区分三个层级：**
  1. **同一个引擎进程内（短 prompt）**：批次大小、请求顺序、短 prompt 的 prefix cache 命中、decode 与 re-feed 的 token id 和 logprob 全部位级一致（差值严格为 0）。长 prompt 的 batch 不变性在“两边都是 fresh prefill”时也是 0.0。
  2. **跨进程 / 跨引擎重启 / 跨 chunk size**：默认的 `torch.compile + CUDA graphs` 模式**不保证**位级一致；使用 `--enforce-eager` 关闭编译图后，所有跨进程、跨 chunk size、跨 prefix-cache 配置的比较也全部位级一致。
  3. **eager + 长 prompt + prefix cache 命中**：缓存命中与全量重新 prefill **不是**位级一致。对“下一个 token 几乎确定”的饱和 prompt，只表现为 2.4e-7 的 logprob 差；对开放结尾的 prompt，首发 logprob 可差 0.035，并会翻转生成 token。compiled 模式没有这个现象。**严格复现建议 eager + 关闭 prefix caching。**

---

## 1. 结论摘要

| 验证项 | 结果 | 关键数字 |
|---|---|---|
| 离线同进程：bs=1 vs bs=32、逆序、decode vs re-feed | ✅ 通过 | max abs logprob diff = **0.0** |
| 离线同进程：短 prompt 前缀缓存命中 vs 未命中 | ✅ 通过 | **0.0** |
| 离线跨进程：eager 模式，同配置 ×2 / chunk 8192 vs 2048 / PC on vs off | ✅ 通过 | **0.0** |
| 离线跨进程：compiled 模式，warm 同配置 ×2 | ✅ 通过 | **0.0** |
| 离线跨进程：compiled 模式，冷启动 vs warm | ❌ 不通过 | token 在第 14 个生成 token 分叉；logprob 最大差 **1.227** |
| 离线跨进程：compiled 模式，chunk 8192 vs 2048（均 warm） | ❌ 不通过 | token 在第 14 个生成 token 分叉；logprob 最大差 **1.227** |
| **在线 HTTP**：短 prompt bs1/bs32、逆序、短 prompt 缓存命中、decode vs re-feed | ✅ 通过 | **0.0** |
| **在线 HTTP**：长 prompt 7600 token，bs1 vs bs4（两边 fresh） | ✅ 通过 | **0.0** |
| **在线 HTTP**：eager 跨进程重启（短 + 长 fresh） | ✅ 通过 | **0.0** |
| **在线 HTTP**：eager chunk 8192 vs 2048（短 + 长 fresh） | ✅ 通过 | **0.0** |
| **在线 HTTP**：eager PC on vs off（短 + 长 fresh） | ✅ 通过 | **0.0** |
| **在线 HTTP**：compiled 单进程内全部 6 项 | ✅ 通过 | **0.0** |
| **长 prompt + eager：prefix cache 命中 vs fresh** | ❌ 不通过（eager） | 饱和 prompt **2.384e-07**；开放结尾 prompt 首发 logprob 差 **0.035**，token 翻转；compiled 下为 0.0 |
| **在线 HTTP**：compiled vs eager | ❌ 不通过 | 短 prompt logprob 从第 0 个 token 起不同，最大差 **1.334**，token 在第 14 个分叉；长 prompt token 相同，logprob 差 **7.153e-07** |
| 对照组：关闭 BI（`VLLM_BATCH_INVARIANT=0`，离线/在线） | ❌ 如预期失败 | 在线 bs1/bs32 最大差 **1.273**（token 也分叉）；逆序 **0.131**；decode/re-feed **0.131** |

**一句话部署建议**：

- 只要“同一个服务进程内，输出不随 batch 组成/顺序变化”：默认编译模式 + `VLLM_BATCH_INVARIANT=1` 即可。
- 只要“跨进程、跨重启、跨 chunk size 也必须位级一致”（RL rollout、回归基线、审计复现）：加 `--enforce-eager`，并固定所有配置。
- 如果同时使用 **eager + 长 prompt + prefix cache**：再加 `--no-enable-prefix-caching`。否则同一个长 prompt 可能因为“缓存命中/未命中”返回不同的 token；这是本次在线测试发现的最大部署风险（见 6.3 与 7.8）。

---

## 2. BI 是什么，以及本报告的比较口径

vLLM 官方文档对 Batch Invariance 的定义是：开启后，模型输出是确定性的，并且与 batch size、请求在 batch 中的顺序无关。该功能目前仍是 beta。[^1]

需要特别区分：

- **BI 文档明确保证的是**：同一套引擎配置下，batch 大小、batch 组成、请求顺序不改变结果。短 prompt 的 prefix cache 命中/未命中、eager 下的 chunked prefill 分块方式，在本报告实测中保持位级一致；但 **eager + 长 prompt 的 prefix cache 命中**不是位级一致（见 6.3、7.8），compiled 模式下 chunk size 也会出现差异（见第 7 节）。因此不能把“跨 chunk size / 跨缓存状态位级一致”当成无条件的官方保证。
- **BI 不自动保证的是**：跨 GPU 型号、跨 TP 大小、跨 vLLM 构建/版本、跨编译模式、跨进程的位级一致；随机采样仍需要固定 seed；也不保证“缓存命中”和“全量重新 prefill”走同一条数值路径。
- **本报告的“位级一致”口径**：
  - 生成 token id 完全相等；
  - 每个生成 token 的 logprob 用 IEEE-754 双精度位模式比较（`struct.pack(">d", float(x))`）完全相等；
  - 因此“max abs logprob diff = 0.0”意味着逐 bit 相同，而不是“四舍五入后相同”。
- **主证据是离线 `LLM.generate`**：直接读取 Python float，不经过 HTTP/JSON 浮点序列化，避免序列化精度损失掩盖真实差异。第 6.3 节另外增加了一层**在线 HTTP 验证**；同配置下，离线和在线的 token id 与 32 个 logprob 逐 token 完全相同（见 6.3.4），说明 HTTP 序列化没有在这组数据上引入可观察差异。

---

## 3. 实验环境

### 3.1 主机与 GPU

| 项目 | 值 |
|---|---|
| 主机名 | `oscar-Super-Server` |
| 操作系统 | Ubuntu 22.04.4 LTS |
| 内核 | 6.5.0-26-generic |
| CPU | Intel Xeon Silver 4316 @ 2.30GHz，2 socket × 20 core × 2 thread = 80 线程 |
| 内存 | 503 GiB |
| 磁盘可用 | 约 937 GiB |
| GPU | 4 × NVIDIA RTX A6000，49140 MiB/卡 |
| Compute Capability | **8.6（SM86，Ampere）** |
| 驱动 | 550.120 |
| GPU 互联 | PCIe，无 NVLink、无 InfiniBand |

### 3.2 软件栈

| 组件 | 版本 |
|---|---|
| vLLM | **0.23.0+cu129** |
| PyTorch | 2.11.0+cu129 |
| CUDA Runtime | 12.9 |
| Transformers | 5.16.1 |
| Triton | 3.6.0 |
| NumPy | 2.5.3 |
| NCCL | 2.28.9 |
| 模型下载工具 | ModelScope 1.29.2（`vllm` 环境） |

### 3.3 模型

本地路径：`$HOME/models/Qwen3-30B-A3B`（ModelScope 下载，16 个 safetensors 分片）。

| 项目 | 值 |
|---|---|
| 架构 | `Qwen3MoeForCausalLM` |
| 总参数 / 激活参数 | 30.5B / 3.3B |
| 层数 | 48 |
| 专家数 / 每 token 激活 | 128 / 8 |
| Hidden size | 2048 |
| Intermediate size | 6144 |
| MoE intermediate size | 768 |
| Attention heads / KV heads | 32 / 4 |
| Head dim | 128 |
| 词表 | 151936 |
| 最大位置 | 40960 |
| 精度 | BF16 |
| Checkpoint 大小 | 56.87 GiB（vLLM 日志） |

运行时（TP=4）日志给出的实际占用：

- 每卡模型权重：**14.25 GiB**
- KV cache：**1,066,720 tokens**
- 8,192 token/请求时的最大并发：**130.21×**

---

## 4. 为什么这个组合“理论上”应该可行

这一节不是凭感觉，而是逐条对应官方文档和安装包源码。

### 4.1 官方 tested models 列表包含该模型

vLLM 0.23.0 的 BI 文档在 “Tested Models → Qwen3 (MoE)” 中明确列出：

- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`

同一页面写明硬件要求：**NVIDIA GPU with compute capability 8.0 or higher**。[^1]

A6000 是 CC 8.6，满足要求。

### 4.2 源码层面：SM86 上的矩阵乘走 Triton persistent matmul 路径

安装包中的 `vllm/model_executor/layers/batch_invariant.py`：

- `enable_batch_invariant_mode()`（第 897 行起）：
  - 对 `current_platform.is_device_capability_family(80)`（即 8.x，含 8.6）安装 `mm_batch_invariant`、`addmm_batch_invariant`、`matmul_batch_invariant`、`linear_batch_invariant` 的 CUDA dispatch override；
  - 对 Hopper/Blackwell 则走 cuBLAS workspace 配置路径。
- `override_envs_for_invariance()`（第 953 行起）设置：
  - `VLLM_ALLREDUCE_USE_SYMM_MEM=0`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - NCCL 确定性相关变量（`NCCL_ALGO=allreduce:tree`、`NCCL_PROTO=Simple`、单 channel、单线程等）
  - `VLLM_USE_AOT_COMPILE=0`
- `init_batch_invariance()`（第 974 行）在 `gpu_worker.py:1173` 被调用；还会关闭 TF32。

实测启动日志中可以直接看到 override 生效：

```text
Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::mm(Tensor self, Tensor mat2) -> Tensor
  dispatch key: CUDA
  new kernel: registered at .../vllm/model_executor/layers/batch_invariant.py:905
```

> 注意：`vllm/envs.py` 中 `VLLM_BATCH_INVARIANT` 的注释仍写着 “Requires NVIDIA GPU with compute capability >= 9.0”，这与官方文档和源码中的 8.x 分支不一致。本报告以文档 + 源码 + 实测为准：**CC 8.6 实际可用**。

### 4.3 TP=4 的 all-reduce 路径是确定的

- `vllm/config/parallel.py:940`：当 `VLLM_BATCH_INVARIANT=1` 时强制 `disable_custom_all_reduce = True`。
- 实测启动日志：

```text
disable_custom_all_reduce=True
Using ['PYNCCL'] all-reduce backends (in dispatch order) ... 
  out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']
```

- 本模型 hidden size = 2048，BF16 每 token 的 all-reduce payload = 2048 × 2 = **4096 bytes**，是 16 的整数倍，满足 vLLM 0.32.0 文档中“TP all-reduce size 必须是 16 字节整数倍”的约束。[^2]

### 4.4 Attention 与 MoE 后端都声明支持 BI

- `FLASH_ATTN`：`supports_batch_invariance() = True`；BI 下 `num_splits=1`（`flash_attn.py:1048`），避免 split-KV 带来的归约顺序变化。
- Triton MoE：`TritonExperts._supports_batch_invariance()` 返回 `True`（`triton_moe.py:125`）。
- BI 下 MoE 不使用 tuned config，而是固定 `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, SPLIT_K=1`（`fused_moe.py:1212`）。
- BI 下会关闭 cascade attention（`config/vllm.py:1407`）。

实测启动日志：

```text
Using AttentionBackendEnum.FLASH_ATTN backend.
Using FlashAttention version 2
Using TRITON Unquantized MoE backend out of potential backends:
  ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']
Using MoEPrepareAndFinalizeNoDPEPModular
```

---

## 5. 测试设计

### 5.1 为什么以离线位级比较为主，并补充 HTTP 在线验证

HTTP 返回的 JSON 会对浮点数做十进制序列化，默认精度可能掩盖最后几位差异，也可能把不同 bit 模式打印成同一个十进制字符串。为了确认“位级”一致，主测试直接使用：

```python
from vllm import LLM, SamplingParams
```

并直接读取 `RequestOutput.outputs[0].logprobs` 里的 Python float，再用 `struct.pack(">d", float(x))` 比较 IEEE-754 位模式。

但“推荐脚本能不能启动、服务端路径是否仍然成立”必须用真实 `vllm serve` + OpenAI-compatible API 验证。因此第 6.3 节增加了一套完整的在线矩阵：用 `/tokenize` 固定 prompt token ids，用 `return_token_ids=true` 取回生成 token，用 `logprobs` / `prompt_logprobs` 取回 logprob，并用 `cache_salt` 精确控制 prefix cache 命中/未命中，避免测试请求之间互相污染。离线与在线同配置的结果会逐 token 对照（6.3.4）。

### 5.2 采样参数

| 参数 | 值 |
|---|---|
| temperature | 0.0（greedy） |
| top_p | 1.0 |
| max_tokens | 32 |
| logprobs | 5 |
| seed | 0 |
| prompt_logprobs（仅 re-feed 测试） | 1 |

> 环境里没有安装 `ninja`，FlashInfer 的 top-p/top-k sampler 在初始化时会尝试 JIT 编译并失败，因此设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 使用原生 sampler。greedy 解码只做 argmax，这个开关不影响本报告的 logits/logprob 比较。

### 5.3 测试矩阵

| 编号 | 测试 | 说明 |
|---|---|---|
| T1 | bs=1 vs bs=32 | 目标 prompt 单独跑；再与 31 条 filler 一起跑，目标在 batch 第 0 位 |
| T2 | bs=1 vs 逆序 | 目标放到 batch 最后一位 |
| T3 | bs=1 冷启动 vs 重复请求 | 同一进程内，第二次请求会命中 prefix cache |
| T4 | decode vs re-feed | 把“prompt + 已生成 token”重新 prefill，用 `prompt_logprobs` 与逐 token decode 的 logprob 比较 |
| T5 | 长 prompt | 约 7600 token，触发 chunked prefill |
| T6 | 跨进程 chunk size | 同一进程配置分别用 `max_num_batched_tokens=8192` 和 `2048` |
| T7 | 跨进程 prefix cache | 同一进程配置分别开启/关闭 `enable_prefix_caching` |
| T8 | 同配置重复进程 | 完全相同的参数跑两次，检查跨进程可复现性 |
| T9 | BI 关闭对照 | `VLLM_BATCH_INVARIANT=0`，验证测试对非 BI 行为足够敏感 |
| T10 | TP=1 小模型对照 | Qwen3-0.6B，TP=1，检查测试脚本与非 TP 路径 |

测试 prompt：

- 目标：`The capital of France is`（5 个 token：`[785, 6722, 315, 9625, 374]`）
- 31 条 filler：`Question i: describe the number i in one short sentence.`
- 长 prompt：`"The following is a long passage. " + "The quick brown fox jumps over the lazy dog. " * 900`，截断到 7600 token

### 5.4 运行配置

除特别说明外：

```text
model=$HOME/models/Qwen3-30B-A3B
tensor_parallel_size=4
max_model_len=8192
gpu_memory_utilization=0.88
enable_chunked_prefill=True
attention_config.backend=FLASH_ATTN
VLLM_BATCH_INVARIANT=1
NCCL_IB_DISABLE=1
VLLM_USE_FLASHINFER_SAMPLER=0
```

### 5.5 在线测试矩阵（HTTP）

服务端统一用 `vllm serve`，TP=4，`max_model_len=8192`，`gpu_memory_utilization=0.88`，`FLASH_ATTN`，`NCCL_IB_DISABLE=1`，`VLLM_USE_FLASHINFER_SAMPLER=0`。每次只启动一个配置，跑完即停。

| 编号 | 配置 | BI | eager | chunk | prefix cache |
|---|---|---|---|---|---|
| H1 | eager 8192 PC1 run1 | 1 | 是 | 8192 | on |
| H2 | eager 8192 PC1 run2（跨进程重启） | 1 | 是 | 8192 | on |
| H3 | eager 2048 PC1 | 1 | 是 | 2048 | on |
| H4 | eager 8192 PC0 | 1 | 是 | 8192 | off |
| H5 | BI=0 eager 8192 PC1 | 0 | 是 | 8192 | on |
| H6 | compiled 8192 PC1（warm cache） | 1 | 否 | 8192 | on |

在线请求参数：`temperature=0`、`top_p=1`、`seed=0`、`max_tokens=32`、`logprobs=5`、`return_token_ids=true`；re-feed 用 `prompt_logprobs=1`。`cache_salt` 为每个 case 指定唯一值；需要制造“缓存命中”时，第二次请求复用第一次的 salt。

| 编号 | 比较 | 说明 |
|---|---|---|
| H-a | bs=1 vs bs=32 | 31 条 filler，目标在第 0 位 |
| H-b | bs=1 vs 逆序 | 目标在第 31 位 |
| H-c | 短 prompt fresh vs cache hit | 同一 salt，第二次命中 |
| H-d | 长 prompt bs=1 vs bs=4（两边 fresh） | 不同 salt，排除缓存影响 |
| H-e | 长 prompt fresh vs cache hit | 第一次 fresh，第二次同 salt 命中 |
| H-f | decode vs re-feed | 用 `prompt_logprobs` 对照 |

另有诊断 D1/D2：用 “Once upon a time ...” 这类开放结尾的约 7584-token prompt，比较 fresh / fresh / cache-hit 三种请求的首发 logprob 与 token，用来暴露“饱和 prompt”掩藏的小差异。

---

## 6. 结果

### 表 A：同一进程内的 BI 测试

表中数值为“最大绝对 logprob 差”；✅ 表示 token id 全部相等且 logprob 逐 bit 相等。

| 运行 | bs=1 vs bs=32 | 逆序 | 前缀缓存命中（短 prompt） | decode vs re-feed |
|---|---|---|---|---|
| compiled 8192/PC1（冷启动首次编译） | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| compiled 8192/PC1（warm #2） | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| compiled 8192/PC1（warm #3） | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| compiled 2048/PC1（冷） | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| compiled 8192/PC0（冷） | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| eager 8192/PC1 #1 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| eager 8192/PC1 #2 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| eager 2048/PC1 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| eager 8192/PC0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| **BI=0 对照（compiled 8192/PC1）** | **6.042e-02 ❌** | **7.360e-02 ❌** | 0 ✅ | **4.201e-02 ❌** |
| TP=1 Qwen3-0.6B smoke | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |

**结论**：只要开启 BI，在同一个引擎进程内，无论 batch 大小、请求顺序、**短 prompt 的** prefix cache 命中与否、decode 还是重新 prefill，结果都是逐 bit 一致的。BI 关闭时，同样的测试立刻出现 10⁻² 量级的 logprob 差异。

> **边界**：表 A 的“前缀缓存命中”列用的是 5-token 短 prompt。长 prompt（约 7600 token）的缓存命中 vs 全量 fresh 在 **eager 下不是 0.0**（饱和 prompt 2.384e-07；开放结尾 prompt 首发差 0.035 且 token 翻转）；compiled 下仍是 0.0。详见 6.3.5 与 7.8。

### 表 B：跨进程 / 跨配置比较

“single” 指目标 prompt 单独生成 32 个 token；“long” 指 7600 token 长 prompt 生成 32 个 token。✅ 表示 token id 全等且 logprob 逐 bit 全等。

> 说明：表中的“首个差异位置”指 **logprob 第一个不等的生成 token 下标**（从 0 开始）。token id 的分叉位置可能更晚，例如冷/热 compiled 对比中 logprob 在第 10 位就不同，但 argmax 直到第 14 位才翻转。

> “cold / 冷启动”指该配置第一次运行、`torch_compile_cache` 为空，进程内即时编译；“warm”指复用已有 `torch_compile_cache`，启动日志会显示 `Directly load the compiled graph(s) ... from the cache`。

| 比较 | single token | single logprob | 首个差异位置 | 最大差 | long token | long logprob | 首个差异位置 | 最大差 |
|---|---|---|---|---|---|---|---|---|
| compiled 同配置：冷启动 vs warm | ❌ | ❌ | 10 | **1.227e+00** | ✅ | ❌ | 4 | 4.768e-07 |
| compiled 同配置：warm #2 vs #3 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| compiled 同配置：2048 冷 vs warm #2 | ✅ | ❌ | 30 | 7.756e-03 | ✅ | ❌ | 4 | 5.960e-07 |
| compiled：8192 vs 2048（两者 warm） | ❌ | ❌ | 10 | **1.227e+00** | ✅ | ❌ | 4 | 7.153e-07 |
| compiled：PC on vs off（两者 warm） | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager 同配置 #1 vs #2 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager：8192 vs 2048 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager：PC on vs off | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| TP=1 同配置 ×2（Qwen3-0.6B） | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |

### 表 C：BI 关闭对照的详细数字

配置：compiled，chunk=8192，PC=1，TP=4，`VLLM_BATCH_INVARIANT=0`。

| 比较 | token id 相等 | logprob 位级相等 | 首个差异 | 最大绝对差 |
|---|---|---|---|---|
| bs=1 vs bs=32 | 是 | **否** | 第 2 个生成 token | **6.042e-02** |
| bs=1 vs 逆序 | 是 | **否** | 第 0 个生成 token | **7.360e-02** |
| bs=1 vs 重复请求 | 是 | 是 | — | 0 |
| decode vs re-feed | — | **否** | 第 0 个生成 token | **4.201e-02** |

这个对照非常重要：它说明本报告的测试对“非 BI 行为”是敏感的，表 A 里的 0.0 不是因为测试太弱。

### 表 D：耗时（秒，单次 `generate` 调用）

这些数字包含 prefix cache、warmup、CUDA graph 等影响，**不是严格 benchmark**，只用于量级对比。

| 运行 | single | batch32 | reverse32 | repeat | long | total |
|---|---|---|---|---|---|---|
| compiled 8192/PC1（冷启动首次编译） | 2.05 | 1.38 | 1.31 | 0.55 | 1.56 | 126.0 |
| compiled 8192/PC1（warm #2） | 0.92 | 1.19 | 1.21 | 0.73 | 1.58 | 59.0 |
| compiled 8192/PC1（warm #3） | 0.74 | 1.28 | 1.38 | 0.73 | 1.65 | 57.4 |
| compiled 2048/PC1（冷） | 0.77 | 1.21 | 1.29 | 0.57 | 1.81 | 106.1 |
| compiled 8192/PC0（冷） | 0.73 | 1.30 | 1.30 | 0.57 | 1.57 | 57.6 |
| eager 8192/PC1 #1 | 3.72 | 3.51 | 3.20 | 2.91 | 2.92 | 37.9 |
| eager 8192/PC1 #2 | 5.89 | 3.72 | 3.08 | 2.35 | 2.62 | 46.6 |
| eager 2048/PC1 | 3.11 | 2.79 | 3.08 | 2.88 | 3.21 | 34.5 |
| eager 8192/PC0 | 2.96 | 2.92 | 3.02 | 2.32 | 2.89 | 33.7 |
| BI=0 对照（compiled 8192/PC1） | 1.32 | 0.98 | 0.81 | 0.23 | 0.69 | 81.7 |
| TP=1 Qwen3-0.6B smoke | 0.99 | 0.70 | 0.36 | 0.37 | 0.48 | 24.7 |

量级结论：

- compiled warm 模式：single 约 0.7–0.9 s，batch32 约 1.2–1.3 s。
- eager 模式：single 约 3.0–5.9 s，batch32 约 2.8–3.7 s，约为 compiled 的 **2–3 倍慢**。
- BI 关闭的 compiled 模式：single 1.32 s，batch32 0.98 s；BI 开启的 compiled 模式与之接近，说明 BI 的主要代价不在这些短序列/小 batch 上，而在 eager 模式下才明显。

### 6.1 生成文本示例

目标 prompt：`The capital of France is`。

| 运行 | 生成文本（截断） |
|---|---|
| eager 8192/PC1（BI on） | ` Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. The capital of Brazil is Brasília. The` |
| compiled 2048/PC1（BI on） | 同上（`United States` 分支） |
| compiled 8192/PC1（冷启动） | 同上（`United States` 分支） |
| compiled 8192/PC1（warm #2） | ` Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome.` |
| BI=0 对照 | ` Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. ...` |

这说明：不同编译状态下的差异不只是最后几位 logprob，而是在一个接近“平局”的位置改变了 argmax，最终生成完全不同的后续句子。只看 token id 的测试很容易漏掉这种风险，必须同时比较 logprob。

### 6.2 服务端 API 冒烟测试

为了确认推荐的部署脚本可用，使用 `serve_bi_eager.sh` 启动了真实服务：

```text
VLLM_BATCH_INVARIANT=1
NCCL_IB_DISABLE=1
VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve ... --tensor-parallel-size 4 --enforce-eager --port 8000
```

- `/health` 返回 200；
- 单请求 `The capital of France is` 与“同一 prompt + 2 条 filler 的 batch”相比：
  - 生成文本完全一致；
  - `logprobs.tokens` 完全一致；
  - `logprobs.token_logprobs` 的 8 个值完全一致，`max abs diff = 0.0`。

```text
single text: ' Paris. The capital of the United Kingdom'
batch  text: ' Paris. The capital of the United Kingdom'
logprobs equal: True
max abs diff: 0.0
```

这说明第 8 节的 `vllm serve` 配置不仅能启动，而且在真实 HTTP API 路径上也保持了 batch 不变性。6.3 节把这一验证扩展为完整的服务端矩阵。

### 6.3 服务端在线 BI 矩阵（HTTP API）

#### 6.3.1 方法

按 5.5 的 H1–H6 依次启动真实 `vllm serve`，每个配置跑完即停，避免 GPU 争用。所有请求通过 `/v1/completions` 发出：

```json
{
  "prompt": [[785, 6722, 315, 9625, 374]],
  "max_tokens": 32,
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": 0,
  "logprobs": 5,
  "return_token_ids": true,
  "cache_salt": "case_single"
}
```

- `prompt` 用 `/tokenize` 返回的 token id，避免不同 tokenizer 配置带来的歧义；
- `return_token_ids=true` 取回生成 token；
- `logprobs.token_logprobs` 取回逐 token logprob；
- `prompt_logprobs=1` 用于 decode vs re-feed；
- `cache_salt` 保证每个 case 的 prefix cache 键独立；要制造缓存命中时，第二次请求复用第一次的 salt。

HTTP 返回的 logprob 是十进制文本。相同 float32 值会序列化成相同文本，再被客户端解析成相同 float64；因此表中 `0.0` 表示“两次请求返回的 logprob 文本完全一致”。这不是离线路径对原始 float32 内存的位比较，但对“服务端是否返回同一个数”是充分的。6.3.4 会用离线结果做一次交叉校验。

启动耗时：eager 各配置 `/health` 约 50 s；compiled 约 80 s（`torch_compile_cache` 命中，日志显示 `Directly load the compiled graph(s) ... from the cache, took 2.052 s`）。

#### 6.3.2 表 E：单进程内在线结果

列含义：H-a = bs1 vs bs32；H-b = 逆序；H-c = 短 prompt fresh vs cache hit；H-d = 长 prompt bs1 vs bs4（两边 fresh）；H-e = 长 prompt fresh vs cache hit；H-f = decode vs re-feed。✅ 表示 token id 全等且 logprob 文本全等。

| 配置 | H-a | H-b | H-c | H-d | H-e | H-f |
|---|---|---|---|---|---|---|
| eager 8192/PC1 run1 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | **2.384e-07 ❌** | 0 ✅ |
| eager 8192/PC1 run2 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | **2.384e-07 ❌** | 0 ✅ |
| eager 2048/PC1 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | **2.384e-07 ❌** | 0 ✅ |
| eager 8192/PC0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| BI=0 eager 8192/PC1 | **1.273 ❌**（token 分叉） | **0.1307 ❌** | 0 ✅ | **3.576e-07 ❌** | **3.576e-07 ❌** | **0.1311 ❌** |
| compiled 8192/PC1 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |

几个要点：

- **短 prompt 的全部在线测试都是 0.0**，包括 batch 大小、顺序、缓存命中和 decode/re-feed；compiled 与 eager 都一样。
- **长 prompt 的 batch 不变性在“两边都 fresh”时是 0.0**（H-d），说明 batch 组成本身没有破坏 BI。
- 唯一的例外是 **eager 下长 prompt 的 fresh vs cache hit**（H-e）：token id 仍然完全相同，但第 4 个生成 token 起 logprob 出现 2.384e-07 的差。两个独立进程、chunk 8192 和 2048 都复现同一数字。
- 关闭 prefix cache 后（H4/PC0），H-e 的两边实际上都是 fresh，结果回到 0.0。这直接把差异定位到 **缓存命中路径**，而不是 batch size 或 chunk size。
- compiled 模式下 H-e 也是 0.0；说明这个差异是 eager 的 attention/prefill 路径特有的（见 6.3.5 与 7.8）。

> BI=0 的长 prompt 行只有 3.576e-07，比短 prompt 的 1.273 小得多，不是因为长 prompt 更稳定，而是因为这条长 prompt 的下一个 token 几乎确定（logprob ≈ 0），差异被 softmax 饱和压缩了。6.3.5 用开放结尾 prompt 会把同一类差异放大到 1e-2–1e-1。

#### 6.3.3 表 F：在线跨进程 / 跨配置比较

比较双方都取各自配置下的 `single`（短 prompt）和 `long_single`（7600-token 长 prompt，fresh）。

| 比较 | 短 prompt token | 短 prompt logprob | 首个差异 | 最大差 | 长 prompt token | 长 prompt logprob | 首个差异 | 最大差 |
|---|---|---|---|---|---|---|---|---|
| eager run1 vs run2（同配置重启） | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager 8192 vs 2048 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager PC1 vs PC0 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| BI=1 vs BI=0（均为 eager 8192/PC1） | ✅ | ❌ | 0 | **1.364e-01** | ✅ | ❌ | 4 | **4.768e-07** |
| compiled vs eager（均 8192/PC1） | ❌ | ❌ | 0 | **1.334e+00** | ✅ | ❌ | 4 | **7.153e-07** |

结论与离线表 B 一致：

- **eager 跨进程、跨 chunk size、跨 PC 配置都是 0.0**；
- **compiled 与 eager 不同**：短 prompt 的 logprob 从第 0 个生成 token 起就不同，最大差 1.334，token id 在第 14 个分叉；长 prompt 虽然 token id 相同，logprob 仍差 7.15e-07；
- BI 关闭的对照在短 prompt 上 logprob 差 0.136，长 prompt 差 4.77e-07（长 prompt 的分布更饱和，logprob 差被压缩）。

compiled vs eager 的短 prompt 文本差异很直观：

| 配置 | 生成文本（截断） |
|---|---|
| eager 8192/PC1 | ` Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. ...` |
| compiled 8192/PC1 | ` Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. ...` |

两者在第 14 个生成 token 从 “ the United States” 分叉到 “ Germany”，与离线表 B 的翻转位置一致。

#### 6.3.4 离线与在线结果交叉校验

同一配置（eager 8192/PC1，BI=1，TP=4）下，把离线 `bi_eager_1.json` 与在线 H1 的返回逐 token 比较：

| 对象 | token id | logprob（32 个） | 最大差 |
|---|---|---|---|
| 短 prompt `The capital of France is` | 完全相同 | 逐 bit 相同 | **0.0** |
| 长 prompt（7600 token） | 完全相同 | 逐 bit 相同 | **0.0** |

也就是说，**HTTP 的十进制序列化没有在这组数据上引入可观察差异**；在线结果与离线引擎结果可以互相印证。这一点也解释了为什么 6.3.5 里 1e-7 量级的差异不是 JSON 精度造成的。

#### 6.3.5 表 G：开放结尾 prompt 的缓存命中诊断

6.3.2 的长 prompt 是重复文本，下一个 token 几乎确定（logprob ≈ 0），会把差异压缩到 1e-7。为了确认这不是“差异本来就很小”，用一条开放结尾的 7584-token prompt 复测：

```text
...（约 7570 token 的重复文本）...
Once upon a time, in a small village, there was a
```

每个配置依次发出：fresh A（salt X）、hit 1（salt X）、hit 2（salt X）、fresh B（salt Y）、hit 3（salt Y）。`/metrics` 的 `vllm:prefix_cache_hits_total` 确认每次 hit 复用了 **7568** 个 token。

| 配置 | 请求 | 首发 logprob | 生成 token（前 8 个） |
|---|---|---|---|
| eager | fresh A | **-1.929295** | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| eager | hit 1 | **-1.964247** | `[27699, 38835, 6941, 55923, 13, 55923, 572, 3881]` |
| eager | hit 2 | -1.964247 | 同 hit 1 |
| eager | fresh B | -1.929295 | 同 fresh A |
| compiled | fresh / hit | **-1.927184** | 全部相同 |

| 比较 | token id | logprob | 首个差异 | 最大差 |
|---|---|---|---|---|
| eager fresh A vs fresh B | ✅ | ✅ | — | **0** |
| eager hit 1 vs hit 2 | ✅ | ✅ | — | **0** |
| eager fresh vs hit | ❌ | ❌ | 0 | **5.970e-01** |
| compiled fresh vs hit | ✅ | ✅ | — | **0** |

关键数字：

- eager 首发 logprob 差：`-1.929295 - (-1.964247) = 0.034952`；
- 由于首发 token 的 argmax 改变，后续序列整体分叉，最大 logprob 差达到 0.597；
- 1900-token 版本同样存在：fresh `-1.563195` vs hit `-1.566805`，首发差 0.00361，最大差 0.0955；
- compiled 模式下 fresh 与 hit 全部 0.0，但 compiled 的首发 logprob（-1.927184）与 eager（-1.929295）本身不同，这是 6.3.3 已经看到的跨模式差异。

也就是说：**eager 的 prefix cache 命中路径与全量 prefill 路径不是同一条数值路径**；饱和 prompt 只暴露 1e-7 的尾巴，开放结尾 prompt 会暴露 1e-2–1e-1 的差异并翻转 token。compiled 模式把两条路径压到同一套 CUDA graph/kernel 上，所以没有这个现象。

#### 6.3.6 在线测试小结

1. **推荐部署脚本可用**：`/health` 正常，`/v1/completions` 的 token id、top logprobs、prompt logprobs 都能取回。
2. **短 prompt 的在线 BI 完全成立**：bs1/bs32、逆序、短 prompt 缓存命中、decode/re-feed 全部 0.0；跨进程重启、chunk 8192/2048、PC on/off 也全部 0.0。
3. **长 prompt 的 batch 不变性成立**：两边都 fresh 时 bs1 vs bs4 = 0.0。
4. **eager + 长 prompt + 缓存命中是例外**：token id 可能改变；必须把缓存状态当成一个输入变量，或关闭 prefix caching。
5. **compiled 单进程内全 0.0**，但与 eager 跨模式不同；如果要求跨重启位级一致，仍然选 eager。
6. **BI=0 对照失败**：在线 bs1/bs32 的 logprob 最大差 1.273，且 token id 分叉，证明测试足够敏感。

---

## 7. 关键发现与解释

### 7.1 同一引擎进程内：BI 成立（短 prompt 全项；长 prompt 的 fresh 路径）

这是本次验证最核心、最可靠的结论。

在表 A 中的 9 组 TP=4 BI 运行中（compiled/eager × chunk 8192/2048 × PC on/off；另有两组 warm 变体见表 D），以下三项全部为 **0.0**：

1. bs=1 vs bs=32；
2. 目标请求从 batch 第 0 位移到第 31 位；
3. 逐 token decode vs 重新 prefill（`prompt_logprobs`）。

第 3 项尤其强：它说明 prefill 路径和 decode 路径对同一个 token 的 logprob 计算是逐 bit 一致的，这通常是最容易出问题的部分。

短 prompt（5 token）的 prefix cache 命中 vs 未命中也是 **0.0**（表 A 第 3 列、在线表 E 的 H-c）。但这条结论**不能外推到长 prompt**：eager 下 7600-token 长 prompt 的“缓存命中 vs 全量 fresh”会出现 2.4e-7 到 0.035 的差异，甚至翻转 token（见 6.3.5、7.8）。表 A 的 prefix cache 列只覆盖短 prompt，这是它的边界。

### 7.2 跨进程：eager 是稳定解，compiled 不是

这是本次实验最有价值的“额外发现”，也是官方文档没有承诺、但对实际复现非常重要的部分。

- **eager 模式**（`enforce_eager=True`，关闭 torch.compile 和 CUDA graphs）：
  - 同配置两次进程：0.0；
  - chunk 8192 vs 2048：0.0；
  - prefix cache on vs off：0.0；
  - 与 TP=1 小模型的跨进程结果一致：0.0。

- **compiled 模式**（默认）：
  - warm 的同配置两次进程：0.0（说明同一编译产物下是稳定的）；
  - 但 **冷启动首次编译的进程** 与 warm 进程之间：token 在第 14 个生成 token 分叉，logprob 最大差 1.227；
  - 同配置 chunk=2048 的冷进程与 warm 进程之间：token id 相同，但第 30 个 token 的 logprob 差 7.756e-03；
  - chunk 8192 vs 2048（两者都 warm）：token 在第 14 个生成 token 分叉，logprob 最大差 1.227；
  - compiled 与 eager 之间也不同（短 prompt logprob 从第 0 个生成 token 起不同，最大差 1.334；token 在第 14 个分叉）。

**一个合理的假设**（未做进一步根因定位，谨慎表述）：

> 冷启动进程是在当前进程内即时编译，warm 进程是从 `torch_compile_cache` 反序列化加载同一份图。两者在 Inductor 的 kernel 选择/内存布局上可能并不完全等价；CUDA graph 的显存池布局也可能影响 NCCL all-reduce 的算法选择。于是同一个配置在不同编译状态下得到不同的数值路径，最终在接近平局的 token 上翻转 argmax。

不管是哪种具体机制，**工程结论是明确的**：

- 如果只要求“同一服务进程内 batch 不变”，compiled 模式足够；
- 如果要求“跨进程/跨重启/跨 chunk size 位级一致”，使用 `--enforce-eager`。

### 7.3 chunk size 的影响

在 eager 模式下，`max_num_batched_tokens=8192` 与 `2048` 的结果逐 bit 相同，包括 7600 token 的长 prompt。

在 compiled 模式下，两者在短 prompt 上就出现了 token 分叉（第 14 个生成 token），长 prompt 的 logprob 也有 7.153e-07 的差异。差异量级不大，但足以改变 argmax。

这说明：**BI 的 chunk-size 不变性在 eager 模式下成立；在 compiled 模式下被编译图/内存布局破坏了。** 官方文档没有承诺“跨 `max_num_batched_tokens` 配置”的位级一致，因此这不算文档违约，但对“跨配置复现”是实质风险。

### 7.4 prefix cache：配置本身不破坏 BI，但“命中路径”在长 prompt 上会

离线 warm 对比 PC on/off 都是 0.0（两边都是 fresh），说明**开不开 prefix caching 这个配置**不改变 fresh 路径的数值。之前看到的“PC 冷启动 run 不同”，本质是冷启动编译状态的问题。

但 6.3 的 H-e 和 6.3.5 表明：**eager 下“缓存命中”与“全量重新 prefill”不是同一条数值路径**。短 prompt 只有 5 个 token，缓存命中后几乎没有剩余 tail，所以看不出差异；长 prompt 命中的是前 7568/7584 个 token，只剩 16-token tail 需要 prefill，attention 的 q/kv 形状从“7600 个 query”变成“16 个 query 对 7584 个 cached key”，eager 会选到不同的 kernel/归约顺序，产生约 4e-3–3.5e-2 的 logit 偏移（1900 token 首发差 0.0036，7584 token 首发差 0.035）。这个偏移对饱和分布不可见（logprob ≈ 0，只剩 2.4e-7 的尾巴），对开放结尾的分布则足以翻转 argmax。

compiled 模式下 CUDA graph 把两条路径压到同一套捕获形状/kernel 上，所以 H-e 是 0.0。因此准确的说法是：**PC 配置本身不是问题，eager 长 prompt 的缓存命中路径是问题。**

### 7.5 BI 关闭对照证明测试足够敏感

BI 关闭时：

- bs=1 vs bs=32：第 2 个 token 起 logprob 不同，最大 6.042e-02；
- 逆序：第 0 个 token 起不同，最大 7.360e-02；
- decode vs re-feed：第 0 个 token 起不同，最大 4.201e-02。

有趣的是，这个短 prompt 的 **token id 仍然全部相同**，只是 logprob 不同。这再次说明：**只比较生成文本/token id 的 BI 测试是不充分的**，必须比较 logprob。

在线 BI=0 对照的差异更大（eager 8192/PC1，6.3.2）：bs=1 vs bs=32 不仅 logprob 最大差 **1.273**，token id 也直接分叉；逆序差 0.131，decode vs re-feed 差 0.131。两种模式都证明测试对非 BI 行为足够敏感。

### 7.6 环境坑：NCCL 2.28.9 在本机默认会 segfault

这是与 BI 无关、但部署必须知道的问题。

在本机上，任何 4-GPU（甚至 2-GPU）NCCL 初始化都会在 `ncclNetPluginInit` 处 segfault：

```text
!!!!!!! Segfault encountered !!!!!!!
  File "plugin/net.cc", line 216, in ncclNetPluginInit
  File "plugin/net.cc", line 362, in ncclNetInit(ncclComm*)
  ...
  File ".../nccl/src/init.cc", line 2055, in ncclCommInitRank
```

- 该主机没有 InfiniBand；
- 设置 **`NCCL_IB_DISABLE=1`** 后，4-GPU all-reduce 正常（本文所有 TP=4 结果都在该变量下产生）；
- `NCCL_NET=Socket` 也能绕开；
- 关闭 BI 的对照组同样会 segfault，因此**与 BI 无关**。

复现命令（最小 NCCL 测试）：

```bash
NCCL_IB_DISABLE=1 torchrun --nproc_per_node=4 nccl_test3.py
```

### 7.7 性能代价

从表 D 看：

- compiled warm：single ≈ 0.7–0.9 s，batch32 ≈ 1.2–1.3 s；
- eager：single ≈ 3.0–5.9 s，batch32 ≈ 2.8–3.7 s；
- 相对 compiled warm，batch32 大约慢 **2–3 倍**；single 因为 batch 很小、固定开销占比更高，慢得更多（约 4–7 倍）。

这不是严格 benchmark：测试只生成 32 个 token，且包含 prefix cache 命中、warmup 等影响。生产环境应按实际输入/输出长度重新测吞吐。

在线 HTTP 的耗时量级一致：eager 每个请求约 2.0–3.1 s（含长 prompt），compiled 约 0.7–1.8 s；BI=0 eager 与 BI=1 eager 接近。HTTP 开销（JSON、tokenize）相对模型前向很小。

### 7.8 eager 长 prompt 的缓存命中路径：本次在线测试的主要新发现

这是 6.3 相对离线矩阵新增的最重要结论，也是部署时最容易踩的坑。

**现象**：

- 同一个 7584-token prompt，在 eager + PC1 的服务上：
  - 第一次请求（fresh）首发 logprob = **-1.929295**，生成 token 为 A；
  - 第二次请求（同 salt，命中 7568 token 缓存）首发 logprob = **-1.964247**，生成 token 为 B；
  - 第三次请求（继续命中）与第二次完全相同；换一个 salt 重新 fresh，又回到 A；
  - `/metrics` 的 `prefix_cache_hits_total` 每次增加 7568，确认不是“假命中”。
- fresh vs hit 的首发 logprob 差 **0.034952**，token id 从第一个 token 起不同，最大 logprob 差 **0.597**。
- 1900-token 版本同样存在：fresh -1.563195 vs hit -1.566805，首发差 **0.00361**，最大差 **0.0955**。
- 6.3.2 的 7600-token 重复文本 prompt 因为下一个 token 几乎确定（logprob ≈ 0），只表现为 **2.384e-07** 的差，token id 不变；这正是“饱和分布掩盖小 logit 偏移”的典型例子。
- compiled 模式没有这个现象：fresh、hit、hit2 的首发 logprob 都是 -1.927184，全部 0.0。

**解释（假设，未逐 kernel 定位）**：

fresh 请求对整个 7584-token prompt 做一次 prefill；cache hit 只对最后 16 个未缓存 token 做一次短 prefill，前面 7568 个 token 走 paged KV。两者的 attention q/kv 形状和 kernel 选择不同，eager 下会产生约 4e-3–3.5e-2 的 logit 偏移（首发 logprob 差 0.0036–0.035）。compiled 模式的 CUDA graph 把形状补齐到捕获尺寸，两条路径走同一套图，所以数值一致。

**工程结论**：

1. 如果使用 **eager + prefix caching + 长 prompt**，不能假设“同一个 prompt 一定返回同一个 token”；缓存命中与否是一个会改变输出的隐藏变量。
2. 需要严格位级复现时，最稳妥的是 **`--enforce-eager --no-enable-prefix-caching`**；本报告的 H4/PC0 在线配置在长 prompt 上全部 0.0。
3. 如果必须保留 prefix caching，且只要求“单进程内 batch 不变”，可以用 compiled 模式；但它不保证跨进程/跨重启位级一致（冷/热编译、chunk size 都可能改变结果）。
4. 只比较生成文本或只看饱和 prompt 的 logprob，都会漏掉这个问题；要用开放结尾的 prompt 做回归。

---

## 8. 部署建议

### 8.1 严格位级可复现（推荐用于 RL rollout / 回归测试 / 审计）

```bash
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve $HOME/models/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-config.backend FLASH_ATTN \
  --enforce-eager \
  --port 8000
```

本目录中对应脚本：`bi_validation/serve_bi_eager.sh`。

要点：

- `--enforce-eager` 关闭 torch.compile 和 CUDA graphs；
- **`--no-enable-prefix-caching` 是本次在线测试后新增的推荐**：eager 下长 prompt 的缓存命中与全量 prefill 不是同一条数值路径，可能改变 token（7.8）。短 prompt 或只做 batch 不变性测试时，开 PC 也可以，但要么保证所有请求的缓存状态一致，要么接受这个隐藏变量；
- 所有配置必须固定：TP、`max_model_len`、`max_num_batched_tokens`、prefix cache、attention backend、dtype、seed；
- 如果要跨机器比较，还需要固定 GPU 型号、驱动、vLLM 版本、NCCL 版本。

### 8.2 高吞吐，只要“同一进程内 BI”

去掉 `--enforce-eager` 即可：

```bash
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve $HOME/models/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-config.backend FLASH_ATTN \
  --port 8000
```

适用：

- 在线服务，同一个进程长期运行；
- 关注“同一请求在不同 batch 组成下结果一致”；
- 必须保留 prefix caching，且 prompt 可能很长：compiled 单进程内连“长 prompt 缓存命中 vs fresh”都是 0.0（6.3.5）；
- 不要求跨重启/跨 chunk size 的位级复现。

注意：compiled 的“单进程内稳定”不等于“跨进程稳定”。它与 eager 之间在短 prompt 上首个 token 就分叉（1.334），所以不要把 compiled 服务和 eager 服务混在同一套复现基线里。

### 8.3 不建议的做法

- 只比较生成文本，不比较 logprob；
- 只用饱和/重复文本做长 prompt 回归：2.4e-7 的差异会掩盖 1e-2 量级的缓存命中差异（7.8）；
- 在 eager + prefix caching + 长 prompt 下，假设“同一个 prompt 必返回同一个 token”；
- 在不同 chunk size / 不同编译模式 / 不同 vLLM 版本之间做“位级复现”断言；
- 在随机采样下期待 BI 自动保证可复现（仍需固定 seed）；
- 在本机忘记设置 `NCCL_IB_DISABLE=1`，然后误以为是 BI 导致初始化失败。

---

## 9. 局限与后续工作

本次验证的边界：

- **单一硬件**：4×RTX A6000（SM86），未测试 A100/H100/Blackwell/RTX 5090 等；
- **单一模型**：Qwen3-30B-A3B BF16，未测试量化版、Thinking 版、其他 MoE/稠密模型；
- **单一 vLLM 版本**：0.23.0+cu129，未测试 0.32.0 的 breakable CUDA graph / 新版 custom all-reduce 路径；
- **单机 TP=4**：未测试 DP、EP、PP、多节点；
- **未测试**：LoRA、speculative decoding、structured output、其他 attention backend（如 `TRITON_ATTN`）；
- **在线测试只覆盖 `/v1/completions`**：没测 streaming、chat completions、tool calling、多轮对话模板；采样只测 greedy；
- **在线输出长度只有 32 token**：长生成下的缓存命中差异可能更大或更小，未做统计；
- **未做严格吞吐 benchmark**：表 D、表 E 只是量级参考；
- **跨进程差异的根因未完全定位**：目前只能归因于“编译/CUDA graph 状态或内存布局”，未逐层 dump kernel 选择；
- **长 prompt 缓存命中差异的根因未完全定位**：目前归因于“eager 下 fresh prefill 与 16-token tail prefill 的 attention kernel/归约路径不同”，未用 Nsight/torch profiler 逐 kernel 验证；
- **长 prompt 只测到 7600 token**：没有覆盖 8192 上限、滑动窗口模型或超长上下文。

建议的后续实验：

1. 在 vLLM 0.32.0 上重复同一矩阵，验证 breakable CUDA graph 是否修复编译模式的跨进程差异；
2. 用 `TRITON_ATTN` 替换 `FLASH_ATTN`，比较 BI 结果，并验证长 prompt 缓存命中差异是否与 attention backend 有关；
3. 在 A100 80GB（SM80）与 RTX 5090（SM120）上重复，验证跨 SM 的差异；
4. 用 `torch.compile` 的 `TORCH_LOGS=output_code` 对比冷/热进程生成的 kernel，定位差异来源；
5. 对 MoE 路由增加 hook，确认 token 分叉是否只来自 all-reduce 数值差；
6. 用 Nsight Systems / `torch.profiler` 对比“7600-token 全量 prefill”和“7568 cached + 16 tail prefill”实际启动的 attention kernel，确认 7.8 的假设；
7. 在 vLLM 0.32.0 或更新版本上复测 prefix cache 命中路径，确认是否已修复。

---

## 10. 复现步骤

### 10.1 下载模型

```bash
$HOME/miniconda3/envs/vllm/bin/modelscope download \
  --model Qwen/Qwen3-30B-A3B \
  --local_dir $HOME/models/Qwen3-30B-A3B
```

### 10.2 运行 BI 矩阵

```bash
cd $HOME
export NCCL_IB_DISABLE=1
bash run_all_bi.sh
```

`run_all_bi.sh` 会依次运行：

1. `CHUNK=8192, PC=1`
2. `CHUNK=2048, PC=1`
3. `CHUNK=8192, PC=0`
4. BI 关闭对照
5. 跨配置比较

### 10.3 运行 eager 复现实验

```bash
bash run_eager_repeat.sh
bash run_eager_variants.sh
```

### 10.4 运行在线 HTTP 矩阵

```bash
cd $HOME
export NCCL_IB_DISABLE=1
bash run_http_matrix.sh
```

脚本会依次启动 H1–H6 六个服务配置，每个配置等待 `/health` 后运行 `bi_http_test.py`，结果写到 `bi_http_<tag>.json`。运行前确认 4 张 GPU 空闲；脚本会检查显存并在配置之间释放进程。

缓存命中诊断（开放结尾 prompt）：

```bash
bash run_diag2.sh   # eager, port 8021
bash run_diag3.sh   # compiled, port 8022
```

### 10.5 关键输出字段

离线结果 JSON 包含：

- `env`：模型、TP、chunk、PC、BI 开关、attention backend、是否 eager；
- `single`：目标 prompt 的 token ids 与逐 token logprobs；
- `long_prompt`：长 prompt 的 token ids 与 logprobs；
- `comparisons`：同进程比较的 token/logprob 是否相等、首个差异位置、最大绝对差；
- `timing_sec`：各阶段耗时；
- `errors`：异常信息（本次全部为空）。

在线结果 JSON 额外包含：

- `single` / `long_single`：短 prompt 和长 prompt 的原始返回；
- `comparisons.server_*`：H-a 到 H-f 的逐项比较；
- `env.tag`：H1–H6 的配置标识；
- `timing_sec`：每个 HTTP 请求的耗时。

### 10.6 文件清单

| 文件 | 说明 |
|---|---|
| `bi_test_qwen3_30b.py` | 离线 BI 测试脚本 |
| `compare_bi.py` | 两个离线结果 JSON 的位级比较脚本 |
| `run_all_bi.sh` | compiled 模式测试矩阵 |
| `run_eager_repeat.sh` | eager 同配置重复实验 |
| `run_eager_variants.sh` | eager chunk/PC 变体实验 |
| `serve_bi_eager.sh` | 严格 BI 的 `vllm serve` 启动脚本 |
| `bi_http_test.py` | 在线 HTTP BI 测试脚本 |
| `run_http_matrix.sh` | H1–H6 在线矩阵编排脚本 |
| `bi_long_diag.py` / `bi_long_diag2.py` | 开放结尾长 prompt 的缓存命中诊断 |
| `run_diag.sh` / `run_diag2.sh` / `run_diag3.sh` | 长 prompt 缓存命中诊断（1900/7600，eager 7584，compiled 7584） |
| `results/*.json` | 离线原始结果 |
| `results/http/*.json` | 在线原始结果与诊断结果 |
| `results/http/analysis_tables.md` | 自动生成的在线表 E–H |
| `results/analysis_tables.md` | 自动生成的离线表 A–D |

---

## 11. 参考资料

1. vLLM 0.23.0 Batch Invariance 文档（硬件要求、tested models、实现说明）  
   https://docs.vllm.ai/en/v0.23.0/features/batch_invariance.html
2. vLLM 0.32.0 Batch Invariance 文档（TP all-reduce 16 字节约束、breakable CUDA graph、更多 tested models）  
   https://docs.vllm.ai/en/v0.32.0/features/batch_invariance.html
3. vLLM BI 追踪 issue #27433  
   https://github.com/vllm-project/vllm/issues/27433
4. Qwen3-30B-A3B 模型卡（Hugging Face）  
   https://huggingface.co/Qwen/Qwen3-30B-A3B
5. Qwen3-30B-A3B（ModelScope）  
   https://modelscope.cn/models/Qwen/Qwen3-30B-A3B
6. NVIDIA RTX A6000 产品页  
   https://www.nvidia.com/en-us/design-visualization/rtx-a6000/
7. NVIDIA CUDA GPU Compute Capability 列表  
   https://developer.nvidia.com/cuda-gpus
8. FlashAttention 仓库  
   https://github.com/Dao-AILab/flash-attention
9. vLLM 仓库  
   https://github.com/vllm-project/vllm

---

## 附录 A：启动日志关键摘录

### A.1 BI 在 worker 中启用

```text
Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::mm(Tensor self, Tensor mat2) -> Tensor
  dispatch key: CUDA
  new kernel: registered at .../vllm/model_executor/layers/batch_invariant.py:905
```

### A.2 TP=4 与 all-reduce 后端

```text
world_size=4 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:55127 backend=nccl
world_size=4 rank=1 local_rank=1 ...
world_size=4 rank=2 local_rank=2 ...
world_size=4 rank=3 local_rank=3 ...
vLLM is using nccl==2.28.9
Using ['PYNCCL'] all-reduce backends (in dispatch order) ...
  out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']
```

### A.3 Attention / MoE 后端

```text
Using AttentionBackendEnum.FLASH_ATTN backend.
Using FlashAttention version 2
Using TRITON Unquantized MoE backend out of potential backends:
  ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']
Using MoEPrepareAndFinalizeNoDPEPModular
```

### A.4 显存与 KV cache

```text
Model loading took 14.25 GiB memory and 7.03 seconds
GPU KV cache size: 1,066,720 tokens
Maximum concurrency for 8,192 tokens per request: 130.21x
```

### A.5 编译与图捕获

```text
torch.compile and initial profiling/warmup run together took 70.37 s in total
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 51/51
Capturing CUDA graphs (FULL): 100%|...| 35/35
```

---

## 附录 B：原始结果文件 SHA256

| 文件 | SHA256 |
|---|---|
| `bi_results_chunk8192_pc1.json` | `621fb9110a847f367cca3c96470ba434dc9424634c5bf11ceeb910341a45a674` |
| `bi_results_chunk8192_pc1_repeat.json` | `159ac378f6cf5dc631f1ad0771ba5a44ab956c7a58470f25bea5069d63b9a0cf` |
| `bi_results_chunk2048_pc1.json` | `8699aeb8054626bcbbe228acb3c935c9e79c4678725b7c7be42e897b8c3ec2b5` |
| `bi_results_chunk8192_pc0.json` | `c9e6508f02e9bb50e67e276b92cd9b503b778d2d568015b4e17016e75e0f68b8` |
| `bi_off_chunk8192_pc1.json` | `0bf2ef030fd6637391565bf50e13a1d2b7ec1689d1c8f048c52b57c890a2226a` |
| `bi_eager_1.json` | `d93c4e7ce5b212b04c1ba227bda0f16ca8a142c80bf6697ad1ab781d973399d0` |
| `bi_eager_2.json` | `9d2284fd270916fde6a89b4e778c52608a6aaed51773ede9e6c51aef7c44cdbc` |
| `bi_eager_chunk2048.json` | `e6787fdd07fd2a38af1e30a7a3a0b144546e088fe1c8e165f1b3f1d8f2d5e07b` |
| `bi_eager_pc0.json` | `119c0880f70c96e75248f039310a14f262f157a10681cd8a80c61c693c21ebe1` |
| `bi_compiled_warm_2.json` | `e6cae91e33ff5fc811245745d1383c77bf3f0e17544b8783c72584953b406d1c` |
| `bi_compiled_warm_3.json` | `6938c6e52d00f23ae24e57fdc6c88ae0f3922f449dadbff42ce50892362c8042` |
| `bi_compiled_chunk2048_warm2.json` | `0303710cdb92ce0832bedc498608a436b2d57c738e3f89455dbee00860be4275` |
| `bi_compiled_pc0_warm2.json` | `eff378ec2a5281f808ddc27ccf99d01665db19dbc81d0391002572fb6cb769ee` |
| `bi_smoke_qwen3_06b.json` | `4cdb3fb1496104e5eb70b48b963997bd2de6b081895ba19dad826ff5d1a9aea7` |
| `bi_smoke_qwen3_06b_repeat.json` | `480d3825fa5b38d4126106f1da3ea61d068946596f11ddd7d6618ac051d5da01` |

在线 HTTP 结果（`results/http/`）：

| 文件 | SHA256 |
|---|---|
| `bi_http_eager_8192_pc1_run1.json` | `56ff8db6759c9e48ba1b2f41b258debdb9add640eb20f495650ef7f4a27a2c77` |
| `bi_http_eager_8192_pc1_run2.json` | `787b6584d81bc45aca1c52507a4c35272c1d2bc5d5d806ec7d5508eaaa4c01cb` |
| `bi_http_eager_2048_pc1.json` | `bfd54b6ff97a3535be1eb417e1e56cba03c6063c0b455fded6d5e61be143779a` |
| `bi_http_eager_8192_pc0.json` | `3c3a37e3e23b1536ba32b34365f10e01283b505f2f3e964042e258523cc61636` |
| `bi_http_bi0_eager_8192_pc1.json` | `83a7d155281239c0bf80e2c431e617c6b0a7b5f230e452d7641704e1f1c2f7a6` |
| `bi_http_compiled_8192_pc1.json` | `b43cce0a9188f0885262280573658d7636d4e3e16e6fda04a1aed8b80b582778` |
| `bi_long_diag.json`（eager，1900/7600） | `2e402013e2dcd9171c83653af2108caa56d0b6585fa63c4e57cbce3b6ee85302` |
| `bi_long_diag2.json`（eager，7584 hit 诊断） | `4bfd1bf3d3f7e7301f8bf2ccb33e6e7941fee10fb544c94e3b3545c7e7cde528` |
| `bi_long_diag3.json`（compiled，7584 hit 诊断） | `c451db050f597b471f33e85eea1e9d697bd4cd5976923802efe70855c84b0ef6` |

---

## 附录 C：HTTP 在线测试的精度说明

HTTP 返回的 logprob 是十进制文本，不是内存位模式。因此 6.3 表中的 `0.0` 只能说明“两次请求返回的十进制值完全相同”，不能单独替代离线位比较；HTTP 层也可能对 `NaN`、`-0.0`、次正规数做特殊处理。

本次用同配置的离线结果做了交叉校验（6.3.4）：短 prompt 和 7600-token 长 prompt 的 token id 与 32 个 logprob 都逐 bit 相同。说明在这个模型、这段数值范围内，float32 → 最短十进制文本 → float64 的往返没有引入可观察差异，在线表中的 2.384e-07 等差异也不是 JSON 序列化造成的。对于 `NaN`、`-0.0`、次正规数等边界值，仍以离线 `RequestOutput` 的位比较为准。

---

*报告结束。*
