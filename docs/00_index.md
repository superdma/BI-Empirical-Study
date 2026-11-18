# Qwen3-30B-A3B × 4×RTX A6000 × vLLM 0.23.0

## 批次不变性（Batch Invariance, BI）实测文档集

- 版本：2.0
- 日期：2026-11-17
- 主机：4×NVIDIA RTX A6000 48 GB（SM86），无 InfiniBand
- 软件：vLLM 0.23.0+cu129，PyTorch 2.11.0+cu129，CUDA 12.9，NCCL 2.28.9
- 模型：`Qwen/Qwen3-30B-A3B`，BF16，TP=4
- 开关：`VLLM_BATCH_INVARIANT=1`

> **v2 更正声明**
>
> v1 单文件报告的“短 prompt 位级一致”结论，只对测试中那一条 5-token 的 target
> prompt 成立。v2 把比较对象扩展到 32 条不同的普通 prompt，发现 eager 模式下
> **31/32 条**的单独请求与批内请求不一致（logprobs 不同，部分 token id 也不同）。
> compiled 模式（`torch.compile` + CUDA Graphs）在同一组测试中 **32/32 一致**。
> v1 的 target-only 结论不再作为部署依据；本目录以 v2 为准。

---

## 一页结论

### 1. 可部署的 BI 配置

在本次硬件与模型上，**compiled 模式是唯一通过全部进程内测试的配置**：

```bash
# 不加 --enforce-eager；其余严格性参数照旧
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
  --port 8000
```

compiled 模式在以下场景全部位级一致（0 差异）：

- 32 条不同 prompt：单独请求 vs 一次批量提交；
- 同样的 32 条：独立并发请求、突发请求、另一请求正在解码时的突发；
- 批内 prompt 数 8/16/32，单步 prefill 总 token 160/253/284/455（N=32 的
  L=5/8/9 与 N=16 的 L=30）；
- 批大小 256（decode），以及 256-token 生成后的 decode vs re-feed；
- 同一配置重复批请求；跨 3 个 warm-cache 服务进程的单独请求输出。

### 2. eager 模式的真实边界

eager（`--enforce-eager`）的 BI 只在部分形状上成立。本次测到：

- 32 条普通 prompt 一起提交，**只有第一条（5-token target）单独 vs 批内一致，其余 31 条不一致**；
- 单步 prefill 总 token 为 253 时 32/32 一致；284 时 1/32 一致；
- 16 条 × 30 token（总 455）时 2/16 一致；
- 8 条 × 40 token（总 285）时 8/8 一致；
- decode 批大小 255 一致，256 开始 logprobs 偏离；
- 生成 192 token 后 re-feed 仍一致，生成 256 token（总长 261）后 48/256 一致；
- 换 `TRITON_ATTN`、关 chunked prefill、换显式 `triton` MoE 后端、换稠密
  Qwen3-0.6B，均不能消除 31/32 的失败；换 compiled 才能消除。

eager 的 BI 只在小步、少序列、同步提交时成立。此前所有
target-only 的 0.0 结果都是真的，但不能外推到其他 prompt。离线
`LLM.generate` 复核得到同样结果：31/32 不一致，且两次相同批请求之间有 1 条
不一致（说明 eager 连“重复同一批”都不保证）。

vLLM 文档给出了一个解释：BI 支持 compute capability ≥8.0，但实现细节写明
“tuned matmul table entries”只覆盖 **Ada / Hopper / Blackwell**。A6000 是
Ampere（SM86），名义上能开 BI，却不在官方 tuned 路径里。官方调优覆盖的
平台是 L40S、H100、B200 这一类。

### 3. compiled 与 eager 不能混用

compiled 与 eager 对同一个 prompt 的输出从第一个 token 起就不同：

- target prompt：token id 不同，最大 logprob 差约 `1.57e-01`；
- 32 条 prompt 的单独基线：compiled vs eager **32/32 不同**；
- 但 compiled 在 3 个独立 warm-cache 进程之间 **32/32 相同**，eager 在 3 个
  独立进程之间也 **32/32 相同**。

两种模式各自是确定的，但数值路径不同。选一种并固定，不能拿 compiled
的结果去对 eager 的日志，也不能在升级/重编译后假设输出不变。

### 4. prefix cache 与长 prompt

- target（5-token）短 prompt 的缓存命中在 eager/compiled 下都可位级一致；
- eager + 长 prompt + 缓存命中仍是最危险的组合：7600-token 饱和 prompt 有
  `2.384e-07` 的差异，开放结尾 7584-token prompt 会改变第一个生成 token；
- 8180-token 近最大上下文在本次测试中 PC0/PC1 都恰好一致，说明“长 prompt 必坏”
  也不成立，结果依赖具体的 chunk/tail 几何；
- compiled 在 7584-token 开放结尾的诊断中，fresh 与 cache hit 全部一致。

### 5. 性能

compiled 的吞吐也更高：

- 单流 400-token 生成：eager 约 25 s，compiled 约 8 s；
- 32-prompt 批量请求：eager 约 1.3 s，compiled 约 0.6 s。

### 6. 换成 SGLang

同样 32 条 prompt，SGLang 0.5.6 的结果取决于并行方式：

| 配置 | 32-prompt 位级一致 |
|---|---|
| `--enable-deterministic-inference`，TP=4 | 7/32 |
| 同上，attention 换 triton | 15/32 |
| `--enable-deterministic-inference`，PP=4 | **32/32** |

跨卡归约是分界点：同一个稠密小模型，TP=1 与 TP=2 通过，TP=4 失败，PP=4 通过。
PP=4 的在线四类场景也是 96/96 一致，代价是长请求耗时从 7.9 s 涨到 17.6 s。
细节见 [12_sglang_results.md](12_sglang_results.md)。

---

## 文档地图

| 文件 | 内容 |
|---|---|
| [01_scope.md](01_scope.md) | BI 的定义、比较口径、能保证与不能保证什么 |
| [02_environment.md](02_environment.md) | 硬件、软件、模型、环境坑与理论依据 |
| [03_design.md](03_design.md) | 离线/在线测试设计，T1–T10、H1–H6、B1–B12 矩阵 |
| [04_offline_results.md](04_offline_results.md) | v1 离线结果（target-only）及其适用范围 |
| [05_online_results.md](05_online_results.md) | v1 在线 H1–H6 结果与“批量提交”口径 |
| [06_boundary_results.md](06_boundary_results.md) | v2 新增：32-prompt、长度、并发、256 边界、re-feed |
| [12_sglang_results.md](12_sglang_results.md) | SGLang 0.5.6 复现：TP=4 与 PP=4 的对照 |
| [07_findings.md](07_findings.md) | 发现 F1–F13：事实、解释、证据强度 |
| [08_deployment.md](08_deployment.md) | 部署决策树、脚本、反模式、监控 |
| [09_limitations.md](09_limitations.md) | 局限、未解释现象、后续工作 |
| [10_reproduction.md](10_reproduction.md) | 复现命令、文件清单、SHA256 |
| [11_references.md](11_references.md) | 官方文档、源码、模型卡与相关链接 |
| [appendix_a_logs.md](appendix_a_logs.md) | 启动日志关键摘录 |
| [appendix_b_precision.md](appendix_b_precision.md) | 位级比较与 HTTP 精度说明 |

原始数据：

- `results/*.json`：v1 离线结果；
- `results/http/*.json`：v1 在线 H1–H6；
- `results/extra/*.json`：批大小、TP、attention 后端、re-feed 等；
- `results/http/concurrency/*.json`：32-prompt、并发、突发、长度扫描；
- `results/extra/analysis_tables.md`、`results/http/concurrency/analysis_tables.md`：
  自动生成的汇总表；
- `sglang/results/*.json`：SGLang 复现的原始数据，脚本在 `sglang/` 下。

---

## 怎么读

- 要做部署决策：读 00 → 08 → 06 的表 M/N/O/P。
- 要复现实验：读 03 → 10，按脚本顺序跑。
- 要理解为什么：读 01 → 02 → 07 → 09。
- 要核对旧结论：读 04/05，注意它们全部是 target-only 口径。
