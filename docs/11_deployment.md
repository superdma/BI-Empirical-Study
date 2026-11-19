# 11. 部署建议

## 11.1 先选目标：你要哪一种“可复现”？

| 目标 | 推荐 | 原因 |
|---|---|---|
| 在线服务，要求不同并发/到达方式下同一请求输出一致 | **compiled + 固定编译产物** | 本次唯一通过 32/32、并发、256 边界、re-feed 的模式 |
| 离线 RL rollout / 打分，批次形状固定且很小 | eager 可作为候选，但必须逐 prompt 验证 | 小步（≤16 条、总 token ≤253）通过；一旦跨形状就失败 |
| 跨进程/跨机器严格一致 | compiled + 固定版本/缓存/配置；或固定 eager + 固定 prompt 集并验证 | compiled 跨 warm-cache 进程稳定，但 cold/warm 与 chunk 会变 |
| 只追求吞吐，不要求 BI | 默认 compiled（不要开 `--enforce-eager`） | 本次 compiled 还快 2–3 倍 |

**不要把“v1 的 target-only 0.0”当作通用 BI 证明。** 先跑 11.4 的验证。

另外，vLLM 官方文档的 tuned matmul 表只覆盖 Ada/Hopper/Blackwell；A6000 是
Ampere，名义支持但不在 tuned 路径。这也是为什么本机需要“实测通过”而不是
“文档说支持”的原因。若你在 L40S/H100/B200 上部署，行为可能不同，仍需重跑
11.4。

## 11.2 推荐配置 A：compiled（通用 BI）

```bash
#!/usr/bin/env bash
set -euo pipefail

export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-$HOME/models/Qwen3-30B-A3B}"
PORT="${PORT:-8000}"

exec $HOME/miniconda3/envs/vllm_update/bin/vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-config.backend FLASH_ATTN \
  --port "$PORT"
```

配置要点：

- **不要加 `--enforce-eager`**；
- 固定 `max-model-len`、`max-num-batched-tokens`、TP、attention backend；
- 固定 vLLM/PyTorch/CUDA/NCCL 版本；
- 固定编译缓存（见 11.3）；
- 建议 PC off：compiled 在本次长 prompt 诊断中 fresh/hit 一致，但关掉可以
  少一个变量；如果吞吐需要 PC，请用长 prompt 场景单独验证。

对应的本地脚本：`serve_bi_compiled.sh`。

## 11.3 固定编译产物

compiled 的 BI 只在**同一份编译产物**内成立。v1 观察到：

- cold compile vs warm：target 第 10 位 logprob 不同，第 14 位 argmax 翻转；
- chunk 8192 vs 2048（均 warm）：同样会改变 token 序列；
- warm #2 vs #3：一致。

建议：

1. 构建镜像时预热一次，把 `VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` /
   `TRITON_CACHE_DIR` 一起打进镜像；
2. 运行时不要清缓存，不要改 chunk/TP/attention backend；
3. 升级任何组件后，重新跑 11.4 的验证并重新基线；
4. 如果无法固定缓存，把“compiled 的首次冷启动输出”视为一个独立版本，
   不要与 warm 结果比较。

## 11.4 部署前验证清单（必须逐 prompt，不是只看 target）

### 1）32-prompt 批不变性

```bash
BASE=http://127.0.0.1:8000 N=32 \
  python bi_http_batchall.py
```

判定：`solo_vs_batch_bad == 0` 且 `batch_repeat_bad == 0`。

本机结果：

- eager：1/32；
- compiled：32/32。

### 2）真实并发

```bash
BASE=http://127.0.0.1:8000 N=32 \
  python bi_http_concurrent.py
```

判定：所有 `prompt*_seq_vs_concurrent` 的 `ids` 与 `bitwise` 都为 true。

本机结果：

- eager PC0/PC1：1/32；
- compiled PC1：32/32。

### 3）长度边界

```bash
BASE=http://127.0.0.1:8000 N=32 L=9 \
  python bi_http_len.py
```

判定：`bad == 0`。`L=9`、N=32 时总 prefill 约 284 token，是 eager 的失败区；
compiled 应通过。

### 4）decode vs re-feed

生成长度 ≥256 时，用 `bi_refeed_diag.py` 检查 `n_mismatch == 0`。

### 5）长 prompt / 缓存

如果开 PC，用 `bi_long_diag2.py` 检查 fresh vs hit；eager 下 7584-token 开放
结尾会翻转首 token。

## 11.5 配置 B：eager（仅限小步、同步、已验证的 prompt 集）

```bash
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-$HOME/models/Qwen3-30B-A3B}"
exec $HOME/miniconda3/envs/vllm_update/bin/vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-config.backend FLASH_ATTN \
  --enforce-eager \
  --max-num-seqs 128 \
  --port 8000
```

说明：

- `--max-num-seqs 128` 能把 bs=256 的 decode 失败修掉（表 J），但**不能**
  保证 32 条 prompt 的 prefill 失败消失；它只是把并发波次变小；
- eager 跨进程对同一个（已验证的）prompt 是稳定的（表 B），适合“固定小批、固定 prompt 集”
  的离线任务；
- 必须用 11.4 的 32-prompt 验证你的实际 prompt 集；如果 1/32，不要用 eager。

## 11.6 反模式

| 反模式 | 为什么错 |
|---|---|
| 只测一条 target prompt | v1 的 1/32 就是这样漏掉的 |
| 把“一个请求里 32 条 prompt”当作“32 路并发” | 两者都不等于真实到达，但失败机制不同 |
| 用 token id 相同判定 BI 通过 | logprob 可能已经差 0.09–2.7 |
| 认为 `--enforce-eager` 更“严格” | 本次相反：eager 失败，compiled 通过 |
| 在 eager 下开 PC 跑长 prompt | 7584-token 会翻转首 token |
| 清编译缓存后直接对比 | cold vs warm 是不同数值函数 |
| 混用 compiled 与 eager 的日志/分数 | 两者首 token 就不同 |
| 把 `max_num_seqs=128` 当通用修复 | 只修 decode bs256，不修 prefill 形状问题 |

## 11.7 监控建议

- 记录每次请求的 `token_ids` + 每个 token 的 logprob（至少对审计样本）；
- 定期用固定 prompt 集做“影子请求”，比较 solo vs 当前批内输出；
- 在升级 vLLM/PyTorch/CUDA、改 TP/chunk/attention backend、清缓存后，
  强制重跑 11.4；
- 对长 prompt + PC 的命中率单独监控；命中率变化可能意味着 tail 几何变化。

## 11.8 结论

**要 BI：compiled，固定产物，逐 prompt 验证；eager 只在你能证明的小形状内使用。**
