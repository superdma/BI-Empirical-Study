# 10. 复现步骤与文件清单

## 10.1 前置条件

- 4×RTX A6000 48 GB（或等价 SM86 机器）；
- vLLM 0.23.0+cu129，PyTorch 2.11.0+cu129，CUDA 12.9；
- 模型 `$HOME/models/Qwen3-30B-A3B`；
- 环境 `$HOME/miniconda3/envs/vllm_update`；
- 无 InfiniBand。

```bash
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
```

## 10.2 v1：离线矩阵

```bash
cd BI-Empirical-Study
bash run_all_bi.sh
```

产出：`results/bi_*.json`，以及 `results/analysis_tables.md`（表 A–D）。脚本默认把新结果写到 `$HOME`；可用 `OUT=` 指到 `results/`。

eager fallback 复现（仅限已验证的小批）：

```bash
bash serve_bi_eager.sh
```

eager 只建议用于小批、同步、已验证的 prompt 集；通用 BI 用
`serve_bi_compiled.sh`。

## 10.3 v1：在线 H1–H6

```bash
bash run_http_matrix.sh
```

六个配置依次启停：eager 8192/PC1 ×2、eager 2048/PC1、eager 8192/PC0、
BI=0、compiled 8192/PC1。

产出：`results/http/bi_http_*.json` 与
`results/http/analysis_tables.md`（表 E–H）。

## 10.4 v1：长 prompt 缓存命中诊断

```bash
bash run_diag2.sh    # eager 1900 / 7600
bash run_diag3.sh    # eager / compiled 7584
```

产出：`results/http/bi_long_probe*.json` 等。

## 10.5 v2：批大小、TP、attention、re-feed

```bash
# 12 个配置的批大小/TP/attention/chunk/长 prompt 矩阵
bash run_extra_matrix.sh

# bs=256 条件隔离
bash run_diag_bs256.sh
bash run_diag_bs256b.sh

# 边界与采样
bash run_diag_bs256c.sh

# 256-token re-feed 扫描
bash run_refeed_b.sh
```

产出：`results/extra/bi_extra_*.json`、`results/extra/bi_diag_*.json`、
`results/extra/bi_refeed_*.json`。

汇总：

```bash
python analyze_extra.py
# -> results/extra/analysis_tables.md（表 I–L、Q）
```

## 10.6 v2：32-prompt、并发、突发、长度

```bash
# 单次批量请求：逐 prompt 单独 vs 批内
bash run_batchall.sh          # eager + compiled
bash run_batchall2.sh         # TRITON_ATTN、无 chunked

# 独立并发
bash run_concurrency.sh       # eager PC0/PC1、compiled PC1、长 filler

# 突发 / 背景解码
bash run_burst.sh             # eager PC0、compiled PC1

# 单个请求插入解码
bash run_mixed.sh             # eager chunked/nochunk、compiled

# 批内长度扫描
bash run_len.sh               # eager：N=32/L=5,8,9; N=16/30; N=8/40
bash run_len_compiled.sh      # compiled 对照
```

产出：`results/http/concurrency/*.json`。

汇总：

```bash
python analyze_online_extra.py
# -> results/http/concurrency/analysis_tables.md（表 M–P）
```

## 10.7 判读标准

| 字段 | 通过条件 |
|---|---|
| `ids` / `token_ids_equal` | true |
| `bitwise` / `logprobs_bitwise_equal` | true |
| `max` / `max_abs_logprob_diff` | 0.0 |
| `repeat_bad` | 0 |
| `solo_vs_batch_bad` | 0 |
| `n_mismatch`（re-feed） | 0 |

任何“小于某个阈值”的判读都不算通过；本报告要求位模式相同。

## 10.8 文件清单

### 脚本

| 文件 | 用途 |
|---|---|
| `bi_test_qwen3_30b.py` | v1 离线 BI harness |
| `compare_bi.py` | 两份离线 JSON 的位级比较 |
| `run_all_bi.sh` | v1 离线矩阵 |
| `serve_bi_eager.sh` | eager 服务（小批 fallback，默认 MNS=128） |
| `serve_bi_compiled.sh` | 推荐的通用 BI 服务 |
| `bi_http_test.py` | v1 在线 H1–H6 |
| `run_http_matrix.sh` | v1 在线矩阵 |
| `bi_long_diag.py` / `bi_long_diag2.py` | 长 prompt 缓存诊断 |
| `run_diag.sh` / `run_diag2.sh` / `run_diag3.sh` | 上述诊断的启动脚本 |
| `bi_extra_qwen3_30b.py` | v2 批大小/TP/attention 矩阵 |
| `bi_extra_diag.py` | v2 bs256 条件隔离 |
| `run_extra_matrix.sh` | v2 12 配置矩阵 |
| `run_diag_bs256.sh` / `b` / `c` | v2 bs256 诊断 |
| `bi_refeed_diag.py` | decode vs re-feed 长度扫描 |
| `run_refeed_b.sh` | 上述扫描的启动脚本 |
| `bi_http_concurrent.py` | 32 路独立并发 |
| `run_concurrency.sh` | 并发矩阵 |
| `bi_http_mixed.py` | 单请求插入解码 |
| `run_mixed.sh` | 混合矩阵 |
| `bi_http_burst.py` | 突发 32 请求 |
| `run_burst.sh` | 突发矩阵 |
| `bi_http_batchall.py` | 32-prompt 单独 vs 批内 |
| `run_batchall.sh` / `run_batchall2.sh` | 批量与后端对照 |
| `bi_http_len.py` | 批内长度扫描 |
| `run_len.sh` / `run_len_compiled.sh` | 长度扫描 |
| `bi_batchall_offline.py` | 离线 32-prompt 复核 |
| `analyze_extra.py` | 表 I–L、Q |
| `analyze_online_extra.py` | 表 M–P |

### 结果目录

| 路径 | 内容 |
|---|---|
| `results/*.json` | v1 离线 |
| `results/http/*.json` | v1 在线、长 prompt 诊断 |
| `results/extra/*.json` | 批大小、TP、attention、re-feed、离线 32-prompt |
| `results/http/concurrency/*.json` | 32-prompt、并发、突发、混合、长度 |
| `results/extra/analysis_tables.md` | 表 I–L、Q |
| `results/http/concurrency/analysis_tables.md` | 表 M–P |

### 文档

| 文件 | 内容 |
|---|---|
| `docs/00_index.md` | 索引与一页结论 |
| `docs/01_scope.md` | 定义与口径 |
| `docs/02_environment.md` | 环境与理论 |
| `docs/03_design.md` | 设计 |
| `docs/04_offline_results.md` | v1 离线 |
| `docs/05_online_results.md` | v1 在线 |
| `docs/06_boundary_results.md` | v2 边界 |
| `docs/07_findings.md` | F1–F12 |
| `docs/08_deployment.md` | 部署 |
| `docs/09_limitations.md` | 局限 |
| `docs/10_reproduction.md` | 本文 |
| `docs/11_references.md` | 参考 |
| `docs/appendix_a_logs.md` | 日志 |
| `docs/appendix_b_precision.md` | 精度 |

## 10.9 生成 SHA256

```bash
cd bi_validation
find results -name '*.json' -print0 | sort -z | \
  xargs -0 sha256sum > results/SHA256SUMS
```

结果哈希见 [appendix_b_precision.md](appendix_b_precision.md)。
