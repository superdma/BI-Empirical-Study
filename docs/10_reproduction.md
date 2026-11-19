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
| `bi_allreduce_bench.py` | 不加载模型的 NCCL all-reduce 形状探针 |
| `run_nccl_sweep.sh` | 固定 `NCCL_ALGO` 重跑 32-prompt |
| `bi_tail_scan.py` | 前缀缓存命中 vs fresh 的 tail 长度扫描 |
| `bi_bucket_scan.py` | CUDA graph bucket 边界的批大小扫描 |
| `run_tier1.sh` | 上述四项按顺序跑一遍 |
| `analyze_tier1.py` | 把四项结果渲染成表 T1–T4 |
| `bi_bi_coverage.py` | BI 覆盖了哪些 aten 算子、tile 配置随 M 的扫描 |
| `bi_matmul_shape.py` | 线性层单独跑：同一行在不同 M 下是否逐位相同 |
| `bi_op_shape.py` | norm / silu / rotary / MoE 单独跑，含对齐对照 |
| `bi_attn_shape.py` | 分页 decode attention 单独跑 |
| `bi_pynccl_bench.py` | 用引擎自己的 pynccl 通信器重做 all-reduce 探针 |
| `bi_layer_probe.py` | 引擎内逐层打点，驱动脚本 |
| `bi_hook/sitecustomize.py` | 注入 worker 进程的模块级捕获 |
| `analyze_layer_dump.py` | 两次打点按调用顺序比对 |
| `extract_layer_map.py` | 把两次打点压成 步 × 层 的差异矩阵 |
| `analyze_tier2.py` | 把第二档结果渲染成表 T5–T11 |
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
| `results/tier1/*.json` | 第 13 章四项机制检查的原始结果 |
| `results/tier1/analysis_tables.md` | 表 T1、T1b、T2、T3、T4 |
| `results/tier2/*.json` | 第 14 章的配置扫描、逐算子与逐层打点结果 |
| `results/tier2/analysis_tables.md` | 表 T5–T11 |

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
| `docs/07_findings.md` | F1–F13 |
| `docs/08_deployment.md` | 部署 |
| `docs/09_limitations.md` | 局限 |
| `docs/10_reproduction.md` | 本文 |
| `docs/11_references.md` | 参考 |
| `docs/12_sglang_results.md` | SGLang 复现 |
| `docs/13_mechanism_checks.md` | 机制验证（第 13 章） |
| `docs/14_kernel_checks.md` | 第二档：覆盖面、tile 扫描、逐层定位（第 14 章） |
| `docs/appendix_a_logs.md` | 日志 |
| `docs/appendix_b_precision.md` | 精度 |

## 10.9 SGLang 复现

SGLang 装在独立 venv 里，不覆盖 vLLM 环境。

```bash
# 依赖 nvcc >= 12.4：用 conda 环境里的 cuda-nvcc 12.8
export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH

~/venvs/sglang054/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --pre 'sglang[all]==0.5.6'
```

三组主配置（TP=4 确定性 / TP=4 关闭确定性 / TP=4 关 CUDA graph），随后是
两组在线（TP=4 确定性、TP=4 关闭确定性）：

```bash
setsid nohup bash sglang/run_sglang_30b.sh > $HOME/sgl_results/run_30b.log 2>&1 &
```

消融与并行方式对照：

```bash
bash sglang/run_sglang_diag.sh    # 0.6B TP=4、30B 关 custom all-reduce、triton attention
bash sglang/run_sglang_diag2.sh   # 0.6B TP=2、0.6B TP=4 + triton
bash sglang/run_sglang_diag3.sh   # 0.6B PP=4、30B PP=4
bash sglang/run_sglang_pp2.sh     # 30B PP=4 + CUDA graph、PP=4 在线
```

汇总成 Markdown 表：

```bash
python3 sglang/analyze_sglang.py sglang/results
```

| 文件 | 用途 |
|---|---|
| `sglang/bi_sgl_offline_suite.py` | 五组离线测试，一次引擎加载 |
| `sglang/bi_sgl_http_suite.py` | 四类在线场景，走原生 `/generate` |
| `sglang/analyze_sglang.py` | 把 JSON 渲染成表格 |
| `sglang/run_sglang_30b.sh` | 三组离线 + 两组在线主配置 |
| `sglang/run_sglang_diag.sh` / `diag2` / `diag3` | 并行方式与后端消融 |
| `sglang/run_sglang_pp2.sh` | PP=4 的 CUDA graph 与在线复现 |
| `sglang/results/*.json` | 原始结果 |

脚本里的路径都用 `$HOME`，换机器时只需保证模型与 venv 在同样的相对位置。

## 10.10 图表复现

站内 9 张图都由 `scripts/figures/` 下的脚本从原始 JSON 现算，不复用表格里的
手抄数字。

```bash
# 独立的绘图环境，不碰 vLLM / SGLang 的 venv
uv venv .venv-figs --python 3.12
uv pip install --python .venv-figs/bin/python -r requirements-figs.txt

for f in scripts/figures/fig_*.py; do
  .venv-figs/bin/python "$f"
done
```

| 脚本 | 图 | 数据来源 |
|---|---|---|
| `fig_overview.py` | 图 1（首页） | `results/http/concurrency/*.json`、`sglang/results/*.json` |
| `fig_method.py` | 图 2（第 1 章） | `results/extra/bi_batchall_offline_eager.json`、`bi_diag_bs256_pc0.json` |
| `fig_env.py` | 图 3（第 2 章） | 第 2、12 章的版本表 |
| `fig_v1_matrix.py` | 图 4（第 4 章） | `results/bi_*.json` |
| `fig_v1_online.py` | 图 5（第 5 章） | `results/http/bi_http_*.json` |
| `fig_boundary.py` | 图 6（第 6 章） | `results/extra/bi_diag_*.json`、`bi_refeed_*.json`、`results/http/concurrency/bi_http_len_*.json` |
| `fig_mechanism.py` | 图 7（第 7 章） | 第 4–6、12 章的结论 |
| `fig_sgl_offline.py` | 图 8（第 12 章） | `sglang/results/*batchall*.json` |
| `fig_sgl_online.py` | 图 9（第 12 章） | `sglang/results/*_http_http.json` |

`figstyle.py` 统一了字号、配色与导出流程：每张图同时写出 SVG、PDF 与
600 dpi PNG，多面板图在导出前跑一遍面板对齐检查，导出后对 PDF 跑文字与线段
碰撞检查。检查产物写到 `scripts/figures/qa/`，不进站点。站点用 PNG 显示，
因为 PNG 已把中文字体栅格化，读者机器上不需要安装对应字体；SVG 与 PDF 留在
`docs/assets/figs/` 里备打印用。

每张图旁边的 CSV 是画图时取用的数值，方便核对图与表是否一致。

## 10.11 机制验证（第 13 章）

四项检查按顺序跑，前两项不需要模型，后两项各需要一次 30B 加载：

```bash
cd BI-Empirical-Study
bash run_tier1.sh          # all-reduce 探针 → NCCL 算法 → tail → bucket
python3 analyze_tier1.py results/tier1
```

单独跑其中一项也可以：

```bash
export NCCL_IB_DISABLE=1 H=2048 MAX_BS=300

# 1. 不加载模型，4 卡 all-reduce
python -m torch.distributed.run --nproc_per_node=4 bi_allreduce_bench.py \
  --out results/tier1/bi_ar_thresh_w4.json --tag default

# 2. 钉死 NCCL 算法后的 32 条 prompt
bash run_nccl_sweep.sh

# 3. tail 长度扫描（eager，PC=1）
OUT=results/tier1/bi_tail_scan.json python bi_tail_scan.py

# 4. bucket 边界扫描（把 EAGER=1 换成 eager 对照）
OUT=results/tier1/bi_bucket_scan.json python bi_bucket_scan.py
```

all-reduce 探针的输出里 `n_diff` 是位模式不同的元素数，`total` 是比较的元素
总数；`--tests` 可以只跑 `reshape`、`pad`、`threshold`、`sparse`、`pos` 中的
某几项。

## 10.12 第二档（第 14 章）

前五项不加载模型，最后一项要加载一次 30B：

```bash
cd BI-Empirical-Study
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
PY=$HOME/miniconda3/envs/vllm_update/bin/python

# 1. BI 覆盖与配置扫描
$PY bi_bi_coverage.py --out results/tier2/bi_bi_coverage.json

# 2. 线性层、其他算子、分页 attention
$PY bi_matmul_shape.py --out results/tier2/bi_matmul_shape.json
$PY bi_op_shape.py     --out results/tier2/bi_op_shape.json
$PY bi_attn_shape.py   --out results/tier2/bi_attn_shape.json

# 3. 引擎通信器：BI 环境 / tree + 8 通道 / NCCL 自动
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py \
  --out results/tier2/bi_pynccl_bi_env.json --tag bi_env
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py --channels 8 \
  --out results/tier2/bi_pynccl_ch8.json --tag ch8
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py --no-bi-env \
  --out results/tier2/bi_pynccl_auto.json --tag auto

# 4. 逐层打点（需要 PYTHONPATH 指向 bi_hook，vLLM 的 worker 才会导入）
export PYTHONPATH=$PWD/bi_hook
$PY bi_layer_probe.py --out results/tier2/bi_layer_probe.json
$PY extract_layer_map.py --dump $HOME/bi_layer_dump --a a256 --b a255 \
  --out results/tier2/bi_layer_map.json

python3 analyze_tier2.py results/tier2
```

`bi_layer_probe.py` 会清空并重建 `$HOME/bi_layer_dump`，每次 `generate()`
写到 `<dump>/<tag>/rank0/` 下，一个模块调用一个 `.pt`。`analyze_layer_dump.py`
按调用顺序比两份，`extract_layer_map.py` 只保留 DecoderLayer 的第 0 行。

## 10.13 生成 SHA256

```bash
cd bi_validation
find results -name '*.json' -print0 | sort -z | \
  xargs -0 sha256sum > results/SHA256SUMS
```

结果哈希见[附录 B](appendix_b_precision.md)。
