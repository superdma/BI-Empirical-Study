# 14. 复现步骤与文件清单

## 14.1 前置条件

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

## 14.2 v1：离线矩阵

```bash
cd BI-Empirical-Study
bash scripts/runners/run_all_bi.sh
```

产出：`results/bi_*.json`，以及 `results/analysis_tables.md`（表 A–D）。脚本默认把新结果写到 `$HOME`；可用 `OUT=` 指到 `results/`。

eager fallback 复现（仅限已验证的小批）：

```bash
bash scripts/runners/serve_bi_eager.sh
```

eager 只建议用于小批、同步、已验证的 prompt 集；通用 BI 用
`serve_bi_compiled.sh`。

## 14.3 v1：在线 H1–H6

```bash
bash scripts/runners/run_http_matrix.sh
```

六个配置依次启停：eager 8192/PC1 ×2、eager 2048/PC1、eager 8192/PC0、
BI=0、compiled 8192/PC1。

产出：`results/http/bi_http_*.json` 与
`results/http/analysis_tables.md`（表 E–H）。

## 14.4 v1：长 prompt 缓存命中诊断

```bash
bash scripts/runners/run_diag2.sh    # eager 1900 / 7600
bash scripts/runners/run_diag3.sh    # eager / compiled 7584
```

产出：`results/http/bi_long_probe*.json` 等。

## 14.5 v2：批大小、TP、attention、re-feed

```bash
# 12 个配置的批大小/TP/attention/chunk/长 prompt 矩阵
bash scripts/runners/run_extra_matrix.sh

# bs=256 条件隔离
bash scripts/runners/run_diag_bs256.sh
bash scripts/runners/run_diag_bs256b.sh

# 边界与采样
bash scripts/runners/run_diag_bs256c.sh

# 256-token re-feed 扫描
bash scripts/runners/run_refeed_b.sh
```

产出：`results/extra/bi_extra_*.json`、`results/extra/bi_diag_*.json`、
`results/extra/bi_refeed_*.json`。

汇总：

```bash
python scripts/experiments/v1/analyze_extra.py
# -> results/extra/analysis_tables.md（表 I–L、Q）
```

## 14.6 v2：32-prompt、并发、突发、长度

```bash
# 单次批量请求：逐 prompt 单独 vs 批内
bash scripts/runners/run_batchall.sh          # eager + compiled
bash scripts/runners/run_batchall2.sh         # TRITON_ATTN、无 chunked

# 独立并发
bash scripts/runners/run_concurrency.sh       # eager PC0/PC1、compiled PC1、长 filler

# 突发 / 背景解码
bash scripts/runners/run_burst.sh             # eager PC0、compiled PC1

# 单个请求插入解码
bash scripts/runners/run_mixed.sh             # eager chunked/nochunk、compiled

# 批内长度扫描
bash scripts/runners/run_len.sh               # eager：N=32/L=5,8,9; N=16/30; N=8/40
bash scripts/runners/run_len_compiled.sh      # compiled 对照
```

产出：`results/http/concurrency/*.json`。

汇总：

```bash
python scripts/experiments/online/analyze_online_extra.py
# -> results/http/concurrency/analysis_tables.md（表 M–P）
```

## 14.7 判读标准

| 字段 | 通过条件 |
|---|---|
| `ids` / `token_ids_equal` | true |
| `bitwise` / `logprobs_bitwise_equal` | true |
| `max` / `max_abs_logprob_diff` | 0.0 |
| `repeat_bad` | 0 |
| `solo_vs_batch_bad` | 0 |
| `n_mismatch`（re-feed） | 0 |

任何“小于某个阈值”的判读都不算通过；本报告要求位模式相同。

## 14.8 文件清单

### 脚本

| 文件 | 用途 |
|---|---|
| `scripts/experiments/v1/bi_test_qwen3_30b.py` | v1 离线 BI harness |
| `scripts/experiments/v1/compare_bi.py` | 两份离线 JSON 的位级比较 |
| `run_all_bi.sh` | v1 离线矩阵 |
| `serve_bi_eager.sh` | eager 服务（小批 fallback，默认 MNS=128） |
| `serve_bi_compiled.sh` | 推荐的通用 BI 服务 |
| `scripts/experiments/online/bi_http_test.py` | v1 在线 H1–H6 |
| `run_http_matrix.sh` | v1 在线矩阵 |
| `scripts/experiments/v1/bi_long_diag.py` / `bi_long_diag2.py` | 长 prompt 缓存诊断 |
| `run_diag.sh` / `run_diag2.sh` / `run_diag3.sh` | 上述诊断的启动脚本 |
| `scripts/experiments/v1/bi_extra_qwen3_30b.py` | v2 批大小/TP/attention 矩阵 |
| `scripts/experiments/v1/bi_extra_diag.py` | v2 bs256 条件隔离 |
| `run_extra_matrix.sh` | v2 12 配置矩阵 |
| `run_diag_bs256.sh` / `b` / `c` | v2 bs256 诊断 |
| `scripts/experiments/v1/bi_refeed_diag.py` | decode vs re-feed 长度扫描 |
| `run_refeed_b.sh` | 上述扫描的启动脚本 |
| `scripts/experiments/online/bi_http_concurrent.py` | 32 路独立并发 |
| `run_concurrency.sh` | 并发矩阵 |
| `scripts/experiments/online/bi_http_mixed.py` | 单请求插入解码 |
| `run_mixed.sh` | 混合矩阵 |
| `scripts/experiments/online/bi_http_burst.py` | 突发 32 请求 |
| `run_burst.sh` | 突发矩阵 |
| `scripts/experiments/online/bi_http_batchall.py` | 32-prompt 单独 vs 批内 |
| `run_batchall.sh` / `run_batchall2.sh` | 批量与后端对照 |
| `scripts/experiments/online/bi_http_len.py` | 批内长度扫描 |
| `run_len.sh` / `run_len_compiled.sh` | 长度扫描 |
| `scripts/experiments/v1/bi_batchall_offline.py` | 离线 32-prompt 复核 |
| `scripts/experiments/tier1/bi_allreduce_bench.py` | 不加载模型的 NCCL all-reduce 形状探针 |
| `run_nccl_sweep.sh` | 固定 `NCCL_ALGO` 重跑 32-prompt |
| `scripts/experiments/tier1/bi_tail_scan.py` | 前缀缓存命中 vs fresh 的 tail 长度扫描 |
| `scripts/experiments/tier1/bi_bucket_scan.py` | CUDA graph bucket 边界的批大小扫描 |
| `run_tier1.sh` | 上述四项按顺序跑一遍 |
| `scripts/experiments/tier1/analyze_tier1.py` | 把四项结果渲染成表 T1–T4 |
| `scripts/experiments/tier2/bi_bi_coverage.py` | BI 覆盖了哪些 aten 算子、tile 配置随 M 的扫描 |
| `scripts/experiments/tier2/bi_matmul_shape.py` | 线性层单独跑：同一行在不同 M 下是否逐位相同 |
| `scripts/experiments/tier2/bi_op_shape.py` | norm / silu / rotary / MoE 单独跑，含对齐对照 |
| `scripts/experiments/tier2/bi_attn_shape.py` | 分页 decode attention 单独跑 |
| `scripts/experiments/tier1/bi_pynccl_bench.py` | 用引擎自己的 pynccl 通信器重做 all-reduce 探针 |
| `scripts/experiments/tier2/bi_layer_probe.py` | 引擎内逐层打点，驱动脚本 |
| `scripts/bi_hook/sitecustomize.py` | 注入 worker 进程的模块级捕获 |
| `scripts/experiments/tier2/analyze_layer_dump.py` | 两次打点按调用顺序比对 |
| `scripts/experiments/tier2/extract_layer_map.py` | 把两次打点压成 步 × 层 的差异矩阵 |
| `scripts/experiments/tier2/analyze_tier2.py` | 把第二档结果渲染成表 T5–T11 |
| `scripts/experiments/tier3/bi_composition.py` | 请求组成矩阵，逐层捕获 hidden/residual 与张量布局 |
| `scripts/experiments/tier3/bi_replay_rmsnorm.py` | 用引擎捕获的张量在独立进程回放第 42 层的 norm |
| `scripts/experiments/tier3/bi_launch_diff.py` | torch profiler 抓 255/256 的 kernel 启动参数 |
| `scripts/experiments/tier3/extract_launch_blocks.py` | 从 trace 里汇总 fused add+norm 的 grid/block |
| `scripts/experiments/tier3/bi_tp2_matrix.py` | TP=2 的 32-prompt、255/256 与重复批 |
| `scripts/experiments/tier3/analyze_tier3.py` | 把第三档结果渲染成表 T12–T16 |
| `scripts/experiments/tier4/bi_inproc_replay.py` | 在 worker 进程里现场重算第 2548 次 norm 调用 |
| `scripts/experiments/tier4/bi_replay_probe.py` | 独立进程回放，记录 BI 开/关各自启动的内核实例与 grid/block |
| `scripts/experiments/tier4/bi_boundary_sweep.py` | decode 批大小 255–512 的边界扫描 |
| `scripts/experiments/tier4/bi_shape_matrix.py` | 32 条与 255/256/448/512 的形状矩阵（eager / compiled） |
| `scripts/experiments/tier4/bi_prompt_control.py` | 两组 prompt 集在两种生成长度下的单独 vs 批内 |
| `scripts/experiments/tier4/bi_nsys_capture.py` / `analyze_nsys.py` | Nsight Systems 采集与对比（本机组合未跑通，保留脚本） |
| `scripts/experiments/tier4/analyze_tier4.py` | 把第四档结果渲染成表 T17–T20 |
| `scripts/experiments/v1/analyze_extra.py` | 表 I–L、Q |
| `scripts/experiments/online/analyze_online_extra.py` | 表 M–P |

### 结果目录

| 路径 | 内容 |
|---|---|
| `results/*.json` | v1 离线 |
| `results/http/*.json` | v1 在线、长 prompt 诊断 |
| `results/extra/*.json` | 批大小、TP、attention、re-feed、离线 32-prompt |
| `results/http/concurrency/*.json` | 32-prompt、并发、突发、混合、长度 |
| `results/extra/analysis_tables.md` | 表 I–L、Q |
| `results/http/concurrency/analysis_tables.md` | 表 M–P |
| `results/tier1/*.json` | 第 8 章四项机制检查的原始结果 |
| `results/tier1/analysis_tables.md` | 表 T1、T1b、T2、T3、T4 |
| `results/tier2/*.json` | 第 9 章的配置扫描、逐算子与逐层打点结果 |
| `results/tier2/analysis_tables.md` | 表 T5–T11 |
| `results/tier3/*.json` | 第 10 章的组成、回放、启动参数、TP=2 结果 |
| `results/tier3/analysis_tables.md` | 表 T12–T16 |
| `results/tier4/*.json` | 第 10 章的进程内回放、BI 开/关回放、边界扫描与形状矩阵 |
| `results/tier4/analysis_tables.md` | 表 T17–T20 |

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
| `docs/07_sglang_results.md` | SGLang 复现 |
| `docs/08_mechanism_checks.md` | 机制验证（第 8 章） |
| `docs/09_kernel_checks.md` | 第二档：覆盖面、tile 扫描、逐层定位（第 9 章） |
| `docs/10_launch_isolation.md` | 第三档：组成、回放、启动参数、TP=2（第 10 章） |
| `docs/11_findings.md` | F1–F14 |
| `docs/12_deployment.md` | 部署 |
| `docs/13_limitations.md` | 局限 |
| `docs/14_reproduction.md` | 本文 |
| `docs/15_references.md` | 参考 |
| `docs/appendix_a_logs.md` | 日志 |
| `docs/appendix_b_precision.md` | 精度 |

## 14.9 SGLang 复现

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

## 14.10 图表复现

站内 15 张图都由 `scripts/figures/` 下的脚本从原始 JSON 现算，不复用表格里的
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
| `scripts/figures/fig_overview.py` | 图 1（首页） | `results/http/concurrency/*.json`、`sglang/results/*.json` |
| `scripts/figures/fig_method.py` | 图 2（第 1 章） | `results/extra/bi_batchall_offline_eager.json`、`bi_diag_bs256_pc0.json` |
| `scripts/figures/fig_env.py` | 图 3（第 2 章） | 第 2、7 章的版本表 |
| `fig_v1_matrix.py` | 图 4（第 4 章） | `results/bi_*.json` |
| `fig_v1_online.py` | 图 5（第 5 章） | `results/http/bi_http_*.json` |
| `scripts/figures/fig_boundary.py` | 图 6（第 6 章） | `results/extra/bi_diag_*.json`、`bi_refeed_*.json`、`results/http/concurrency/bi_http_len_*.json` |
| `scripts/figures/fig_sgl_offline.py` | 图 7（第 7 章） | `sglang/results/*batchall*.json` |
| `scripts/figures/fig_sgl_online.py` | 图 8（第 7 章） | `sglang/results/*_http_http.json` |
| `scripts/figures/fig_ar_sensitivity.py` | 图 9（第 8 章） | `results/tier1/bi_ar_*.json` |
| `scripts/figures/fig_engine_checks.py` | 图 10（第 8 章） | `results/tier1/bi_nccl_*.json`、`bi_tail_scan.json`、`bi_bucket_scan*.json` |
| `scripts/figures/fig_bi_coverage.py` | 图 11（第 9 章） | `results/tier2/bi_bi_coverage.json`、`bi_pynccl_*.json` |
| `scripts/figures/fig_layer_probe.py` | 图 12（第 9 章） | `results/tier2/bi_layer_map.json` |
| `fig_tier3_composition.py` | 图 13（第 10 章） | `results/tier3/bi_composition.json`、`bi_launch_blocks.json`、`bi_block_probe.json` |
| `fig_tier4_replay.py` | 图 14（第 10 章） | `results/tier4/probe_bi0.json`、`probe_bi1.json`、`bi_boundary_sweep.json` |
| `scripts/figures/fig_mechanism.py` | 图 15（第 11 章） | 第 4–6、7 章的结论 |

`figstyle.py` 统一了字号、配色与导出流程：每张图同时写出 SVG、PDF 与
600 dpi PNG，多面板图在导出前跑一遍面板对齐检查，导出后对 PDF 跑文字与线段
碰撞检查。检查产物写到 `scripts/figures/qa/`，不进站点。站点用 PNG 显示，
因为 PNG 已把中文字体栅格化，读者机器上不需要安装对应字体；SVG 与 PDF 留在
`docs/assets/figs/` 里备打印用。

每张图旁边的 CSV 是画图时取用的数值，方便核对图与表是否一致。

## 14.11 机制验证（第 8 章）

四项检查按顺序跑，前两项不需要模型，后两项各需要一次 30B 加载：

```bash
cd BI-Empirical-Study
bash scripts/runners/run_tier1.sh          # all-reduce 探针 → NCCL 算法 → tail → bucket
python3 scripts/experiments/tier1/analyze_tier1.py results/tier1
```

单独跑其中一项也可以：

```bash
export NCCL_IB_DISABLE=1 H=2048 MAX_BS=300

# 1. 不加载模型，4 卡 all-reduce
python -m torch.distributed.run --nproc_per_node=4 bi_allreduce_bench.py \
  --out results/tier1/bi_ar_thresh_w4.json --tag default

# 2. 钉死 NCCL 算法后的 32 条 prompt
bash scripts/runners/run_nccl_sweep.sh

# 3. tail 长度扫描（eager，PC=1）
OUT=results/tier1/bi_tail_scan.json python bi_tail_scan.py

# 4. bucket 边界扫描（把 EAGER=1 换成 eager 对照）
OUT=results/tier1/bi_bucket_scan.json python bi_bucket_scan.py
```

all-reduce 探针的输出里 `n_diff` 是位模式不同的元素数，`total` 是比较的元素
总数；`--tests` 可以只跑 `reshape`、`pad`、`threshold`、`sparse`、`pos` 中的
某几项。

## 14.12 第二档（第 9 章）

前五项不加载模型，最后一项要加载一次 30B：

```bash
cd BI-Empirical-Study
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
PY=$HOME/miniconda3/envs/vllm_update/bin/python

# 1. BI 覆盖与配置扫描
$PY scripts/experiments/tier2/bi_bi_coverage.py --out results/tier2/bi_bi_coverage.json

# 2. 线性层、其他算子、分页 attention
$PY scripts/experiments/tier2/bi_matmul_shape.py --out results/tier2/bi_matmul_shape.json
$PY scripts/experiments/tier2/bi_op_shape.py     --out results/tier2/bi_op_shape.json
$PY scripts/experiments/tier2/bi_attn_shape.py   --out results/tier2/bi_attn_shape.json

# 3. 引擎通信器：BI 环境 / tree + 8 通道 / NCCL 自动
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py \
  --out results/tier2/bi_pynccl_bi_env.json --tag bi_env
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py --channels 8 \
  --out results/tier2/bi_pynccl_ch8.json --tag ch8
$PY -m torch.distributed.run --nproc_per_node=4 bi_pynccl_bench.py --no-bi-env \
  --out results/tier2/bi_pynccl_auto.json --tag auto

# 4. 逐层打点（需要 PYTHONPATH 指向 bi_hook，vLLM 的 worker 才会导入）
export PYTHONPATH=$PWD/scripts/bi_hook
$PY scripts/experiments/tier2/bi_layer_probe.py --out results/tier2/bi_layer_probe.json
$PY scripts/experiments/tier2/extract_layer_map.py --dump $HOME/bi_project/data/bi_layer_dump --a a256 --b a255 \
  --out results/tier2/bi_layer_map.json

python3 scripts/experiments/tier2/analyze_tier2.py results/tier2
```

`bi_layer_probe.py` 会清空并重建 `$HOME/bi_project/data/bi_layer_dump`，每次 `generate()`
写到 `<dump>/<tag>/rank0/` 下，一个模块调用一个 `.pt`。`analyze_layer_dump.py`
按调用顺序比两份，`extract_layer_map.py` 只保留 DecoderLayer 的第 0 行。

## 14.13 第三档（第 10 章）

三次模型加载，一次纯回放。回放与 trace 汇总可以放在同一次会话里：

```bash
cd BI-Empirical-Study
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
PY=$HOME/miniconda3/envs/vllm_update/bin/python

# 1. 请求组成 + 逐层捕获（PYTHONPATH 必须指向 bi_hook）
export PYTHONPATH=$PWD/scripts/bi_hook
export BI_NORM_META=1 BI_DUMP=$HOME/bi_project/data/bi_norm_meta BI_FULL_RANGE=2400:2650
$PY scripts/experiments/tier3/bi_composition.py --out results/tier3/bi_composition.json
unset PYTHONPATH BI_NORM_META BI_DUMP BI_FULL_RANGE

# 2. 回放（不需要加载模型）
$PY scripts/experiments/tier3/bi_replay_rmsnorm.py --out results/tier3/bi_replay_rmsnorm.json

# 3. 启动参数（vLLM 0.23 需要显式打开 profiler）
export VLLM_TORCH_PROFILER_DIR=$HOME/bi_project/data/bi_traces
$PY scripts/experiments/tier3/bi_launch_diff.py --out results/tier3/bi_launch_diff.json

# 先跑 256，后跑 255；ls -t 最新的那个是 255
$PY scripts/experiments/tier3/extract_launch_blocks.py \
  --traces p256=$(ls -t $HOME/bi_project/data/bi_traces/*rank0* | sed -n 2p) \
           p255=$(ls -t $HOME/bi_project/data/bi_traces/*rank0* | sed -n 1p) \
  --out results/tier3/bi_launch_blocks.json

# 4. TP=2
$PY scripts/experiments/tier3/bi_tp2_matrix.py --out results/tier3/bi_tp2_matrix.json

python3 scripts/experiments/tier3/analyze_tier3.py results/tier3
```

`bi_composition.py` 会清空 `$HOME/bi_project/data/bi_norm_meta`，每个 tag 写一份层捕获；只有
c256 与 c255 两个 tag 写盘。`BI_FULL_RANGE` 指定保存全量张量的调用序号区间，
窗口按第 9 章的分叉位置选。`extract_launch_blocks.py` 的 `--traces` 支持多个
`tag=path`，这里只取 rank 0。

## 14.14 第四档（第 10 章的更正与补充）

第四档有整套启动脚本，也可以逐条手动跑。回放与探针不需要加载模型，其余命令
各加载一次：

```bash
cd BI-Empirical-Study
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export VLLM_BATCH_INVARIANT=1
PY=$HOME/miniconda3/envs/vllm_update/bin/python

bash scripts/runners/run_tier4.sh          # 进程内回放、边界扫描、三套形状矩阵、nsys、prompt 集对照
```

`run_tier4.sh` 把结果写到 `results/tier4/`，日志和返回码写到 `logs/tier4/`，
捕获的张量写到 `${BI_DATA:-$HOME/bi_project/data}`。单独重跑某一项：

```bash
# 独立进程回放，BI 开与关各一遍（不加载模型）
$PY scripts/experiments/tier4/bi_replay_probe.py --bi 0 --out results/tier4/probe_bi0.json
$PY scripts/experiments/tier4/bi_replay_probe.py --bi 1 --out results/tier4/probe_bi1.json

# 汇总
python3 scripts/experiments/tier4/analyze_tier4.py results/tier4
```

`bi_replay_probe.py` 在导入 vLLM 之前决定 `VLLM_BATCH_INVARIANT` 是否设置，并从
它自己导出的 chrome trace 里读回内核实例名与 grid/block。`bi_shape_matrix.py`
用 `--compiled` 决定是否加 `--enforce-eager`，其余参数与其他脚本一致。

Nsight Systems 的采集脚本（`bi_nsys_capture.py`、`analyze_nsys.py`）一并保留：
nsys 2023.2 在这套 CUDA 12.9 + 四个 worker 的组合下无法完成引擎初始化，
本轮没有拿到可用 trace。

## 14.15 生成 SHA256

```bash
cd BI-Empirical-Study
find results -name '*.json' | sort | while read -r r; do sha256sum "$r"; done \
  > results/SHA256SUMS
```

结果哈希见[附录 B](appendix_b_precision.md)。
