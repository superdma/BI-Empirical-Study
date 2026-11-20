# 3. 测试设计

## 3.1 两条测试路径

| 路径 | 入口 | 优点 | 缺点 |
|---|---|---|---|
| 离线 | `LLM.generate`（Python API） | 可控、快、可直接拿 `logprobs` | 不经过 HTTP 调度器 |
| 在线 | `/v1/completions` + `/tokenize` | 覆盖真实服务路径 | 请求到达方式会影响批次 |

v1 以离线为主，在线补充；v2 的 32-prompt 实验以在线为主，并用离线
`LLM.generate` 复核（见 04 与 06）。

## 3.2 采样参数

除非特别注明：

```python
SamplingParams(
    temperature=0.0,
    top_p=1.0,
    seed=0,
    max_tokens=32,
    logprobs=5,
)
```

- `temperature=0`：贪心，排除采样随机性；
- `logprobs=5`：每个生成位置返回 top-5，取实际 token 的 logprob；
- `seed=0`：固定；
- v2 的 sampling 实验使用 `temperature=0.7, top_p=0.95, seed=42`。

## 3.3 v1 离线矩阵（T1–T10）

| 编号 | 配置 | 目的 |
|---|---|---|
| T1 | eager 8192 PC1 | 基线 |
| T2 | eager 8192 PC1 重复 | 进程内重复性 |
| T3 | eager 8192 PC0 | 关 prefix cache |
| T4 | eager 2048 PC1 | chunk size 影响 |
| T5 | eager 2048 PC0 | 组合 |
| T6 | compiled 8192 PC1 冷/暖 | 编译与图捕获 |
| T7 | compiled 2048 PC1 | 编译 + chunk |
| T8 | BI=0 eager 8192 PC1 | 灵敏度对照 |
| T9 | TP=2 eager | TP 影响（v2 补） |
| T10 | TRITON_ATTN eager | attention 后端影响（v2 补） |

每个 T 配置都运行：

1. `single`：target 单独；
2. `batch32`：target + 31 个 filler；
3. `reverse`：反序；
4. `repeat`：再单独跑一次（PC=1 时为缓存命中）；
5. `long_batch` → `long_single`：约 7600-token 长 prompt，先批后单；
6. `decode_vs_refeed`：把生成结果拼回，比较 prompt logprobs。

## 3.4 v1 在线矩阵（H1–H6）

| 编号 | 服务配置 |
|---|---|
| H1 | eager 8192 PC1 第一次启动 |
| H2 | eager 8192 PC1 第二次启动 |
| H3 | eager 2048 PC1 |
| H4 | eager 8192 PC0 |
| H5 | eager 8192 PC1，BI=0 |
| H6 | compiled 8192 PC1 |

每个 H 配置通过 `/v1/completions` 测：

- 短 prompt：bs1 vs bs32、逆序、短 prompt 缓存命中；
- 长 prompt：bs1 vs bs4 fresh、fresh vs cache hit；
- decode vs re-feed。

口径：v1 的“bs32”是**一个 HTTP 请求里放 32 条 prompt**
（`prompt: [p0, p1, ..., p31]`），不是 32 个独立并发请求。v2 表明这个区别
会改变结果。

## 3.5 v2 新增矩阵（B1–B12）

| 编号 | 实验 | 变量 |
|---|---|---|
| B1 | 批大小扫描 | bs 128/192/224/240/255/256/257 |
| B2 | bs256 条件隔离 | PC0/PC1、ignore_eos、全 target、max_num_seqs |
| B3 | bs256 长生成 | 256 token，greedy |
| B4 | 采样 | temperature=0.7，seed=42 |
| B5 | 256-token re-feed | MT=64/128/192/256 |
| B6 | TP=2 vs TP=4 | 跨进程 |
| B7 | TRITON_ATTN vs FLASH_ATTN | 跨后端 |
| B8 | chunk=2048 | 跨 chunk size |
| B9 | 8180-token 近最大上下文 | PC0/PC1 |
| B10 | 32-prompt 全量比较 | 单独 vs 批内，逐条 |
| B11 | 独立并发 / 突发 / 混合 | 真实到达方式 |
| B12 | 稠密模型 / MoE 后端 / no-chunk 对照 | 机制隔离 |

### B10 的协议

1. 32 条 prompt：1 条 target + 31 条 “Question i: ...”；
2. 对每条 prompt 单独发一次请求（`max_tokens=16`），记为 solo；
3. 一次请求提交全部 32 条（`prompt=[...]`），记为 batch；
4. 再提交一次同样的 batch，记为 batch2；
5. 对 i=0..31 比较 `solo[i]` vs `batch[i]`，以及 `batch[i]` vs `batch2[i]`。

协议同时检查：

- 单条 prompt 的批不变性；
- 整批请求的可重复性；
- 批内第一条与其他条的差异。

### B11 的协议

- **独立并发**：32 个线程各发一个请求；
- **突发（有背景解码）**：先启动一个 400-token 的长请求，1 s 后突发 32 个请求；
- **突发（无背景解码）**：直接突发 32 个请求（第一条仍会先进入解码）；
- **单个插入**：长请求解码期间，只发 1 个短请求，测 4 个延迟（0.2/0.6/1.5/3.0 s）。

### B12 的协议

在 B10 基础上替换一个变量：

- attention 后端：`TRITON_ATTN`；
- 关 chunked prefill；
- 显式 `moe_backend=triton`；
- 模型换成稠密 `Qwen3-0.6B`（TP=1）；
- 模式换成 compiled。

## 3.6 数据完整性

- 所有 JSON 原样保存，未做四舍五入；
- 表 I–Q、M–P、T1–T20 由脚本从 JSON 生成；表 A–H 是整理时从
  `compare_bi.py` 的输出手工汇总的，原始 JSON 为准；
- 任何“0.0”都表示位模式相同，不是“小于某个阈值”；
- 失败配置的日志只保留在实验机上，未纳入公开仓库。
