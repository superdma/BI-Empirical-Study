# 4. v1 离线结果（target-only 口径）

> **适用范围**
>
> 本章所有“短 prompt ✅”都只比较了一条 5-token 的 target prompt
> `The capital of France is`。v2 的 32-prompt 实验（[06](06_boundary_results.md)）
> 证明这个结论不能外推到其他 prompt：eager 模式下 31/32 条普通 prompt
> 单独 vs 批内不一致。本章保留 v1 数据，用于说明“同一进程、同一个 target
> 的内部一致性”，不作为部署的充分证据。

## 4.1 表 A：同一进程内

数值为最大绝对 logprob 差；✅ 表示 token id 全等且 logprob 逐 bit 相等。

| 运行 | bs=1 vs bs=32 | 逆序 | 前缀缓存命中（短） | decode vs re-feed |
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
| BI=0 对照（compiled 8192/PC1） | 6.042e-02 ❌ | 7.360e-02 ❌ | 0 ✅ | 4.201e-02 ❌ |
| TP=1 Qwen3-0.6B smoke | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |

对这条 5-token target，BI 开启后所有配置在进程内都是 0.0；BI 关闭时出现
10⁻² 量级差异，说明测试能检测到非 BI 行为。

## 4.2 表 B：跨进程 / 跨配置

| 比较 | single token | single logprob | 首个差异 | 最大差 | long token | long logprob | 首个差异 | 最大差 |
|---|---|---|---|---|---|---|---|---|
| compiled 同配置：冷启动 vs warm | ❌ | ❌ | 10 | **1.227e+00** | ✅ | ❌ | 4 | 4.768e-07 |
| compiled 同配置：warm #2 vs #3 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| compiled 同配置：2048 冷 vs warm #2 | ✅ | ❌ | 30 | 7.756e-03 | ✅ | ❌ | 4 | 5.960e-07 |
| compiled：8192 vs 2048（均 warm） | ❌ | ❌ | 10 | **1.227e+00** | ✅ | ❌ | 4 | 7.153e-07 |
| compiled：PC on vs off（均 warm） | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager 同配置 #1 vs #2 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager：8192 vs 2048 | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| eager：PC on vs off | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |
| TP=1 同配置 ×2（Qwen3-0.6B） | ✅ | ✅ | — | **0** | ✅ | ✅ | — | **0** |

结果：

- eager 对这条 target 是跨进程稳定的；
- compiled 在 warm #2/#3 之间稳定，但**冷启动与 warm 不同**；
- compiled 的 chunk 8192 vs 2048 会改变 target 的 token 序列（第 10 位 logprob
  开始不同，第 14 位 argmax 翻转）；
- compiled 的输出依赖编译状态；要严格复现，必须固定编译缓存和版本。

## 4.3 表 C：BI 关闭对照

配置：compiled，chunk=8192，PC=1，TP=4，`VLLM_BATCH_INVARIANT=0`。

| 比较 | token id | logprob | 首个差异 | 最大差 |
|---|---|---|---|---|
| bs=1 vs bs=32 | 是 | **否** | 第 2 个 | **6.042e-02** |
| bs=1 vs 逆序 | 是 | **否** | 第 0 个 | **7.360e-02** |
| bs=1 vs 重复 | 是 | 是 | — | 0 |
| decode vs re-feed | — | **否** | 第 0 个 | **4.201e-02** |

## 4.4 表 D：耗时（秒）

包含 cache、warmup、图捕获，不是严格 benchmark。

| 运行 | single | batch32 | reverse32 | repeat | long | total |
|---|---|---|---|---|---|---|
| compiled 8192/PC1（冷） | 2.05 | 1.38 | 1.31 | 0.55 | 1.56 | 126.0 |
| compiled 8192/PC1（warm #2） | 0.92 | 1.19 | 1.21 | 0.73 | 1.58 | 59.0 |
| compiled 8192/PC1（warm #3） | 0.74 | 1.28 | 1.38 | 0.73 | 1.65 | 57.4 |
| compiled 2048/PC1（冷） | 0.77 | 1.21 | 1.29 | 0.57 | 1.81 | 106.1 |
| compiled 8192/PC0（冷） | 0.73 | 1.30 | 1.30 | 0.57 | 1.57 | 57.6 |
| eager 8192/PC1 #1 | 3.72 | 3.51 | 3.20 | 2.91 | 2.92 | 37.9 |
| eager 8192/PC1 #2 | 5.89 | 3.72 | 3.08 | 2.35 | 2.62 | 46.6 |
| eager 2048/PC1 | 3.11 | 2.79 | 3.08 | 2.88 | 3.21 | 34.5 |
| eager 8192/PC0 | 2.96 | 2.92 | 3.02 | 2.32 | 2.89 | 33.7 |
| BI=0（compiled 8192/PC1） | 1.32 | 0.98 | 0.81 | 0.23 | 0.69 | 81.7 |

compiled warm 比 eager 快约 2–3 倍。v2 的 HTTP 测试进一步显示，compiled 的
单流 400-token 生成约 8 s，eager 约 25 s；32-prompt 批量请求 compiled 约
0.6 s，eager 约 1.3 s。

## 4.5 生成文本示例

target：`The capital of France is`

| 运行 | 生成文本（截断） |
|---|---|
| eager 8192/PC1（BI on） | ` Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. ...` |
| compiled 8192/PC1（cold） | 同上（`United States` 分支） |
| compiled 8192/PC1（warm #2） | ` Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Spain is Madrid. ...` |
| BI=0 对照 | ` Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. ...` |

同一个 target，cold compiled 与 warm compiled 在一个接近平局的位置改变了
argmax，后续句子完全不同。只看 token id 的测试很容易漏掉这种风险，必须同时
比较 logprob。

## 4.6 本章的适用边界

本章能证明：

- 在同一条 5-token prompt 上，BI 的进程内重复性确实存在；
- BI 关闭时测试能检测到差异；
- compiled 的编译状态/ chunk 会改变输出。

本章不能证明：

- 其他 prompt 也有同样的 0.0；
- eager 是“严格可复现”的通用配置；
- 长 prompt 的缓存命中一致。

后三点由 v2 的 [06](06_boundary_results.md) 回答。
