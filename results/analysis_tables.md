## 表 A：同一进程内的 BI 测试

| 运行 | bs=1 vs bs=32 | 逆序 | 前缀缓存命中 | decode vs re-feed |
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

## 表 B：跨进程 / 跨配置比较

| 比较 | single token | single logprob | 首个差异 | 最大差 | long token | long logprob | 首个差异 | 最大差 |
|---|---|---|---|---|---|---|---|---|
| compiled 同配置：冷启动 vs warm | ❌ | ❌ | 10 | 1.227e+00 | ✅ | ❌ | 4 | 4.768e-07 |
| compiled 同配置：warm #2 vs #3 | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |
| compiled 同配置：2048 冷 vs warm #2 | ✅ | ❌ | 30 | 7.756e-03 | ✅ | ❌ | 4 | 5.960e-07 |
| compiled：8192 vs 2048（两者 warm） | ❌ | ❌ | 10 | 1.227e+00 | ✅ | ❌ | 4 | 7.153e-07 |
| compiled：PC on vs off（两者 warm） | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |
| eager 同配置 #1 vs #2 | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |
| eager：8192 vs 2048 | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |
| eager：PC on vs off | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |
| TP=1 同配置 ×2（Qwen3-0.6B） | ✅ | ✅ | None | 0 | ✅ | ✅ | None | 0 |

## 表 C：BI 关闭对照（compiled 8192/PC1）

- bs1_vs_bs32: token_ids_equal=True, logprobs_bitwise=False, first_mismatch=2, max_abs_diff=6.042e-02
- bs1_vs_reverse_order: token_ids_equal=True, logprobs_bitwise=False, first_mismatch=0, max_abs_diff=7.360e-02
- bs1_vs_repeat_prefix_cache: token_ids_equal=True, logprobs_bitwise=True, first_mismatch=None, max_abs_diff=0
- decode_vs_refeed: all_bitwise=False, first=0, max_abs_diff=4.201e-02

## 表 D：耗时（秒）

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
| compiled 2048/PC1（warm #2） | 0.77 | 1.28 | 1.30 | 0.56 | 1.63 | 57.2 |
| compiled 8192/PC0（warm #2） | 0.74 | 1.29 | 1.31 | 0.73 | 1.76 | 57.9 |
