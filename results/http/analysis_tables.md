# 在线 HTTP 结果表（H1–H6）

数据来自 `bi_http_*.json`；表中数值为最大绝对 logprob 差，✅ 表示 token id 与 logprob 均相等。

## 表 E：单进程内

| 配置 | bs1/bs32 | 逆序 | 短 prompt 缓存命中 | 长 prompt bs1/bs4 fresh | 长 prompt fresh/缓存命中 | decode/re-feed |
|---|---|---|---|---|---|---|
| eager_8192_pc1_run1 | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 2.384e-07 ❌ | 0.0 ✅ |
| eager_8192_pc1_run2 | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 2.384e-07 ❌ | 0.0 ✅ |
| eager_2048_pc1 | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 2.384e-07 ❌ | 0.0 ✅ |
| eager_8192_pc0 | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ |
| bi0_eager_8192_pc1 | 1.273e+00 ❌ | 1.307e-01 ❌ | 0.0 ✅ | 3.576e-07 ❌ | 3.576e-07 ❌ | 1.311e-01 ❌ |
| compiled_8192_pc1 | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ |

## 表 F：跨进程 / 跨配置（single / long_single）

| 比较 | 短 token | 短 logprob | 短首个差 | 短最大差 | 长 token | 长 logprob | 长首个差 | 长最大差 |
|---|---|---|---|---|---|---|---|---|
| eager 同配置重启 | ✅ | ✅ | — | 0.0 | ✅ | ✅ | — | 0.0 |
| eager 8192 vs 2048 | ✅ | ✅ | — | 0.0 | ✅ | ✅ | — | 0.0 |
| eager PC1 vs PC0 | ✅ | ✅ | — | 0.0 | ✅ | ✅ | — | 0.0 |
| BI=1 vs BI=0 | ✅ | ❌ | 0 | 1.364e-01 | ✅ | ❌ | 4 | 4.768e-07 |
| compiled vs eager | ❌ | ❌ | 0 | 1.334e+00 | ✅ | ❌ | 4 | 7.153e-07 |

## 表 G：在线 BI=0 对照

| 比较 | token id | logprob | 首个差 | 最大差 |
|---|---|---|---|---|
| bs1/bs32 | ❌ | ❌ | 0 | 1.273e+00 |
| 逆序 | ✅ | ❌ | 0 | 1.307e-01 |
| 短 prompt 缓存命中 | ✅ | ✅ | — | 0.0 |
| 长 prompt bs1/bs4 fresh | ✅ | ❌ | 6 | 3.576e-07 |
| 长 prompt fresh/缓存命中 | ✅ | ❌ | 4 | 3.576e-07 |
| decode/re-feed | — | ❌ | 0 | 1.311e-01 |

## 表 H：开放结尾长 prompt 的缓存命中诊断

### eager 7584

| 请求 | 首发 logprob | 前 8 个 token |
|---|---|---|
| freshA | -1.929295 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| hit1 | -1.964247 | `[27699, 38835, 6941, 55923, 13, 55923, 572, 3881]` |
| hit2 | -1.964247 | `[27699, 38835, 6941, 55923, 13, 55923, 572, 3881]` |
| freshB | -1.929295 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| hit3 | -1.964247 | `[27699, 38835, 6941, 55923, 13, 55923, 572, 3881]` |

- freshA__freshB: ids=✅, logprob=✅, first=None, max=0.0
- freshA__hit1: ids=❌, logprob=❌, first=0, max=5.970e-01
- hit1__hit2: ids=✅, logprob=✅, first=None, max=0.0
- freshB__hit3: ids=❌, logprob=❌, first=0, max=5.970e-01
- freshA__hit3: ids=❌, logprob=❌, first=0, max=5.970e-01

### compiled 7584

| 请求 | 首发 logprob | 前 8 个 token |
|---|---|---|
| freshA | -1.927184 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| hit1 | -1.927184 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| hit2 | -1.927184 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| freshB | -1.927184 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |
| hit3 | -1.927184 | `[3908, 3743, 6941, 47290, 13, 2932, 572, 3881]` |

- freshA__freshB: ids=✅, logprob=✅, first=None, max=0.0
- freshA__hit1: ids=✅, logprob=✅, first=None, max=0.0
- hit1__hit2: ids=✅, logprob=✅, first=None, max=0.0
- freshB__hit3: ids=✅, logprob=✅, first=None, max=0.0
- freshA__hit3: ids=✅, logprob=✅, first=None, max=0.0

