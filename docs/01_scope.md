# 1. BI 是什么，以及本报告的比较口径

## 1.1 定义

批次不变性（Batch Invariance，BI）指：**同一个请求的输出，不随同批次中其他请求的
数量、顺序、内容或到达方式而改变**（[vLLM BI 文档](https://docs.vllm.ai/en/latest/features/batch_invariance)）。

在 vLLM 中通过环境变量开启：

```bash
export VLLM_BATCH_INVARIANT=1
```

开启后，vLLM 会：

- 在 EngineCore worker 进程中初始化 BI 模式；
- 覆盖若干 `aten` 算子（如 `aten::mm`）为固定归约顺序的实现；
- 在启动日志中打印 `batch_invariant.py` 的覆盖信息。

BI 不改变模型权重的确定性，也不承诺跨版本或跨后端的兼容性。

## 1.2 本报告的比较口径

### 位级比较

所有 logprob 比较都用 IEEE-754 双精度位模式：

```python
import struct
struct.pack(">d", float(x)).hex()
```

两个数的位模式完全相同才算相等；打印到 6 位小数相同不算。

### 三类比较

| 记号 | 含义 | 典型用途 |
|---|---|---|
| `bs1 vs bsN` | 同一 prompt 单独跑 vs 与 N-1 个其他 prompt 同批跑 | BI 主断言 |
| `reverse` | 同批但顺序反转 | 顺序不变性 |
| `fresh vs cache-hit` | 同 prompt 首次 prefill vs 命中 prefix cache | 缓存状态不变性 |
| `decode vs re-feed` | 逐 token 生成得到的 logprob vs 把整段拼回后一次 prefill 的 prompt logprob | 生成/打分一致性 |
| `solo vs concurrent` | 每条 prompt 单独跑 vs 多路并发请求 | 真实服务口径 |

### 比较字段

- **token ids**：贪心解码下生成的 token 序列；
- **logprobs**：每个生成 token 的 log 概率（`logprobs=5` 中该 token 的值）；
- **prompt logprobs**：re-feed 时整段 prompt 每个位置的 log 概率。

token ids 相同但 logprobs 不同，仍然判定为 **BI 失败**：对 RL/打分/审计来说，
logprob 是输出的一部分。

## 1.3 vLLM 官方声明的范围

官方文档把 BI 标为 beta，并明确它只保证**同一引擎配置、同一进程内**的输出不随
batch 组成变化；不保证：

- 跨 TP size；
- 跨 attention backend；
- 跨 compilation mode（eager vs compiled）；
- 跨 vLLM/PyTorch/CUDA 版本；
- 跨 prefix cache 状态（文档只承诺 batch invariance，不承诺 cache invariance）。

本报告在这些边界上逐项测量。

## 1.4 v1 与 v2 的口径差异

v1 只比较了一条 5-token 的 target prompt。v2 发现必须区分：

1. **“单条 prompt 的 BI”**：A 单独跑 vs A 在批内跑；
2. **“批内每一条 prompt 的 BI”**：对批内所有 prompt 逐条做上面的比较。

只做第 1 条、且只选一条短 prompt，会得到“全 0”的乐观结论；对 32 条 prompt
逐条做第 2 条，eager 模式会暴露 31/32 不一致。v1 的 0.0 结论因此不能外推。

## 1.5 证据强度分级

| 级别 | 含义 | 本报告例子 |
|---|---|---|
| A | 多次重复、跨进程、跨配置复现 | compiled 32/32 一致；eager 31/32 不一致；两者各自的 solo 基线跨 3 个进程一致 |
| B | 单次但方向明确、有对照 | 256-token 边界；TP2 vs TP4 差异 |
| C | 观察到的现象，机制未完全隔离 | 长 prompt cache-hit 的 2.384e-07；8180 恰好一致 |

正文在每条发现后会标注级别。
