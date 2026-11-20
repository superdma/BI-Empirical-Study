# 13. 局限与后续工作

## 13.1 实验范围

| 维度 | 本次覆盖 | 未覆盖 |
|---|---|---|
| GPU | 4×RTX A6000（SM86） | SM120（5090/RTX PRO 6000）、A100、H100、MI300 |
| 互联 | PCIe，无 IB | NVLink、InfiniBand、多机 |
| TP | 4，少量 TP=2 对照 | TP=1（除 0.6B）、TP=8 |
| 模型 | Qwen3-30B-A3B（MoE）、Qwen3-0.6B（稠密） | 其他 MoE、量化模型、长上下文模型 |
| 上下文 | ≤8192，最长 8180 | 16k/32k/128k |
| 生成长度 | ≤256 token | 更长 |
| 并发 | HTTP ≤32；离线 bs≤257 | 数百路真实并发 |
| 采样 | greedy；temp=0.7/top_p=0.95 | top_k、min_p、beam、结构化输出 |
| 负载 | 空载/轻载 | KV 压满、抢占、PD 分离 |

## 13.2 尚未解释的现象

### N=8、总 285 token 通过，N=16/32 失败

如果只看“总 prefill token”，285 应该落在失败区；但 N=8×40 全过。说明阈值
不是单一的总 token 数，而是批形状（序列数 × 每序列长度）的组合。第 9 章测过
单个算子的 tile 配置，在 255 与 256 上不变，所以这个组合影响的不像是哪一个
算子的配置。本文只给出经验边界，没有给出闭式公式。

### 5-token target 在 32-prompt 批内恰好一致

target 单独 vs 批内是 0.0，而其他 31 条失败。可能的解释是 5-token 序列在
prefill 中走了不同的 query tile 路径；但没有逐 kernel 验证。

### 8180-token 近最大上下文 PC0/PC1 都一致

7600-token 饱和 prompt 有 2.384e-07，7584-token 开放结尾翻转首 token，8180
却全 0。说明“长 prompt 必坏”不成立，tail/chunk 对齐影响结果；具体几何未建模。

第 8 章把 tail 长度从 1 扫到 512，只有 256 与 448 分叉，320/384/512 又一致，
把“越长越坏”这个读法排掉了。剩下的解释仍然是 chunk 形状，但没有闭式模型。

### 256 条时某一层 norm 的输出变了，而输入逐位相同

第 9 章逐层打点：同一批 256 条跑两遍，4340 次模块调用全部逐位相同；换成
255 条，从第 5 个解码步、第 42 层 input_layernorm 开始出现差异。这一层第 0 行
的 hidden 与 residual 输入逐位相同，输出有 25 个元素不同，形态像是归一化因子
差了一个 ulp（其余行在此前已经分叉）。

同一个 `fused_add_rms_norm` 单独跑 1 到 300 行、不同 storage offset，行 0
都逐位不变。第 10 章把这件事拆开了：边界只认解码步里同时活跃的序列数，请求
顺序与瞬时多一条都不算；`VLLM_BATCH_INVARIANT=1` 时算子走标量实例
`<bf16, 0>`，255 行的 block 是 1024、256 行起是 256，归约树随线程数改变，
第 0 行 25 个元素差一个 ulp。开着 BI 的回放能逐位复现两次运行；上一版“复现
不出 256”的记录，是回放进程没开 BI、跑成向量化实例 `<bf16, 8>` 造成的。
没测到的是把 block 单独钉住：这套代码里行数与 block 同变。

### compiled 为什么通过

可能是 Inductor 融合、CUDA Graph 的固定形状、attention 后端选择，或几者
共同作用；没有做逐 kernel 对比。

## 13.3 方法上的局限

- **只测了 32 条 prompt**：足以推翻“全 0”的外推，但不能代表所有分布；
- **prompt 都是英文短句**：没有测中文、代码、chat template、系统提示；
- **HTTP 客户端在同一台主机**：线程池的到达抖动可能与真实网络不同；
- **没有做 latency/throughput 的严格 benchmark**：表 D 的耗时只用于量级；
- **没有在 KV cache 压满/抢占下测试**：preemption 可能引入新的形状变化；
- **没有做多进程/多副本一致性**：只测了单机 TP=4；
- **没有测 tensor parallel 的通信确定性在拥塞下的表现**。

## 13.4 与 v1 的关系

v1 的以下结论仍然成立（在其口径内）：

- 同一条 5-token target 的进程内重复性；
- eager 跨进程对同一 target 稳定；
- 长 prompt + eager + cache hit 会分叉；
- BI=0 对照敏感；
- NCCL/FlashInfer 环境坑。

v1 的以下结论被 v2 修正：

- “短 prompt 位级一致”不能外推到其他 prompt；
- “eager 是严格可复现的推荐配置”不成立；
- “在线 BI 成立”只对“单请求批量提交 + target-only”成立。

## 13.5 后续工作（按优先级）

1. **把 block 单独钉住**：第 10 章已经确认 BI 下标量实例的 block 在 256 行处
   从 1024 换成 256，但行数与 block 在这套代码里同变。需要一个改过启动逻辑的
   vLLM 版本，在 255 行强制 block 256（或在 256 行强制 1024），才能把因果链
   的最后一环合上；顺带可以确认 `num_tokens` 之外还有没有入参影响结果；
2. **扩大 prompt 分布**：中文、代码、chat template、长系统提示、不同长度；
3. **扩大并发**：128/256 路真实 HTTP，观察失败率与批形状的关系；
4. **KV 压力**：把 gpu_memory_utilization 拉高、制造抢占，测试 BI 是否仍成立；
5. **compiled 冷/暖系统复现**：固定/清空编译缓存，量化 cold vs warm 的差异；
6. **SM120 复测**：RTX 5090 / RTX PRO 6000 Blackwell 上的 kernel 选择可能不同；
7. **TP=8 / 多机**：all-reduce 与通信确定性；
8. **量化模型**：FP8/INT8 的 BI 行为；
9. **上游反馈**：把 32-prompt 的复现脚本与结果整理成 vLLM issue，询问
   eager 模式下 custom kernel 的 BI 覆盖范围。

## 13.6 结论的适用边界

本报告的所有“通过/失败”都是**在这套硬件、这个 vLLM 版本、这个模型、这些
prompt 和这些形状上**的实测结果。它们足以说明：

- 仅设置 `VLLM_BATCH_INVARIANT=1` 不足以保证 BI；
- eager 与 compiled 的行为差异很大；
- 必须逐 prompt、逐形状验证。

它们不足以说明：

- 其他 vLLM 版本一定相同；
- 其他 GPU 的阈值相同；
- 所有 MoE 模型都这样；
- compiled 永远优于 eager。

部署前请用 [08](12_deployment.md) 的清单在你的实际配置上重跑。
