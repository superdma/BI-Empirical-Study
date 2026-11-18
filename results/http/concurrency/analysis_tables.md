# 补充实验汇总表（在线并发、批内长度、re-feed）

## 表 M：HTTP 到达方式与并发

| 场景 | 模式 | N | 比较对象 | 通过/总数 | 最大 \|Δlogprob\| |
|---|---|---|---|---|---|
| 独立并发请求（短 prompt） | eager PC0 | 32 | 逐 prompt：单独 vs 并发 | 1/32 | 2.723e+00 |
| 独立并发请求（短 prompt） | eager PC1 | 32 | 逐 prompt：单独 vs 并发 | 1/32 | 2.723e+00 |
| 独立并发请求（短 prompt） | compiled PC1 | 32 | 逐 prompt：单独 vs 并发 | 32/32 | 0.000e+00 |
| 突发 32 请求（另一请求正在解码） | eager PC0 | 32 | 逐 prompt：单独 vs 突发 | 1/32 | 2.723e+00 |
| 突发 32 请求（无背景解码） | eager PC0 | 32 | 逐 prompt：单独 vs 突发 | 1/32 | 2.723e+00 |
| 突发 32 请求（另一请求正在解码） | compiled PC1 | 32 | 逐 prompt：单独 vs 突发 | 32/32 | 0.000e+00 |
| 突发 32 请求（无背景解码） | compiled PC1 | 32 | 逐 prompt：单独 vs 突发 | 32/32 | 0.000e+00 |
| 单次批量请求（32 个 prompt 一起提交） | eager PC0 | 32 | 逐 prompt：单独 vs 批内 | 1/32 | 2.723e+00 |
| 单次批量请求（32 个 prompt 一起提交） | compiled PC1 | 32 | 逐 prompt：单独 vs 批内 | 32/32 | 0.000e+00 |
| 单个请求插入解码中的批次（4 个延迟） | eager PC0 | 1 | 单独 vs 混合 | 4/4 | 0.000e+00 |
| 单个请求插入解码中的批次（4 个延迟） | eager 无 chunked | 1 | 单独 vs 混合 | 4/4 | 0.000e+00 |
| 单个请求插入解码中的批次（4 个延迟） | compiled PC0 | 1 | 单独 vs 混合 | 4/4 | 0.000e+00 |

## 表 N：批内 prompt 长度扫描（eager，FLASH_ATTN，PC=0）

| N | 每个 filler 长度 | 总 prefill token | 通过/总数 | 最大 \|Δlogprob\| | 重复批一致 |
|---|---|---|---|---|---|
| 16 | 30 | 455 | 2/16 | 9.614e-02 | 是 |
| 32 | 5 | 160 | 32/32 | 0.000e+00 | 是 |
| 32 | 8 | 253 | 32/32 | 0.000e+00 | 是 |
| 32 | 9 | 284 | 1/32 | 1.590e+00 | 否（1） |
| 8 | 40 | 285 | 8/8 | 0.000e+00 | 是 |

### 对照：compiled 模式

| N | 每个 filler 长度 | 总 prefill token | 通过/总数 | 最大 \|Δlogprob\| |
|---|---|---|---|---|
| 16 | 30 | 455 | 16/16 | 0.000e+00 |
| 32 | 8 | 253 | 32/32 | 0.000e+00 |
| 32 | 9 | 284 | 32/32 | 0.000e+00 |
| 8 | 40 | 285 | 8/8 | 0.000e+00 |

## 表 O：decode vs re-feed 的生成长度扫描（eager，PC=0）

| 生成 token 数 | 拼回后总长度 | 通过/比较数 | 首个不一致 | 最大 \|Δlogprob\| |
|---|---|---|---|---|
| 64 | 69 | 64/64 | — | — |
| 128 | 133 | 128/128 | — | — |
| 192 | 197 | 192/192 | — | — |
| 256 | 261 | 48/256 | 5 | 1.377e-01 |
| 256（compiled） | 261 | 256/256 | — | — |

## 表 P：模型 / attention / MoE 后端对照（单次批量请求，32 prompt）

| 配置 | 通过/总数 | 批重复一致 | 说明 |
|---|---|---|---|
| Qwen3-30B-A3B, eager, FLASH, chunked | 1/32 | 是 | 基线 |
| Qwen3-30B-A3B, eager, TRITON_ATTN | 1/32 | 是 | 换 attention 后端 |
| Qwen3-30B-A3B, eager, FLASH, 无 chunked | 1/32 | 否（1） | 关 chunked prefill |
| Qwen3-30B-A3B, eager, moe=triton（显式） | 2/32 | 是 | 默认 MoE 后端 |
| Qwen3-0.6B 稠密模型, eager, FLASH | 2/32 | 否（1） | 排除 MoE 因素 |
| Qwen3-30B-A3B, compiled, FLASH | 32/32 | 是 | torch.compile + CUDA graphs |
