### 表 T5：BI 在 SM86 上替换了哪些算子

| aten 算子 | 实际执行的内核 |
|---|---|
| aten::mm | matmul_kernel_persistent |
| aten::addmm | matmul_kernel_persistent |
| aten::matmul | matmul_kernel_persistent |
| aten::linear | matmul_kernel_persistent |
| aten::softmax | aten 逐元素内核（由几个 aten 算子拼成） |
| aten::_log_softmax | _log_softmax_kernel |
| aten::mean.dim | aten 逐元素内核（由几个 aten 算子拼成） |
| aten::bmm | bmm_kernel |

### 表 T6：BI 线性层的 tile 配置

K=2048，bf16。M 从 1 走到 512，配置始终是 BLOCK_M=128、BLOCK_N=128、BLOCK_K=64、GROUP_M=8、warps=8，不同配置数 1。

| 形状 | K | N | 测过的 M 个数 | 行 0 改位的 M 个数 |
|---|---|---|---|---|
| qkv（每卡） | 2048 | 1280 | 12 | 0 |
| o_proj | 2048 | 2048 | 12 | 0 |
| lm_head | 2048 | 151936 | 12 | 0 |

### 表 T7：MoE 默认配置随 M 的切换点

| VLLM_BATCH_INVARIANT | 切换次数 | 切换发生在 M | 255 时的配置 | 255 与 256 相同 |
|---|---|---|---|---|
| 开 | 0 | 无 | 64/64/32 | 是 |
| 关 | 5 | 33, 65, 97, 129, 513 | 64/128/64 | 是 |

### 表 T8：BI 启动时自己写入的 NCCL 环境

| 变量 | 值 |
|---|---|
| `NCCL_ALGO` | allreduce:tree |
| `NCCL_PROTO` | Simple |
| `NCCL_MIN_NCHANNELS` | 1 |
| `NCCL_MAX_NCHANNELS` | 1 |
| `NCCL_NVLS_ENABLE` | 0 |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | 0 |

调用前用户设的是 `NCCL_ALGO=None`。

### 表 T9：引擎自带 pynccl 通信器的批大小敏感性

| 运行 | NCCL_ALGO | 通道数 | 299 步里改位的步数 | 255 与 256 改位元素 | 同一批重复 |
|---|---|---|---|---|---|
| BI 环境 | allreduce:tree | 1 | 0/299 | 0 / 522240 | 0 |
| tree，8 通道 | allreduce:tree | 8 | 30/299 | 0 / 522240 | 0 |
| NCCL 自动选择 | 自动 | 默认 | 299/299 | 1061 / 522240 | 0 |

### 表 T10：每个算子单独跑，行 0 在 255 与 256 之间是否变

| 算子 | 测过的批大小个数 | 行 0 变位的个数 | 每行元素 |
|---|---|---|---|
| rms_norm | 8 | 0 | 2048 |
| fused_add_rms_norm | 8 | 0 | 2048 |
| silu_and_mul | 8 | 0 | 4096 |
| rotary_embedding | 8 | 0 | 4096 |
| fused_experts | 8 | 0 | 2048 |
| paged decode attention | 16 | 0 | 1024 |

批大小取 1、64、128、192、254、256、257、300，基准是 255。线性层另测了三组形状（表 T6）。

### 表 T11：逐层打点（行 0）

| 比较 | 捕获次数 | 改位的模块数 | 首个改位的模块 | 幅度 |
|---|---|---|---|---|
| a256 vs b256 | 4340 | 0 | — | — |
| a256 vs a255 | 4340 | 1028 | RMSNorm | 25/2048（1.95e-03） |

