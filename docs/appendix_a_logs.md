# 附录 A：启动日志关键摘录

以下日志来自 4×A6000、vLLM 0.23.0、Qwen3-30B-A3B、TP=4 的实际运行。

## A.1 BI 在 worker 中启用

```text
Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::mm(Tensor self, Tensor mat2) -> Tensor
  dispatch key: CUDA
  new kernel: registered at .../vllm/model_executor/layers/batch_invariant.py:905
```

说明：`init_batch_invariance()` 在 EngineCore worker 进程里执行；driver 进程的
`_batch_invariant_MODE` 仍然是 `False`，这是设计如此，不代表 BI 没开。

## A.2 TP=4 与 all-reduce

```text
world_size=4 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:55127 backend=nccl
world_size=4 rank=1 local_rank=1 ...
world_size=4 rank=2 local_rank=2 ...
world_size=4 rank=3 local_rank=3 ...
vLLM is using nccl==2.28.9
Using ['PYNCCL'] all-reduce backends (in dispatch order) ...
  out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']
```

## A.3 Attention / MoE 后端

```text
Using AttentionBackendEnum.FLASH_ATTN backend.
Using FlashAttention version 2
Using TRITON Unquantized MoE backend out of potential backends:
  ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']
Using MoEPrepareAndFinalizeNoDPEPModular
```

### TRITON MoE 的 JIT 警告

```text
WARNING ... [jit_monitor.py:103] Triton kernel JIT compilation during inference:
fused_moe_kernel. This causes a latency spike; consider extending warmup to cover
this shape/config.
```

这条警告说明 `fused_moe_kernel` 会在推理中触发 JIT，可能随形状换配置。
它与“同一 prompt 在不同 batch 形状下 logits 不同”的现象一致；这不是唯一
可能原因，稠密模型也复现了失败。

## A.4 显存与 KV cache

```text
Model loading took 14.25 GiB memory and 7.03 seconds
GPU KV cache size: 1,066,720 tokens
Maximum concurrency for 8,192 tokens per request: 130.21x
```

## A.5 编译与图捕获（compiled 模式）

```text
torch.compile and initial profiling/warmup run together took 70.37 s in total
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 51/51
Capturing CUDA graphs (FULL): 100%|...| 35/35
```

warm 启动时会显示从 cache 直接加载编译图；cold 启动会重新编译。v1 观察到
cold vs warm 的输出可能不同（见第 11 章 F7）。

## A.6 MoE 后端启动失败（SM86）

`flashinfer_cutlass`：

```text
ValueError: Unquantized MoE backend FlashInfer CUTLASS does not support the
deployment configuration since kernel does not support current device cuda.
```

`flashinfer_trtllm`：同类失败。完整启动日志未纳入公开仓库。

## A.7 环境变量

```bash
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

不设置 `NCCL_IB_DISABLE=1` 时，NCCL 2.28.9 在本机无 IB 环境会在
`ncclNetPluginInit` segfault。
