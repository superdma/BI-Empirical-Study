# 12. SGLang 复现实验

前面的章节都在 vLLM 上完成。SGLang 是另一个主流推理引擎，它提供了自己的
确定性推理开关 `--enable-deterministic-inference`，实现路线与 vLLM 不同：
它用 Triton 重写 `aten::mm`、`aten::addmm`、`aten::bmm`、`log_softmax`、`mean`，
并给 attention 指定固定的 split 大小。本章用同一套测试脚本，在同样的
4×RTX A6000 上把实验重做一遍，回答两个问题：

1. 同一套口径，SGLang 能不能复现 vLLM 的结论；
2. SGLang 的确定性开关在这台机器上要满足什么条件才真的做到位级一致。

结论：

- 在 TP=4 下，SGLang 的确定性模式只能做到 7/32 条 prompt 与单独运行位级
  一致，远不到“可用”的程度，但明显好于关闭开关时的 0/32；
- 把 TP=4 换成 PP=4（流水线并行，跨卡只传激活、不做归约）后，32/32 条
  prompt 位级一致，批大小 128 到 257 全部为 0.0，在线四类场景 96/96 一致；
- 同一台机器、同一个模型，TP=1 与 TP=2 通过，TP=4 失败。差异随并行方式
  出现，与批大小无关。

## 12.1 环境

SGLang 装在独立的 venv 里，与 vLLM 的 conda 环境互不干扰。

| 组件 | 版本 |
|---|---|
| SGLang | 0.5.6 |
| PyTorch | 2.9.1+cu128 |
| FlashInfer | 0.5.3（python + cubin） |
| sgl-kernel | 0.3.18.post2 |
| Triton | 3.5.1 |
| Transformers | 4.57.1 |
| Python | 3.12 |
| CUDA 编译工具 | conda 环境中的 `cuda-nvcc` 12.8 |

硬件与前面各章相同：4 × RTX A6000 48 GB（SM86），驱动 550.120，PCIe 互联，
无 InfiniBand。模型仍是 `Qwen/Qwen3-30B-A3B`（BF16，MoE）。

### 版本选择的过程

最初装的是 0.5.9（torch 2.9.1+cu128、flashinfer 0.6.3、sgl-kernel 0.3.21）。
它能加载模型、打印 `engine up`，但在第一次 `generate` 时进程收到 SIGSEGV。
复现范围很宽：

- attention 后端 flashinfer / triton 都崩；
- CUDA graph 开、关都崩；
- 确定性开关开、关都崩；
- 离线 `sgl.Engine` 与 `python -m sglang.launch_server` 都崩；
- 单独调用各个 kernel（flashinfer 的 ragged prefill、paged decode，sgl-kernel
  的 `rmsnorm`、`silu_and_mul`，JIT rope，batch-invariant 的 `aten::mm`，
  torch 的 SDPA、linear、lmhead、topk）全部正常。

`compute-sanitizer --tool memcheck` 跑了数分钟没有报出越界或非法访问，gdb 抓到的
栈也停在 Python 调用层，没有指向某一个具体 kernel（这两项都没能给出确定结论，
只排除了明显的访存错误）。由于单 kernel 全部通过、换后端与开关都不起作用，
判断问题出在 0.5.9 这套依赖组合与 A6000（SM86）加 550 驱动的配合上，而不是
模型、也不是测试代码。这个判断是排除法得到的，没有再往下定位到具体库。

换成 0.5.6（flashinfer 0.5.3、sgl-kernel 0.3.18.post2）后同一份脚本一次跑通，
后续全部实验都基于 0.5.6。0.5.4 也试过，但它依赖 `apache-tvm-ffi==0.1.0b15`
这个预发布包，镜像站没有，安装直接失败，没有继续。

FlashInfer 的 JIT 需要两样东西：`ninja` 在 `PATH` 里，以及 nvcc ≥ 12.4
（它用 `-std=c++20` 编译）。系统自带的 CUDA 12.2 不够用，解决办法是在 conda
环境里装 `cuda-nvcc=12.8` 和 `cuda-cudart-dev`，运行时把 `CUDA_HOME` 指向它。

## 12.2 测试方法

测试代码在 `sglang/bi_sgl_offline_suite.py` 与 `sglang/bi_sgl_http_suite.py`，
与 vLLM 版本一一对应：一次引擎加载跑完五组离线测试，在线部分用原生 `/generate`
接口发请求。比较方式不变，仍是把 logprob 转成 IEEE-754 位模式逐位比。

换成 SGLang 需要改三处细节：

| 项目 | vLLM | SGLang 0.5.6 |
|---|---|---|
| 引擎 | `LLM(...)` | `sgl.Engine(...)` |
| 生成 | `llm.generate(prompts, SamplingParams(...))` | `engine.generate(input_ids=..., sampling_params={...}, return_logprob=True)` |
| logprob 字段 | `outputs[0].logprobs` | `meta_info["output_token_logprobs"]`，形如 `[[logprob, token_id, text], ...]` |
| prompt logprob | `prompt_logprobs` | `logprob_start_len=0` 时的 `meta_info["input_token_logprobs"]`，首 token 为 `null` |

另外两个行为需要提前知道，它们由 SGLang 自己决定，不是我们的配置：

- 打开确定性推理后，采样后端被强制改为 `pytorch`；
- flashinfer 后端不支持确定性模式下的 radix cache，所以引擎会强制
  `disable_radix_cache=True`，日志里有对应警告。这意味着第 5 组“长 prompt
  缓存命中”在 SGLang 上退化成“同一长 prompt 连续请求三次”，测的是调度与
  重复执行的一致性，而不是前缀缓存命中。

## 12.3 离线结果

四台 GPU 全部参与，Qwen3-30B-A3B，32 token 生成，greedy。“全等”指 logprob
逐位相同且 token id 相同；“ids 变化”统计 32 条里有多少条连 token 都变了。

### 表 S1：TP=4 与 PP=4

| 配置 | 32-prompt 全等 | ids 变化 | 批重复 | bs 扫描最大差 | 长度 284 全等 | decode vs re-feed | 长 prompt |
|---|---|---|---|---|---|---|---|
| 确定性 TP=4，CUDA graph 开 | 7/32 | 7 | 32/32 | 4.083e-02 | 0/32 | 64/64 不一致 | 0.0 ✅ |
| 确定性 TP=4，关 CUDA graph | 7/32 | 7 | 32/32 | 4.083e-02 | 0/32 | 64/64 不一致 | 0.0 ✅ |
| 确定性 TP=4，关 custom all-reduce | 7/32 | 7 | 32/32 | 4.083e-02 | 0/32 | 64/64 不一致 | 0.0 ✅ |
| 确定性 TP=4，triton attention | 15/32 | 3 | 32/32 | 5.003e-02 | 1/32 | 64/64 不一致 | 1.192e-07 ❌ |
| 关闭确定性 TP=4 | 0/32 | 13 | 32/32 | 2.187e-01 | 0/32 | 62/64 不一致 | 1.192e-07 ❌ |
| 确定性 PP=4，关 CUDA graph | 32/32 | 0 | 32/32 | 0.0 | 32/32 | 51/64 不一致 | 0.0 ✅ |
| 确定性 PP=4，CUDA graph 开 | 32/32 | 0 | 32/32 | 0.0 | 32/32 | 51/64 不一致 | 0.0 ✅ |

表里有三处需要解释：

- TP=4 下，25 条 prompt 的 logprob 不同，其中 7 条连生成的 token 都变了，
  文本会从这些位置分叉。vLLM 的 eager 模式在同样 32 条 prompt 上是 31 条
  logprob 不同、10 条 token 变化，两个引擎的失败量级接近。
- 关掉确定性开关后，ids 变化从 7 条涨到 13 条，最大差也翻倍，说明这个开关
  确实起了作用，只是不完整。
- 换 attention 后端、关 custom all-reduce、关 CUDA graph，都没有改变 25/32
  这个数字中的任何一个，三组配置的数值完全一样（4.083e-02）。能改变结果的
  只有并行方式。

### 表 S2：批大小扫描（target-only）

| bs | TP=4 确定性 | TP=4 关闭确定性 | PP=4 确定性 |
|---|---|---|---|
| 128 | 4.083e-02 ❌ | 6.740e-02 ❌ | 0.0 ✅ |
| 192 | 4.083e-02 ❌ | 2.187e-01 ❌（ids ✗） | 0.0 ✅ |
| 224 | 4.083e-02 ❌ | 2.187e-01 ❌（ids ✗） | 0.0 ✅ |
| 240 | 4.083e-02 ❌ | 2.187e-01 ❌（ids ✗） | 0.0 ✅ |
| 255 | 4.083e-02 ❌ | 2.187e-01 ❌（ids ✗） | 0.0 ✅ |
| 256 | 4.083e-02 ❌ | 2.187e-01 ❌（ids ✗） | 0.0 ✅ |
| 257 | 4.083e-02 ❌ | 6.324e-02 ❌ | 0.0 ✅ |

vLLM 的 eager 模式在 255 与 256 之间有干净的分界（表 I），SGLang 没有：
TP=4 下 128 到 257 的差值一直是 4.083e-02，等于说**批量之间彼此一致，只有
bs=1 与批量不同**。单条请求与批量请求走了两条不同的数值路径，差异只随并行
方式消失，不随批大小变化。

### 表 S3：小模型上的并行方式对照

为了把“模型大 / 有 MoE / 并行方式”三个因素分开，又用稠密模型
Qwen3-0.6B 做了一组同样的测试（其余参数相同，确定性开，CUDA graph 关）。

| 并行方式 | 32-prompt 全等 | ids 变化 | bs 扫描 | 长度 284 全等 |
|---|---|---|---|---|
| TP=1 | 32/32 | 0 | 0.0 ✅ | 32/32 |
| TP=2 | 32/32 | 0 | 0.0 ✅ | 32/32 |
| TP=4 | 0/32 | 6 | 1.347e+00 ❌ | 0/32 |
| TP=4，triton attention | 0/32 | 9 | 3.070e+00 ❌ | 0/32 |
| PP=4 | 32/32 | 0 | 0.0 ✅ | 32/32 |

稠密模型排除了 MoE 的影响：它在 TP=4 上同样是 0/32。TP=1 与 TP=2 通过、TP=4 失败，
说明分界在跨卡归约的规模上。PP=4 不做归约，同样通过。

这组数据也决定了 30B 的并行方式选择：BF16 权重约 61 GB，单卡 48 GB 放不下，
TP=1 不可行；TP=2 时每卡约 30.5 GB 权重，留给 KV cache 的只有约 17 GB，
长上下文的几组测试跑不完，所以 30B 只测了 TP=4 与 PP=4。稠密小模型的 TP=2
结果用来说明趋势：分界不在 TP>1，而在 TP 规模更大的情形。

## 12.4 在线结果

服务用 `python -m sglang.launch_server` 启动，其余参数与离线一致。在线测试
沿用第 5 章的四类场景：32 条顺序基线、32 条并发、32 条相同 prompt 并发、
以及一条长请求解码期间再发 32 条。

| 配置 | 顺序 vs 并发 | 相同 prompt 并发 | 顺序 vs 突发 | 长请求耗时 |
|---|---|---|---|---|
| 确定性 TP=4 | 4/32 通过 | 0/32 通过 | 7/32 通过 | 7.94 s |
| 关闭确定性 TP=4 | 0/32 通过 | 0/32 通过 | 0/32 通过 | 3.31 s |
| 确定性 PP=4 | 32/32 通过 | 32/32 通过 | 32/32 通过 | 17.63 s |

PP=4 的 96 次比较全部为 0.0，没有一条 token id 变化。也就是说，在流水线
并行下，SGLang 的在线服务确实做到了“同一请求，不管和谁一起进来，输出位级
相同”。

代价是速度。同样一条 400 token 的长请求，TP=4 用 7.94 s，PP=4 用 17.63 s，
慢了约一倍多；32 条并发的墙钟时间从 0.91 s 变成 2.22 s。流水线并行在
解码阶段天然有气泡，这个代价符合预期。

## 12.5 decode vs re-feed

五组离线测试里，唯一在 PP=4 下也没能通过的是 decode 与 re-feed 的一致性：
先逐 token 生成 64 个 token，再把它们拼回 prompt 一次性 prefill，比较同一批
位置的 logprob。

| 配置 | 不一致数 | 首个不一致位置 | 最大 \|Δlogprob\| |
|---|---|---|---|
| 确定性 TP=4 | 64/64 | 0 | 8.106e-02 |
| 关闭确定性 TP=4 | 62/64 | 0 | 1.577e-01 |
| 确定性 TP=4，triton attention | 64/64 | 0 | 2.049e-01 |
| 确定性 PP=4 | 51/64 | 12 | 1.141e-01 |
| 稠密 0.6B，TP=1 | 64/64 | 0 | 9.853e-02 |

这和 vLLM 的结论一致：prefill 与 decode 走的是不同的 attention kernel，
`--enable-deterministic-inference` 只保证“同一条路径在不同批形状下一致”，
不保证两条路径之间一致。PP=4 下首个不一致出现在第 12 位而不是第 0 位，
说明前 12 个 token 的 prefill 与 decode 恰好落在相同的数值路径上，之后才分开，
这属于具体的几何巧合，不能当成部分保证。

## 12.6 与 vLLM 的对照

| 引擎 / 配置 | 32-prompt 全等 | ids 变化 | 说明 |
|---|---|---|---|
| vLLM compiled，TP=4 | 32/32 | 0 | `torch.compile` + CUDA Graphs，BI kernel |
| vLLM eager，TP=4 | 1/32 | 10 | 只有 5-token target 通过 |
| SGLang 确定性，TP=4 | 7/32 | 7 | 默认配置 |
| SGLang 确定性，TP=4，triton attention | 15/32 | 3 | 最好的 TP=4 结果 |
| SGLang 关闭确定性，TP=4 | 0/32 | 13 | 对照组 |
| SGLang 确定性，PP=4 | 32/32 | 0 | 跨卡不做归约 |

两个引擎都能做到位级一致，但门槛不一样：

- vLLM 的路径是编译 + CUDA Graphs，在 TP=4 下成立；
- SGLang 的路径是 aten 覆写 + 固定 split 的 attention，在 TP=4 下不成立，
  换成 PP=4 才成立；
- 失败的形状依赖不同：vLLM eager 在 256 附近有清晰的批大小边界，SGLang
  TP=4 则是单条请求与任何批量之间都有差异，与批大小无关；
- decode 与 re-feed 在两边都不成立，属于共同短板。

## 12.7 结论与建议

- 想在 A6000 上用 SGLang 得到位级一致的输出，用 `--pp-size 4` 加
  `--enable-deterministic-inference`，不要用 TP=4；
- 如果必须用 TP=4（例如 PP 的吞吐代价不可接受），就只能接受“调度层确定、
  数值层不确定”的状态，不能对外声称位级可复现；
- decode 与 re-feed 的不一致在两个引擎上都存在，任何“用 re-feed 复核线上
  输出”的做法都需要单独验证；
- 版本影响很大：0.5.9 在 A6000 上直接段错误，0.5.6 可用；换版本后所有结论
  都应在目标机器上重跑一遍。

复现命令、原始 JSON 与脚本见[复现与文件清单](10_reproduction.md)。
