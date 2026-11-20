# 附录 B：精度说明与结果文件 SHA256

## B.1 为什么用位模式比较

logprob 是浮点数。两种常见但不够严格的比较：

- 打印到 6 位小数再比较：会漏掉 1e-7 级差异；
- 用 `abs(a-b) < 1e-6`：会漏掉 1e-7 级差异，也会把精确的 0 和“接近 0”混为一谈。

本报告统一用 IEEE-754 双精度的位模式：

```python
import struct
def bits(x):
    return struct.pack(">d", float(x)).hex()
```

只有 `bits(a) == bits(b)` 才算相等。表中的 `0.0` 表示位模式完全相同。

## B.2 HTTP JSON 的精度

vLLM 返回的 logprob 在 Python 中是 `float`（通常是 float32 计算后转成
Python float）。JSON 序列化使用最短可回读表示；Python 的 `json` 解析会把
它还原成同一个双精度值。因此：

- 同一个服务进程内，两次 HTTP 返回的 logprob 逐 bit 比较是有效的；
- 不同进程/不同版本之间，JSON 往返本身不会引入额外误差（除非上游改了
  精度或序列化）；
- 但 HTTP 层可能把 `None` 序列化为 `null`，比较脚本必须处理缺失值。

## B.3 token_logprobs 与 prompt_logprobs

- `choices[].logprobs.token_logprobs[i]`：第 i 个生成 token 的 logprob；
- `choices[].logprobs.top_logprobs[i]`：该位置 top-k 的字典；
- `choices[].prompt_logprobs[pos]`：把整段 prompt 一次 prefill 时，位置 `pos`
  的 token logprob；`pos = len(original_prompt) + i` 对应第 i 个生成 token；
- 某些实现会在 `prompt_logprobs[0]` 放 `None`（第一个 token 没有前文）。

## B.4 为什么 `max_abs_logprob_diff` 可能是 `null`

如果所有位置都缺失（例如 logprobs 未请求、或 token 不在 top-k 且实现未补），
差值为空。报告中的比较会同时给出 `n_compared`/`n_mismatch`，不能只看 max。

## B.5 采样与 seed

- greedy（temperature=0）：argmax，seed 不影响结果；
- temperature>0：vLLM 按请求 seed 采样；本次 temp=0.7 的实验在 bs32/bs256
  下 token id 仍然相同，但 logprob 已经不同；
- 不能因为“token id 相同”就判定 BI 通过。

## B.6 结果文件 SHA256

```text
766132334e4172304420908ff50c462b36a3b9b56aae6eb356dd9242f8931bcc  results/extra/bi_batchall_offline_eager.json
9c3888194e739c850f71d4013dd0192857b32276d197b0f857607711d069a48e  results/http/concurrency/batchall_eager_pc0.json
9f45c14853ae8aec82f831af5cfe19702315533aac61bf986ba22df08de70f4e  results/http/concurrency/batchall_compiled_pc1.json
8832e07dfabb418ec1b4d284e8fe1278f92cf3ae50b500eaddc9804a939b316b  results/http/concurrency/bi_http_len_eager_n32_l9.json
376d82ccdb40b8a0605c4374b3aa7bd933d7bef39fc57a4a07798a5dce146a39  results/http/concurrency/bi_http_len_compiled_n32_l9.json
92c49bcbe8e9c204bb796506b837ca552c6659a0dfa295fb156d416ceeddeb48  results/http/concurrency/bi_http_conc_compiled_pc1.json
da3fc62a0965bba74cc01d5b05834982b35b61aabdf0d5b18ae39b43650ea1bb  results/http/concurrency/burst_eager_pc0.json
0b15147b1a6289cc100402087ed7db8cd435397bd9b7e32ec12ef46bc7a1767f  results/http/concurrency/burst_compiled_pc1.json
66d560c28c02ae252707ea718b9e3ecba32e17d6e0ff1f5f110b9f7733b230d8  results/extra/bi_diag_bs256_pc0.json
4de798613587a42e672be732f8c53eab7fed4f21c16bb0bf7ae4133f8ce94ead  results/extra/bi_diag_bs256_comp_pc0.json
f83efb66ce44a3406fb7c0b31183c951680be7df312b4e5b1971bfbdaa5faacf  results/extra/bi_refeed_eager_pc0_mt256.json
702b0fdb82f0e88850a3ce8eb2538f433f279ed345da4a56320b32046cc1d041  results/extra/bi_refeed_compiled_pc0_mt256.json
b626cd1a5241075e6b45af988d1d8a92bec240caddaebc89a4606e35fce3f352  results/extra/analysis_tables.md
7e2032ee1b31fae0cc9b658c643ff0dc45bdaccfa743b3d6b96dbbc04050602a  results/http/concurrency/analysis_tables.md
```

生成全部哈希：

```bash
find results -name '*.json' | sort | while read -r r; do sha256sum "$r" \
  done > results/SHA256SUMS
```

## B.7 表中的“0.0”不等于“没有比较”

每个比较表都同时给出：

- `n_compared`：参与比较的位置数；
- `n_mismatch`：不一致的位置数；
- `first_mismatch`：第一个不一致的位置（None 表示没有）。

如果 `n_compared` 为 0，不能把 max 的缺失当作通过。
