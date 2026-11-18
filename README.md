# BI validation: Qwen3-30B-A3B on 4×RTX A6000 (vLLM 0.23.0)

This directory contains the scripts and raw results for the batch-invariance (BI)
validation. The current, authoritative write-up is the document collection in
[`docs/00_index.md`](docs/00_index.md). The original single-file report is kept as
a v1 archive; its target-only conclusions were superseded.

## Results (v2, 2026-11-17)

- Compiled mode (`torch.compile` + CUDA Graphs, no `--enforce-eager`) passed
  every in-process test: 32 distinct prompts solo-vs-batch (32/32), independent
  concurrent and bursty HTTP requests (32/32), in-batch prompt counts 8/16/32
  with 160/253/284/455 total prefill tokens, bs=256, 256-token
  decode-vs-refeed, and repeated batches. Solo baselines were identical
  across three warm-cache server processes.
- Eager mode is not batch-invariant for general prompts on this stack:
  with 32 distinct prompts, only the 5-token target matched (1/32); the other 31
  differed, some with different token ids. The same failure appears in a single
  batched request, in independent concurrent requests, and in offline
  `LLM.generate` (31/32; even two identical batch calls differed for 1 prompt).
- The eager boundary is shape-dependent: 32 prompts with 253 total prefill
  tokens passed, 284 failed; 16×30 tokens failed, 8×40 passed; bs=255 decode
  passed, bs=256 failed; re-feed passed at 197 total tokens, failed at 261.
- Eager + long prompt + prefix-cache hit can still change the first token
  (7584-token open-ended prompt); `--no-enable-prefix-caching` removes it.
  At 8180 tokens PC0 and PC1 happened to match in this run.
- Compiled and eager produce different outputs from the first token
  (max Δlogprob ~1.57e-01 for the target; 32/32 prompts differ). Each mode is
  deterministic across warm-cache restarts, but they must not be mixed.
- Official vLLM docs support CC≥8.0 and list Qwen3-30B-A3B, but the tuned
  matmul tables cover only Ada/Hopper/Blackwell. A6000 is Ampere (SM86), so it
  is nominally supported but outside the tuned path.

## SGLang replication (2026-11-18)

The same suites were re-run on SGLang 0.5.6 (torch 2.9.1+cu128, FlashInfer
0.5.3) with `--enable-deterministic-inference`; see
[`docs/12_sglang_results.md`](docs/12_sglang_results.md).

- With TP=4 the deterministic mode matched only 7/32 prompts (7 of them also
  changed token ids). Disabling custom all-reduce, disabling CUDA graphs, or
  switching to the Triton attention backend did not close the gap (25/32,
  25/32, 17/32 mismatches).
- With PP=4 (pipeline parallel, no cross-GPU reduction) the same model matched
  32/32 offline, all batch sizes 128-257 at 0.0, and 96/96 online comparisons.
- A dense 0.6B model isolates the cause: TP=1 and TP=2 pass, TP=4 fails
  (32/32 mismatches), PP=4 passes.
- Decode-vs-re-feed fails under every configuration in SGLang as well.
- SGLang 0.5.9 segfaults on the first `generate` on this box (all backends,
  CUDA-graph and determinism combinations); 0.5.6 works. Scripts and raw JSONs
  are in [`sglang/`](sglang/).

## Publish as a website

The report can be built into a static site with MkDocs Material:

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r requirements-docs.txt
bash scripts/build_site.sh     # -> site/
bash scripts/serve_site.sh     # preview at http://127.0.0.1:8001
```

See [`WEBSITE.md`](WEBSITE.md) for GitHub Pages, Cloudflare Pages, intranet
hosting and the repository privacy notes.

## Environment

- 4× NVIDIA RTX A6000 48GB (SM86, CC 8.6), driver 550.120
- vLLM 0.23.0+cu129, PyTorch 2.11.0+cu129, CUDA 12.9, NCCL 2.28.9
- Qwen/Qwen3-30B-A3B, BF16, TP=4
- No InfiniBand; `NCCL_IB_DISABLE=1` is required (NCCL 2.28.9 segfaults in its
  IB plugin otherwise). `VLLM_USE_FLASHINFER_SAMPLER=0` because `ninja` is absent.

## Recommended serving script

- `serve_bi_compiled.sh` — general BI, recommended.
- `serve_bi_eager.sh` — eager fallback for small, synchronized, verified
  prompt sets only. It defaults to `--max-num-seqs 128` and prints a warning.

## Files

| File | Purpose |
|---|---|
| `docs/00_index.md` | Document collection index and one-page conclusions |
| `docs/06_boundary_results.md` | v2 boundary experiments (tables I–Q) |
| `docs/08_deployment.md` | Deployment decision tree and verification checklist |
| `bi_test_qwen3_30b.py` | v1 offline BI harness |
| `compare_bi.py` | Bitwise comparison of two offline result JSONs |
| `run_all_bi.sh` | v1 compiled-mode matrix + BI-off control |
| `serve_bi_compiled.sh` | Recommended BI serving command |
| `serve_bi_eager.sh` | Eager fallback serving command |
| `bi_http_test.py` | v1 online BI test over `/v1/completions` |
| `run_http_matrix.sh` | v1 online H1–H6 servers |
| `bi_extra_qwen3_30b.py`, `bi_extra_diag.py` | v2 batch/TP/attention/bs256 tests |
| `run_extra_matrix.sh`, `run_diag_bs256*.sh` | v2 matrix launchers |
| `bi_refeed_diag.py`, `run_refeed_b.sh` | decode-vs-refeed length scan |
| `bi_http_batchall.py` | 32-prompt solo-vs-batch test |
| `bi_http_concurrent.py`, `bi_http_burst.py`, `bi_http_mixed.py` | arrival/concurrency tests |
| `bi_http_len.py` | in-batch prompt-length scan |
| `bi_batchall_offline.py` | offline 32-prompt confirmation |
| `analyze_extra.py`, `analyze_online_extra.py` | table generators |
| `results/*.json` | v1 offline results |
| `results/http/*.json` | v1 online results and long-prompt diagnostics |
| `results/extra/*.json` | batch/TP/attention/refeed results |
| `results/http/concurrency/*.json` | 32-prompt/concurrency/length results |
| `results/extra/analysis_tables.md` | Tables I–L, Q |
| `results/http/concurrency/analysis_tables.md` | Tables M–P |
| `mkdocs.yml`, `requirements-docs.txt` | Static site configuration |
| `scripts/build_site.sh`, `scripts/serve_site.sh` | Build / preview the website |
| `WEBSITE.md` | Hosting guide (GitHub Pages, Cloudflare, intranet) |

## Reproduce

```bash
# On the GPU host, with the model at $HOME/models/Qwen3-30B-A3B
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

bash run_all_bi.sh          # v1 offline
bash run_http_matrix.sh     # v1 online
bash run_extra_matrix.sh    # v2 batch/TP/attention
bash run_batchall.sh        # v2 32-prompt solo-vs-batch
bash run_concurrency.sh     # v2 independent concurrency
bash run_len.sh             # v2 in-batch length scan
bash run_refeed_b.sh        # v2 re-feed length scan

python analyze_extra.py
python analyze_online_extra.py
```

## Notes

- Comparisons are IEEE-754 bit patterns (`struct.pack(">d", float(x)).hex()`),
  not decimal text.
- Matching token ids is not sufficient: eager fails first on logprobs and can
  later flip the argmax.
- `init_batch_invariance()` runs in the EngineCore worker; the driver process's
  `_batch_invariant_MODE` stays `False` by design.
