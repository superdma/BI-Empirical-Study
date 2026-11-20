"""Dump per-layer hidden states from inside vLLM's worker processes.

Put this directory on PYTHONPATH before the engine starts. Every Python
process that vLLM spawns imports sitecustomize, so the capture also happens
in the four tensor-parallel workers. vLLM does not forward every environment
variable to its workers, so the dump directory has a default here.

The capture is off unless <dump>/current_tag exists; the driver rewrites that
file before each generate() call, so runs can be told apart. Only rank 0
writes, and only for the module types listed in NAMES.
"""
import json
import os

DUMP = os.environ.get("BI_DUMP") or os.path.expanduser("~/bi_project/data/bi_layer_dump")
NAMES = {"Qwen3MoeDecoderLayer", "Qwen3MoeAttention",
         "Qwen3MoeSparseMoeBlock", "Qwen3MoeForCausalLM",
         "Attention", "QKVParallelLinear", "RowParallelLinear", "FusedMoE",
         "RMSNorm"}
LIMIT = int(os.environ.get("BI_DUMP_LIMIT", "6000"))
META = os.environ.get("BI_NORM_META") == "1"
FULL = os.environ.get("BI_FULL_RANGE", "")
INPROC_SEQ = os.environ.get("BI_INPROC_SEQ", "")
INPROC_OUT = os.environ.get("BI_INPROC_OUT", "")


def _in_full(seq):
    if not FULL or ":" not in FULL:
        return False
    lo, hi = FULL.split(":")
    return int(lo) <= seq <= int(hi)

try:
    import torch
    import torch.nn as nn
except Exception:  # no torch in this interpreter
    torch = None

if torch is not None:
    try:
        os.makedirs(DUMP, exist_ok=True)
        with open(os.path.join(DUMP, f"_hook_loaded_{os.getpid()}"), "w") as handle:
            handle.write("ok\n")
    except OSError:
        pass

    _orig_call = nn.Module.__call__
    _state = {"tag": None, "seq": 0, "rank": None, "seen": set()}

    def _current_tag():
        try:
            with open(os.path.join(DUMP, "current_tag")) as handle:
                return handle.read().strip()
        except OSError:
            return None

    def _note(name):
        if name in _state["seen"] or len(_state["seen"]) > 400:
            return
        _state["seen"].add(name)
        if "Qwen" in name or "MoE" in name or "Attention" in name:
            try:
                with open(os.path.join(DUMP, f"_names_{os.getpid()}.txt"), "a") as h:
                    h.write(name + "\n")
            except OSError:
                pass

    def _dump(self, out, snapshot=None, full=None, full1=None):
        name = type(self).__name__
        _note(name)
        if name not in NAMES:
            return
        tag = _current_tag()
        if not tag:
            return
        if _state["rank"] is None:
            try:
                _state["rank"] = torch.distributed.get_rank()
            except Exception:
                _state["rank"] = 0
        if _state["rank"] != 0:
            return
        if tag != _state["tag"]:
            _state["tag"] = tag
            _state["seq"] = 0
            _state["inproc"] = False
        if _state["seq"] >= LIMIT:
            return
        first = out[0] if isinstance(out, tuple) else out
        if not isinstance(first, torch.Tensor) or first.dim() != 2:
            return
        folder = os.path.join(DUMP, tag, "rank0")
        os.makedirs(folder, exist_ok=True)
        payload = {
            "name": name,
            "shape": list(first.shape),
            "row0": first[0].detach().to("cpu", copy=True),
        }
        if snapshot is not None:
            payload["in0"] = snapshot
        if full is not None:
            payload["full_in"] = full
        if full1 is not None:
            payload["full_in1"] = full1
        # DecoderLayer returns (hidden_states, residual); the residual is the
        # tensor that actually carries the information to the next layer.
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
            if out[1].dim() == 2:
                payload["row1"] = out[1][0].detach().to("cpu", copy=True)
        torch.save(payload, os.path.join(folder, f"{_state['seq']:05d}.pt"))
        used = _state["seq"]
        _state["seq"] += 1
        return used

    def _meta(module, x, seq, out):
        """Record where the norm's tensors actually live.

        The replay needs to know whether the engine hands the kernel a plain
        contiguous buffer or a slice with a different base address, and which
        weight tensor it uses.
        """
        try:
            weight = getattr(module, "weight", None)
            wid = None
            if isinstance(weight, torch.Tensor):
                wid = id(weight)
                wfolder = os.path.join(DUMP, "weights")
                os.makedirs(wfolder, exist_ok=True)
                wpath = os.path.join(wfolder, f"{wid}.pt")
                if not os.path.exists(wpath):
                    torch.save(weight.detach().to("cpu", copy=True), wpath)
            rec = {
                "tag": _state["tag"], "seq": seq, "rank": _state["rank"],
                "x_shape": list(x.shape), "x_stride": list(x.stride()),
                "x_offset": x.storage_offset(), "x_ptr_mod": x.data_ptr() % 4096,
                "dtype": str(x.dtype),
                "eps": float(getattr(module, "variance_epsilon", -1.0)),
                "weight_id": wid,
                "out_shape": list(out.shape) if isinstance(out, torch.Tensor) else None,
                "out_stride": list(out.stride()) if isinstance(out, torch.Tensor) else None,
                "out_offset": out.storage_offset() if isinstance(out, torch.Tensor) else None,
                "out_ptr_mod": out.data_ptr() % 4096 if isinstance(out, torch.Tensor) else None,
            }
            with open(os.path.join(DUMP, "meta", f"{_state['tag']}_{os.getpid()}.jsonl"), "a") as h:
                h.write(json.dumps(rec) + "\n")
        except Exception:
            pass


    def _bits_view(t):
        return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)

    def _inproc_replay(module, pre, out, seq):
        """Repeat the norm call at one sequence index, inside the worker.

        The standalone replay already showed that the engine's own tensors
        produce the 255 result in a fresh process. Repeating the call here
        says whether the engine process itself is what differs.
        """
        if _state["rank"] != 0 or not INPROC_OUT:
            return
        # The clones were taken before the module ran: fused_add_rms_norm
        # updates its arguments in place, so reading them afterwards would
        # feed the result back in.
        x, residual, _ = pre
        from vllm import _custom_ops as ops
        weight = module.weight
        eps = float(module.variance_epsilon)
        fused = residual is not None
        if fused:
            xx, rr = x, residual
            ops.fused_add_rms_norm(xx, rr, weight, eps)
            ref = out[0] if isinstance(out, tuple) else out
            ref_r = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        else:
            xx = torch.empty_like(x)
            ops.rms_norm(xx, x, weight, eps)
            rr = ref_r = None
            ref = out
        rec = {
            "tag": _state["tag"], "seq": seq, "rank": _state["rank"],
            "rows": int(x.shape[0]), "fused": fused, "pid": os.getpid(),
            "x_row0_vs_engine_row0": int((_bits_view(x)[0] != _bits_view(ref)[0]).sum()),
            "replay_vs_engine_row0": int((_bits_view(xx)[0] != _bits_view(ref)[0]).sum()),
            "replay_vs_engine_all": int((_bits_view(xx) != _bits_view(ref)).sum()),
        }
        if ref_r is not None and rr is not None:
            rec["replay_vs_engine_res_row0"] = int((_bits_view(rr)[0] != _bits_view(ref_r)[0]).sum())
        folder = os.path.dirname(INPROC_OUT) or "."
        os.makedirs(folder, exist_ok=True)
        with open(INPROC_OUT, "a") as h:
            h.write(json.dumps(rec) + "\n")
        stem = os.path.join(folder, f"{_state['tag']}_{seq}_{os.getpid()}")
        torch.save({"x_row0": x[0].detach().to("cpu", copy=True),
                    "engine_row0": ref[0].detach().to("cpu", copy=True),
                    "replay_row0": xx[0].detach().to("cpu", copy=True),
                    "engine_res_row0": None if ref_r is None else ref_r[0].detach().to("cpu", copy=True),
                    "replay_res_row0": None if rr is None else rr[0].detach().to("cpu", copy=True)},
                   stem + ".pt")

    def __call__(self, *args, **kwargs):
        snapshot = None
        xin = None
        if type(self).__name__ in NAMES:
            try:
                first_arg = args[0] if args else None
                if isinstance(first_arg, torch.Tensor) and first_arg.dim() == 2:
                    snapshot = first_arg[0].detach().to("cpu", copy=True)
                    xin = first_arg
            except Exception:
                snapshot = None
        full = full1 = None
        if FULL and type(self).__name__ in NAMES:
            try:
                if _in_full(_state["seq"]):
                    full = args[0].detach().to("cpu", copy=True)
                    if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].dim() == 2:
                        full1 = args[1].detach().to("cpu", copy=True)
            except Exception:
                full = full1 = None
        pre = None
        if INPROC_SEQ and not _state.get("inproc") and type(self).__name__ == "RMSNorm":
            try:
                if _current_tag() and _state["seq"] == int(INPROC_SEQ):
                    a0 = args[0] if args else None
                    if isinstance(a0, torch.Tensor):
                        a1 = args[1] if len(args) > 1 else None
                        pre = (a0.clone(),
                               a1.clone() if isinstance(a1, torch.Tensor) else None,
                               _state["seq"])
            except Exception:
                pre = None
        out = _orig_call(self, *args, **kwargs)
        try:
            seq = _dump(self, out, snapshot, full, full1)
        except Exception:
            seq = None
        if META and seq is not None and type(self).__name__ == "RMSNorm" and xin is not None:
            try:
                _meta(self, xin, seq, out)
            except Exception:
                pass
        if pre is not None and seq is not None and seq == pre[2]:
            try:
                _state["inproc"] = True
                _inproc_replay(self, pre, out, seq)
            except Exception:
                _state["inproc"] = True
        return out

    nn.Module.__call__ = __call__
