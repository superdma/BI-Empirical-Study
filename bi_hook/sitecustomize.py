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

DUMP = os.environ.get("BI_DUMP") or os.path.expanduser("~/bi_layer_dump")
NAMES = {"Qwen3MoeDecoderLayer", "Qwen3MoeAttention",
         "Qwen3MoeSparseMoeBlock", "Qwen3MoeForCausalLM",
         "Attention", "QKVParallelLinear", "RowParallelLinear", "FusedMoE",
         "RMSNorm"}
LIMIT = int(os.environ.get("BI_DUMP_LIMIT", "6000"))
META = os.environ.get("BI_NORM_META") == "1"
FULL = os.environ.get("BI_FULL_RANGE", "")


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
        return out

    nn.Module.__call__ = __call__
