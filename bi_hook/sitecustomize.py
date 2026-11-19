"""Dump per-layer hidden states from inside vLLM's worker processes.

Put this directory on PYTHONPATH before the engine starts. Every Python
process that vLLM spawns imports sitecustomize, so the capture also happens
in the four tensor-parallel workers. vLLM does not forward every environment
variable to its workers, so the dump directory has a default here.

The capture is off unless <dump>/current_tag exists; the driver rewrites that
file before each generate() call, so runs can be told apart. Only rank 0
writes, and only for the module types listed in NAMES.
"""
import os

DUMP = os.environ.get("BI_DUMP") or os.path.expanduser("~/bi_layer_dump")
NAMES = {"Qwen3MoeDecoderLayer", "Qwen3MoeAttention",
         "Qwen3MoeSparseMoeBlock", "Qwen3MoeForCausalLM",
         "Attention", "QKVParallelLinear", "RowParallelLinear", "FusedMoE",
         "RMSNorm"}
LIMIT = int(os.environ.get("BI_DUMP_LIMIT", "6000"))

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

    def _dump(self, out, snapshot=None):
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
        # DecoderLayer returns (hidden_states, residual); the residual is the
        # tensor that actually carries the information to the next layer.
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
            if out[1].dim() == 2:
                payload["row1"] = out[1][0].detach().to("cpu", copy=True)
        torch.save(payload, os.path.join(folder, f"{_state['seq']:05d}.pt"))
        _state["seq"] += 1

    def __call__(self, *args, **kwargs):
        snapshot = None
        if type(self).__name__ in NAMES:
            try:
                first_arg = args[0] if args else None
                if isinstance(first_arg, torch.Tensor) and first_arg.dim() == 2:
                    snapshot = first_arg[0].detach().to("cpu", copy=True)
            except Exception:
                snapshot = None
        out = _orig_call(self, *args, **kwargs)
        try:
            _dump(self, out, snapshot)
        except Exception:
            pass
        return out

    nn.Module.__call__ = __call__
