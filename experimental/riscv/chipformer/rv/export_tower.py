"""Export the RV32 machine and its text bridge as a standalone tower (layout "rv1-tower").

    python -m chipformer.rv.export_tower HOST --out dist/tower-27b \\
        [--firmware tests/data/rv/calc-bridge.elf]

HOST is a directory with the host model's ``config.json`` and ``tokenizer.json`` (for Bonsai 2:
Qwen3.8-27B's). The circuit is compiled for the host's DeltaNet shape (key heads, value heads,
MLP width), but the host's weights are not read and not changed: the tower has its own residual
stream (no host dims: ``chip_dims_offset`` 0), its own embedding table and its own states. A
decoder runs every token through the tower and reads the output code; text tokens also run
through the host, untouched, exactly as rv/chat.py's fast path does with a merged checkpoint.

Model ids are the merged layout's: a host token keeps its id, chip token t is ``vocab + t``.

Written files:

- ``model.safetensors``: the chip layers (HF names, bf16) and ``model.embed_tokens.weight``.
  The MLPs are cut to the units the circuit uses (the rest of the host's width is zero). With
  ``--firmware``, also ``firmware.NAME.state``: the parked firmware as the DeltaNet states,
  shape (layers, value heads, key dim, value dim), fp32.
- ``config.json``: architecture ``ChipTower``, the tower's text config, and ``chip_config``: the
  layout (rv/merge.py's ``chipformer.json`` fields) and the firmware names.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chipformer.q35.emit import Alloc, Layout, allocate, check_bf16, emit_layers
from chipformer.q35.ir import HD, Target
from chipformer.q35.merge import pick_hidden
from chipformer.q35.tiny import BASE
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.merge import text_rows
from chipformer.rv.tiny import embedding_matrix

MLP = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


@dataclass
class Tower:
    c: core.RvCore
    lay: Layout
    alloc: Alloc
    T: BR.Text
    v0: int  # the host's vocabulary size: chip token t has model id v0 + t
    host: dict  # the host's text config

    @property
    def keep(self) -> list[int]:
        return list(self.c.keep)

    @property
    def code_dims(self) -> list[int]:
        return [self.lay.dim(next(iter(e.terms))) for e in self.c.code]

    def model_id(self, t: int) -> int:
        """Stream token (chip token, or TEXT_TOKEN | host id) -> model id."""
        return t - S.TEXT_TOKEN if S.is_text(t) else self.v0 + t

    def text_config(self, intermediate: int) -> dict:
        n_chip = self.c.n_layers()
        tg = self.c.target
        return dict(BASE, hidden_size=self.lay.hidden,
                    vocab_size=-(-(self.v0 + S.VOCAB) // 64) * 64,
                    num_hidden_layers=n_chip + 1,
                    layer_types=["linear_attention"] * n_chip + ["full_attention"],
                    linear_num_key_heads=tg.key_heads, linear_num_value_heads=tg.value_heads,
                    intermediate_size=intermediate, tie_word_embeddings=False)


def build(host) -> Tower:
    host = Path(host)
    cfg = json.loads((host / "config.json").read_text())
    tc0 = cfg.get("text_config", cfg)
    T = BR.Text(host / "tokenizer.json")
    c = core.build(Target(tc0["linear_num_key_heads"], tc0["linear_num_value_heads"],
                          tc0["intermediate_size"]), T)
    alloc = allocate(c, None, set(c.keep), tc0["intermediate_size"])
    hidden, m = pick_hidden(0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=0, m=m, slots=alloc.slots)
    assert m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    assert T.n <= tc0["vocab_size"]
    return Tower(c, lay, alloc, T, tc0["vocab_size"], tc0)


def used_units(layers: dict, n_layers: int) -> int:
    """MLP units the circuit uses (they are emitted from unit 0 up)."""
    top = 0
    for L in range(n_layers):
        gate, up, down = (layers[f"model.layers.{L}.{k}"] for k in MLP)
        live = (gate != 0).any(1) | (up != 0).any(1) | (down != 0).any(0)
        top = max(top, int(live.nonzero().max()) + 1 if bool(live.any()) else 0)
    return top


def emit(tw: Tower) -> tuple[dict, dict]:
    """(state dict in bf16, text config): the chip layers, cut to their units, and the
    embedding table."""
    c, lay = tw.c, tw.lay
    n_chip = c.n_layers()
    full = tw.text_config(c.target.intermediate)
    layers = emit_layers(c, lay, full, 0, "model.", tw.alloc, set(c.keep))
    inter = -(-used_units(layers, n_chip) // 64) * 64
    sd = {}
    for k in list(layers):
        w = layers.pop(k)
        if k.endswith(MLP[:2]):
            assert not bool((w[inter:] != 0).any()), k
            w = w[:inter]
        elif k.endswith(MLP[2]):
            assert not bool((w[:, inter:] != 0).any()), k
            w = w[:, :inter]
        check_bf16(k, w)
        sd[k] = w.to(torch.bfloat16).contiguous()
    tc = tw.text_config(inter)
    emb = torch.zeros(tc["vocab_size"], lay.hidden, dtype=torch.bfloat16)
    emb[:tw.v0] = text_rows(c, lay, tw.T, torch.zeros(tw.v0, 0, dtype=torch.bfloat16), tw.v0)
    emb[tw.v0:tw.v0 + S.VOCAB] = embedding_matrix(c, lay, S.VOCAB).to(torch.bfloat16)
    check_bf16("embedding", emb.float())
    sd["model.embed_tokens.weight"] = emb
    return sd, tc


def meta(tw: Tower, tc: dict) -> dict:
    c, lay = tw.c, tw.lay
    chip = {"layout": "rv1-tower", "chip_token_offset": tw.v0, "chip_vocab": S.VOCAB,
            "chip_layers": c.n_layers(), "hidden": lay.hidden, "chip_dims_offset": 0,
            "ballast_dims": lay.m, "log2_ballast": lay.log2_ballast, "circuit": c.stats(),
            "slots": tw.alloc.n_slots, "code_dims": tw.code_dims,
            "tau_dim": lay.dim(next(iter(c.tau.terms))),
            "mu_dim": lay.dim(next(iter(c.mu.terms))), "nfam": core.NFAM, "npay": core.NPAY,
            "tool_call": [tw.T.TCO, tw.T.TCC], "printable": tw.T.printable,
            "eos": sorted({248046, tw.host.get("eos_token_id", 248044)}), "firmware": [],
            "host": {k: tw.host[k] for k in ("hidden_size", "intermediate_size", "vocab_size",
                                             "linear_num_key_heads", "linear_num_value_heads")}}
    return {"architectures": ["ChipTower"], "model_type": "chip_tower", **tc, "chip_config": chip}


def state(c) -> torch.Tensor:
    """The heads' initial states (set by core.snapshot) as (layers, value heads, key dim, value
    dim) fp32: what hf.inject writes into the HF cache."""
    nv = c.target.value_heads
    r = nv // c.target.key_heads
    st = np.zeros((c.n_layers(), nv, HD, HD), dtype=np.float32)
    for ni, (L, j) in hf.head_slots(c).items():
        init = c.nodes[ni].init or []
        for x in range(len(c.nodes[ni].vheads())):
            for row, vec in (init[x] if x < len(init) else {}).items():
                st[L, j * r + x, row] = vec
    return torch.from_numpy(st)


def parked(firmware) -> "ref.Machine":
    m = ref.Machine.from_program(elf.program(str(firmware)))
    BR.park(m)
    return m


def save(tw: Tower, out, firmware=None) -> dict:
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd, tc = emit(tw)
    info = meta(tw, tc)
    if firmware is not None:
        core.snapshot(tw.c, parked(firmware))
        name = Path(firmware).stem
        sd[f"firmware.{name}.state"] = state(tw.c)
        info["chip_config"]["firmware"].append(name)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(info, indent=1))
    return info


def load_reference(path, dtype=torch.bfloat16):
    """(text model, chip_config): the tower in stock HF modules, with the one zero full-attention
    layer HF's cache needs to count positions. ``hidden_states[chip_layers]`` is the residual
    after the chip layers."""
    from safetensors.torch import load_file
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    info = cfg.pop("chip_config")
    assert cfg.pop("architectures") == ["ChipTower"] and cfg.pop("model_type") == "chip_tower"
    model = q.Qwen3_5TextModel(Qwen3_5TextConfig(**cfg))
    for p in model.parameters():
        p.data.zero_()
    sd = {k.removeprefix("model."): v
          for k, v in load_file(str(path / "model.safetensors")).items()
          if not k.startswith("firmware.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_chip = info["chip_layers"]
    assert not unexpected, unexpected[:5]
    assert all(k.startswith(f"layers.{n_chip}.") or k == "norm.weight" for k in missing), missing[:5]
    return model.to(dtype).eval(), info


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export the RV32 machine as a standalone tower")
    ap.add_argument("host", help="directory with the host's config.json and tokenizer.json")
    ap.add_argument("--out", default="dist/tower")
    ap.add_argument("--firmware")
    args = ap.parse_args()
    info = save(build(args.host), args.out, args.firmware)
    chip = info["chip_config"]
    print(json.dumps({k: chip[k] for k in ("hidden", "chip_layers", "slots", "circuit")}
                     | {"intermediate": info["intermediate_size"]}))


if __name__ == "__main__":
    main()
