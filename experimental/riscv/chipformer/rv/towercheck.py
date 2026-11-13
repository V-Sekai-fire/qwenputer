"""Check an exported tower (rv/export_tower.py) in stock HF modules against the IR evaluator and
the reference machine, with the host's text teacher-forced (no host weights are needed).

    python -m chipformer.rv.towercheck TOWER HOST --firmware tests/data/rv/calc-bridge.elf \\
        [--mode decode|prefill] [--served] [--golden out.parquet] [--control rom|weight|fp16]

The stream is the one a decoder would produce: the chat prompt (with the calc tool), then for
each expression the host's tool call, the machine's run with its echoes, and a few text tokens.
At every position the residual after the chip layers must equal the IR's on the kept dims (output
code, tau, mu) bit for bit; the code must be set exactly where the reference says the circuit
owns the next token, and must name that token.

``--served`` runs the prompt from zero states and injects the snapshot after it, as a decoder
does (rv/chat.py); otherwise the snapshot is injected first, as the IR has it.

A control must fail: ``rom`` flips the low bit of each byte of the ROM word the parked machine
runs next, ``weight`` flips one MLP output weight, ``fp16`` keeps the DeltaNet states in fp16 (a
state row's ballast, 2^24, does not fit). With ``--control`` the exit status is 0 only if the check fails.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.rv import bridge as BR
from chipformer.rv import core, export_tower, hf
from chipformer.rv.chat import CALC_TOOL

EXPRS = ["123456789*987654321", "2*(3+4)-50", "(1.2 - 3) * 4.5 - 0.75", "-84/18", "7**77 % 1000003"]


def random_exprs(n: int, seed: int = 0) -> list[str]:
    """Integer, decimal and fraction expressions within the calc tool's limits."""
    import random

    rng = random.Random(seed)

    def integer(digits: int) -> str:
        return str(rng.randint(1, 10 ** rng.randint(1, digits)))

    def decimal() -> str:
        return f"{integer(7)}.{rng.randint(0, 99999):0{rng.randint(1, 5)}d}"

    out = []
    for _ in range(n):
        kind = rng.randrange(5)
        if kind == 0:
            out.append(f"{integer(28)}{rng.choice('+-*')}{integer(28)}")
        elif kind == 1:
            out.append(f"({integer(12)}{rng.choice('+-*')}{integer(12)}){rng.choice('+-*')}"
                       f"{integer(12)}")
        elif kind == 2:
            out.append(f"({decimal()}{rng.choice('+-*')}{decimal()}){rng.choice('+-*')}{decimal()}")
        elif kind == 3:
            out.append(f"{integer(9)}/{integer(9)}{rng.choice('+-')}{integer(6)}/{integer(6)}")
        else:
            out.append(f"{integer(20)}{rng.choice(['//', '%'])}{integer(9)}")
    return out


def call(T, expr: str) -> list[int]:
    return T.encode(f"<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n"
                    "</parameter>\n</function>\n</tool_call>")


def prompt_ids(host, ask: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(host)
    out = tok.apply_chat_template([{"role": "user", "content": ask}], tools=[CALC_TOOL],
                                  add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    if not isinstance(out, list):
        out = out["input_ids"]
    return [int(x) for x in out]


def stream(tw, firmware, prompt: list[int], exprs: list[str]):
    """(session, skip): the reference's stream, and the owned positions whose next token is
    prefilled history."""
    ses = BR.Session(tw.T, export_tower.parked(firmware))
    ses.history(prompt)
    skip = {i for i, q in enumerate(prompt) if q == tw.T.TCC}
    for e in exprs:
        ses.qwen(call(tw.T, e))
        ses.qwen(tw.T.encode(f"\n\nThe result is {e} = "))
    return ses, skip


def want(tw, firmware, toks: list[int]) -> np.ndarray:
    """The IR's values of the kept dims at every position."""
    core.snapshot(tw.c, export_tower.parked(firmware))
    carry, mats = {}, []
    for a in range(0, len(toks), 2048):
        mats.append(ir.evaluate(tw.c, toks[a:a + 2048], carry=carry)[:, tw.keep])
    return np.concatenate(mats)


def code_token(tw, code: np.ndarray) -> int | None:
    """rv/chat.py's code_token, as a stream token."""
    if not (code != 0).any():
        return None
    bits = (code > 0).astype(int).tolist()
    fid = sum(b << i for i, b in enumerate(bits[:core.NFAM]))
    p = sum(b << i for i, b in enumerate(bits[core.NFAM:]))
    return core.decode_code(fid, p, tw.T)


@torch.no_grad()
def run(tw, model, info, ids: list[int], inject_at: int, mode: str, state_dtype=None,
        spoil=None, verbose: bool = False) -> np.ndarray:
    """The residual after the chip layers on the kept dims, at every position of ``ids``.
    The snapshot is injected before position ``inject_at``."""
    n_chip = info["chip_layers"]
    cols = [tw.lay.dim(d) for d in tw.keep]
    zero = info["chip_token_offset"] + info["chip_vocab"]  # a padding row: all zero
    hs = []
    dev = next(model.parameters()).device

    def fwd(x, cache):
        o = model(input_ids=torch.tensor([x], device=dev), past_key_values=cache, use_cache=True,
                  output_hidden_states=True)
        return o.hidden_states[n_chip][0][:, cols].float().cpu().double().numpy(), \
            o.past_key_values

    def inject(cache):
        hf.inject(tw.c, cache)
        if spoil is not None:
            spoil(cache)
        if state_dtype is not None:
            for L in range(n_chip):
                st = cache.layers[L].recurrent_states
                st[0] = st[0].to(state_dtype)

    if inject_at == 0:
        _, cache = fwd([zero], None)
    else:
        h, cache = fwd(ids[:inject_at], None)
        hs.append(h)
    inject(cache)
    rest = ids[inject_at:]
    t0 = time.time()
    if mode == "prefill":
        for a in range(0, len(rest), 1024):
            h, cache = fwd(rest[a:a + 1024], cache)
            hs.append(h)
    else:
        for t, x in enumerate(rest):
            h, cache = fwd([x], cache)
            hs.append(h)
            if verbose and (t + 1) % 500 == 0:
                print(f"  {t + 1}/{len(rest)} tokens, {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(hs)


def compare(tw, ses, skip: set[int], mat: np.ndarray, got: np.ndarray, first: int = 0) -> dict:
    """``got`` against the IR (``mat``) and the reference's stream, from position ``first``."""
    toks, w = ses.toks, mat
    n_code = len(tw.c.code)
    inexact = int((got[first:] != w[first:]).any(1).sum())
    bad_own, bad_tok, owned = [], [], 0
    for i in range(first, len(toks) - 1):
        t = code_token(tw, got[i, :n_code])
        if (t is not None) != bool(ses.own[i]):
            bad_own.append(i)
        elif ses.own[i] and i not in skip:
            owned += 1
            if t != toks[i + 1]:
                bad_tok.append(i)
    return {"positions": len(toks) - first, "owned": owned, "inexact_positions": inexact,
            "owner_mismatch": len(bad_own), "wrong_tokens": len(bad_tok),
            "first_bad": min(bad_own + bad_tok, default=None),
            "ok": not inexact and not bad_own and not bad_tok}


def spoil_rom(tw, firmware):
    """Flip the low bit of each byte of the ROM word the parked machine runs next (the layout
    of core.byte_rows)."""
    m = export_tower.parked(firmware)
    wi = m.anchor + m.n
    per = 2048  # words per key head (core.snapshot: ROM_WORDS >> 11 heads)
    x, wl = (wi % per) // (per // 2), wi % (per // 2)
    row, half, k = (wl >> 4) + 64 * (wl >> 3 & 1), wl >> 3 & 1, wl & 7
    L, j = next(v for i, v in hf.head_slots(tw.c).items()
                if tw.c.nodes[i] is tw.c.tables[f"rom{wi // per}"])
    r = tw.c.target.value_heads // tw.c.target.key_heads

    def f(cache):
        st = cache.layers[L].recurrent_states[0][0, j * r + x]
        for b in range(4):
            col = 2 + 32 * half + 4 * k + b
            st[row, col] = float(int(st[row, col]) ^ 1)
    return f


def spoil_weight(model, info) -> None:
    w = model.layers[info["chip_layers"] // 2].mlp.down_proj.weight
    i, j = (w != 0).nonzero()[0].tolist()
    w.data[i, j] = -w.data[i, j]


def write_golden(path, tw, ses, got: np.ndarray, info: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {"position": pa.array(np.arange(len(ses.toks), dtype=np.int32)),
            "model_id": pa.array(np.array([tw.model_id(t) for t in ses.toks], dtype=np.int32)),
            "owned": pa.array(np.array(ses.own, dtype=np.bool_))}
    for k, d in enumerate([tw.lay.dim(d) for d in tw.keep]):
        cols[f"dim_{d}"] = pa.array(got[:, k].astype(np.float32))
    table = pa.table(cols).replace_schema_metadata(
        {"layout": info["layout"], "chip_token_offset": str(info["chip_token_offset"])})
    pq.write_table(table, str(path), compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check a tower in HF modules against the IR")
    ap.add_argument("tower")
    ap.add_argument("host")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--ask", default="Use calc: what is 123456789 * 987654321?")
    ap.add_argument("--exprs", nargs="*", default=EXPRS[:1])
    ap.add_argument("--random", type=int, default=0, help="add this many random expressions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--served", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--golden")
    ap.add_argument("--control", choices=["rom", "weight", "fp16"])
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tw = export_tower.build(args.host)
    model, info = export_tower.load_reference(args.tower)
    model = model.to(args.device)
    assert info["code_dims"] == tw.code_dims and info["hidden"] == tw.lay.hidden
    prompt = prompt_ids(args.host, args.ask)
    exprs = args.exprs + random_exprs(args.random, args.seed)
    ses, skip = stream(tw, args.firmware, prompt, exprs)
    mat = want(tw, args.firmware, ses.toks)
    core.snapshot(tw.c, export_tower.parked(args.firmware))
    ids = [tw.model_id(t) for t in ses.toks]
    first = len(prompt) if args.served else 0
    if args.control == "weight":
        spoil_weight(model, info)
    t0 = time.time()
    got = run(tw, model, info, ids, first, args.mode,
              state_dtype=torch.float16 if args.control == "fp16" else None,
              spoil=spoil_rom(tw, args.firmware) if args.control == "rom" else None, verbose=True)
    r = compare(tw, ses, skip, mat, got, first)
    r.update(mode=args.mode, served=args.served, control=args.control, runs=len(ses.runs),
             device=args.device, seconds=round(time.time() - t0, 1))
    print(json.dumps(r))
    if args.golden and r["ok"] and not args.control:
        write_golden(Path(args.golden), tw, ses, got, info)
    sys.exit(int(r["ok"]) if args.control else int(not r["ok"]))


if __name__ == "__main__":
    main()
