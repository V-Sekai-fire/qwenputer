"""A tower's matrices as sums of ternary planes (docs/tower.md)."""

import json
from pathlib import Path

import torch

F16_MAX, F16_MIN = 65504.0, 2.0**-24


def planes(w: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    g = -(-cols // group)
    a = torch.zeros(rows, g * group, dtype=torch.float32)
    a[:, :cols] = w.float().abs()
    a = a.view(rows, g, group)
    s, idx = a.sort(-1)
    new = s != torch.cat([torch.zeros(rows, g, 1), s[..., :-1]], -1)
    rank_sorted = new.cumsum(-1)  # 0 on zeros, 1.. on the distinct magnitudes
    k = int(rank_sorted.max())
    rank = torch.zeros_like(rank_sorted).scatter_(-1, idx, rank_sorted)
    scales = torch.zeros(rows, g, k + 1, dtype=torch.float32)
    scales.scatter_(-1, rank_sorted, s)
    scales = scales[..., 1:].permute(2, 0, 1).contiguous()
    rank = rank.view(rows, g * group)[:, :cols]
    sign = torch.sign(w.float()).to(torch.int8)
    trits = torch.stack([sign * (rank == p + 1) for p in range(k)]) if k else \
        torch.zeros(0, rows, cols, dtype=torch.int8)
    return trits, scales.to(w.dtype)


def rebuild(trits: torch.Tensor, scales: torch.Tensor, group: int = 128) -> torch.Tensor:
    cols = trits.shape[2]
    out = torch.zeros(trits.shape[1:], dtype=torch.float32)
    for t, s in zip(trits, scales):
        out += t.float() * s.float().repeat_interleave(group, 1)[:, :cols]
    return out.to(scales.dtype)


def report(name: str, w: torch.Tensor, group: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    trits, scales = planes(w, group)
    same = bool((rebuild(trits, scales, group).float() == w.float()).all())
    nz = scales.float()[scales != 0]
    row = {"tensor": name, "shape": list(w.shape), "planes": trits.shape[0],
           "nonzero": round(float((w != 0).float().mean()), 5),
           "magnitudes": int(w.float().abs().unique().numel() - bool((w == 0).any())),
           "scale_min": float(nz.min()) if nz.numel() else 0.0,
           "scale_max": float(nz.max()) if nz.numel() else 0.0,
           "fp16": bool(nz.numel() == 0 or (nz.max() <= F16_MAX and nz.min() >= F16_MIN
                                           and bool((nz.half().float() == nz).all()))),
           "exact": same}
    return row, trits, scales


def best(name: str, w: torch.Tensor, group: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    r, trits, scales = report(name, w, group)
    r["scale_axis"] = "row"
    if r["planes"] > 3:
        r2, t2, s2 = report(name, w.T.contiguous(), group)
        if r2["planes"] < r["planes"]:
            r2.update(shape=list(w.shape), scale_axis="column")
            return r2, t2, s2
    return r, trits, scales


def main() -> None:
    import argparse

    from safetensors import safe_open
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(description="Ternary planes of an exported tower")
    ap.add_argument("tower")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--max-planes", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--control", action="store_true",
                    help="flip one sign in the first plane set: its rebuild must differ")
    args = ap.parse_args()
    path = Path(args.tower)
    rows, out = [], {}
    dense_bits = tern_bits = 0
    with safe_open(str(path / "model.safetensors"), "pt") as fh:
        for k in sorted(fh.keys()):
            w = fh.get_tensor(k)
            if w.ndim != 2:
                rows.append({"tensor": k, "shape": list(w.shape), "planes": None})
                if args.out:
                    out[k] = w
                dense_bits += w.numel() * 16
                tern_bits += w.numel() * 16
                continue
            r, trits, scales = best(k, w, args.group)
            if args.control:
                i = tuple((trits != 0).nonzero()[0].tolist())
                trits[i] = -trits[i]
                ref = w.T if r["scale_axis"] == "column" else w
                r["exact"] = bool((rebuild(trits, scales, args.group).float() == ref.float()).all())
                print(json.dumps({"control": k, "rebuild_equal": r["exact"]}))
                raise SystemExit(int(r["exact"]))
            rows.append(r)
            dense_bits += w.numel() * 16
            if r["planes"] <= args.max_planes and r["exact"]:
                tern_bits += trits.numel() * 2 + scales.numel() * 16
                if args.out:
                    axis = "t" if r["scale_axis"] == "column" else ""
                    out[f"{k}.trits{axis}"] = trits
                    out[f"{k}.scales{axis}"] = scales
            else:
                tern_bits += w.numel() * 16
                if args.out:
                    out[k] = w
            print(json.dumps(r), flush=True)
    two_d = [r for r in rows if r["planes"] is not None]
    summary = {"group": args.group, "matrices": len(two_d),
               "max_planes": max(r["planes"] for r in two_d),
               "planes_histogram": {str(p): sum(r["planes"] == p for r in two_d)
                                    for p in sorted({r["planes"] for r in two_d})},
               "all_exact": all(r["exact"] for r in two_d),
               "all_scales_fp16": all(r["fp16"] for r in two_d),
               "not_fp16": [r["tensor"] for r in two_d if not r["fp16"]][:8],
               "dense_gib": round(dense_bits / 8 / 2**30, 3),
               "ternary_gib": round(tern_bits / 8 / 2**30, 3)}
    print(json.dumps(summary))
    (path / "ternary.json").write_text(json.dumps({"summary": summary, "tensors": rows}, indent=1))
    if args.out:
        save_file(out, args.out, metadata={"group": str(args.group)})


if __name__ == "__main__":
    main()
