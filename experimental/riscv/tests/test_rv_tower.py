"""The standalone tower's exported state and ternary planes (docs/tower.md)."""

import pytest
import torch

from chipformer.rv import core, export_tower, hf, ref, ternary
from chipformer.rv.programs import program


def test_exported_state_is_the_injected_state():
    c = core.build()
    p = program("signext")
    core.snapshot(c, ref.Machine.from_program(p))
    toks = ref.Machine.from_program(p).run()
    model, _, ids = hf.build_model(c, toks)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([[ids[None]]]), use_cache=True).past_key_values
    hf.inject(c, cache)
    st = export_tower.state(c)
    assert st.dtype == torch.float32 and st.shape[0] == c.n_layers()
    assert bool((st != 0).any())
    for L in range(c.n_layers()):
        assert torch.equal(cache.layers[L].recurrent_states[0][0].float(), st[L])


def circuit_like(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mags = torch.tensor([2.0**-12, 1.0, 7.5, 128.0, 2.0**30, 2.0**40])
    w = mags[torch.randint(0, len(mags), (rows, cols), generator=g)]
    w = w * (torch.rand(rows, cols, generator=g) < 0.05)
    return (w * (1 - 2 * torch.randint(0, 2, (rows, cols), generator=g))).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(7, 300), (130, 128), (5, 64)])
def test_planes_sum_back_bit_for_bit(shape):
    w = circuit_like(*shape, seed=shape[0])
    trits, scales = ternary.planes(w, 128)
    assert set(trits.unique().tolist()) <= {-1, 0, 1}
    assert int((trits != 0).sum(0).max()) == 1  # each weight sits in one plane
    assert torch.equal(ternary.rebuild(trits, scales, 128).float(), w.float())
    i = tuple((trits != 0).nonzero()[0].tolist())
    trits[i] = -trits[i]
    assert not torch.equal(ternary.rebuild(trits, scales, 128).float(), w.float())


def test_column_scales_are_used_where_they_need_fewer_planes():
    w = torch.zeros(8, 256)
    w[:, :6] = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])  # one coefficient per column
    r, trits, scales = ternary.best("down", w.to(torch.bfloat16), 128)
    assert r["scale_axis"] == "column" and r["planes"] == 1 and r["exact"]
    assert torch.equal(ternary.rebuild(trits, scales, 128).float(), w.T)
