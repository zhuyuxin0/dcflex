import numpy as np
from dcflex.config import Facility
from dcflex import model, protocol


def _inp(T=48, seed=0):
    rng = np.random.default_rng(seed)
    fac = Facility()
    a = {"trn": np.full(T, 20.0) * rng.uniform(0.8, 1.2, T), "bat": np.full(T, 7.5) * rng.uniform(0.8, 1.2, T)}
    p0 = np.full(T, 47.5)
    temp = 30 + 8 * np.sin(np.arange(T) / 24 * 2 * np.pi)
    return fac, model.inputs(fac, a, p0, temp, np.zeros(T))


def test_deadlines_and_conservation():
    fac, inp = _inp()
    o = model.optimize(inp, fac, np.linspace(100, 300, 48))
    for c in ("trn", "bat"):
        ca, cy, D = np.cumsum(inp["a_" + c]), np.cumsum(o["y_" + c]), fac.deadline_h[c]
        assert np.all(cy <= ca + 1e-4) and abs(cy[-1] - ca[-1]) < 1e-3
        assert np.all(cy[D:] >= ca[:-D] - 1e-4)


def test_shed_positive_and_bau_balance():
    fac, inp = _inp()
    ref = model.bau(inp, fac)["pg"]
    o = model.optimize(inp, fac, np.full(48, 200.0), shed=(np.arange(19, 23), ref))
    assert o["r"] > 0


def test_commit_reveal_roundtrip():
    c, rec = protocol.commit("e1", "t", range(19, 23), [1, 2, 3, 4], b"m", b"x")
    assert protocol.verify(c, rec, b"m", b"x")
    assert not protocol.verify(c, dict(rec, baseline_mw=[1, 2, 3, 5]), b"m", b"x")


def test_no_deferral_follows_bau_when_capacity_binds():
    fac, inp = _inp()
    inp["a_trn"] = inp["a_trn"].copy()
    inp["a_trn"][18:21] += 40.0          # arrivals exceed the 100 MW IT capacity for three hours
    assert (inp["p0"] + inp["a_trn"] + inp["a_bat"] > fac.it_mw).any()
    ref = model.bau(inp, fac)
    for lev in (("tes",), ("bess",)):
        o = model.optimize(inp, fac, np.full(48, 200.0), levers=lev, shed=(np.arange(19, 23), ref["pg"]))
        assert np.allclose(o["pit"], ref["pit"], atol=1e-4)
        assert o["r"] >= 0
