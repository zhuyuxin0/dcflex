"""Tests for the learned baselines (B5), cross-building pretraining, forecast-driven MPC and real-data helpers."""
import numpy as np
import pandas as pd
import torch

from dcflex import learned_batch as lb, pretrain as pt, mpc, model, realdata, baselines
from dcflex.config import Facility, System, Study
from dcflex.signal import system_signal, price_from_net
from tests.test_core import _inp


def _toy(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * 24)
    hour, dow = t % 24, (t // 24) % 7
    theta = 30 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.15 * (t / 24) + rng.normal(0, 0.5, len(t))
    base = 60 + 10 * np.sin(2 * np.pi * (hour - 14) / 24)
    y = base * (1 + 0.01 * np.maximum(theta - 25, 0)) + rng.normal(0, 0.5, len(t))
    return hour, dow, theta, np.full(len(t), 15.0), np.zeros(len(t)), y


def test_pgnb_response_is_monotone_and_convex_for_any_weights():
    torch.manual_seed(0)
    net = lb.PGNB(3, use_dew=False)
    with torch.no_grad():
        for q in (net.ar, net.ah, net.r0):
            q.copy_(torch.randn_like(q) * 3)
    th = torch.linspace(0, 55, 200)
    n = len(th)
    cal = torch.tensor(lb.calendar_features(np.full(n, 15), np.full(n, 2)))
    out = net(cal, lb._hinge_basis(th, lb.KNOTS), lb._hinge_basis(torch.zeros(n), lb.DEW_KNOTS), th,
              torch.zeros(n), torch.zeros(n)).detach().numpy()
    d1 = np.diff(out, axis=1)
    assert np.all(d1 >= -1e-5)                              # increasing in temperature
    assert np.all(np.diff(d1, axis=1) >= -1e-4)             # convex


def test_fit_many_learns_and_extrapolates_upward():
    hour, dow, theta, dew, pv, y = _toy()
    train = np.arange(len(y)) < 30 * 24
    pred, info = lb.fit_many("pgnb", hour, dow, theta, dew, pv, y[:, None], train, seeds=(0,), epochs=400,
                             use_dew=False)
    p = pred(hour, dow, theta, dew, pv)[:, 0]
    assert np.mean(np.abs(p[train] - y[train])) / y.mean() < 0.03
    hot = pred(np.full(2, 15), np.full(2, 2), np.array([35.0, 45.0]), np.full(2, 15.0), np.zeros(2))[:, 0]
    assert hot[1] > hot[0]


def test_committed_b5_is_cached_and_serializes_deterministically():
    hour, dow, theta, dew, pv, y = _toy(n_days=10)
    train = np.arange(len(y)) < 7 * 24
    f = lb.committed(hour, dow, theta, dew, pv, train, seeds=(0,), epochs=30)
    a = f(y)
    y2 = y.copy(); y2[~train] += 5.0                         # changes after the training window only
    assert f(y2) is a                                       # same pre-season data: the committed model is reused
    b1, b2 = lb.model_bytes(f.info["net"]), lb.model_bytes(f.info["net"])
    assert b1 == b2 and len(b1) > 1000


def test_shared_grid_matches_row_forward():
    for kind in ("pgnb", "mlp"):
        net = pt.Shared(5, kind=kind)
        N = 30
        hour, dow = np.arange(N) % 24, (np.arange(N) // 24) % 7
        cal, BT, BD, th, dw = pt._inputs(hour, dow, np.linspace(10, 45, N), np.linspace(0, 25, N))
        E = net.emb[:2]
        g = net.grid(E, cal, BT, BD, th, dw)
        r = torch.stack([net(E[i].expand(N, -1), cal, BT, BD, th, dw) for i in range(2)])
        assert torch.allclose(g, r, atol=1e-5)


def test_mpc_with_perfect_forecasts_meets_deadlines_and_restores_storage():
    fac, inp = _inp(T=96)
    price = 100 + 100 * (np.arange(96) % 24 >= 12)
    actual = dict(p0=inp["p0"], cop=inp["cop"], pv=inp["pv"], a_trn=inp["a_trn"], a_bat=inp["a_bat"])
    for c in ("trn", "bat"):
        actual["a_" + c] = actual["a_" + c].copy()
        actual["a_" + c][60:] = 0.0
    fc = {k: mpc.forecast_fn(v, H=12) for k, v in actual.items()}
    o = mpc.simulate(fac, actual, fc, mpc.forecast_fn(price, H=12), 0, 96, H=12)
    assert max(o["late_trn"].max(), o["late_bat"].max()) < 1e-6
    for c in ("trn", "bat"):
        assert abs(o["y_" + c].sum() - actual["a_" + c].sum()) < 1e-3
    cap = fac.tes_hours * fac.peak_heat_mw
    assert o["soc"][-1] >= 0.5 * cap - 1e-3 and o["soe"][-1] >= 0.9 * fac.bess_mwh - 1e-3


def test_mos_is_causal():
    rng = np.random.default_rng(1)
    f = rng.normal(0, 1, 20 * 24)
    a = f + 1.0
    base = mpc.mos(f, a)
    a2 = a.copy(); a2[10 * 24:] += 50.0                    # change the actuals from day 10 on
    alt = mpc.mos(f, a2)
    assert np.allclose(base[:11 * 24], alt[:11 * 24])      # forecasts up to day 11 cannot see them
    assert np.allclose(base[5 * 24:], f[5 * 24:] + 1.0)


def test_price_from_net_matches_system_signal():
    s = System()
    t = 25 + 15 * np.sin(np.arange(8760) / 8760 * 2 * np.pi) + 5 * np.sin(np.arange(8760) / 24 * 2 * np.pi)
    pvn = np.clip(np.sin((np.arange(8760) % 24 - 6) / 12 * np.pi), 0, None)
    sig = system_signal(t, pvn, s)
    assert np.allclose(price_from_net(sig["net"], s, sig["calib"]), sig["price"])
    again = system_signal(t, pvn, s, calib=sig["calib"])
    assert np.allclose(again["price"], sig["price"])


def test_expected_price_is_mean_over_error_samples():
    f = lambda n: np.where(np.asarray(n) >= 1.0, 100.0, 10.0)
    ex = mpc.expected_price(np.array([0.9, 2.0]), np.array([[0.2, -0.2], [np.nan, np.nan]]), f)
    assert np.allclose(ex, [55.0, 100.0])


def test_clean_marks_long_constant_runs():
    y = np.r_[np.arange(10.0), np.zeros(30), np.arange(1.0, 6.0), np.full(10, 3.0)]
    c = realdata.clean(y, max_run=24)
    assert np.isnan(c[10:40]).all() and np.isfinite(c[40:]).all()


def test_event_days_skip_weekends_and_holidays():
    idx = pd.date_range("2017-01-01", "2017-12-31 23:00", freq="h")
    temp = np.where(idx.month.isin([6, 7, 8, 9]), 30.0, 10.0) + (idx.dayofyear == 185) * 20.0   # 4 July
    ev = realdata.event_days(temp, idx, n=5)
    days = pd.DatetimeIndex([idx[d * 24] for d in ev])
    assert (days.dayofweek < 5).all() and not (days == pd.Timestamp("2017-07-04")).any()
    assert days.month.isin([6, 7, 8, 9]).all()


def test_pre_season_mask_is_sixty_days_before_summer():
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    m = baselines.pre_season(idx, Study())
    days = pd.DatetimeIndex(idx[m]).normalize().unique()
    assert len(days) == 60 and days[-1] == pd.Timestamp("2025-05-31")
