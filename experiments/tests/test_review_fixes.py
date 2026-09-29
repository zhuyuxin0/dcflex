"""Regression tests for bugs found in code review."""
import json

import numpy as np
import pandas as pd

import run_all
from dcflex import analysis, baselines, export, model
from dcflex.config import Facility


def _inp(T=48, seed=0, pv_mw=0.0):
    rng = np.random.default_rng(seed)
    fac = Facility()
    a = {"trn": np.full(T, 20.0) * rng.uniform(0.8, 1.2, T), "bat": np.full(T, 7.5) * rng.uniform(0.8, 1.2, T)}
    p0 = np.full(T, 47.5)
    temp = 30 + 8 * np.sin(np.arange(T) / 24 * 2 * np.pi)
    pv = np.clip(pv_mw * np.sin((np.arange(T) % 24 - 6) / 12 * np.pi), 0, None)
    return fac, model.inputs(fac, a, p0, temp, pv)


def test_pv_is_self_consumed_when_the_price_is_zero():
    fac, inp = _inp(pv_mw=30.0)
    price = np.where(inp["pv"] > 0, 0.0, 150.0)       # zero price in PV hours, as in S2 surplus hours
    o = model.optimize(inp, fac, price)
    load = o["pg"] + o["pv_used"]
    assert np.allclose(o["pv_used"], np.minimum(inp["pv"], load), atol=1e-4)


def test_run_monthly_returns_chronological_order():
    fac, inp = _inp()
    idx = pd.date_range("2025-12-31 00:00", periods=48, freq="h")    # December then January
    out = model.run_monthly(inp, fac, np.full(48, 100.0), idx, strategy="bau")
    assert np.allclose(out["pit"], model.bau(inp, fac)["pit"])


def test_deferrable_share_sensitivity_is_symmetric():
    fac, inp = _inp()
    base = (inp["a_trn"] + inp["a_bat"]).sum()
    total = (inp["p0"] + inp["a_trn"] + inp["a_bat"]).sum()
    for f in (0.67, 1.33):
        out = analysis._shift_deferrable(inp, fac, f)
        assert abs((out["a_trn"] + out["a_bat"]).sum() / base - f) < 1e-9
        assert abs((out["p0"] + out["a_trn"] + out["a_bat"]).sum() - total) < 1e-6


def test_tornado_labels_have_plain_text_names():
    for label, *_ in analysis.SENS_CASES:
        assert label in run_all.TORNADO_TEXT and "$" not in run_all.TORNADO_TEXT[label]


def test_macro_number_format(tmp_path):
    assert export.fmt(-3.4) == "$-$3.4"
    assert export.fmt(-0.04) == "0.0"
    assert export.fmt(-17596.4) == "$-$17,596"
    export.write_macros({"RbauPUE": 1.3109, "RcommitBytes": 371.0, "RcapPayBreakeven": 50.0}, tmp_path)
    text = (tmp_path / "macros.tex").read_text()
    assert r"\newcommand{\RbauPUE}{1.31}" in text
    assert r"\newcommand{\RcommitBytes}{371}" in text
    assert r"\newcommand{\RcapPayBreakeven}{50.0}" in text


def test_window_ending_in_overload_is_feasible():
    fac, inp = _inp(T=24)
    inp["a_trn"] = inp["a_trn"].copy()
    inp["a_trn"][20:] += 45.0           # arrivals exceed IT capacity in the last hours of the window
    ref = model.bau(inp, fac)
    assert ref["y_trn"].sum() < inp["a_trn"].sum() - 1.0     # BAU cannot finish either
    o = model.optimize(inp, fac, np.full(24, 100.0))
    assert o["y_trn"].sum() >= ref["y_trn"].sum() - 1e-4
    assert np.all(o["pit"] <= fac.it_mw + 1e-6)


def test_baseline_inflation_is_not_compounded():
    L = np.ones((40, 24))
    ev_days, peak = [30, 32, 35], np.full(40, 19)
    Lg = baselines.inflate(L, ev_days, peak, 0.10, 10)
    assert np.isclose(Lg.max(), 1.10)                 # days 25-29 precede all three events: inflated once
    assert np.isclose(Lg[20:35].min(), 1.10) and Lg[19].max() == 1.0 and Lg[36].max() == 1.0
    Ls = baselines.inflate(L, [30], peak, 0.10, 0)    # pre-window hours only: 15:00-18:00 on day 30
    assert np.isclose(Ls[30, 15:19], 1.10).all() and Ls[30, 14] == 1.0 and Ls[30, 19] == 1.0
    Lm = baselines.inflate(L, [30], np.full(40, 1), 0.10, 0)   # event at 01:00: pre-hours reach day 29
    assert np.isclose(Lm[29, 21], 1.10) and np.isclose(Lm[30, 0], 1.10) and Lm[29, 20] == 1.0


def test_day_of_adjustment_uses_previous_evening_for_early_events():
    rng = np.random.default_rng(1)
    L = 100 + rng.normal(0, 1, (20, 24))
    hrs = [1, 2, 3, 4]                                 # event starting at 01:00 on day 15
    b0 = baselines.b_high(L, 15, hrs, set())
    L2 = L.copy()
    L2[14, 22] *= 1.1                                  # load at 22:00 the evening before
    assert not np.allclose(baselines.b_high(L2, 15, hrs, set()), b0)


def test_shopping_is_measured_against_the_true_reduction():
    credits = {"b1": [12.0, 10.0], "b2": [9.0, 11.0], "b3": [8.0, 8.0], "b4": [10.0, 10.0]}
    assert np.isclose(baselines.shopping(credits, 10.0), 15.0)   # (12-10)/10 and (11-10)/10


def test_verify_rejects_a_recomputed_baseline_of_the_wrong_shape():
    from dcflex import protocol
    c, rec = protocol.commit("e1", "t", range(19, 23), [5, 5, 5, 5], b"m", b"x")
    assert protocol.verify(c, rec, b"m", b"x", recompute=lambda: [5, 5, 5, 5])
    assert not protocol.verify(c, rec, b"m", b"x", recompute=lambda: [5.0])     # would broadcast
    c2, rec2 = protocol.commit("e2", "t", range(19, 21), [5, 5, 5, 5], b"m", b"x")
    assert not protocol.verify(c2, rec2, b"m", b"x")                              # window/baseline mismatch


def test_rule_based_tes_state_stays_nonnegative():
    _, inp = _inp(T=24 * 5)
    fac = Facility(tes_hours=1.0, tes_loss_h=0.01)     # small tank: emptied in the 17-22 h discharge window
    o = model.rule_based(inp, fac, np.arange(24 * 5) % 24)
    S = 0.5 * fac.tes_hours * fac.peak_heat_mw
    for qi, qo in zip(o["qin"], o["qout"]):
        S = (1 - fac.tes_loss_h) * S + fac.tes_eff * qi - qo
        assert S >= -1e-9


def test_facility_energy_includes_battery_losses():
    fac, inp = _inp()
    o = model.optimize(inp, fac, np.linspace(100, 300, 48))
    sig = dict(peak=np.arange(48) % 24 == 20, surplus=np.arange(48) % 24 == 12, ef=np.zeros(48))
    class S:
        ef_avg = 0.4
    sm = analysis.summarize(o, fac, sig, S, None)
    assert np.isclose(sm["energy_gwh"] * 1000, (o["pg"] + o["pv_used"]).sum())


def test_flex_envelope_uses_the_reliability_quantile():
    fac, inp = _inp(T=24 * 4)
    net = np.tile(np.r_[np.zeros(19), np.ones(5)], 4)          # daily net-load peak at 19:00
    days = [0, 1]
    lo = analysis.flex_envelope(inp, fac, net, days, durations=(4,), q=0.0)[0]["All levers"][4]
    hi = analysis.flex_envelope(inp, fac, net, days, durations=(4,), q=1.0)[0]["All levers"][4]
    _, raw = analysis.flex_envelope(inp, fac, net, days, durations=(4,))
    assert np.isclose(lo, min(raw["All levers"][4])) and np.isclose(hi, max(raw["All levers"][4]))


def test_weather_radiation_is_aligned_to_hour_start(tmp_path):
    from dcflex import data
    idx = pd.date_range("2025-06-21", periods=48, freq="h")
    raw = pd.DataFrame({"temp": np.arange(48.0), "rh": 50.0, "ghi": np.arange(48.0) * 10,
                        "dni": np.arange(48.0), "dhi": np.arange(48.0)}, index=idx)
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx], "temperature_2m": raw["temp"].tolist(),
              "relative_humidity_2m": raw["rh"].tolist(), "shortwave_radiation": raw["ghi"].tolist(),
              "direct_normal_irradiance": raw["dni"].tolist(), "diffuse_radiation": raw["dhi"].tolist()}
    # cache file = the raw API response, in the API's hour-ending labels
    data.weather_cache_path(1.0, 2.0, "a", "b", "UTC", cache_dir=tmp_path).write_text(json.dumps({"hourly": hourly}))
    w = data.fetch_weather(1.0, 2.0, "a", "b", "UTC", cache_dir=tmp_path)
    assert np.allclose(w["ghi"].to_numpy()[:-1], raw["ghi"].to_numpy()[1:])     # label t = mean over [t, t+1)
    assert np.allclose(w["temp"], raw["temp"])                                     # instantaneous values unchanged
    assert not w.isna().any().any()


def test_hourly_trace_without_value_column_is_rejected(tmp_path):
    import pytest
    from dcflex import data
    idx = pd.date_range("2024-05-13", periods=168, freq="h")
    pd.DataFrame({"time_gst": idx, "requests": np.arange(168) % 24 + 1.0}).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError):
        data.profile_from_csv(data.hourly_index(2025), tmp_path / "t.csv")
    p = data.profile_from_csv(data.hourly_index(2025), tmp_path / "t.csv", "requests")
    assert p.std() > 0.1


def test_pv_output_does_not_hide_pvlib_errors():
    import pytest
    from dcflex import data
    idx = pd.date_range("2025-06-21", periods=24, freq="h")
    w = pd.DataFrame({"temp": 35.0, "ghi": 500.0, "dni": 600.0, "dhi": 100.0}, index=idx)
    with pytest.raises(Exception):
        data.pv_output(w.drop(columns="dni"), 24.45, 54.38, 1.0, "Asia/Dubai")   # bad input must not fall back


def test_system_signal_rejects_degenerate_calibration():
    import pytest
    from dcflex.config import System
    from dcflex.signal import system_signal
    temp = 20 + 10 * np.sin(np.arange(8760) / 24 * 2 * np.pi)
    with pytest.raises(ValueError):
        system_signal(temp, np.zeros(8760), System(t_base=50.0))       # no hour above the base temperature


def test_profile_reads_offset_timestamps_on_the_gulf_clock(tmp_path):
    from dcflex import data
    idx = pd.date_range("2024-05-13 00:00", periods=168, freq="h", tz="UTC")
    val = np.where(idx.tz_convert("Asia/Dubai").hour == 15, 10.0, 1.0)        # peak at 15:00 Gulf time
    pd.DataFrame({"time": idx.astype(str), "v": val}).to_csv(tmp_path / "t.csv", index=False)
    index = data.hourly_index(2025)
    p = data.profile_from_csv(index, tmp_path / "t.csv", "v")
    assert index.hour[np.argmax(p)] == 15


def test_load_ercot_keeps_the_most_recent_hours_and_reads_offsets(tmp_path):
    from dcflex import data
    t = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    pd.DataFrame({"time": t.tz_convert("US/Central").astype(str), "rt_price": np.arange(48.0),
                  "as_price": 1.0}).to_csv(tmp_path / "e.csv", index=False)
    idx, rt, asp = data.load_ercot(tmp_path / "e.csv", 24)
    assert idx[0] == pd.Timestamp("2024-01-02 00:00") and rt[0] == 24.0


def test_placeholders_reset_existing_data_figures(tmp_path):
    (tmp_path / "figures").mkdir()
    old = tmp_path / "figures" / "fig-duck.pdf"
    old.write_bytes(b"stale figure from an earlier run")
    export.write_placeholders(tmp_path)
    assert old.read_bytes() != b"stale figure from an earlier run"


def test_synthetic_run_refuses_the_manuscript(tmp_path):
    import pytest
    (tmp_path / "main.tex").write_text("")
    with pytest.raises(SystemExit):
        run_all.check_root(tmp_path, synthetic=True)
    run_all.check_root(tmp_path, synthetic=False)
    run_all.check_root(tmp_path / "elsewhere", synthetic=False)    # allowed, with a note
    run_all.check_root(tmp_path / "elsewhere", synthetic=True)


def test_tables_and_macros_handle_missing_and_boundary_values(tmp_path):
    export.write_table("tab-value", [["ERCOT market", None, "S4"]], tmp_path)
    assert r"\tbd{n/a}" in (tmp_path / "tab-value.tex").read_text()
    assert export.fmt(float("nan")) is None
    assert export.fmt(999.96) == "1,000"


def test_cooling_share_counts_fans_and_pumps():
    fac, inp = _inp()
    b = model.bau(inp, fac)
    mask = np.ones(48, bool)
    expect = 100 * (b["pch"] + fac.fan_frac * b["pit"]).sum() / (b["pg"] + b["pv_used"]).sum()
    assert np.isclose(analysis.cooling_share(b, fac, mask), expect)
    assert analysis.cooling_share(b, fac, mask) > 100 * b["pch"].sum() / (b["pg"] + b["pv_used"]).sum()


def test_base_commitment_requires_a_sweep_point():
    import pytest
    sweep = {0: 0.0, 50: 36.7, 350: 36.7}
    assert analysis.base_commitment(sweep, 350.0) == 36.7
    with pytest.raises(ValueError):
        analysis.base_commitment(sweep, 300.0)
