"""Tests for the modelling decisions taken after the provisional run (deferrable share, ERA5 weather, peaker quantile, trace mapping)."""
import numpy as np
import pandas as pd

from dcflex import data
from dcflex.config import Facility


def test_deferrable_work_is_20_percent_of_mean_it_load():
    fac = Facility()
    idx = pd.date_range("2025-01-01", periods=24 * 28, freq="h")
    flat = np.ones(len(idx))
    a, p0 = data.build_arrivals(idx, fac, np.random.default_rng(0), flat, flat)
    mean_it = (p0 + a["trn"] + a["bat"]).mean()
    assert abs(mean_it - fac.util_mean * fac.it_mw) < 1e-6
    assert abs((a["trn"] + a["bat"]).mean() / mean_it - 0.20) < 1e-9


def _small_inp(T=48, p0=47.5, trn=20.0, bat=7.5):
    from dcflex import model
    fac = Facility()
    a = {"trn": np.full(T, trn), "bat": np.full(T, bat)}
    temp = 30 + 8 * np.sin(np.arange(T) / 24 * 2 * np.pi)
    return fac, model.inputs(fac, a, np.full(T, p0), temp, np.zeros(T))


def test_sensitivity_ranges_match_appendix_a():
    import pathlib
    from dcflex import analysis
    from dcflex.config import Study
    r = Study().sens
    assert {key for _, key, _ in analysis.SENS_CASES} == set(r)
    appendix = pathlib.Path(__file__).resolve().parents[2] / "sections" / "A-parameters.tex"
    if not appendix.exists():   # the code-only public release has no manuscript
        import pytest
        pytest.skip("manuscript (sections/A-parameters.tex) not present")
    app = appendix.read_text()
    fac = Facility()
    d = {c: fac.deadline_h[c] for c in ("trn", "bat")}
    lo_d, hi_d = r["deadline"]
    expected = [
        f"{r['cop_slope'][0]:.2f}--{r['cop_slope'][1]:.2f} K",
        f"{r['cop_ref'][0]:g}--{r['cop_ref'][1]:g}",
        f"{r['tes_hours'][0]:g}--{r['tes_hours'][1]:g} h",
        f"{r['bess_mw'][0]:g}--{r['bess_mw'][1]:g} MW",
        f"{d['trn'] * lo_d:g}/{d['bat'] * lo_d:g}--{d['trn'] * hi_d:g}/{d['bat'] * hi_d:g} h",
        f"{r['share_pct'][0]:g}--{r['share_pct'][1]:g}\\%",
        f"{r['pv_gw'][0]:g}--{r['pv_gw'][1]:g} GW"]
    for text in expected:
        assert text in app, text


def test_base_deferrable_share_pct():
    from dcflex import analysis
    assert abs(analysis.deferrable_share_pct(Facility()) - 20.0) < 1e-9


def test_zero_deferrable_share_never_pushes_p0_above_capacity():
    from dcflex import analysis
    fac, inp = _small_inp(p0=80.0, trn=20.0, bat=10.0)     # 110 MW of work in some hours
    out = analysis._shift_deferrable(inp, fac, 0.0)
    assert out["p0"].max() <= fac.it_mw + 1e-9
    total = inp["p0"] + inp["a_trn"] + inp["a_bat"]
    assert np.allclose(out["p0"] + out["a_trn"] + out["a_bat"], total)
    fac, inp = _small_inp()                                  # enough room: all work becomes inflexible
    out = analysis._shift_deferrable(inp, fac, 0.0)
    assert np.allclose(out["a_trn"] + out["a_bat"], 0.0)


def test_range_ends_are_feasible():
    import copy
    from dcflex import model
    from dcflex.config import Study
    fac, inp = _small_inp()
    r = Study().sens
    f = copy.deepcopy(fac); f.tes_hours = r["tes_hours"][0]
    model.optimize(inp, f, np.full(48, 100.0))
    f = copy.deepcopy(fac); f.bess_mw, f.bess_mwh = r["bess_mw"][0], fac.bess_mwh * r["bess_mw"][0] / fac.bess_mw
    model.optimize(inp, f, np.full(48, 100.0))


def test_weather_request_pins_era5_and_caches_by_model(tmp_path, monkeypatch):
    import json
    import requests
    from dcflex.config import Study
    assert Study().weather_model == "era5"
    seen = {}

    class Resp:
        ok, status_code = True, 200
        text = json.dumps({"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                                      "temperature_2m": [20.0, 21.0], "relative_humidity_2m": [50.0, 50.0],
                                      "shortwave_radiation": [0.0, 0.0], "direct_normal_irradiance": [0.0, 0.0],
                                      "diffuse_radiation": [0.0, 0.0]}})

    def fake_get(url, params=None, timeout=None):
        seen.update(params)
        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    w = data.fetch_weather(1.0, 2.0, "2025-01-01", "2025-01-01", "Asia/Dubai", cache_dir=tmp_path)
    assert seen["models"] == "era5" and list(w["temp"]) == [20.0, 21.0]
    path = data.weather_cache_path(1.0, 2.0, "2025-01-01", "2025-01-01", "Asia/Dubai", "era5", tmp_path)
    assert path.exists() and "era5" in path.name
    # a second read uses the cache, not the network
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    data.fetch_weather(1.0, 2.0, "2025-01-01", "2025-01-01", "Asia/Dubai", cache_dir=tmp_path)


def test_cached_api_error_is_not_read_as_weather(tmp_path):
    import pytest
    path = data.weather_cache_path(1.0, 2.0, "a", "b", "UTC", cache_dir=tmp_path)
    path.write_text('{"reason":"Daily API request limit exceeded.","error":true}')
    with pytest.raises(RuntimeError):
        data.fetch_weather(1.0, 2.0, "a", "b", "UTC", cache_dir=tmp_path)


def test_open_cycle_tier_is_reached_at_the_net_load_peak():
    from dcflex.config import System
    from dcflex.signal import system_signal
    rng = np.random.default_rng(1)
    idx = data.hourly_index(2025)
    w = data.synthetic_weather(idx, rng)
    pv_norm = np.clip(np.sin((idx.hour.to_numpy() - 6) / 12 * np.pi), 0, None)
    s = System()
    sig = system_signal(w["temp"].to_numpy(float), pv_norm, s)
    top = sig["net"] > sig["net"].max() - s.ocgt_gw
    assert top.any() and np.isclose(sig["n_cc"], sig["net"].max() - s.ocgt_gw)
    assert np.all(sig["mc"][top] == s.cost_ocgt) and np.all(sig["ef"][top] == s.ef_ocgt)
    mid = (sig["net"] > 0) & ~top
    assert mid.any()
    # merit order: within the combined-cycle band cost and emissions rise with net load
    order = np.argsort(sig["net"][mid])
    assert np.all(np.diff(sig["mc"][mid][order]) >= -1e-12) and np.all(np.diff(sig["ef"][mid][order]) >= -1e-12)
    lo, hi = s.fuel_aed_kwh(s.eta_ccgt_max), s.fuel_aed_kwh(s.eta_ccgt_min)
    assert lo - 1e-12 <= sig["mc"][mid].min() and sig["mc"][mid].max() <= hi + 1e-12 < s.cost_ocgt
    # S2 price = energy cost + peaking-capacity cost spread over the peak set K
    adder = sig["price"] - 1000 * sig["mc"]
    assert np.allclose(adder[~sig["peak"]], 0) and np.allclose(adder[sig["peak"]], 1000 * s.cap_value / s.peak_hours)
    assert sig["peak"].sum() == s.peak_hours


def test_system_parameters_follow_their_sources():
    from dcflex.config import System, AED_PER_USD
    s = System()
    assert abs(s.energy_2024_gwh / (s.peak_2024_mw / 1000 * 8784) - 0.6586) < 1e-4       # 2024 Global load factor
    assert abs(s.energy_twh - 151.0) < 0.1
    assert abs(s.cost_ccgt - 3.0 * 3.412 / 0.55 / 1000 * AED_PER_USD) < 1e-12              # band mean
    assert abs(s.ef_ccgt - 0.3672) < 1e-3 and abs(s.ef_ocgt - 0.594) < 1e-3                # IPCC / efficiency
    assert abs(s.cap_value / AED_PER_USD - 78.57) < 0.01                                   # EIA Case 4 at 7%, 25 yr
    assert abs(s.nuclear_gw - 4 * 1.337) < 1e-9


def test_delays_per_class_and_pooled_by_energy():
    from dcflex import analysis
    T = 48
    a = {"trn": np.zeros(T), "bat": np.zeros(T)}
    a["trn"][0], a["bat"][0] = 30.0, 10.0
    y = {"trn": np.zeros(T), "bat": np.zeros(T)}
    y["trn"][3], y["bat"][10] = 30.0, 10.0                 # training waits 3 h, batch 10 h
    d = analysis.delays(a, y)
    assert d["trn"] == 3 and d["bat"] == 10
    assert d["all"] == 10                                   # batch is 25% of the energy, above the 5% tail
    y["bat"][:] = 0; y["bat"][1] = 10.0                     # batch now waits 1 h: training dominates
    assert analysis.delays(a, y)["all"] == 3
    y["trn"][:] = 0; y["trn"][3] = 15.0                     # half of training never served in the window
    assert analysis.delays(a, y)["trn"] == 3


def test_dr_contract_returns_the_schedule_at_the_base_payment():
    from dcflex import analysis
    from dcflex.config import Study
    fac, inp = _small_inp(T=24 * 8)
    idx = pd.date_range("2025-07-01", periods=24 * 8, freq="h")
    net = np.tile(np.r_[np.zeros(18), np.ones(6)], 8)
    st = Study(events=2, cap_pay_sweep=(0, 350))
    bau = __import__("dcflex.model", fromlist=["bau"]).bau(inp, fac)["pg"]
    ev, ph, sweep, sched, _ = analysis.dr_contract(inp, fac, np.full(len(idx), 200.0), idx, net, st, bau, keep_pay=350)
    assert sched is not None and len(sched["y_trn"]) == sched["window"].stop - sched["window"].start
    served = sched["y_trn"].sum() + sched["y_bat"].sum()
    assert served <= (inp["a_trn"] + inp["a_bat"]).sum() + 1e-6


def test_breakeven_is_refined_below_the_first_sweep_step():
    from dcflex import analysis
    true_threshold = 1.7                                    # commitment jumps from 0 to 30 MW at 1.7 AED/kW-yr
    commit = lambda pay: 30.0 if pay >= true_threshold else 0.0
    sweep = {p: commit(p) for p in (0, 5, 10, 50)}
    be = analysis.breakeven(sweep, commit, iters=10)
    assert true_threshold <= be < true_threshold + 5 / 2 ** 10 + 1e-12
    assert analysis.breakeven({0: 0.0, 5: 0.0}, commit, 5) is None
    assert analysis.breakeven({0: 30.0, 5: 30.0}, commit, 5) == 0


def test_ercot_workload_follows_the_texas_local_clock(tmp_path):
    idx_tx = pd.date_range("2025-09-01 05:00", periods=24 * 7 * 20, freq="h")        # UTC, as in load_ercot
    loc = data.local_clock(idx_tx, "America/Chicago")
    assert loc[0] == pd.Timestamp("2025-09-01 00:00")                                  # CDT = UTC-5
    jan = np.where(idx_tx == pd.Timestamp("2026-01-15 06:00"))[0][0]
    assert loc[jan] == pd.Timestamp("2026-01-15 00:00")                                # CST = UTC-6
    # a trace whose load peaks at 14:00 on its own clock peaks at 14:00 Texas time in S4
    t = pd.date_range("2024-05-13", periods=24 * 7, freq="h")
    pd.DataFrame({"time_gst": t, "requests": np.where(t.hour == 14, 10.0, 1.0)}).to_csv(tmp_path / "t.csv", index=False)
    shape = data.profile_from_csv(loc, tmp_path / "t.csv", "requests")
    assert set(loc[shape > 2].hour) == {14}


def test_cop_curve_matches_the_manufacturer_fit():
    import importlib.util, pathlib
    here = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fit_chiller_cop", here / "scripts" / "fit_chiller_cop.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    r = mod.fit(pd.read_csv(mod.CSV), 10)
    fac = Facility()
    assert r["n_sizes"] == 19
    assert abs(fac.cop_ref - r["cop_ref"]) < 0.005 and abs(fac.cop_slope - r["cop_slope"]) < 0.0005
    assert fac.t_ref == mod.T_REF
    # the Carrier source table: unit 200 at 8 C leaving water, 30 C air = 732.9 kW / 204.8 kW
    d = pd.read_csv(mod.CSV, dtype={"unit_size": str})
    row = d[(d.lcwt_c == 8) & (d.unit_size == "200") & (d.t_air_c == 30)].iloc[0]
    assert (row.cap_kw, row.input_kw) == (732.9, 204.8)


def test_archived_weather_is_used_before_the_api(tmp_path, monkeypatch):
    import json
    import requests
    from dcflex.config import Site, Study
    site, st = Site(), Study()
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[1]
    w = data.fetch_weather(site.lat, site.lon, f"{st.year}-01-01", f"{st.year}-12-31", site.tz,
                           cache_dir=tmp_path, model=st.weather_model, archive_dir=here / "data" / "weather")
    assert len(w) == 8760 and not w.isna().any().any() and 10 < w["temp"].min() < w["temp"].max() < 50
