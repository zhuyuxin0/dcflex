#!/usr/bin/env python3
"""Run every experiment in the paper and write its numbers, tables and figures to results/ and figures/.

    python run_all.py                                             # real data (downloads weather)
    python run_all.py --ercot-csv data/ercot_hb_west.csv \
        --inference-trace data/azure_llm.csv --inference-col requests \
        --deferrable-trace data/alibaba_jobs.csv --deferrable-col gpu_hours
    python run_all.py --synthetic --quick --root /tmp/test        # software test only (never into ..)
"""
import argparse, dataclasses, hashlib, json, pathlib, time
import numpy as np
import pandas as pd
from dcflex.config import Site, Facility, System, Tariffs, Study, AED_PER_USD
from dcflex import data, model, analysis, baselines, protocol, export, figures, learned_batch
from dcflex.signal import system_signal


def check_root(root, synthetic):
    """Synthetic numbers must never reach the manuscript: refuse to write them into a folder
    that holds main.tex. A real run elsewhere (e.g. a sensitivity case in /tmp) only prints a note."""
    has_main = (pathlib.Path(root) / "main.tex").exists()
    if synthetic and has_main:
        raise SystemExit(f"--synthetic output must not go into the manuscript ({root}); use e.g. /tmp/test")
    if not synthetic and not has_main:
        print(f"note: {root} has no main.tex; results and figures are written to {root}/results and {root}/figures")


# Plain-text names of the sensitivity cases for the sentence "The most influential is the ..."
TORNADO_TEXT = {"COP slope $k$": "COP slope", "Reference COP": "reference COP", "TES size": "TES size",
                "Battery size": "battery size", "Deadlines": "service deadlines",
                "Deferrable share": "deferrable share", "System PV": "system PV capacity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..", help="folder that receives results/ and figures/ (default: the repository root)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--synthetic", action="store_true", help="software test only")
    ap.add_argument("--quick", action="store_true", help="fewer sampled days")
    ap.add_argument("--inference-trace"); ap.add_argument("--inference-col")
    ap.add_argument("--deferrable-trace"); ap.add_argument("--deferrable-col")
    ap.add_argument("--inference-trace-alt", help="same trace with the absolute-time (UTC) mapping, for sensitivity")
    ap.add_argument("--deferrable-trace-alt", help="same trace with the absolute-time (UTC) mapping, for sensitivity")
    ap.add_argument("--ercot-csv")
    a = ap.parse_args()
    t0 = time.time()
    site, fac, sysp, tar, st = Site(), Facility(), System(), Tariffs(), Study()
    st.cap_pay_sweep = tuple(sorted(set(st.cap_pay_sweep) | {sysp.cap_value}))   # the base payment is a sweep point
    if a.quick:
        st.flex_days, st.placebo_days = 4, 8
    rng = np.random.default_rng(st.seed)
    out, root = pathlib.Path(a.out), pathlib.Path(a.root)
    check_root(root, a.synthetic)
    out.mkdir(parents=True, exist_ok=True)
    export.write_placeholders(root)
    idx = data.hourly_index(st.year)
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    # 1. weather, PV, workload, system signal
    if a.synthetic:
        w = data.synthetic_weather(idx, rng)
    else:
        w = data.fetch_weather(site.lat, site.lon, f"{st.year}-01-01", f"{st.year}-12-31", site.tz,
                               model=st.weather_model)
        w = w.reindex(idx).interpolate().bfill().ffill()
    temp = w["temp"].to_numpy(float)
    pv = data.pv_output(w, site.lat, site.lon, fac.pv_mwp, site.tz)
    pv_norm = data.pv_output(w, site.lat, site.lon, 1.0, site.tz)
    inf_shape = (data.profile_from_csv(idx, a.inference_trace, a.inference_col) if a.inference_trace
                 else data.parametric_profile(idx, 0.15, 15))
    def_shape = (data.profile_from_csv(idx, a.deferrable_trace, a.deferrable_col) if a.deferrable_trace
                 else data.parametric_profile(idx, 0.10, 11))
    # the job-size noise has its own random stream, so the trace-mapping sensitivity sees the same noise
    arr, p0 = data.build_arrivals(idx, fac, np.random.default_rng(st.seed + 2), inf_shape, def_shape)
    inp = model.inputs(fac, arr, p0, temp, pv)
    sig = system_signal(temp, pv_norm, sysp)
    net_fn = lambda gw: system_signal(temp, pv_norm, dataclasses.replace(sysp, pv_gw=gw))["net"]
    log("inputs ready")

    # 2. annual scenarios
    P = analysis.tariff_vectors(idx, tar)
    S = {"S0": model.run_monthly(inp, fac, P["addc_commercial"], idx, strategy="bau"),
         "R": model.rule_based(inp, fac, idx.hour.to_numpy())}
    S["S1c"] = model.run_monthly(inp, fac, P["addc_commercial"], idx)
    S["S1i"] = model.run_monthly(inp, fac, P["addc_industrial"], idx)
    S["S2"] = model.run_monthly(inp, fac, sig["price"], idx)
    # H1 control: the same S2 run with a constant chiller COP, to separate the effect of ambient-dependent
    # chiller efficiency on the timing of load from that of the price signal and the service deadlines
    fac_k = dataclasses.replace(fac, cop_slope=0.0)
    inp_k = model.inputs(fac_k, arr, p0, temp, pv)
    s0_k = model.run_monthly(inp_k, fac_k, P["addc_commercial"], idx, strategy="bau")
    s2_const_cop = model.run_monthly(inp_k, fac_k, sig["price"], idx)["pg"] - s0_k["pg"]
    log("annual scenarios solved")

    # 3. flexibility envelope and DR contract
    days = analysis.sample_days(idx, st, rng, st.flex_days)
    F, _ = analysis.flex_envelope(inp, fac, sig["net"], days, q=st.reliability_q)
    ev_days, ph, sweep, s3, breakeven = analysis.dr_contract(inp, fac, P["addc_commercial"], idx, sig["net"], st, S["S0"]["pg"],
                                                  keep_pay=sysp.cap_value)
    itl = analysis.it_level_shed(inp, fac, sig["net"], days, dur=3, q=st.reliability_q)
    log("flexibility envelope and DR contract done")

    # 4. ERCOT benchmark (S4)
    ercot = None
    if a.ercot_csv or a.synthetic:
        if a.ercot_csv:
            idx_tx, rt, asp = data.load_ercot(a.ercot_csv, len(idx))
            wtx = data.fetch_weather(st.ercot_lat, st.ercot_lon, str(idx_tx[0].date()), str(idx_tx[-1].date()), "UTC",
                                     model=st.weather_model)
            wtx = wtx.reindex(idx_tx).interpolate().bfill().ffill()
        else:
            idx_tx = idx
            wtx = data.synthetic_weather(idx_tx, rng, lat=st.ercot_lat, t_mean=19, t_seas=10, t_diur=7)
            rt, asp = data.synthetic_ercot(len(idx_tx), rng)
        pv_tx = data.pv_output(wtx, st.ercot_lat, st.ercot_lon, fac.pv_mwp, "UTC")
        # workload replayed at the same hours of the Texas day (Central time), as in Abu Dhabi; own random stream
        loc_tx = data.local_clock(idx_tx, st.ercot_tz)
        inf_tx = (data.profile_from_csv(loc_tx, a.inference_trace, a.inference_col) if a.inference_trace
                  else data.parametric_profile(loc_tx, 0.15, 15))
        def_tx = (data.profile_from_csv(loc_tx, a.deferrable_trace, a.deferrable_col) if a.deferrable_trace
                  else data.parametric_profile(loc_tx, 0.10, 11))
        arr_tx, p0_tx = data.build_arrivals(idx_tx, fac, np.random.default_rng(st.seed + 1), inf_tx, def_tx)
        inp_tx = model.inputs(fac, arr_tx, p0_tx, wtx["temp"].to_numpy(float), pv_tx)
        s0tx = model.run_monthly(inp_tx, fac, rt, idx_tx, strategy="bau")
        s4 = model.run_monthly(inp_tx, fac, rt, idx_tx, as_price=asp)
        as_rev = float(asp @ s4["ras"])
        ercot = dict(value_usd=float(rt @ s0tx["pg"] - rt @ s4["pg"]) + as_rev, as_rev_usd=as_rev)
        # Negative prices: the LP can charge and discharge storage at once to burn energy it is paid to take
        # (a relaxation artifact). Re-dispatch on prices floored at zero and value it at the real prices.
        s4f = model.run_monthly(inp_tx, fac, np.maximum(rt, 0.0), idx_tx, as_price=asp)
        val_f = float(rt @ s0tx["pg"] - rt @ s4f["pg"]) + float(asp @ s4f["ras"])
        ercot.update(neg_hours=int((rt < 0).sum()),
                     simul_hours=int(((s4["bc"] > 1e-3) & (s4["bd"] > 1e-3)).sum()
                                     + ((s4["qin"] > 1e-3) & (s4["qout"] > 1e-3)).sum()),
                     neg_value_pct=100 * (ercot["value_usd"] - val_f) / ercot["value_usd"] if ercot["value_usd"] else None)
        log("ERCOT benchmark done")

    # 5. baselines and protocol
    ft = dataclasses.replace(fac, cop_slope=fac.cop_slope * 1.15)
    k = np.ones(3) / 3
    arr_s = {c: np.convolve(arr[c], k, mode="same") for c in ("trn", "bat")}
    twin = model.bau(model.inputs(ft, arr_s, p0, temp, pv), ft)["pg"]
    F4 = F["All levers"][4]
    r_true = analysis.base_commitment(sweep, sysp.cap_value) or F4
    # B5: physics-guided network trained on the 60 pre-season days only (calendar, temperature, dew point, PV)
    dew = data.dew_point(temp, w["rh"].to_numpy(float))
    b5 = learned_batch.committed(idx.hour.to_numpy(), idx.dayofweek.to_numpy(), temp, dew, pv,
                                 baselines.pre_season(idx, st), epochs=300 if a.quick else 1500)
    acc, shop = baselines.experiment(S["S0"]["pg"], temp, twin, idx, ev_days, ph, st, max(r_true, 1e-3), rng,
                                     learned=b5)
    b5_pred, b5_net = b5(S["S0"]["pg"]), b5.info["net"]
    b5_bytes = learned_batch.model_bytes(b5_net)
    micros, nbytes = protocol.benchmark()
    # one event of the inflation test, for the figure; and the commitment the operator would publish
    # for it before the season (committed regression B3: model = coefficients, features = the event
    # window's temperatures). The nonce is seeded so the appendix record is reproducible; in operation
    # it is drawn from a secure random source.
    infl_ex = baselines.inflation_example(S["S0"]["pg"], temp, twin, idx, ev_days, ph, st, max(r_true, 1e-3), sig["net"],
                                          learned_pred=b5_pred)
    c_ex, rec_ex = protocol.commit(f"AD-DR-{infl_ex['date']}", (pd.Timestamp(infl_ex["date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00+04:00"),
                                   [infl_ex["event_day"] * 24 + h for h in infl_ex["hrs"]], infl_ex["clean"]["b3"],
                                   np.asarray(infl_ex["coef"], float).tobytes(), np.asarray(infl_ex["features"], float).tobytes(),
                                   nonce=np.random.default_rng(st.seed + 7).bytes(32).hex())
    assert protocol.verify(c_ex, rec_ex, np.asarray(infl_ex["coef"], float).tobytes(),
                           np.asarray(infl_ex["features"], float).tobytes())
    log("baselines and protocol done")

    # 6. sensitivity
    # same sampled days as F(d), so the tornado reference value equals F(4) in Table ablation
    sdays = days[:3] if a.quick else days
    base_F4, sens = analysis.sensitivity(inp, fac, net_fn, sdays, sysp, st.sens, q=st.reliability_q)
    log("sensitivity done")

    # 6b. trace time mapping: absolute (UTC) time instead of the same hours of the Gulf day (Appendix F)
    alt = None
    if a.inference_trace_alt or a.deferrable_trace_alt:
        inf_a = data.profile_from_csv(idx, a.inference_trace_alt, a.inference_col) if a.inference_trace_alt else inf_shape
        def_a = data.profile_from_csv(idx, a.deferrable_trace_alt, a.deferrable_col) if a.deferrable_trace_alt else def_shape
        arr_a, p0_a = data.build_arrivals(idx, fac, np.random.default_rng(st.seed + 2), inf_a, def_a)
        inp_a = model.inputs(fac, arr_a, p0_a, temp, pv)
        s0a = model.run_monthly(inp_a, fac, P["addc_commercial"], idx, strategy="bau")
        s2a = model.run_monthly(inp_a, fac, sig["price"], idx)
        F4a = analysis.flex_envelope(inp_a, fac, sig["net"], days, durations=(4,),
                                     lever_sets={"all": model.ALL}, q=st.reliability_q)[0]["all"][4]
        st1 = dataclasses.replace(st, cap_pay_sweep=(sysp.cap_value,))
        sw_a = analysis.dr_contract(inp_a, fac, P["addc_commercial"], idx, sig["net"], st1, s0a["pg"])[2]
        np0 = analysis.summarize(s0a, fac, sig, sysp, idx)["netpeak_mw"]
        np2 = analysis.summarize(s2a, fac, sig, sysp, idx)["netpeak_mw"]
        sys_a = float(sig["mc"] * 1000.0 @ (s0a["pg"] - s2a["pg"]) + sysp.cap_value * 1000 * (np0 - np2))
        alt = dict(F4=F4a, dr=analysis.base_commitment(sw_a, sysp.cap_value),
                   sysv=sys_a / (F4a * 1000) if F4a > 0 else None)
        log("trace-mapping sensitivity done")

    # 7. metrics
    sm = {k: analysis.summarize(v, fac, sig, sysp, idx) for k, v in S.items()}
    ja = np.isin(idx.month, (7, 8))
    s0 = S["S0"]
    fac_e = s0["pg"] + s0["pv_used"]
    mc = sig["mc"] * 1000.0
    sys_num = lambda x: float(mc @ (s0["pg"] - S[x]["pg"]) + sysp.cap_value * 1000 * (sm["S0"]["netpeak_mw"] - sm[x]["netpeak_mw"]))
    per_kw = lambda aed: aed / (F4 * 1000) if F4 > 0 else None
    save_c = float(P["addc_commercial"] @ (s0["pg"] - S["S1c"]["pg"]))
    save_i = float(P["addc_industrial"] @ (s0["pg"] - S["S1i"]["pg"]))
    # a flat tariff has the same optimal schedule at any price level, so the transmission tariff reuses S1c
    save_t = float(P["addc_transmission"] @ (s0["pg"] - S["S1c"]["pg"]))
    save_d = float(P["dewa"] @ (s0["pg"] - S["S1c"]["pg"]))
    # private value as a range: the campus's tariff class (flat commercial or industrial TOU) is unconfirmed
    v = {"private": per_kw(save_c), "private_tou": per_kw(save_i), "private_trans": per_kw(save_t),
         "system": per_kw(sys_num("S2"))}
    v["ercot"] = per_kw(ercot["value_usd"] * AED_PER_USD) if ercot else None
    r_base = analysis.base_commitment(sweep, sysp.cap_value)
    # the S3 event with the highest net-load peak (dispatch figure): mean change in import over its event
    # hours by lever, and the largest rise in import above BAU in the 24 h before it (pre-charging)
    dmax = np.asarray(sig["net"]).reshape(-1, 24).max(1)
    e_pk = max(ev_days, key=lambda d: dmax[d])
    w0 = s3["window"].start
    evh = np.arange(e_pk * 24 + ph[e_pk], e_pk * 24 + ph[e_pk] + st.event_hours)
    pre_h = np.arange(e_pk * 24 + ph[e_pk] - 24, e_pk * 24 + ph[e_pk])
    s3h = {k: np.asarray(x)[evh - w0] for k, x in s3.items() if isinstance(x, np.ndarray)}
    s0h = {k: np.asarray(x)[evh] for k, x in S["S0"].items() if isinstance(x, np.ndarray)}
    parts = figures._lever_parts(s0h, s3h, np.asarray(inp["cop"])[evh], fac, slice(None))
    pre_rise = float(np.max(np.asarray(s3["pg"])[pre_h - w0] - np.asarray(S["S0"]["pg"])[pre_h]))
    # service delay under the S3 contract schedule at the base payment, and under BAU on the same window
    ws = s3["window"]
    arr_s3 = {c: arr[c][ws] for c in ("trn", "bat")}
    dl = analysis.delays(arr_s3, {c: s3["y_" + c] for c in ("trn", "bat")})
    bau_s3 = model.bau(model.window(inp, ws), fac)
    dl_bau = analysis.delays(arr_s3, {c: bau_s3["y_" + c] for c in ("trn", "bat")})
    best = min(baselines.NAMES, key=lambda b: acc[b]["nmae"])
    top = max(sens, key=lambda r: abs(r[4] - r[3]))[0] if sens else None
    vals = {
        "RpeakEventDate": pd.Timestamp(idx[e_pk * 24]).strftime("%-d %B"),
        "RpeakEventComputeMW": -float(parts["Compute deferral"].mean()), "RpeakEventTesMW": -float(parts["Thermal storage"].mean()),
        "RpeakEventBessMW": -float(parts["Battery"].mean()), "RpeakEventRedMW": float((s0h["pg"] - s3h["pg"]).mean()),
        "RpeakEventPreRiseMW": pre_rise,
        "RinflEventDate": pd.Timestamp(idx[infl_ex["event_day"] * 24]).strftime("%-d %B"),
        "RinflCreditHighMW": float(np.mean(infl_ex["gamed"]["b1"] - infl_ex["metered"])),
        "RinflCreditHighCleanMW": float(np.mean(infl_ex["clean"]["b1"] - infl_ex["metered"])),
        "RinflCreditCommitMW": float(np.mean(infl_ex["gamed"]["b3"] - infl_ex["metered"])),
        "RinflTrueMW": float(infl_ex["r_true"]),
        "RbauEnergyGWh": fac_e.sum() / 1000, "RbauPUE": sm["S0"]["pue"],
        "RbauPUEsummer": fac_e[ja].sum() / s0["pit"][ja].sum(),
        "RcoolShareSummer": analysis.cooling_share(s0, fac, ja), "RbauPeakMW": s0["pg"].max(),
        "RshedOneHourMW": F["All levers"][1], "RshedTwoHourMW": F["All levers"][2], "RshedFourHourMW": F4,
        "RshedFourHourPct": 100 * F4 / s0["pg"].max(), "RflexComputeMW": F["Compute deferral only"][4],
        "RflexTesMW": F["TES only"][4], "RflexBessMW": F["Battery only"][4],
        "RslaDelayPctl": dl["all"], "RslaDelayTrnPctl": dl["trn"], "RslaDelayBatPctl": dl["bat"],
        "RslaDelayBauPctl": dl_bau["all"],
        "RsolarShareBau": sm["S0"]["solar_share"], "RsolarShareFlat": sm["S1c"]["solar_share"],
        "RsolarShareSys": sm["S2"]["solar_share"],
        "RsurplusAbsorbedGWh": float((S["S2"]["pg"] - s0["pg"])[sig["surplus"]].sum() / 1000),
        "RnetPeakRedMW": sm["S0"]["netpeak_mw"] - sm["S2"]["netpeak_mw"],
        "RgrossPeakChangePct": 100 * (sm["S2"]["peak_mw"] / sm["S0"]["peak_mw"] - 1),
        "RcoolChangeFlatPct": 100 * (sm["S1c"]["chiller_gwh"] / sm["S0"]["chiller_gwh"] - 1),
        "RcoolChangeSysPct": 100 * (sm["S2"]["chiller_gwh"] / sm["S0"]["chiller_gwh"] - 1),
        "RcoTwoAvgRed": 1000 * (sm["S0"]["co2_avg_kt"] - sm["S2"]["co2_avg_kt"]),
        "RcoTwoMargRed": 1000 * (sm["S0"]["co2_marg_kt"] - sm["S2"]["co2_marg_kt"]),
        "RbillSaveCommAED": save_c / 1000, "RbillSaveIndAED": save_i / 1000,
        "RprivateValueKWyr": v["private"], "RprivateValueTouKWyr": v["private_tou"],
        "RprivateValueTransKWyr": v["private_trans"],
        "RsystemValueKWyr": v["system"], "RercotValueKWyr": v["ercot"],
        "RvalueGapKWyr": (v["system"] - v["private"]) if v["system"] is not None and v["private"] is not None else None,
        "RvalueGapTouKWyr": (v["system"] - v["private_tou"]) if v["system"] is not None and v["private_tou"] is not None else None,
        "RercotAsShare": 100 * ercot["as_rev_usd"] / ercot["value_usd"] if ercot and ercot["value_usd"] else None,
        "RercotNegHours": float(ercot["neg_hours"]) if ercot else None,
        "RercotSimulHours": float(ercot["simul_hours"]) if ercot else None,
        "RercotNegValuePct": ercot["neg_value_pct"] if ercot else None,
        "RdrFirmMW": r_base, "RdrTargetSharePct": 100 * r_base / 200, "RdrInterimSharePct": 100 * r_base / 80,
        "RcapPayBreakeven": breakeven, "RbaselineBest": baselines.NAMES[best].lower(),
        "RbaselineBestNMAE": acc[best]["nmae"], "RbaselineHighNMAE": acc["b1"]["nmae"],
        "RgamingOverpayHighPct": acc["b1"]["overcredit"], "RgamingOverpayCommitPct": acc["b3"]["overcredit"],
        "RgamingOverpayRollPct": acc["b2"]["overcredit"], "RbaselineCommitNMAE": acc["b3"]["nmae"],
        "RpreTmax": float(temp[baselines.pre_season(idx, st)].max()),
        "RsummerTmax": float(temp[np.isin(idx.month, st.summer_months)].max()),
        "RbaselineLearnedNMAE": acc["b5"]["nmae"], "RbaselineLearnedBias": acc["b5"]["bias"],
        "RgamingOverpayLearnedPct": acc["b5"]["overcredit"],
        "RbfiveParams": float(sum(q.numel() for q in b5_net.parameters())),
        "RbfiveKB": len(b5_bytes) / 1024, "RbfiveDigest": hashlib.sha256(b5_bytes).hexdigest()[:16],
        "RshoppingOverpayPct": shop, "RcommitMicros": micros, "RcommitBytes": float(nbytes),
        "RtornadoTop": TORNADO_TEXT.get(top, top) if top else None,
        "RruleGapPct": 100 * sys_num("R") / sys_num("S2") if sys_num("S2") else None,
        # H1: chiller energy of S2 above the cooling-seeking flat-tariff schedule, in points of S0 chiller energy
        "RcoolPenaltyPts": 100 * (sm["S2"]["chiller_gwh"] - sm["S1c"]["chiller_gwh"]) / sm["S0"]["chiller_gwh"],
        # mean change in summer import over the solar window (10:00-15:00), with the ambient-dependent COP and without
        "RsysMiddayMW": float(analysis.summer_profile(S["S2"]["pg"] - s0["pg"], idx)[10:15].mean()),
        "RsysMiddayConstCopMW": float(analysis.summer_profile(s2_const_cop, idx)[10:15].mean()),
        "RsysShiftHourConstCop": float(np.argmax(analysis.summer_profile(s2_const_cop, idx))),
        # summer hour of day with the largest average added import, and of the daily net-load peak
        "RsysShiftHour": float(np.argmax(analysis.summer_profile(S["S2"]["pg"] - s0["pg"], idx))),
        "RflatShiftHour": float(np.argmax(analysis.summer_profile(S["S1c"]["pg"] - s0["pg"], idx))),
        "RnetPeakHour": float(np.bincount(np.asarray(ph)[[d for d in range(len(ph)) if idx[d * 24].month in st.summer_months]],
                                          minlength=24).argmax()),
        "RsurplusHours": float(sig["surplus"].sum()),
        "RsurplusSummerHours": float(sig["surplus"][np.isin(idx.month, st.summer_months)].sum()),
        "RopenCycleHours": float((sig["net"] > sig["n_cc"]).sum()),
        "RitShedThreePct": itl["q20_pct"], "RitShedThreeMedPct": itl["median_pct"],
        "RutcShedFourHourMW": alt["F4"] if alt else None, "RutcDrFirmMW": alt["dr"] if alt else None,
        "RutcSystemValueKWyr": alt["sysv"] if alt else None}

    # 8. tables, figures, export
    rd, fd = root / "results", root / "figures"
    lab = {"S0": "S0 Business as usual", "R": "R Rule-based", "S1c": "S1 ADDC commercial (flat)",
           "S1i": "S1 ADDC industrial TOU", "S2": "S2 System signal"}
    export.write_table("tab-scenarios", [[lab[k], sm[k]["energy_gwh"], f"{sm[k]['pue']:.2f}", sm[k]["peak_mw"],
                                           sm[k]["netpeak_mw"], sm[k]["solar_share"], sm[k]["co2_marg_kt"]] for k in lab],
                       rd, a.synthetic)
    export.write_table("tab-ablation", [[n, F[n][1], F[n][2], F[n][4]] for n in F], rd, a.synthetic)
    vrows = [["Private: ADDC commercial (flat)", v["private"], "Bill saving, S1"],
             ["Private: ADDC transmission-connected (flat)", v["private_trans"], "Bill saving, S1"],
             ["Private: ADDC industrial TOU", per_kw(save_i), "Bill saving, S1"],
             ["Private: DEWA top slab", per_kw(save_d), "Bill saving, S1"],
             ["System (Abu Dhabi 2030)", v["system"], "Avoided marginal cost + peaking capacity, S2"],
             ["DR contract (base payment)", sysp.cap_value * r_base / F4 if F4 else None, "Capacity payment, S3"]]
    if ercot:
        vrows.append(["ERCOT market", v["ercot"], "Real-time energy + ancillary services, S4"])
    export.write_table("tab-value", vrows, rd, a.synthetic)
    export.write_table("tab-baselines", [[baselines.NAMES[k], acc[k]["bias"], acc[k]["nmae"], acc[k]["cvrmse"],
                                           acc[k]["overcredit"]] for k in baselines.NAMES], rd, a.synthetic)
    export.write_macros(vals, rd, a.synthetic)
    export.write_numbers_csv(rd)
    figures.duck(fd / "fig-duck.pdf", S, sig, idx)
    figures.flex(fd / "fig-flex.pdf", F)
    figures.heatmap(fd / "fig-heatmap.pdf", S, idx)
    figures.value(fd / "fig-value.pdf", {"Private (ADDC flat)": v["private"], "Private (ADDC transmission)": v["private_trans"],
                                         "Private (ADDC TOU)": per_kw(save_i), "Private (DEWA)": per_kw(save_d),
                                         "System": v["system"], "ERCOT": v["ercot"]})
    figures.baselines(fd / "fig-baselines.pdf", acc, baselines.NAMES)
    figures.tornado(fd / "fig-tornado.pdf", base_F4, sens)
    figures.cop(fd / "fig-cop.pdf", fac, temp, slopes=st.sens["cop_slope"], points=data.chiller_rating_points())
    figures.dispatch(fd / "fig-dispatch.pdf", S["S0"], s3, sig["net"], idx, ev_days, ph, fac, inp["cop"],
                     st.event_hours, r_base)
    figures.inflation(fd / "fig-inflation.pdf", infl_ex, baselines.NAMES)
    export.write_commit_example(c_ex, rec_ex, rd, a.synthetic)
    # hourly series for the learned-baseline and forecast/MPC experiments (experiments/out, not in git)
    np.savez_compressed(out / "series.npz", time=np.asarray(idx.astype("int64")), load_s0=S["S0"]["pg"], temp=temp,
                        pv=inp["pv"], net=sig["net"], price=sig["price"], ev_days=np.asarray(ev_days),
                        peak_hour=np.asarray(ph), r_true=r_true, twin=twin, p0=p0, a_trn=arr["trn"],
                        a_bat=arr["bat"], inf_shape=inf_shape, def_shape=def_shape, pv_norm=pv_norm,
                        rh=w["rh"].to_numpy(float), ghi=w["ghi"].to_numpy(float), dni=w["dni"].to_numpy(float),
                        dhi=w["dhi"].to_numpy(float), mc=sig["mc"], peak=sig["peak"],
                        price_s1=P["addc_commercial"], b5=b5_pred, b5_curve=b5.info["curve"],
                        calib=np.array([sig["calib"][k] for k in ("k", "b", "n_cc", "peak_thr")]))
    json.dump({k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in vals.items()},
              open(out / "results.json", "w"), indent=2)
    log(f"done: results written to {rd} and {fd}" + ("  [SYNTHETIC TEST DATA]" if a.synthetic else ""))


if __name__ == "__main__":
    main()
