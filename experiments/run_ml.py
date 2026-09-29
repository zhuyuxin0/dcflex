#!/usr/bin/env python3
"""Learned baselines and forecast-driven control: every number, table and figure of the ML extension.

Stages cache their outputs in out/ml/ so that they can run separately (and the heavy ones elsewhere):
    python run_ml.py pretrain                   # cross-building pretraining on BDG2 (leave one site out)
    python run_ml.py realdata                   # out-of-sample validation on BDG2 hot sites
    python run_ml.py mpc                        # forecast-driven control (needs out/series.npz from run_all)
    python run_ml.py all                        # both, then export
Raw BDG2 files must be in data/raw/bdg2/ (see data/README.md). Seeds are fixed; torch runs on the CPU.
"""
import argparse, dataclasses, json, pathlib, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

from dcflex import realdata, validate, pretrain as pt, mpc, model, data, export, export_ml, figures_ml, learned, learned_batch
from dcflex.config import Site, Facility, System, Study
from dcflex.signal import system_signal, price_from_net

OUT = pathlib.Path("out/ml")
NEW = ("b1", "b2", "towt", "gbm", "mlp", "pgnb_free", "pgnb", "pt_mlp", "pt_pgnb")
EST = ("b1", "b2", "towt", "gbm", "mlp", "pgnb")     # established facility: a year of history; no pretraining


def log(m, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {m}", flush=True)


def pretrain_path(kind, site, mk):
    return OUT / "pretrain" / f"{kind}-no{site}-{mk}.pt"


def stage_pretrain(a):
    (OUT / "pretrain").mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        for site in a.sites:
            todo = [mk for mk in ("pgnb", "mlp") if not pretrain_path(kind, site, mk).exists()]
            if not todo:
                continue
            rows, names = pt.pool_rows(realdata.pool(kind, exclude=(site,)))
            log(f"pretrain {kind} without {site}: {len(names)} buildings, {len(rows['y'])} rows")
            for mk in todo:
                net, curve = pt.pretrain(rows, len(names), kind=mk, epochs=a.pt_epochs, seed=0,
                                         log=lambda m: log(f"  {mk}{m}"))
                torch.save(dict(state=net.state_dict(), n_build=len(names), kind=mk, curve=curve, names=names),
                           pretrain_path(kind, site, mk))


def load_pretrained(kind, site, mk):
    d = torch.load(pretrain_path(kind, site, mk), weights_only=False)
    net = pt.Shared(d["n_build"], kind=d["kind"])
    net.load_state_dict(d["state"])
    return net, d


def stage_realdata(a):
    (OUT / "realdata").mkdir(parents=True, exist_ok=True)
    json.dump(dict(seeds=a.seeds, epochs=a.epochs, max_buildings=a.max_buildings),
              open(OUT / "realdata" / "settings.json", "w"))
    for regime in a.regimes:
        for kind in a.kinds:
            for site in a.sites:
                f = OUT / "realdata" / f"{site}-{kind}-{regime}.npz"
                if f.exists():
                    continue
                est = regime == "est"
                # the established regime (a year more of training data per model) is capped to bound compute time
                p = realdata.site_panel(site, kind, established=est, max_buildings=a.max_buildings if est else None)
                if len(p["buildings"]) < 3:
                    log(f"skip {site} {kind} {regime}: {len(p['buildings'])} eligible buildings")
                    continue
                train = (p["pre"] | p["prior"]) if est else p["pre"]
                log(f"{site} {kind} {regime}: {len(p['buildings'])} buildings")
                preds, curves, resp = {}, {}, {}
                ok_w = np.isfinite(p["temp"]) & np.isfinite(p["dew"])
                th, dw = np.nan_to_num(p["temp"]), np.nan_to_num(p["dew"])
                q = validate.response_query(p)
                for m in (EST if est else NEW):
                    t = time.time()
                    if m == "towt":
                        preds[m], resp[m] = validate.towt_committed(p, train, query=q)
                    elif m == "gbm":
                        preds[m], resp[m] = validate.gbm(p, train, query=q)
                    elif m in ("b1", "b2"):
                        if "b1" not in preds:
                            preds["b1"], preds["b2"] = validate.rolling(p)
                    elif m.startswith("pt_"):
                        if not pretrain_path(kind, site, m[3:]).exists():
                            log(f"  {m}: no pretrained model (run the pretrain stage first); skipped")
                            continue
                        net, _ = load_pretrained(kind, site, m[3:])
                        pred, info = pt.adapt(net, p["hour"], p["dow"], th, dw, p["Y"], train & ok_w)
                        P = pred(p["hour"], p["dow"], th, dw)
                        P[~ok_w] = np.nan
                        preds[m] = P
                        resp[m] = pred(q["hour"], q["dow"], q["temp"], q["dew"])
                    else:
                        preds[m], curves[m], resp[m] = validate.neural(p, train, m, seeds=tuple(range(a.seeds)),
                                                                       epochs=a.epochs, query=q)
                    log(f"  {m}: {time.time() - t:.0f}s")
                # response curves and, for the showcase figure, the summer observations at the query hour
                obs = p["season"] & (p["hour"] == q["hour"][0]) & (p["dow"] < 5)
                np.savez_compressed(f, buildings=np.array(p["buildings"]), events=p["events"],
                                    **{"pred_" + k: v.astype(np.float32) for k, v in preds.items()},
                                    **{"curve_" + k: v for k, v in curves.items()},
                                    **{"resp_" + k: v for k, v in resp.items()}, resp_temp=q["temp"],
                                    obs_temp=p["temp"][obs], obs_y=p["Y"][obs].astype(np.float32),
                                    train_temp=p["temp"][train & (p["hour"] == q["hour"][0]) & (p["dow"] < 5)],
                                    train_y=p["Y"][train & (p["hour"] == q["hour"][0]) & (p["dow"] < 5)].astype(np.float32))
                rows = []
                t_hi = float(np.nanmax(p["temp"][train]))
                for m, P in preds.items():
                    sc = validate.scores(p, P, train)
                    for j, b in enumerate(p["buildings"]):
                        rows.append(dict(site=site, kind=kind, regime=regime, building=b, model=m, t_train_max=t_hi,
                                         **{k: float(v[j]) for k, v in sc.items()}))
                pd.DataFrame(rows).to_csv(f.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------- forecast-driven control (MPC)
ECMWF = "data/weather/openmeteo_previous_runs_ecmwf_ifs025_2025.json"
X_H = 48                  # hours simulated after the season so that work arriving in it is finished
R_LEVELS = (0.8, 0.9, 1.0)
MODES = ("none", "conf0.2", "conf0.1")


def series():
    z = dict(np.load("out/series.npz"))
    z["idx"] = pd.to_datetime(z["time"])
    z["calib"] = dict(zip(("k", "b", "n_cc", "peak_thr"), map(float, z["calib"])))
    return z


def weather_forecasts(z, site, fac, sysp):
    """Forecast weather sets on the pipeline's hourly index: raw ECMWF day-ahead, the same after the causal
    hour-of-day bias correction (MOS), and persistence (the same hour yesterday). Each gives COP, PV, the
    system net load and its S2 price under the actual year's calibration."""
    idx = z["idx"]
    raw = mpc.with_decomposition(mpc.ecmwf_day_ahead(ECMWF, idx), site.lat, site.lon, site.tz)
    cor = raw.copy()
    cor["temp"] = mpc.mos(raw["temp"].to_numpy(), z["temp"])
    w = pd.DataFrame({k: z[k] for k in ("temp", "ghi", "dni", "dhi")}, index=idx)
    per = w.shift(24).bfill()
    out = {}
    for name, wf in (("ecmwf", raw), ("ecmwf_mos", cor), ("persistence", per)):
        t = wf["temp"].to_numpy(float)
        sf = system_signal(t, data.pv_output(wf, site.lat, site.lon, 1.0, site.tz), sysp, calib=z["calib"])
        out[name] = dict(temp=t, cop=fac.cop(t), pv=data.pv_output(wf, site.lat, site.lon, fac.pv_mwp, site.tz),
                         net=sf["net"], price=sf["price"])
    out["ghi_raw"] = raw["ghi"].to_numpy(float)
    return out


def _season(z, st):
    s = np.where(np.isin(z["idx"].month, st.summer_months))[0]
    return int(s[0]), int(s[-1]) + 1


def _arrivals(z, fac, st, seed):
    if seed == 0:
        return {"trn": z["a_trn"].copy(), "bat": z["a_bat"].copy()}, z["p0"]
    arr, p0 = data.build_arrivals(z["idx"], fac, np.random.default_rng(st.seed + 2 + 1000 * seed),
                                  z["inf_shape"], z["def_shape"])
    return {"trn": arr["trn"], "bat": arr["bat"]}, p0


def _expected_arrivals(z, fac):
    dyn = (fac.util_mean - fac.idle_frac) * fac.it_mw
    return {c: dyn * fac.shares[c] * z["def_shape"] / z["def_shape"].mean() for c in ("trn", "bat")}


def _mpc_s2_run(args):
    """One S2 season: forecast arrays (issued for each valid hour) are turned into horizon accessors here, in the
    worker, because closures cannot be sent to another process."""
    name, fc, price, actual, t0, t1 = args
    fset = {k: mpc.forecast_fn(v) for k, v in fc.items()}
    return name, mpc.simulate(Facility(), actual, fset, mpc.forecast_fn(price), t0, t1, H=24)


def stage_mpc(a):
    (OUT / "mpc").mkdir(parents=True, exist_ok=True)
    site, fac, sysp, st = Site(), Facility(), System(), Study()
    z = series()
    t0, t1 = _season(z, st)
    wf = weather_forecasts(z, site, fac, sysp)
    arr, p0 = _arrivals(z, fac, st, 0)
    a_fc = _expected_arrivals(z, fac)
    for c in arr:
        arr[c][t1:] = 0.0; a_fc[c][t1:] = 0.0
    actual = dict(p0=p0, cop=fac.cop(z["temp"]), pv=z["pv"], a_trn=arr["trn"], a_bat=arr["bat"])
    T1 = t1 + X_H
    # ---- S2: value retained under forecasts (system price)
    f2 = OUT / "mpc" / "s2.npz"
    if not f2.exists():
        pof = lambda n: price_from_net(n, sysp, z["calib"])
        smp = mpc.error_samples(wf["ecmwf_mos"]["net"], z["net"])
        p_exp = mpc.expected_price(wf["ecmwf_mos"]["net"], smp, pof)
        fsets = lambda w_: dict(p0=p0, cop=w_["cop"], pv=w_["pv"], a_trn=a_fc["trn"], a_bat=a_fc["bat"])
        jobs = [("perfect", dict(actual), z["price"], actual, t0, T1),
                ("ecmwf", fsets(wf["ecmwf"]), wf["ecmwf"]["price"], actual, t0, T1),
                ("ecmwf_mos", fsets(wf["ecmwf_mos"]), wf["ecmwf_mos"]["price"], actual, t0, T1),
                ("ecmwf_expected", fsets(wf["ecmwf_mos"]), p_exp, actual, t0, T1),
                ("persistence", fsets(wf["persistence"]), wf["persistence"]["price"], actual, t0, T1)]
        with Pool(a.procs) as pool:
            runs = dict(pool.map(_mpc_s2_run, jobs))
        inp = model.inputs(fac, arr, p0, z["temp"], z["pv"])
        s = slice(t0, T1)
        runs["pf"] = model.optimize(model.window(inp, s), fac, z["price"][s])
        runs["bau"] = model.bau(model.window(inp, s), fac)
        runs["R"] = model.rule_based(model.window(inp, s), fac, z["idx"].hour.to_numpy()[s])
        np.savez_compressed(f2, t0=t0, t1=t1, T1=T1, price_expected=p_exp,
                            **{f"{k}_{q}": np.asarray(v[q]) for k, v in runs.items() for q in ("pg", "soc", "soe")
                               if q in v},
                            **{f"{k}_late": np.maximum(v["late_trn"], v["late_bat"]) for k, v in runs.items()
                               if "late_trn" in v})
        log("MPC S2 done")
    # ---- workload forecastability (ESIF) and MPC with its real forecast errors on the IT load
    fw = OUT / "mpc" / "workload.npz"
    if not fw.exists():
        from dcflex import workload_fc
        ev = {y: workload_fc.evaluate(y) for y in WF_YEARS}
        E = workload_fc.scaled_errors(WF_YEARS)
        np.savez_compressed(fw, E=E, summary=json.dumps(ev))
        log("workload forecastability done")
    E = np.load(fw)["E"]
    f2w = OUT / "mpc" / "s2_wl.npz"
    if not f2w.exists():
        pof = lambda n: price_from_net(n, sysp, z["calib"])
        smp = mpc.error_samples(wf["ecmwf_mos"]["net"], z["net"])
        p_exp = mpc.expected_price(wf["ecmwf_mos"]["net"], smp, pof)
        fc = dict(p0=_p0_with_error(p0, fac, E), cop=wf["ecmwf_mos"]["cop"], pv=wf["ecmwf_mos"]["pv"],
                  a_trn=a_fc["trn"], a_bat=a_fc["bat"])
        _, o = _mpc_s2_run(("ecmwf_expected_wl", fc, p_exp, actual, t0, T1))
        np.savez_compressed(f2w, pg=o["pg"], late=np.maximum(o["late_trn"], o["late_bat"]))
        log("MPC S2 with workload errors done")
    # ---- S3: DR delivery under forecasts, with and without a conformal margin
    f3 = OUT / "mpc" / "s3.npz"
    if not f3.exists():
        jobs = [(rl, mode, seed) for rl in R_LEVELS for mode in MODES for seed in range(a.dr_seeds)]
        with Pool(a.procs) as pool:
            res = pool.map(_mpc_s3_run, jobs)
        np.savez_compressed(f3, jobs=np.array(jobs, dtype=object), deliv=np.array([r[0] for r in res]),
                            margin=np.array([r[1] for r in res]), cost_pct=np.array([r[2] for r in res]),
                            late=np.array([r[3] for r in res]), r_commit=float(z["r_true"]))
        log("MPC S3 done")
    f3w = OUT / "mpc" / "s3_wl.npz"
    if not f3w.exists():
        jobs = [(rl, mode, seed, True) for rl in R_LEVELS for mode in MODES for seed in range(a.dr_seeds)]
        with Pool(a.procs) as pool:
            res = pool.map(_mpc_s3_run, jobs)
        np.savez_compressed(f3w, jobs=np.array([j[:3] for j in jobs], dtype=object),
                            deliv=np.array([r[0] for r in res]), margin=np.array([r[1] for r in res]),
                            cost_pct=np.array([r[2] for r in res]), late=np.array([r[3] for r in res]),
                            r_commit=float(z["r_true"]))
        log("MPC S3 with workload errors done")


WF_YEARS = (2019, 2021, 2025)


def _it_error(E, n):
    """Scaled IT-load forecast errors for n hours of the local-time series: the ESIF day-ahead error days in
    order, repeated as needed (each row is one local day)."""
    return np.resize(E, (n // 24 + 1, 24)).ravel()[:n]


def _p0_with_error(p0, fac, E):
    """Forecast of the non-deferrable IT load carrying the ESIF day-ahead errors, scaled to the campus's mean IT
    load, util_mean x capacity (the deferrable arrivals keep their own noise)."""
    return np.maximum(p0 + _it_error(E, len(p0)) * fac.util_mean * fac.it_mw, 0.0)


def _mpc_s3_run(job):
    """One DR season under forecasts: the contract commitment times rl, planned with margin `mode`, for one
    draw of the deferrable-work noise; with `wl`, the IT-load forecasts also carry the real day-ahead errors of
    the ESIF data center. Returns the delivered reduction per event hour [events, hours], the mean planning margin
    (MW), the energy-cost change vs BAU (%) and the largest deadline overrun (MWh)."""
    rl, mode, seed, *wl = job
    wl = bool(wl and wl[0])
    site, fac, sysp, st = Site(), Facility(), System(), Study()
    z = series()
    t0, t1 = _season(z, st)
    wf = weather_forecasts(z, site, fac, sysp)["ecmwf_mos"]
    arr, p0 = _arrivals(z, fac, st, seed)
    a_fc = _expected_arrivals(z, fac)
    price = z["price_s1"]
    p0_fc = p0
    if wl:
        p0_fc = _p0_with_error(p0, fac, np.load(OUT / "mpc" / "workload.npz")["E"])
    bau_act = model.bau(model.inputs(fac, arr, p0, z["temp"], z["pv"]), fac)["pg"]
    # the plan's estimate of business-as-usual import: forecast deferrable service, with IT load, COP and PV
    # observed for the current hour (nowcast), as the planner uses them for its first hour
    bfc = model.bau(model.inputs(fac, a_fc, p0_fc, wf["temp"], wf["pv"]), fac)
    ybau = bfc["y_trn"] + bfc["y_bat"]
    ex = 1 + fac.loss_frac + fac.fan_frac
    cop_a = fac.cop(z["temp"])
    bau_now = ex * (p0 + ybau) * (1 + 1 / cop_a) + fac.aux_mw - np.minimum(z["pv"], ex * (p0 + ybau) * (1 + 1 / cop_a))
    ev = [int(d) for d in z["ev_days"]]
    ev_idx = np.concatenate([d * 24 + int(z["peak_hour"][d]) + np.arange(st.event_hours) for d in ev])
    r = rl * float(z["r_true"])
    if mode == "none":
        delta = np.zeros(len(price))
    else:
        eps = float(mode[4:])
        smp = mpc.error_samples(bau_now, bau_act)               # actual - estimate, causal
        delta = np.maximum(np.nan_to_num(np.nanquantile(-smp, 1 - eps, axis=1)), 0.0)
    mask = np.zeros(len(price), bool); mask[ev_idx] = True
    dr = dict(mask=mask, need=np.where(mask, r + delta, 0.0), ybau=ybau)
    for c in arr:
        arr[c][t1:] = 0.0; a_fc[c][t1:] = 0.0
    actual = dict(p0=p0, cop=fac.cop(z["temp"]), pv=z["pv"], a_trn=arr["trn"], a_bat=arr["bat"])
    fset = dict(p0=mpc.forecast_fn(p0_fc), cop=mpc.forecast_fn(wf["cop"]), pv=mpc.forecast_fn(wf["pv"]),
                a_trn=mpc.forecast_fn(a_fc["trn"]), a_bat=mpc.forecast_fn(a_fc["bat"]))
    T1 = t1 + X_H
    o = mpc.simulate(fac, actual, fset, mpc.forecast_fn(price), t0, T1, H=24, dr=dr)
    deliv = (bau_act[ev_idx] - o["pg"][ev_idx - t0]).reshape(len(ev), -1)
    cb, cm = price[t0:T1] @ bau_act[t0:T1], price[t0:T1] @ o["pg"]
    return deliv, float(delta[ev_idx].mean()), float(100 * (cm - cb) / cb), float(max(o["late_trn"].max(),
                                                                                      o["late_bat"].max()))


# ---------------------------------------------------------------- export
def _hparams():
    """Hyperparameters as the code uses them (function defaults and module constants), for Appendix E."""
    import inspect
    fm = inspect.signature(learned_batch.fit_many).parameters
    pr = inspect.signature(pt.pretrain).parameters
    ad = inspect.signature(pt.adapt).parameters
    sh = inspect.signature(pt.Shared).parameters
    cm = inspect.signature(learned_batch.committed).parameters
    pg = inspect.signature(learned_batch.PGNB).parameters
    ml = inspect.signature(learned_batch.MLP).parameters
    kn = learned.KNOTS
    d = lambda P, k: P[k].default
    sf = OUT / "realdata" / "settings.json"
    rs = json.load(open(sf)) if sf.exists() else dict(seeds=None, epochs=None)
    num = lambda x: f"$-${abs(x):g}" if x < 0 else f"{x:g}"
    return [
        ["B5 network", "Calendar input", f"hour and weekday one-hot ({learned.N_CAL})"],
        ["", "Base network", f"2 hidden layers of {d(pg, 'hidden')}, SiLU, softplus output"],
        ["", "Hinge knots (dry bulb)", f"{kn[0]:.0f}--{kn[-1]:.0f} $^\\circ$C every {kn[1] - kn[0]:.1f}"],
        ["", "Hinge knots (dew point)", f"{num(learned_batch.DEW_KNOTS[0])}--{num(learned_batch.DEW_KNOTS[-1])} $^\\circ$C "
                                        f"every {learned_batch.DEW_KNOTS[1] - learned_batch.DEW_KNOTS[0]:.1f}"],
        ["", "Hinge softness $s$; initial $a_k$", f"{learned.HINGE_WIDTH} K; {num(learned_batch.HINGE_INIT)}"],
        ["Training", "Ensemble; optimizer", f"{len(d(cm, 'seeds'))} seeds (synthetic), {rs['seeds']} (buildings); "
                                            "Adam, full batch, cosine schedule"],
        ["", "Epochs; learning rate", f"{d(cm, 'epochs')} (synthetic), {rs['epochs']} (buildings); {d(fm, 'lr')}"
                                      " (5$\\times$ for $a_k$, $r_0$, $c_0$, $\\beta$)"],
        ["", "Weight decay; hinge penalty", f"{d(fm, 'weight_decay')}; {d(fm, 'shape_l2')}"],
        ["", "Early stopping", f"best epoch on the last {100 * d(fm, 'val_frac'):.0f}\\% of the training days"],
        ["Ablations", "Unconstrained MLP", f"2 hidden layers of {d(ml, 'hidden')}, same inputs"],
        ["", "LightGBM", "600 trees, learning rate 0.03, 31 leaves"],
        ["Pretraining", "Embedding; base network", f"{d(sh, 'dim')} dimensions; 2 hidden layers of {d(sh, 'hidden')}"],
        ["", "Epochs; batch; learning rate", f"{PT_EPOCHS}; {d(pr, 'batch')} rows; {d(pr, 'lr')} (10$\\times$ for "
                                             "embeddings and hypernetwork), one-cycle"],
        ["", "Loss", f"Huber ($\\delta={d(pr, 'huber')}$) + {d(pr, 'shape_l2')} $\\times$ hinge penalty"],
        ["Adaptation", "Fitted; epochs; learning rate", f"embedding only; {d(ad, 'epochs')}; {d(ad, 'lr')}, "
                                                        f"{d(ad, 'restarts')} restarts"],
        ["MPC", "Horizon; re-planning", "24 h; every hour"],
        ["", "Bias correction; error samples", "30 days; same hour $\\pm$2 h"],
        ["", "Conformal margin", "risk level $\\alpha\\in\\{0.2, 0.1\\}$, same 30-day window"],
    ]


PT_EPOCHS = 6


def stage_export(a):
    root = pathlib.Path(a.root)
    rd, fd = root / "results", root / "figures"
    export_ml.placeholders(rd)
    site, fac, sysp, st = Site(), Facility(), System(), Study()
    z = series()
    t0, t1 = _season(z, st)
    wf = weather_forecasts(z, site, fac, sysp)
    vals = export_ml.forecast_skill(z, wf, t0, t1, sysp)
    f2, f3 = OUT / "mpc" / "s2.npz", OUT / "mpc" / "s3.npz"
    if f2.exists():
        v, netpk, late = export_ml.mpc_s2(f2, z, sysp)
        ret = {k: v[k] / v["pf"] for k in v}
        vals.update({"RmpcRetainPerfectPct": 100 * ret["perfect"], "RmpcRetainRawPct": 100 * ret["ecmwf"],
                     "RmpcRetainMosPct": 100 * ret["ecmwf_mos"], "RmpcRetainExpPct": 100 * ret["ecmwf_expected"],
                     "RmpcRetainPersistPct": 100 * ret["persistence"], "RmpcRetainRulePct": 100 * ret["R"],
                     "RmpcNetPeakBauMW": netpk["bau"], "RmpcNetPeakPfMW": netpk["pf"],
                     "RmpcNetPeakExpMW": netpk["ecmwf_expected"], "RmpcNetPeakRawMW": netpk["ecmwf"],
                     "RmpcSeasonValueMAED": v["pf"] / 1e6})
        names = [("Perfect foresight (LP over the season)", "pf"), ("MPC, perfect forecasts", "perfect"),
                 ("MPC, expected price (bias-corrected ECMWF)", "ecmwf_expected"),
                 ("MPC, bias-corrected ECMWF", "ecmwf_mos"), ("MPC, persistence", "persistence"),
                 ("MPC, raw ECMWF", "ecmwf"), ("Rule-based R", "R")]
        export_ml.write_table("tab-ml-mpc", [[n, v[k] / 1e6, 100 * ret[k], netpk[k]] for n, k in names], rd)
    if f3.exists():
        rel, nseeds, r0 = export_ml.mpc_s3(f3)
        full = {m: rel[(1.0, m)] for m in ("none", "conf0.2", "conf0.1")}
        vals.update({"RmpcDrSeeds": nseeds, "RmpcDrEventsNonePct": 100 * full["none"]["events"],
                     "RmpcDrEventsTwentyPct": 100 * full["conf0.2"]["events"],
                     "RmpcDrEventsTenPct": 100 * full["conf0.1"]["events"],
                     "RmpcDrHoursNonePct": 100 * full["none"]["hours"],
                     "RmpcDrHoursTwentyPct": 100 * full["conf0.2"]["hours"],
                     "RmpcDrHoursTenPct": 100 * full["conf0.1"]["hours"],
                     "RmpcDrMarginTwentyMW": full["conf0.2"]["margin"], "RmpcDrMarginTenMW": full["conf0.1"]["margin"],
                     "RmpcDrMeanDelivPct": 100 * full["none"]["deliv"],
                     "RmpcDrCostPct": float(np.mean([o["cost"] for o in rel.values()])),
                     "RmpcDrLateMax": max(o["late"] for o in rel.values()),
                     "RmpcDrEventsTenLowPct": 100 * rel[(min(k[0] for k in rel), "conf0.1")]["events"],
                     "RmpcDrHoursTenLowPct": 100 * rel[(min(k[0] for k in rel), "conf0.1")]["hours"],
                     "RmpcDrEventsNoneLowPct": 100 * rel[(min(k[0] for k in rel), "none")]["events"]})
        if f2.exists():
            f3w_ = OUT / "mpc" / "s3_wl.npz"
            figures_ml.mpc(fd / "fig-mpc.pdf", ret, rel, r0,
                           reliability_wl=export_ml.mpc_s3(f3w_)[0] if f3w_.exists() else None)
    fw = OUT / "mpc" / "workload.npz"
    if fw.exists():
        ev = {int(k): v for k, v in json.loads(str(np.load(fw)["summary"])).items()}
        for e_ in ev.values():                      # older caches use the label "Persistence (last day)"
            if "Persistence (last day)" in e_["res"]:
                e_["res"]["Persistence (two days earlier)"] = e_["res"].pop("Persistence (last day)")
        yrs = sorted(ev)
        methods = ["Persistence (two days earlier)", "Weekly persistence", "LightGBM on lags", "Chronos-Bolt (zero-shot)"]
        mean = lambda m: float(np.mean([ev[y]["res"][m]["nmae"] for y in yrs]))
        export_ml.write_table("tab-ml-workload", [[m] + [ev[y]["res"][m]["nmae"] for y in yrs] + [mean(m)]
                                                  for m in methods], rd)
        # the synthetic campus: day-ahead error of its IT load when only deferrable arrivals are uncertain
        arr, p0 = _arrivals(z, fac, st, 0)
        a_fc = _expected_arrivals(z, fac)
        it, it_fc = p0 + arr["trn"] + arr["bat"], p0 + a_fc["trn"] + a_fc["bat"]
        vals.update({"RwfDays": sum(ev[y]["days"] for y in yrs), "RwfChronosNmae": mean("Chronos-Bolt (zero-shot)"),
                     "RwfPersistNmae": mean("Persistence (two days earlier)"),
                     "RwfWeeklyNmae": mean("Weekly persistence"), "RwfGbmNmae": mean("LightGBM on lags"),
                     "RwfCoverPct": float(np.mean([ev[y]["chronos_cover80"] for y in yrs])),
                     "RwfSynthNmae": float(100 * np.abs(it_fc - it).mean() / it.mean())})
    f2w, f3w = OUT / "mpc" / "s2_wl.npz", OUT / "mpc" / "s3_wl.npz"
    if f2w.exists() and f2.exists():
        d2 = np.load(f2); dw = np.load(f2w)
        t0_, T1_ = int(d2["t0"]), int(d2["T1"])
        K = z["peak"][t0_:T1_].astype(bool); mc = z["mc"][t0_:T1_] * 1000.0; bb = d2["bau_pg"]
        vv = float(mc @ (bb - dw["pg"]) + sysp.cap_value * 1000.0 * (bb[K].mean() - dw["pg"][K].mean()))
        vals["RmpcRetainExpWlPct"] = 100 * vv / v["pf"]
    if f3w.exists():
        relw, _, _ = export_ml.mpc_s3(f3w)
        vals.update({"RmpcDrEventsNoneWlPct": 100 * relw[(1.0, "none")]["events"],
                     "RmpcDrEventsTwentyWlPct": 100 * relw[(1.0, "conf0.2")]["events"],
                     "RmpcDrEventsTenWlPct": 100 * relw[(1.0, "conf0.1")]["events"],
                     "RmpcDrHoursTenWlPct": 100 * relw[(1.0, "conf0.1")]["hours"],
                     "RmpcDrMarginTenWlMW": relw[(1.0, "conf0.1")]["margin"],
                     "RmpcDrEventsTenWlLowPct": 100 * relw.get((0.8, "conf0.1"), {}).get("events", np.nan),
                     "RmpcDrEventsTenWlMidPct": 100 * relw.get((0.9, "conf0.1"), {}).get("events", np.nan),
                     "RmpcDrHoursTenWlLowPct": 100 * relw.get((0.8, "conf0.1"), {}).get("hours", np.nan),
                     "RmpcDrHoursNoneWlPct": 100 * relw[(1.0, "none")]["hours"]})
    vals.update(export_realdata(rd, fd))
    # training curves (Appendix E)
    rd_curves = {}
    f = OUT / "realdata" / "Fox-chilledwater-new.npz"
    if f.exists():
        d = np.load(f)
        rd_curves = {k[6:]: d[k] for k in d if k.startswith("curve_")}
    pt_curves = {}
    for g in sorted((OUT / "pretrain").glob("*.pt")):
        kind, rest = g.stem.split("-no")
        site, mk = rest.split("-")
        pt_curves[(kind, site, mk)] = torch.load(g, weights_only=False)["curve"]
    figures_ml.training(fd / "fig-ml-training.pdf", z["b5_curve"] if "b5_curve" in z else None, rd_curves, pt_curves)
    export_ml.write_table("tab-ml-hparams", _hparams(), rd)
    export_ml.write_macros(vals, rd)
    export.write_numbers_csv(rd)
    log(f"export done: {rd}")


def export_realdata(rd, fd):
    files = sorted((OUT / "realdata").glob("*.csv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    vals = {}
    new = df[df.regime == "new"]
    df = df.copy()
    rows = []

    def medians(g):
        med = {}
        for key, _ in export_ml.REALDATA_MODELS:
            gg = g[g.model == key]
            if not gg.empty:
                med[key] = {c: float(gg[c].median()) for c in ("season_cv", "season_nmbe", "event_nmae", "hot_nmae")}
                med[key]["g14"] = float(100 * np.mean((gg.season_cv <= 30) & (gg.season_nmbe.abs() <= 10)))
        return med

    def cell(med, key, c, lower_better=True, absval=False):
        """Value as text; committed models shaded by their change against B3 (blue = better, red = worse)."""
        v, ref = med[key][c], med.get("towt")
        if not np.isfinite(v):
            return r"\tbd{n/a}"
        txt = export_ml.fmt(v)
        if ref is None or key in ("towt", "b1", "b2"):
            return txt
        if absval:   # bias: difference in |NMBE| relative to the 10% ASHRAE limit (a ratio of two small biases misleads)
            d = (abs(v) - abs(ref[c])) / 10.0
        else:
            d = (v - ref[c]) / max(abs(ref[c]), 1e-9) * (1 if lower_better else -1)
        shade = int(min(30, round(60 * abs(d))))
        if shade < 5:
            return txt
        return r"\cellcolor{%s!%d}%s" % ("keyblue" if d < 0 else "keyred", shade, txt)

    # one panel per meter type, medians over the meters of all sites
    for kind, label in (("chilledwater", "Chilled water"), ("electricity", "Electricity")):
        g = new[new.kind == kind]
        if g.empty:
            continue
        med = medians(g)
        n_m, n_s = g.groupby(["site", "building"]).ngroups, g.site.nunique()
        rows.append([(r"\addlinespace" if rows else "") + r"\multicolumn{6}{@{}l}{\emph{%s: %d meters at %d site%s}}"
                     % (label, n_m, n_s, "s" if n_s > 1 else "")])
        for key, name in export_ml.REALDATA_MODELS:
            if key not in med:
                continue
            if key not in ("b1", "b2"):
                rows.append([name, cell(med, key, "season_cv"), cell(med, key, "season_nmbe", absval=True),
                             cell(med, key, "event_nmae"), cell(med, key, "hot_nmae"),
                             cell(med, key, "g14", lower_better=False)])
            else:
                rows.append([name, "--", "--", cell(med, key, "event_nmae"), "--", "--"])
    export_ml.write_table("tab-ml-realdata", rows, rd)
    # macros: the hot-desert chilled-water panel (Tempe), all meters of the new-facility regime, established regime
    def summary(g, prefix):
        m = lambda k, c: float(g[g.model == k][c].median()) if (g.model == k).any() else None
        w = g.pivot_table(index=["site", "kind", "building"], columns="model", values="event_nmae")
        beat = lambda a_, b_: float(100 * (w[a_] < w[b_]).mean()) if a_ in w and b_ in w else None
        out = {f"{prefix}Meters": float(len(w))}
        for k, tag in (("pgnb", "Bfive"), ("towt", "Towt"), ("gbm", "Gbm"), ("mlp", "Mlp"), ("pgnb_free", "Free"),
                       ("pt_pgnb", "Pt"), ("pt_mlp", "PtMlp"), ("b2", "Roll"), ("b1", "High")):
            out[f"{prefix}Ev{tag}"] = m(k, "event_nmae")
            if k in ("pgnb", "towt", "gbm"):          # interquartile range over meters of the event error
                out[f"{prefix}EvLo{tag}"] = float(g[g.model == k].event_nmae.quantile(0.25)) if (g.model == k).any() else None
                out[f"{prefix}EvHi{tag}"] = float(g[g.model == k].event_nmae.quantile(0.75)) if (g.model == k).any() else None
            out[f"{prefix}EvNmbe{tag}"] = m(k, "event_nmbe")
            if k not in ("b1", "b2"):
                out[f"{prefix}Hot{tag}"] = m(k, "hot_nmae")
                out[f"{prefix}HotNmbe{tag}"] = m(k, "hot_nmbe")
                out[f"{prefix}Nmbe{tag}"] = m(k, "season_nmbe")
                out[f"{prefix}Cv{tag}"] = m(k, "season_cv")
                gg = g[g.model == k]
                out[f"{prefix}Gft{tag}"] = float(100 * np.mean((gg.season_cv <= 30) & (gg.season_nmbe.abs() <= 10))) \
                    if len(gg) else None
        for b_, tag in (("towt", "Towt"), ("gbm", "Gbm"), ("mlp", "Mlp"), ("pgnb_free", "Free"), ("b2", "Roll")):
            out[f"{prefix}BeatB{tag}Pct"] = beat("pgnb", b_)
        out[f"{prefix}PtBeatBfivePct"] = beat("pt_pgnb", "pgnb")
        out[f"{prefix}PtBeatMlpPct"] = beat("pt_pgnb", "pt_mlp")
        return {k: v for k, v in out.items() if v is not None}
    tempe = new[(new.site == "Fox") & (new.kind == "chilledwater")]
    if len(tempe):
        vals.update(summary(tempe, "RrdTempe"))
        vals["RrdTempeTrainTmax"] = float(tempe.t_train_max.iloc[0])
        try:
            w_ = realdata.weather("Fox"); _, season, _ = realdata.windows(w_.index)
            vals["RrdTempeSeasonTmax"] = float(np.nanmax(w_.temp.to_numpy()[season]))
        except FileNotFoundError:
            pass
    # every site-meter panel, e.g. RrdAustinChwEvBfive, with the warmest training and season temperatures
    for (site, kind), g in new.groupby(["site", "kind"]):
        tag = f"Rrd{realdata_name(site)}{'Chw' if kind == 'chilledwater' else 'Elec'}"
        vals.update(summary(g, tag))
        vals[f"{tag}TrainTmax"] = float(g.t_train_max.iloc[0])
        try:
            w_ = realdata.weather(site); _, season, _ = realdata.windows(w_.index)
            vals[f"{tag}SeasonTmax"] = float(np.nanmax(w_.temp.to_numpy()[season]))
        except FileNotFoundError:
            pass
    vals.update(summary(new, "Rrd"))
    vals["RrdSiteMeters"] = float(new.groupby(["site", "kind"]).ngroups)
    vals["RrdSites"] = float(new.site.nunique())
    est = df[df.regime == "est"]
    if len(est):
        vals.update(summary(est, "RrdEst"))
    for kind, tag in (("chilledwater", "Chw"), ("electricity", "Elec")):
        for reg, g in (("", new), ("Est", est)):
            gk = g[g.kind == kind]
            if len(gk):
                vals.update(summary(gk, f"Rrd{reg}{tag}"))
                vals[f"Rrd{reg}{tag}Sites"] = float(gk.site.nunique())
        # the same meters as the established regime, trained on the pre-season only (new facility)
        ek = est[est.kind == kind]
        if len(ek):
            same = new[(new.kind == kind) & new.set_index(["site", "building"]).index.isin(
                ek.set_index(["site", "building"]).index.unique())]
            vals.update(summary(same, f"RrdEstBase{tag}"))
    # share of site-meter panels on which B5 has the lowest median event error among the committed models
    committed = ("towt", "gbm", "mlp", "pgnb_free", "pgnb", "pt_mlp", "pt_pgnb")
    panel = new[new.model.isin(committed)].groupby(["site", "kind", "model"]).event_nmae.median().unstack()
    if "pgnb" in panel:
        vals["RrdPanelsBfiveBest"] = float((panel.idxmin(axis=1) == "pgnb").sum())
        vals["RrdPanelsBfiveBeatTowt"] = float((panel["pgnb"] < panel["towt"]).sum()) if "towt" in panel else None
    # showcase: the hot-desert chilled-water panel, the building with the median B5 season CV(RMSE)
    for site, kind in (("Fox", "chilledwater"), *{(r.site, r.kind) for r in new.itertuples()}):
        f = OUT / "realdata" / f"{site}-{kind}-new.npz"
        if f.exists():
            break
    show = dict(np.load(f))
    g = new[(new.site == site) & (new.kind == kind) & (new.model == "pgnb")].sort_values("season_cv")
    b_med = g.building.iloc[len(g) // 2]
    show["j"] = list(show["buildings"]).index(b_med)
    figures_ml.realdata(fd / "fig-realdata.pdf", df, show, f"{realdata_name(site)} chilled-water meter"
                        if kind == "chilledwater" else f"{realdata_name(site)} electricity meter")
    return vals


def realdata_name(site):
    return {"Fox": "Tempe", "Bull": "Austin", "Panther": "Orlando"}.get(site, site)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pretrain", "realdata", "mpc", "export", "all"))
    ap.add_argument("--root", default="..", help="folder that receives results/ and figures/ (default: the repository root)")
    ap.add_argument("--sites", nargs="+", default=list(realdata.HOT_SITES))
    ap.add_argument("--kinds", nargs="+", default=["chilledwater", "electricity"])
    ap.add_argument("--regimes", nargs="+", default=["new", "est"])
    ap.add_argument("--max-buildings", type=int, default=40,
                    help="meters per site and meter type in the established-facility regime (random, fixed seed)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--pt-epochs", type=int, default=PT_EPOCHS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="out/ml")
    ap.add_argument("--procs", type=int, default=3, help="parallel MPC simulations")
    ap.add_argument("--dr-seeds", type=int, default=5, help="noise draws per DR configuration")
    a = ap.parse_args()
    global OUT
    OUT = pathlib.Path(a.out)
    torch.set_num_threads(a.threads)
    if a.stage in ("pretrain", "all"):
        stage_pretrain(a)
    if a.stage in ("realdata", "all"):
        stage_realdata(a)
    if a.stage in ("mpc", "all"):
        stage_mpc(a)
    if a.stage in ("export", "all"):
        stage_export(a)


if __name__ == "__main__":
    main()
