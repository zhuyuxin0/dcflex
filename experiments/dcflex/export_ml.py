"""Summaries of the ML stages (run_ml.py) and their outputs: results/macros-ml.tex and
results/tab-ml-*.tex. Stages that have not run leave their macros as \\tbd placeholders."""
import pathlib
import numpy as np

from .export import fmt

__all__ = ["fmt"]

MACROS_ML = [
    # forecast skill (Abu Dhabi 2025, ECMWF IFS day-ahead against ERA5)
    "RfcTempMaeRaw", "RfcTempBiasRaw", "RfcTempMaeMos", "RfcGhiMaePct", "RfcNetMaeGW", "RfcPeakHitRaw",
    "RfcPeakHitMos", "RfcEventDaysHit", "RfcPriceMaePoint", "RfcPriceMaeExp",
    # MPC, S2 value retained over June-September
    "RmpcRetainPerfectPct", "RmpcRetainRawPct", "RmpcRetainMosPct", "RmpcRetainExpPct", "RmpcRetainPersistPct",
    "RmpcRetainRulePct", "RmpcNetPeakBauMW", "RmpcNetPeakPfMW", "RmpcNetPeakExpMW", "RmpcNetPeakRawMW",
    "RmpcSeasonValueMAED",
    # MPC, S3 delivery under forecasts (commitment = S3 value; pooled over noise draws)
    "RmpcDrSeeds", "RmpcDrEventsNonePct", "RmpcDrEventsTwentyPct", "RmpcDrEventsTenPct",
    "RmpcDrHoursNonePct", "RmpcDrHoursTwentyPct", "RmpcDrHoursTenPct", "RmpcDrMarginTwentyMW",
    "RmpcDrMarginTenMW", "RmpcDrMeanDelivPct", "RmpcDrCostPct", "RmpcDrLateMax",
    "RmpcDrEventsTenLowPct",
    # real data-center IT load: day-ahead forecastability (ESIF) and MPC with its errors
    "RwfDays", "RwfChronosNmae", "RwfPersistNmae", "RwfWeeklyNmae", "RwfGbmNmae", "RwfCoverPct", "RwfSynthNmae",
    "RmpcRetainExpWlPct", "RmpcDrEventsNoneWlPct", "RmpcDrEventsTwentyWlPct", "RmpcDrEventsTenWlPct",
    "RmpcDrMarginTenWlMW", "RmpcDrHoursTenWlPct", "RmpcDrEventsTenWlLowPct", "RmpcDrEventsTenWlMidPct",
]
DECIMALS_ML = {"RwfDays": 0, "RfcPeakHitRaw": 0, "RfcPeakHitMos": 0, "RfcEventDaysHit": 0, "RmpcDrSeeds": 0,
               "RfcTempMaeRaw": 2, "RfcTempBiasRaw": 2, "RfcTempMaeMos": 2, "RmpcDrCostPct": 2, "RmpcDrLateMax": 2,
               "RfcPriceMaePoint": 0, "RfcPriceMaeExp": 0, "RmpcSeasonValueMAED": 2}
# results/tab-ml-*.tex: columns per table (the synthetic-data row rule of export.write_table does not apply:
# these tables come only from real data and real forecasts)
TABLE_COLS_ML = {"tab-ml-mpc": 4, "tab-ml-realdata": 6, "tab-ml-hparams": 3, "tab-ml-workload": 5}


def write_macros(values, results_dir):
    lines = ["% Written by experiments/run_ml.py. Do not edit by hand."]
    for name in MACROS_ML + sorted(k for k in values if k not in MACROS_ML):
        nd = 0 if name.endswith(("Meters", "Sites")) or "Panels" in name else DECIMALS_ML.get(name, 1)
        s = fmt(values.get(name), nd)
        lines.append(r"\newcommand{\%s}{%s}" % (name, r"\tbd{%s}" % name[1:] if s is None else s))
    pathlib.Path(results_dir, "macros-ml.tex").write_text("\n".join(lines) + "\n")


def write_table(name, rows, results_dir):
    body = [" & ".join(x if isinstance(x, str) else (fmt(x) or r"\tbd{n/a}") for x in r) + r" \\" for r in rows]
    pathlib.Path(results_dir, name + ".tex").write_text("\n".join(body) + "\n")


def placeholders(results_dir):
    """Every ML macro and table as a placeholder, before the stages overwrite them."""
    write_macros({}, results_dir)
    for name, n in TABLE_COLS_ML.items():
        write_table(name, [[r"\multicolumn{%d}{c}{\tbd{pending: run\_ml.py}}" % n]], results_dir)


# ------------------------------------------------------------------ summaries
def forecast_skill(z, wf, t0, t1, sysp):
    """Day-ahead forecast errors over the summer window [t0, t1) and peak-set detection over the year."""
    from .signal import price_from_net
    from . import mpc
    s = slice(t0, t1)
    raw, cor = wf["ecmwf"], wf["ecmwf_mos"]
    e_raw = raw["temp"][s] - z["temp"][s]
    ghi_err = np.abs(wf["ghi_raw"][s] - z["ghi"][s])
    day = z["ghi"][s] > 0
    pk = z["peak"].astype(bool)
    thr = z["calib"]["peak_thr"]
    nd = len(z["net"]) // 24
    summer = np.isin(z["idx"][::24].month, (6, 7, 8, 9))[:nd]
    top = lambda net: set(np.flatnonzero(summer)[np.argsort(net.reshape(nd, 24).max(1)[summer])[-10:]])
    smp = mpc.error_samples(cor["net"], z["net"])
    p_exp = mpc.expected_price(cor["net"], smp, lambda n: price_from_net(n, sysp, z["calib"]))
    return {"RfcTempMaeRaw": float(np.abs(e_raw).mean()), "RfcTempBiasRaw": float(e_raw.mean()),
            "RfcTempMaeMos": float(np.abs(cor["temp"][s] - z["temp"][s]).mean()),
            "RfcGhiMaePct": float(100 * ghi_err[day].mean() / z["ghi"][s][day].mean()),
            "RfcNetMaeGW": float(np.abs(raw["net"][s] - z["net"][s]).mean()),
            "RfcPeakHitRaw": float((pk & (raw["net"] >= thr)).sum()),
            "RfcPeakHitMos": float((pk & (cor["net"] >= thr)).sum()),
            "RfcEventDaysHit": float(len(top(z["net"]) & top(cor["net"]))),
            "RfcPriceMaePoint": float(np.abs(cor["price"][s] - z["price"][s]).mean()),
            "RfcPriceMaeExp": float(np.abs(p_exp[s] - z["price"][s]).mean())}


def mpc_s2(f2, z, sysp):
    """System value (Eq. vsys numerator, AED) of each strategy over the simulated summer, and the share of the
    perfect-foresight value it retains."""
    d = np.load(f2)
    t0, T1 = int(d["t0"]), int(d["T1"])
    K = z["peak"][t0:T1].astype(bool)
    mc = z["mc"][t0:T1] * 1000.0
    b = d["bau_pg"]
    val = lambda pg: float(mc @ (b - pg) + sysp.cap_value * 1000.0 * (b[K].mean() - pg[K].mean()))
    names = ("pf", "perfect", "ecmwf_expected", "ecmwf_mos", "persistence", "ecmwf", "R")
    v = {k: val(d[k + "_pg"]) for k in names}
    netpk = {k: float(d[k + "_pg"][K].mean()) for k in names + ("bau",)}
    late = {k: float(d[k + "_late"].max()) for k in names if k + "_late" in d}
    return v, netpk, late


def mpc_s3(f3):
    """Delivery reliability by commitment level and margin mode, pooled over noise draws."""
    d = np.load(f3, allow_pickle=True)
    jobs, deliv = d["jobs"], d["deliv"]
    r0 = float(d["r_commit"])
    out = {}
    for i, (rl, mode, seed) in enumerate(jobs):
        r = float(rl) * r0
        dv = deliv[i]
        o = out.setdefault((float(rl), str(mode)), dict(events=[], hours=[], margin=[], cost=[], late=[], deliv=[]))
        # an event counts as delivered only if the reduction holds in every event hour, as in F(d) and S3
        o["events"].append(np.mean((dv >= r - 1e-6).all(1)))
        o["hours"].append(np.mean(dv >= r - 1e-6))
        o["deliv"].append(dv.mean() / r)
        o["margin"].append(float(d["margin"][i])); o["cost"].append(float(d["cost_pct"][i]))
        o["late"].append(float(d["late"][i]))
    return {k: {q: float(np.mean(v)) if q != "late" else float(np.max(v)) for q, v in o.items()} for k, o in out.items()}, \
        len({int(j[2]) for j in jobs}), r0


# ------------------------------------------------------------------ real data
REALDATA_MODELS = [("b1", "B1 High-5-of-10 (rolling)"), ("b2", "B2 Rolling regression"),
                   ("towt", "B3 Committed regression"), ("gbm", "LightGBM (committed)"),
                   ("mlp", "MLP (committed)"), ("pgnb_free", "B5 without shape constraints"),
                   ("pgnb", "B5 physics-guided network"), ("pt_mlp", "MLP, pretrained"),
                   ("pt_pgnb", "B5, pretrained")]
