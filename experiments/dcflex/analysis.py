"""Scenarios, flexibility envelope, DR contract, sensitivity and metrics (paper Sections III.E-III.H)."""
import copy
import numpy as np
from .model import optimize, bau, window, run_monthly, ALL
from .signal import daily_peak_hour


def tariff_vectors(idx, tar):
    """AED/MWh price vectors for scenario S1."""
    summer = np.isin(idx.month, (6, 7, 8, 9))
    peak = summer & (idx.hour >= 10) & (idx.hour < 22)
    n = len(idx)
    return {"addc_commercial": np.full(n, tar.addc_commercial * 1000),
            "addc_transmission": np.full(n, tar.addc_transmission * 1000),
            "addc_industrial": np.where(peak, tar.addc_ind_peak, tar.addc_ind_off) * 1000,
            "dewa": np.full(n, (tar.dewa_top_slab + tar.dewa_fuel) * 1000)}


def summer_profile(x, idx, months=(6, 7, 8, 9)):
    """Average value by hour of day over the summer months."""
    m = np.isin(idx.month, months)
    h = np.asarray(idx.hour)[m]
    return np.array([np.asarray(x)[m][h == k].mean() for k in range(24)])


def sample_days(idx, study, rng, n):
    days = [d for d in range(len(idx) // 24 - 1) if idx[d * 24].month in study.summer_months]
    return np.sort(rng.choice(days, size=min(n, len(days)), replace=False))


LEVER_SETS = {"Compute deferral only": ("defer",), "TES only": ("tes",),
              "Battery only": ("bess",), "All levers": ALL}


def flex_envelope(inp, fac, net, days, durations=(1, 2, 4), lever_sets=None, q=0.2):
    """F(d): 20th percentile over days of the max reduction held from the daily net-load peak (Eq. flex)."""
    lever_sets = lever_sets or LEVER_SETS
    ph = daily_peak_hour(net)
    vals = {k: {d: [] for d in durations} for k in lever_sets}
    for day in days:
        s = slice(day * 24, day * 24 + 48)
        w = window(inp, s)
        ref = bau(w, fac)["pg"]
        price = np.full(48, 200.0)
        for dur in durations:
            win = np.arange(ph[day], ph[day] + dur)
            for name, lev in lever_sets.items():
                o = optimize(w, fac, price, levers=lev, shed=(win, ref))
                vals[name][dur].append(max(o["r"], 0.0))
    return {k: {d: float(np.quantile(v[d], q)) for d in durations} for k, v in vals.items()}, vals


def it_level_shed(inp, fac, net, days, dur=3, q=0.2):
    """IT-power reduction held for `dur` hours from the daily net-load peak by compute deferral alone, as a
    share of BAU IT power in the window (comparable with IT-side field results such as the Phoenix trial)."""
    ph = daily_peak_hour(net)
    shares, mws = [], []
    for day in days:
        s = slice(day * 24, day * 24 + 48)
        w = window(inp, s)
        ref = bau(w, fac)
        win = np.arange(ph[day], ph[day] + dur)
        o = optimize(w, fac, np.full(48, 200.0), levers=("defer",), shed=(win, ref["pg"]))
        red = (ref["pit"][win] - o["pit"][win]).min()
        mws.append(red)
        shares.append(100 * red / ref["pit"][win].mean())
    return dict(q20_pct=float(np.quantile(shares, q)), median_pct=float(np.median(shares)),
                q20_mw=float(np.quantile(mws, q)), median_mw=float(np.median(mws)), n_days=len(days))


def dr_contract(inp, fac, price, idx, net, study, bau_pg, keep_pay=None):
    """S3: events on the ten summer days with the highest net load; payment sweep (Eq. dr).
    Returns the full schedule and summer window at payment keep_pay (delay metric, dispatch figure)."""
    summer = np.where(np.isin(idx.month, study.summer_months))[0]
    s = slice(summer[0], summer[-1] + 1)
    nd = len(idx) // 24
    dmax = np.asarray(net).reshape(-1, 24).max(1)
    sdays = [d for d in range(nd) if idx[d * 24].month in study.summer_months]
    ev_days = sorted(sorted(sdays, key=lambda d: dmax[d])[-study.events:])
    ph = daily_peak_hour(net)
    ev_idx = np.concatenate([d * 24 + ph[d] + np.arange(study.event_hours) for d in ev_days])
    ev_idx = ev_idx[(ev_idx >= s.start) & (ev_idx < s.stop)]
    w = window(inp, s)
    solve = lambda pay: optimize(w, fac, price[s], events=ev_idx - s.start, baseline=bau_pg[s], cap_pay=pay * 1000.0)
    sweep, sched = {}, None
    for pay in study.cap_pay_sweep:
        o = solve(pay)
        sweep[pay] = max(o["r"], 0.0)
        if keep_pay is not None and abs(pay - keep_pay) < 1e-9:
            sched = dict(o, window=s)
    return ev_days, ph, sweep, sched, breakeven(sweep, lambda pay: max(solve(pay)["r"], 0.0), study.breakeven_iter)


def breakeven(sweep, commit, iters, share=0.95):
    """Lowest payment (AED/kW-yr) at which the commitment reaches share x its maximum: the first sweep
    point that does, refined by bisection against the point below it. The optimal commitment of the LP
    is non-decreasing in the payment, so bisection is valid. None if nothing is committed."""
    rmax = max(sweep.values(), default=0.0)
    if rmax <= 0:
        return None
    pays = sorted(sweep)
    hi = next(p for p in pays if sweep[p] >= share * rmax)
    lo = max((p for p in pays if p < hi), default=None)
    if lo is None:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if commit(mid) >= share * rmax:
            hi = mid
        else:
            lo = mid
    return hi


def base_commitment(sweep, cap_value):
    """Committed reduction (MW) at the base capacity payment; the payment must be a point of the sweep."""
    for pay, r in sweep.items():
        if abs(pay - cap_value) < 1e-9:
            return r
    raise ValueError(f"cap_value {cap_value} is not in Study.cap_pay_sweep {sorted(sweep)}")


def delays(arrivals, served, q=0.95, step_mwh=0.1):
    """q-quantile of the delay (h) of deferred work, per class and pooled, from cumulative arrival and
    service curves (Eq. deadline). Work is cut into equal energy quanta, so the pooled value weights
    classes by energy; work not served by the end of the window is left out."""
    out, pooled = {}, []
    for c in arrivals:
        ca, cy = np.cumsum(arrivals[c]), np.cumsum(served[c])
        levels = np.arange(step_mwh / 2, min(ca[-1], cy[-1]), step_mwh)
        d = np.maximum(np.searchsorted(cy, levels - 1e-6) - np.searchsorted(ca, levels - 1e-9), 0)
        out[c] = float(np.quantile(d, q)) if len(d) else 0.0
        pooled.append(d)
    allv = np.concatenate(pooled) if pooled else np.zeros(0)
    out["all"] = float(np.quantile(allv, q)) if len(allv) else 0.0
    return out


def _shift_deferrable(inp, fac, factor):
    """Move work between inference (non-deferrable) and deferrable classes for sensitivity."""
    out = {k: v.copy() for k, v in inp.items()}
    inf = out["p0"] - fac.idle_frac * fac.it_mw
    dfr = out["a_trn"] + out["a_bat"]
    if factor < 1:   # less deferrable work; any part that would push P0 above IT capacity stays deferrable
        moved = np.minimum((1 - factor) * dfr, np.maximum(fac.it_mw - out["p0"], 0.0))
        keep = np.divide(dfr - moved, dfr, out=np.zeros_like(dfr), where=dfr > 0)
        out["a_trn"] = out["a_trn"] * keep
        out["a_bat"] = out["a_bat"] * keep
        out["p0"] = out["p0"] + moved
    else:            # more deferrable work, taken from inference (at most 90% of it in any hour)
        moved = np.minimum((factor - 1) * dfr, 0.9 * inf)
        share = np.divide(out["a_trn"], dfr, out=np.full_like(dfr, 0.7), where=dfr > 0)
        out["a_trn"] += moved * share
        out["a_bat"] += moved * (1 - share)
        out["p0"] = out["p0"] - moved
    return out


# (label, key in Study.sens, tick text for the range). Ranges come from Study.sens (Appendix A).
SENS_CASES = [("COP slope $k$", "cop_slope", "{:.2f}–{:.2f} K$^{{-1}}$"), ("Reference COP", "cop_ref", "{:.1f}–{:.1f}"),
              ("TES size", "tes_hours", "{:g}–{:g} h"), ("Battery size", "bess_mw", "{:g}–{:g} MW"),
              ("Deadlines", "deadline", "{:g}–{:g}$\\times$"), ("Deferrable share", "share_pct", "{:g}–{:g}%"),
              ("System PV", "pv_gw", "{:g}–{:g} GW")]


def deferrable_share_pct(fac):
    """Deferrable work (training + batch) as a percentage of mean IT load."""
    dyn = fac.util_mean - fac.idle_frac
    return 100.0 * dyn * (fac.shares["trn"] + fac.shares["bat"]) / fac.util_mean


def sensitivity(inp, fac, net_fn, days, sysp, ranges, q=0.2):
    """One-at-a-time variation of F(4) (all levers) over ranges (Study.sens). net_fn(pv_gw) returns net load."""
    base_net = net_fn(sysp.pv_gw)
    base = flex_envelope(inp, fac, base_net, days, durations=(4,), lever_sets={"all": ALL}, q=q)[0]["all"][4]
    rows = []
    for label, key, tick in SENS_CASES:
        lo, hi = ranges[key]
        res = []
        for v in (lo, hi):
            f, i, net = copy.deepcopy(fac), inp, base_net
            if key in ("cop_slope", "cop_ref", "tes_hours"):
                setattr(f, key, v)
                if key != "tes_hours":
                    i = dict(inp, cop=f.cop(inp["temp"]))
            elif key == "bess_mw":
                f.bess_mw, f.bess_mwh = v, fac.bess_mwh * v / fac.bess_mw
            elif key == "deadline":
                f.deadline_h = {c: max(1, int(round(d * v))) for c, d in fac.deadline_h.items()}
            elif key == "share_pct":
                i = _shift_deferrable(inp, fac, v / deferrable_share_pct(fac))
            elif key == "pv_gw":
                net = net_fn(v)
            res.append(flex_envelope(i, f, net, days, durations=(4,), lever_sets={"all": ALL}, q=q)[0]["all"][4])
        rows.append((label, lo, hi, res[0], res[1], tick.format(lo, hi)))
    return base, rows


def cooling_share(res, fac, mask):
    """Share (%) of facility energy used for cooling over the hours in mask: chillers plus fans and
    pumps (the term f of Eq. balance)."""
    cooling = res["pch"] + fac.fan_frac * res["pit"]
    facility = res["pg"] + res["pv_used"]
    return 100 * cooling[mask].sum() / facility[mask].sum()


def summarize(res, fac, sig, sysp, idx, summer_mask=None):
    """Annual indicators for one strategy (Table tab:scenarios)."""
    pg, pit, pch = res["pg"], res["pit"], res["pch"]
    facility = pg + res["pv_used"]      # energy at the facility meter, incl. battery losses
    return dict(energy_gwh=facility.sum() / 1000, grid_gwh=pg.sum() / 1000,
                pue=facility.sum() / pit.sum(), peak_mw=pg.max(),
                netpeak_mw=pg[sig["peak"]].mean(),
                solar_share=100 * pg[sig["surplus"]].sum() / pg.sum(),
                co2_marg_kt=(sig["ef"] * pg).sum() / 1000, co2_avg_kt=sysp.ef_avg * pg.sum() / 1000,
                chiller_gwh=pch.sum() / 1000)
