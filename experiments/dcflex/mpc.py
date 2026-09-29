"""Forecast-driven model predictive control of the facility (receding horizon, hourly).

The annual scenarios of the paper solve the facility LP with perfect foresight of weather, prices and workload.
Here the facility re-plans every hour over the next H hours from forecasts and only the first hour of each plan
is carried out, against the actual weather and workload, by a plant simulator that enforces what the plan
cannot know (the work actually waiting, deadlines, storage limits). Comparing the realized system value with
the perfect-foresight value separates what is lost to the finite horizon (MPC with perfect forecasts) from what
is lost to forecast error (MPC with day-ahead forecasts).

The planning LP is Eqs. 1-8 of the paper over a short horizon, with the state carried in: storage levels,
cumulative work served, and the arrivals whose deadlines fall inside the horizon. Deadlines are soft (a large
penalty) so a forecast miss never makes the plan infeasible; the simulator then serves overdue work first.
It is built once with CVXPY parameters and re-solved with new data each hour.
"""
import numpy as np
import cvxpy as cp

from .model import _solver

CLASSES = ("trn", "bat")
BIG = 1e4                 # AED/MWh penalty on a missed deadline or a DR shortfall in the plan
NO_CAP = 1e6


class Planner:
    def __init__(self, fac, H=24, levers=("defer", "tes", "bess"), eps=0.01, s_ref=0.5, e_ref=0.9):
        self.fac, self.H, self.levers = fac, H, levers
        ex = self.ex = 1 + fac.loss_frac + fac.fan_frac
        P = lambda name, **k: cp.Parameter(H, name=name, **k)
        self.price, self.invcop, self.p0cool = P("price"), P("invcop", nonneg=True), P("p0cool")
        self.p0 = P("p0")
        self.room = P("room", nonneg=True)                    # IT headroom for deferrable work, max(cap - p0, 0)
        self.pv = P("pv", nonneg=True)
        self.cap = P("cap")          # cap on the decision part of import (DR event hours), NO_CAP elsewhere
        self.A = {c: P("A_" + c) for c in CLASSES}            # forecast cumulative arrivals minus work served
        self.L = {c: P("L_" + c) for c in CLASSES}            # cumulative arrivals due by each hour minus served
        self.S0, self.E0 = cp.Parameter(name="S0"), cp.Parameter(name="E0")
        # storage must be back at its reference level at the end of the horizon, or at the end of the simulated
        # period if that comes first (term = 1 at that step), as in the perfect-foresight LP
        self.term = cp.Parameter(H + 1, nonneg=True, name="term")
        y = {c: cp.Variable(H, nonneg=True) for c in CLASSES}
        late = {c: cp.Variable(H, nonneg=True) for c in CLASSES}
        cons = []
        for c in CLASSES:
            cy = cp.cumsum(y[c])
            cons += [cy <= self.A[c], cy + late[c] >= self.L[c]]
            if "defer" not in levers:
                cons += [late[c] == 0]
        cons += [y["trn"] + y["bat"] <= self.room]
        heat_dyn = ex * (y["trn"] + y["bat"])                 # heat from deferrable work; the p0 part is in p0cool
        cap_tes, rate = fac.tes_hours * fac.peak_heat_mw, fac.tes_rate_frac * fac.peak_heat_mw
        qin, qout = cp.Variable(H, nonneg=True), cp.Variable(H, nonneg=True)
        S = cp.Variable(H + 1)
        if "tes" in levers:
            cons += [S[0] == self.S0, cp.multiply(self.term, S - s_ref * cap_tes) >= 0, S >= 0, S <= cap_tes,
                     qin <= rate, qout <= rate,
                     qout <= heat_dyn + ex * self.p0,
                     S[1:] == (1 - fac.tes_loss_h) * S[:-1] + fac.tes_eff * qin - qout]
        else:
            cons += [qin == 0, qout == 0, S == self.S0]
        pch = cp.Variable(H)
        cons += [pch == cp.multiply(self.invcop, heat_dyn + qin - qout) + self.p0cool]
        bc, bd = cp.Variable(H, nonneg=True), cp.Variable(H, nonneg=True)
        E = cp.Variable(H + 1)
        if "bess" in levers:
            res = fac.reserve_min / 60.0 * fac.it_mw
            cons += [E[0] == self.E0, cp.multiply(self.term, E - e_ref * fac.bess_mwh) >= 0, E >= res, E <= fac.bess_mwh,
                     bc <= fac.bess_mw, bd <= fac.bess_mw, E[1:] == E[:-1] + fac.bess_eta * bc - bd / fac.bess_eta]
        else:
            cons += [bc == 0, bd == 0, E == self.E0]
        pvu = cp.Variable(H, nonneg=True)
        short = cp.Variable(H, nonneg=True)
        # the import is split into a part that depends on the decisions and one that does not (p0 and auxiliary
        # load); only the first enters the price term, which keeps the problem DPP (parameter x variable only)
        pg_dec = ex * (y["trn"] + y["bat"]) + pch + bc - bd - pvu
        pg = pg_dec + ex * self.p0 + fac.aux_mw
        cons += [pvu <= self.pv, pg >= 0, pg_dec <= self.cap + short]
        thr = cp.sum(qin) + cp.sum(qout) + cp.sum(bc) + cp.sum(bd)
        obj = (self.price @ pg_dec + eps * thr - eps / 10 * cp.sum(pvu)
               + BIG * (cp.sum(late["trn"]) + cp.sum(late["bat"]) + cp.sum(short)))
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.v = dict(y=y, qin=qin, qout=qout, bc=bc, bd=bd, pg=pg, S=S, E=E)

    def solve(self, fc, state, price, dr=None, steps_left=None):
        """fc: dict of horizon arrays (p0, cop, pv, and cumulative arrivals A_c, due L_c, both net of work
        served); state: S, E; dr: horizon arrays of a DR commitment (mask of event hours, need = committed
        reduction plus margin, ybau = deferrable work served under business as usual); steps_left: hours to the
        end of the simulated period. Returns the first-hour decisions.

        On event hours import must stay `need` below the plan's own estimate of business-as-usual import,
        ex (p0 + ybau)(1 + 1/COP) + aux - PV (storage idle, PV self-consumed), built from the same p0, COP and PV
        as the plan. With nowcasting, the first hour of the plan and of that estimate both use observed values."""
        ex = self.ex
        term = np.zeros(self.H + 1)
        term[self.H if steps_left is None else min(self.H, steps_left)] = 1.0
        self.term.value = term
        self.price.value = np.asarray(price, float)
        self.invcop.value = 1.0 / np.asarray(fc["cop"], float)
        self.p0.value = np.asarray(fc["p0"], float)
        self.room.value = np.maximum(self.fac.it_mw - self.p0.value, 0.0)
        self.p0cool.value = ex * self.p0.value * self.invcop.value
        self.pv.value = np.maximum(np.asarray(fc["pv"], float), 0.0)
        cap = np.full(self.H, NO_CAP)
        if dr is not None:
            m = np.asarray(dr["mask"], bool)
            est = self.p0cool.value + ex * np.asarray(dr["ybau"], float) * (1 + self.invcop.value) - self.pv.value
            cap[m] = (est - np.asarray(dr["need"], float))[m]
        self.cap.value = cap
        for c in CLASSES:
            self.A[c].value = np.asarray(fc["A_" + c], float)
            self.L[c].value = np.asarray(fc["L_" + c], float)
        self.S0.value, self.E0.value = float(state["S"]), float(state["E"])
        self.prob.solve(solver=_solver())
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC plan LP status: {self.prob.status}")
        g = lambda x: float(np.asarray(x.value).ravel()[0])
        return dict(y_trn=g(self.v["y"]["trn"]), y_bat=g(self.v["y"]["bat"]), qin=g(self.v["qin"]),
                    qout=g(self.v["qout"]), bc=g(self.v["bc"]), bd=g(self.v["bd"]),
                    pg_plan=np.asarray(self.v["pg"].value, float))


def simulate(fac, actual, forecast, price_fc, t0, t1, H=24, levers=("defer", "tes", "bess"), dr=None,
             nowcast=True, s_init=0.5, e_init=0.9, log=None):
    """Receding-horizon control over hours [t0, t1). actual: dict of full-length hourly arrays (p0, cop, pv,
    a_trn, a_bat); forecast: dict of accessors, forecast[k](t) -> the horizon values as known at t; price_fc(t)
    likewise. dr: full-length arrays of a DR commitment (mask, need, ybau; see Planner.solve), known a day ahead.
    nowcast: the current hour's IT load, COP and PV are observed, so the first hour of each plan uses them.
    Returns hourly realized arrays."""
    ex = 1 + fac.loss_frac + fac.fan_frac
    plan = Planner(fac, H=H, levers=levers)
    cap_tes = fac.tes_hours * fac.peak_heat_mw
    res = fac.reserve_min / 60.0 * fac.it_mw
    S, E = s_init * cap_tes, e_init * fac.bess_mwh
    D = {c: int(fac.deadline_h[c]) for c in CLASSES}
    cumA = {c: np.concatenate([[0.0], np.cumsum(actual["a_" + c])]) for c in CLASSES}   # cumA[t] = sum a[:t]
    served = {c: cumA[c][t0] for c in CLASSES}       # everything before t0 is taken as served on time
    n = t1 - t0
    out = {k: np.zeros(n) for k in ("y_trn", "y_bat", "pit", "qin", "qout", "bc", "bd", "pch", "pg", "pv_used",
                                     "late_trn", "late_bat", "soc", "soe")}
    T = len(actual["p0"])
    for i, t in enumerate(range(t0, t1)):
        h = np.arange(t, min(t + H, T))
        pad = lambda x: np.concatenate([x, np.repeat(x[-1:], H - len(x))]) if len(x) < H else x
        fc = {"p0": pad(forecast["p0"](t)[: len(h)]).copy(), "cop": pad(forecast["cop"](t)[: len(h)]).copy(),
              "pv": pad(forecast["pv"](t)[: len(h)]).copy()}
        if nowcast:
            for k in ("p0", "cop", "pv"):
                fc[k][0] = actual[k][t]
        for c in CLASSES:
            a_fc = pad(forecast["a_" + c](t)[: len(h)])
            A = cumA[c][t] + np.cumsum(a_fc)                      # through t + tau (actual past, forecast ahead)
            due_idx = np.arange(t, t + H) - D[c] + 1              # arrivals through t + tau - D are due by t + tau
            L = np.where(due_idx <= t, cumA[c][np.clip(due_idx, 0, T)],
                         cumA[c][t] + np.concatenate([[0.0], np.cumsum(a_fc)])[np.clip(due_idx - t, 0, H)])
            fc["A_" + c], fc["L_" + c] = A - served[c], L - served[c]
        pr = pad(np.asarray(price_fc(t))[: len(h)])
        drh = None if dr is None else {k: pad(np.asarray(v)[t:t + H][: len(h)]) for k, v in dr.items()}
        d = plan.solve(fc, dict(S=S, E=E), pr, drh, steps_left=t1 - t)
        # --- plant: actual workload, weather and state
        p0 = actual["p0"][t]
        room = max(fac.it_mw - p0, 0.0)
        y = {}
        for c in CLASSES:
            avail = cumA[c][t + 1] - served[c]
            must = max(cumA[c][max(t - D[c] + 1, 0)] - served[c], 0.0) if "defer" in levers else avail
            y[c] = min(max(d["y_" + c], must), avail)
        tot = y["trn"] + y["bat"]
        if tot > room:                                            # capacity binds: scale down, overdue first
            y = {c: y[c] * room / tot for c in CLASSES}
        for c in CLASSES:
            served[c] += y[c]
            out["late_" + c][i] = max(cumA[c][max(t - D[c] + 1, 0)] - served[c], 0.0)
        pit = p0 + y["trn"] + y["bat"]
        heat = ex * pit
        qin = min(d["qin"], max(cap_tes - (1 - fac.tes_loss_h) * S, 0.0) / fac.tes_eff) if "tes" in levers else 0.0
        qout = min(d["qout"], heat, (1 - fac.tes_loss_h) * S) if "tes" in levers else 0.0
        S = (1 - fac.tes_loss_h) * S + fac.tes_eff * qin - qout
        bc = min(d["bc"], fac.bess_mw, max(fac.bess_mwh - E, 0.0) / fac.bess_eta) if "bess" in levers else 0.0
        bd = min(d["bd"], fac.bess_mw, max(E - res, 0.0) * fac.bess_eta) if "bess" in levers else 0.0
        E = E + fac.bess_eta * bc - bd / fac.bess_eta
        pch = (heat + qin - qout) / actual["cop"][t]
        load = heat + pch + fac.aux_mw + bc - bd
        pvu = min(actual["pv"][t], max(load, 0.0))
        for k, v in (("y_trn", y["trn"]), ("y_bat", y["bat"]), ("pit", pit), ("qin", qin), ("qout", qout),
                     ("bc", bc), ("bd", bd), ("pch", pch), ("pg", load - pvu), ("pv_used", pvu),
                     ("soc", S), ("soe", E)):
            out[k][i] = v
        if log and i % 720 == 0:
            log(f"  MPC hour {i}/{n}")
    return out


def forecast_fn(series, H=24):
    """Forecast accessor for an hourly series of values issued for each valid hour (e.g. the 24 h-ahead ECMWF
    value): at time t the forecast for t..t+H-1 is series[t:t+H]."""
    s = np.asarray(series, float)
    return lambda t: s[t:t + H]


def persistence_fn(series, H=24, lag=24):
    """Naive forecast: the value observed `lag` hours earlier (yesterday, same hour)."""
    s = np.asarray(series, float)
    return lambda t: s[np.clip(np.arange(t, t + H) - lag, 0, len(s) - 1)]


def ecmwf_day_ahead(path, idx, utc_offset_h=4):
    """ECMWF IFS 24 h-ahead forecasts (Open-Meteo Previous Runs, `*_previous_day1`) on the local hourly index idx:
    temperature (deg C) and global horizontal irradiance (W/m2). The file is in UTC; the first utc_offset_h local
    hours of the year, which fall on the previous UTC day, take the first forecast value."""
    import json
    import pandas as pd
    h = json.loads(open(path).read())["hourly"]
    f = pd.DataFrame(h)
    f.index = pd.to_datetime(f.pop("time")) + pd.Timedelta(hours=utc_offset_h)
    f = f.reindex(idx).bfill().ffill()
    return pd.DataFrame({"temp": f["temperature_2m_previous_day1"].to_numpy(float),
                         "ghi": f["shortwave_radiation_previous_day1"].to_numpy(float)}, index=idx)


def with_decomposition(w, lat, lon, tz):
    """Add DNI and DHI to a frame holding only GHI (Erbs et al. 1982 decomposition, pvlib), so that the PV model
    of data.pv_output can run on forecast irradiance."""
    import pandas as pd
    import pvlib
    t = w.index.tz_localize(tz) if w.index.tz is None else w.index
    sp = pvlib.solarposition.get_solarposition(t + pd.Timedelta(minutes=30), lat, lon)
    d = pvlib.irradiance.erbs(w["ghi"].to_numpy(float), sp["zenith"].to_numpy(), t)
    out = w.copy()
    out["dni"], out["dhi"] = np.nan_to_num(np.asarray(d["dni"], float)), np.nan_to_num(np.asarray(d["dhi"], float))
    return out


def mos(forecast, actual, window_days=30):
    """Causal hour-of-day bias correction (model output statistics). The forecast for day d is issued on day d-1,
    when the actuals are complete through day d-2; it is shifted by the mean error of each hour of the day over
    the window_days days ending on day d-2. Series are hourly, starting at midnight."""
    F, A = np.asarray(forecast, float).reshape(-1, 24), np.asarray(actual, float).reshape(-1, 24)
    out = F.copy()
    for d in range(2, len(F)):
        lo, hi = max(0, d - 1 - window_days), d - 1
        out[d] = F[d] + np.mean(A[lo:hi] - F[lo:hi], axis=0)
    return out.ravel()


def error_samples(forecast, actual, window_days=30, spread_h=2):
    """Causal empirical error samples for each hour: the errors (actual - forecast) of the window_days days ending
    on day d-2, at the same hour of the day +- spread_h. Returns [n_hours, n_samples] (NaN-padded early on)."""
    F, A = np.asarray(forecast, float).reshape(-1, 24), np.asarray(actual, float).reshape(-1, 24)
    E = A - F
    nd = len(F)
    ns = window_days * (2 * spread_h + 1)
    out = np.full((nd, 24, ns), np.nan)
    for d in range(2, nd):
        lo, hi = max(0, d - 1 - window_days), d - 1
        W = E[lo:hi]                                                     # [w, 24]
        for h in range(24):
            hs = [(h + k) % 24 for k in range(-spread_h, spread_h + 1)]
            v = W[:, hs].ravel()
            out[d, h, :len(v)] = v
    return out.reshape(nd * 24, ns)


def expected_price(net_fc, samples, price_of_net):
    """Expected S2 price at each hour given a net-load forecast and samples of its error: the planning LP is
    linear in the price, so the expected price is what a risk-neutral plan needs (for the peak adder, this is the
    probability that the hour falls in the net-load peak set). Falls back to the point forecast without samples."""
    net_fc = np.asarray(net_fc, float)
    x = net_fc[:, None] + samples
    p = price_of_net(np.where(np.isnan(x), net_fc[:, None], x))
    ok = np.isfinite(samples).any(1)
    point = price_of_net(net_fc)
    return np.where(ok, np.nanmean(np.where(np.isnan(samples), np.nan, p), axis=1), point)
