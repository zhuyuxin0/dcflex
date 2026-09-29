"""Facility model (paper Eqs. 1-8): LP in CVXPY, solved with HiGHS; BAU and rule-based strategies."""
import numpy as np
import cvxpy as cp

ALL = ("defer", "tes", "bess")


def _solver():
    return cp.HIGHS if "HIGHS" in cp.installed_solvers() else cp.CLARABEL


def inputs(fac, a, p0, temp, pv):
    return {"a_trn": np.asarray(a["trn"], float), "a_bat": np.asarray(a["bat"], float),
            "p0": np.asarray(p0, float), "cop": fac.cop(temp), "pv": np.asarray(pv, float),
            "temp": np.asarray(temp, float)}


def window(inp, s):
    return {k: v[s] for k, v in inp.items()}


def _account(fac, inp, pit, qin, qout, bc, bd):
    ex = 1 + fac.loss_frac + fac.fan_frac
    pch = (ex * pit + qin - qout) / inp["cop"]
    load = ex * pit + pch + fac.aux_mw + bc - bd
    pvu = np.minimum(inp["pv"], np.maximum(load, 0))
    return pch, load - pvu, pvu


def bau(inp, fac):
    """S0: serve work on arrival (FIFO when capacity binds); storage idle; PV self-consumed."""
    T = len(inp["p0"])
    y = {c: np.zeros(T) for c in ("trn", "bat")}
    back = {c: 0.0 for c in y}
    for t in range(T):
        room = max(fac.it_mw - inp["p0"][t], 0.0)
        for c in ("trn", "bat"):
            back[c] += inp["a_" + c][t]
            s = min(back[c], room)
            y[c][t], back[c], room = s, back[c] - s, room - s
    pit = inp["p0"] + y["trn"] + y["bat"]
    z = np.zeros(T)
    pch, pg, pvu = _account(fac, inp, pit, z, z, z, z)
    return dict(pit=pit, y_trn=y["trn"], y_bat=y["bat"], pch=pch, pg=pg, pv_used=pvu,
                qin=z, qout=z, bc=z, bd=z, ras=z, r=None)


def rule_based(inp, fac, hours):
    """R: TES and battery charge 10-15h, discharge 17-22h; batch paused 17-22h then caught up."""
    T = len(inp["p0"])
    ex = 1 + fac.loss_frac + fac.fan_frac
    cap, rate = fac.tes_hours * fac.peak_heat_mw, fac.tes_rate_frac * fac.peak_heat_mw
    res = fac.reserve_min / 60 * fac.it_mw
    S, E = 0.5 * cap, 0.9 * fac.bess_mwh
    y = {c: np.zeros(T) for c in ("trn", "bat")}
    back = {c: 0.0 for c in y}
    qin, qout, bc, bd = (np.zeros(T) for _ in range(4))
    for t in range(T):
        h = hours[t]
        room = max(fac.it_mw - inp["p0"][t], 0.0)
        for c in ("trn", "bat"):
            back[c] += inp["a_" + c][t]
            if c == "bat" and 17 <= h < 22:
                continue
            s = min(back[c], room)
            y[c][t], back[c], room = s, back[c] - s, room - s
        heat = ex * (inp["p0"][t] + y["trn"][t] + y["bat"][t])
        if 10 <= h < 15:
            qin[t] = min(rate, max(cap - S, 0) / fac.tes_eff)
            bc[t] = min(fac.bess_mw, max(fac.bess_mwh - E, 0) / fac.bess_eta)
        elif 17 <= h < 22:
            qout[t] = min(rate, heat, (1 - fac.tes_loss_h) * S)
            bd[t] = min(fac.bess_mw, max(E - res, 0) * fac.bess_eta)
        S = (1 - fac.tes_loss_h) * S + fac.tes_eff * qin[t] - qout[t]
        E = E + fac.bess_eta * bc[t] - bd[t] / fac.bess_eta
    pit = inp["p0"] + y["trn"] + y["bat"]
    pch, pg, pvu = _account(fac, inp, pit, qin, qout, bc, bd)
    return dict(pit=pit, y_trn=y["trn"], y_bat=y["bat"], pch=pch, pg=pg, pv_used=pvu,
                qin=qin, qout=qout, bc=bc, bd=bd, ras=np.zeros(T), r=None)


def optimize(inp, fac, price, levers=ALL, events=None, baseline=None, cap_pay=0.0,
             as_price=None, shed=None, s0=0.5, e0=0.9, eps=0.01):
    """Solve the facility LP. price: currency/MWh. shed=(window_idx, reference_pg) maximizes the
    reduction held over the window (Eq. flex). events/baseline/cap_pay implement the DR contract (Eq. dr)."""
    T = len(price)
    ex = 1 + fac.loss_frac + fac.fan_frac
    cons, y = [], {}
    # Business as usual (serve on arrival, FIFO when the IT capacity binds) is work-conserving, so its
    # cumulative service is the most that can be done by any hour. Without the deferral lever work
    # follows it; with deferral, deadlines apply except where capacity makes them impossible, and at
    # least as much work as under BAU must be done by the end of the window (all of it if BAU can).
    ref = bau(inp, fac)
    for c in ("trn", "bat"):
        a = inp["a_" + c]
        ca = np.cumsum(a)
        y[c] = cp.Variable(T, nonneg=True)
        if "defer" not in levers:
            cons += [y[c] == ref["y_" + c]]
            continue
        cy = cp.cumsum(y[c])
        cb = np.cumsum(ref["y_" + c])
        D = int(fac.deadline_h[c])
        lag = ca if D == 0 else (np.concatenate([np.zeros(D), ca[:-D]]) if D < T else np.zeros(T))
        cons += [cy <= ca + 1e-6, cy >= np.minimum(lag, cb) - 1e-6, cp.sum(y[c]) >= cb[-1] - 1e-6]
    pit = inp["p0"] + y["trn"] + y["bat"]
    cons += [pit <= fac.it_mw]
    heat = ex * pit
    thr, qin, qout, bc, bd = [], None, None, None, None
    qch = heat
    if "tes" in levers:
        cap, rate = fac.tes_hours * fac.peak_heat_mw, fac.tes_rate_frac * fac.peak_heat_mw
        S = cp.Variable(T + 1)
        qin, qout = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
        cons += [S[0] == s0 * cap, S[T] >= s0 * cap, S >= 0, S <= cap, qin <= rate, qout <= rate,
                 qout <= heat, S[1:] == (1 - fac.tes_loss_h) * S[:-1] + fac.tes_eff * qin - qout]
        qch = heat + qin - qout
        thr += [qin, qout]
    cons += [qch >= 0]
    pch = cp.multiply(1.0 / inp["cop"], qch)
    batt = 0
    if "bess" in levers:
        E = cp.Variable(T + 1)
        bc, bd = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
        res = fac.reserve_min / 60.0 * fac.it_mw
        cons += [E[0] == e0 * fac.bess_mwh, E[T] >= e0 * fac.bess_mwh, E >= res, E <= fac.bess_mwh,
                 bc <= fac.bess_mw, bd <= fac.bess_mw, E[1:] == E[:-1] + fac.bess_eta * bc - bd / fac.bess_eta]
        batt = bc - bd
        thr += [bc, bd]
    pvu = cp.Variable(T, nonneg=True)
    cons += [pvu <= inp["pv"]]
    pg = ex * pit + pch + fac.aux_mw + batt - pvu
    cons += [pg >= 0]
    # eps/10 reward on self-consumed PV breaks the tie when the price is zero (solar-surplus hours
    # under S2); otherwise the LP may leave on-site PV unused and import instead.
    obj = price @ pg + (eps * sum(cp.sum(v) for v in thr) if thr else 0) - eps / 10 * cp.sum(pvu)
    r = ras = None
    if events is not None:
        r = cp.Variable(nonneg=True)
        cons += [pg[events] <= baseline[events] - r]
        obj = obj - cap_pay * r
    if as_price is not None:
        ras = cp.Variable(T, nonneg=True)
        cons += [ras <= y["trn"] + y["bat"]]
        obj = obj - as_price @ ras
    if shed is not None:
        w, ref = shed
        r = cp.Variable()
        cons += [pg[w] <= ref[w] - r]
        obj = -1e4 * r + 1e-3 * (price @ pg)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=_solver())
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP status: {prob.status}")
    v = lambda x: np.zeros(T) if x is None else np.asarray(x.value, float).ravel()
    # storage levels at the end of each hour (MWh of cooling; MWh of electricity), for the dispatch figure
    soc = np.asarray(S.value, float)[1:] if "tes" in levers else np.zeros(T)
    soe = np.asarray(E.value, float)[1:] if "bess" in levers else np.zeros(T)
    return dict(pit=np.asarray(pit.value, float), y_trn=v(y["trn"]), y_bat=v(y["bat"]),
                pch=np.asarray(pch.value, float), pg=np.asarray(pg.value, float), pv_used=v(pvu),
                qin=v(qin), qout=v(qout), bc=v(bc), bd=v(bd), ras=v(ras), soc=soc, soe=soe,
                r=None if r is None else float(r.value))


def run_monthly(inp, fac, price, idx, strategy="opt", **kw):
    """Solve month by month (storage returns to its start level each month); concatenate results in
    chronological order, so windows that do not start in January (ERCOT) stay aligned with idx."""
    out = {}
    key = np.asarray(idx.year) * 12 + np.asarray(idx.month)
    for k in dict.fromkeys(key):
        pos = np.where(key == k)[0]
        s = slice(pos[0], pos[-1] + 1)
        w = window(inp, s)
        kws = {k: (v[s] if isinstance(v, np.ndarray) and len(v) == len(price) else v) for k, v in kw.items()}
        o = bau(w, fac) if strategy == "bau" else optimize(w, fac, price[s], **kws)
        for k, val in o.items():
            if isinstance(val, np.ndarray):
                out.setdefault(k, []).append(val)
    return {k: np.concatenate(v) for k, v in out.items()}
