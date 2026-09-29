"""Out-of-sample validation of DR baselines on real metered data (realdata.site_panel).

Every baseline is fitted only on data available before the season (the committed ones) or before each event
(the rolling ones), and scored on the season: hourly CV(RMSE) and NMBE over June-September (the ASHRAE
Guideline 14-2014 hourly calibration limits are 30% and +-10%), and the error of the 14:00-18:00 window on
the hottest weekdays, when DR events are called and a spring-trained model must extrapolate.
"""
import numpy as np
import torch

from . import baselines, learned_batch

# B5 and its ablations; all see the same inputs (calendar, dry bulb, dew point)
NEURAL = {"pgnb": dict(use_dew=True), "pgnb_free": dict(constrained=False, use_dew=True), "mlp": dict(use_dew=True)}


def _days(p):
    nd = len(p["idx"]) // 24
    return nd, baselines.day_of_week(p["idx"], nd)


def _design(p):
    """TOWT design rows for every hour of the panel (the rows baselines.reg_fit builds day by day)."""
    nd, dow = _days(p)
    Tm = p["temp"][:nd * 24].reshape(nd, 24)
    return np.vstack([baselines._reg_rows(Tm, d, range(24), dow) for d in range(nd)])


def _query_rows(q):
    """TOWT design rows for query points (hour, dow, temp)."""
    X = np.zeros((len(q["temp"]), 168 + len(baselines.TOWT_KNOTS) + 1))
    for i, (h, d, t) in enumerate(zip(q["hour"], q["dow"], q["temp"])):
        X[i, int(d) * 24 + int(h)] = 1
        X[i, 168:] = baselines._temp_segments(t)
    return X


def towt_committed(p, train, query=None):
    """B3: TOWT regression fitted once on the complete days of `train`, applied to every hour (and to the query
    points, if given: dict of hour, dow, temp, dew arrays; returns (pred, pred_query))."""
    nd, dow = _days(p)
    Tm = p["temp"][:nd * 24].reshape(nd, 24)
    X = _design(p)
    tr_days = [d for d in range(nd) if train[d * 24]]
    out = np.full(p["Y"].shape, np.nan)
    qo = None if query is None else np.full((len(query["temp"]), p["Y"].shape[1]), np.nan)
    for j in range(p["Y"].shape[1]):
        L = p["Y"][:nd * 24, j].reshape(nd, 24)
        ok = [d for d in tr_days if np.isfinite(L[d]).all() and np.isfinite(Tm[d]).all()]
        coef = baselines.reg_fit(L, Tm, ok, dow)
        out[:nd * 24, j] = X @ coef
        if query is not None:
            qo[:, j] = _query_rows(query) @ coef
    return out if query is None else (out, qo)


def gbm(p, train, seed=0, query=None):
    """LightGBM on hour, day of week, temperature and dew point (the black-box benchmark of Touzani et al.)."""
    import lightgbm as lgb
    X = np.column_stack([p["hour"], p["dow"], p["temp"], p["dew"]])
    out = np.full(p["Y"].shape, np.nan)
    Xq = None if query is None else np.column_stack([query["hour"], query["dow"], query["temp"], query["dew"]])
    qo = None if query is None else np.full((len(Xq), p["Y"].shape[1]), np.nan)
    for j in range(p["Y"].shape[1]):
        y = p["Y"][:, j]
        m = train & np.isfinite(y)
        g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, random_state=seed, verbose=-1,
                              n_jobs=torch.get_num_threads())      # same budget as torch: no oversubscription
        s = np.nanmean(y[m])
        g.fit(X[m], y[m] / s)
        out[:, j] = g.predict(X) * s
        if query is not None:
            qo[:, j] = g.predict(Xq) * s
    return out if query is None else (out, qo)


def neural(p, train, kind, seeds=(0, 1, 2), epochs=1000, chunk=64, query=None, **kw):
    """One PGNB/MLP ensemble per building, trained in vectorized chunks of buildings. With `query`, also returns
    the ensemble predictions at the query points."""
    ok_w = np.isfinite(p["temp"]) & np.isfinite(p["dew"])
    tr = train & ok_w
    th, dw = np.nan_to_num(p["temp"]), np.nan_to_num(p["dew"])
    pv = np.zeros(len(th))
    out = np.full(p["Y"].shape, np.nan)
    qo = None if query is None else np.full((len(query["temp"]), p["Y"].shape[1]), np.nan)
    curves = []
    for a in range(0, p["Y"].shape[1], chunk):
        cols = slice(a, a + chunk)
        pred, info = learned_batch.fit_many(kind, p["hour"], p["dow"], th, dw, pv, p["Y"][:, cols], tr,
                                            seeds=seeds, epochs=epochs, **{**NEURAL.get(kind, {}), **kw})
        out[:, cols] = pred(p["hour"], p["dow"], th, dw, pv)
        if query is not None:
            qo[:, cols] = pred(query["hour"], query["dow"], query["temp"], query["dew"], np.zeros(len(query["temp"])))
        curves.append(info["curve"])
    out[~ok_w] = np.nan
    return (out, np.mean(curves, axis=0)) if query is None else (out, np.mean(curves, axis=0), qo)


def rolling(p):
    """B1 (high 5 of 10 prior working days, day-of adjustment) and B2 (TOWT refitted on the 30 prior non-event
    days) for each event day's window; NaN elsewhere. Days with gaps in the window are skipped."""
    nd, dow = _days(p)
    Tm = p["temp"][:nd * 24].reshape(nd, 24)
    hrs = list(range(14, 18))
    ev = set(int(d) for d in p["events"])
    nonwork = {d for d in range(nd) if dow[d] >= 5}
    b1 = np.full(p["Y"].shape, np.nan)
    b2 = np.full(p["Y"].shape, np.nan)
    X = _design(p)
    for j in range(p["Y"].shape[1]):
        L = p["Y"][:nd * 24, j].reshape(nd, 24)
        bad = {d for d in range(nd) if not (np.isfinite(L[d]).all() and np.isfinite(Tm[d]).all())}
        for d in ev:
            h = d * 24 + np.array(hrs)
            b1[h, j] = baselines.b_high(L, d, hrs, ev | nonwork | bad)
            prior = [q for q in range(d - 1, -1, -1) if q not in ev and q not in bad][:30]
            rows = np.concatenate([np.arange(q * 24, q * 24 + 24) for q in prior])
            b2[h, j] = X[h] @ np.linalg.lstsq(X[rows], L.ravel()[rows], rcond=None)[0]    # = baselines.reg_fit
    return b1, b2


def response_query(p, hour=15, dow=2, t_lo=15.0, t_hi=50.0, n=71):
    """Query points for a temperature-response curve: a fixed weekday hour, dew point at its season median for that
    hour, dry bulb from t_lo to t_hi."""
    m = p["season"] & (p["hour"] == hour)
    dew = float(np.nanmedian(p["dew"][m]))
    t = np.linspace(t_lo, t_hi, n)
    return dict(hour=np.full(n, hour), dow=np.full(n, dow), temp=t, dew=np.full(n, dew))


def scores(p, pred, train):
    """Per-building metrics (arrays of length B), in % of the mean load of the scored hours."""
    Y = p["Y"]
    t_hi = np.nanmax(p["temp"][train])
    ev = np.zeros(len(Y), bool)
    for d in p["events"]:
        ev[d * 24 + 14: d * 24 + 18] = True
    out = {}
    for name, m in (("season", p["season"]), ("event", ev), ("hot", p["season"] & (p["temp"] > t_hi))):
        e = pred[m] - Y[m]
        ok = np.isfinite(e)
        mu = np.array([np.mean(Y[m][ok[:, j], j]) if ok[:, j].any() else np.nan for j in range(Y.shape[1])])
        n = ok.sum(0)
        e0 = np.where(ok, e, 0.0)
        out[name + "_cv"] = 100 * np.sqrt((e0 ** 2).sum(0) / np.maximum(n, 1)) / mu
        out[name + "_nmbe"] = 100 * e0.sum(0) / np.maximum(n, 1) / mu
        out[name + "_nmae"] = 100 * np.abs(e0).sum(0) / np.maximum(n, 1) / mu
        out[name + "_n"] = n
    return out
