"""Day-ahead forecastability of a real data center's IT load (NLR ESIF HPC data center, realdata.esif()).

The MPC of the paper forecasts deferrable arrivals by their expected weekly profile, so its workload forecast error
is the noise of the synthetic arrivals. This module checks that assumption against metered IT power: every day at
12:00 local time (19:00 UTC) the next local day's 24 hourly values are forecast from the history available then,
by persistence (the last 24 hours), weekly persistence, LightGBM on lagged values, and Chronos-Bolt, a pretrained
time-series foundation model used zero-shot.
"""
import numpy as np
import pandas as pd

ISSUE_UTC = 19          # 12:00 MST
START_UTC = 7           # the forecast day runs 07:00-06:00 UTC (00:00-23:00 MST)
LEAD0 = 24 - ISSUE_UTC + START_UTC    # hours from issue to the first forecast hour (12)
H = 24
CONTEXT = 28 * 24


def series(year, col="it_power_kw"):
    """Hourly IT power (kW), UTC, for the year and the 60 days before it; gaps up to 6 h interpolated."""
    from .realdata import esif
    s = esif()[col]
    s = s[(s.index >= pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=60)) & (s.index < pd.Timestamp(f"{year + 1}-01-02"))]
    s = s.asfreq("h")
    return s.interpolate(limit=6, limit_area="inside")


def issues(s, year):
    """Issue times (index positions) for each forecast day of the year with complete context and truth."""
    out = []
    for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
        t_issue = day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_UTC)
        i = s.index.get_indexer([t_issue])[0]
        if i < CONTEXT or i + LEAD0 + H > len(s):
            continue
        ctx, tru = s.values[i - CONTEXT + 1:i + 1], s.values[i + LEAD0:i + LEAD0 + H]
        if np.isfinite(ctx).all() and np.isfinite(tru).all():
            out.append(i)
    return out


def naive(s, idx, lag):
    """Value `lag` hours before each target hour, using only data up to the issue time (lag >= LEAD0 + H)."""
    v = s.values
    return np.array([[v[i + LEAD0 + k - lag] for k in range(H)] for i in idx])


def gbm(s, idx, lags=(36, 48, 72, 168, 336)):
    """LightGBM on lagged values (all at least 36 h, so available at issue time) and calendar; refitted each month
    on the preceding 365 days."""
    import lightgbm as lgb
    v, t = s.values, s.index
    feats = lambda j: [v[j - L] for L in lags] + [t[j].hour, t[j].dayofweek]
    out, model, fitted_month = [], None, None
    for i in idx:
        month = t[i].to_period("M")
        if month != fitted_month:
            rows = [j for j in range(max(max(lags), i - 365 * 24), i + 1 - 36) if np.isfinite(v[j])
                    and all(np.isfinite(v[j - L]) for L in lags)]
            X = np.array([feats(j) for j in rows]); y = v[rows]
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=1,
                                      random_state=0).fit(X, y)
            fitted_month = month
        out.append(model.predict(np.array([feats(i + LEAD0 + k) for k in range(H)])))
    return np.array(out)


def chronos(s, idx, model="amazon/chronos-bolt-small", q=(0.1, 0.5, 0.9)):
    """Zero-shot Chronos-Bolt quantile forecasts; the first LEAD0 steps (issue to forecast day) are discarded."""
    import torch
    from chronos import BaseChronosPipeline
    torch.manual_seed(0)
    pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu", torch_dtype=torch.float32)
    ctx = [torch.tensor(s.values[i - CONTEXT + 1:i + 1], dtype=torch.float32) for i in idx]
    qs, _ = pipe.predict_quantiles(ctx, prediction_length=LEAD0 + H, quantile_levels=list(q))
    return qs[:, LEAD0:, :].numpy()                         # [days, 24, len(q)]


def truth(s, idx):
    return np.array([s.values[i + LEAD0:i + LEAD0 + H] for i in idx])


def evaluate(year):
    s = series(year)
    idx = issues(s, year)
    y = truth(s, idx)
    f = {"Persistence (two days earlier)": naive(s, idx, 48), "Weekly persistence": naive(s, idx, 168),
         "LightGBM on lags": gbm(s, idx)}
    qc = chronos(s, idx)
    f["Chronos-Bolt (zero-shot)"] = qc[:, :, 1]
    res = {k: dict(nmae=float(100 * np.abs(p - y).mean() / y.mean()),
                   bias=float(100 * (p - y).mean() / y.mean())) for k, p in f.items()}
    cover = float(np.mean((y >= qc[:, :, 0]) & (y <= qc[:, :, 2])))
    return dict(year=year, days=len(idx), mean_kw=float(y.mean()), res=res, chronos_cover80=100 * cover,
                cv_hourly=float(100 * y.std() / y.mean()))


def scaled_errors(years=(2019, 2021, 2025)):
    """Chronos-Bolt day-ahead errors of the ESIF IT load, (forecast - truth) divided by the mean load of the 28-day
    context: one row of 24 hours per forecast day (MST hours 0-23), years stacked in time order. Scaling by the
    typical level, as nMAE does, keeps outage days (true load near zero) from producing unbounded relative errors.
    Used to perturb the MPC's IT-load forecasts."""
    rows = []
    for y in years:
        s = series(y)
        idx = issues(s, y)
        level = np.array([s.values[i - CONTEXT + 1:i + 1].mean() for i in idx])
        rows.append((chronos(s, idx)[:, :, 1] - truth(s, idx)) / level[:, None])
    return np.vstack(rows)
