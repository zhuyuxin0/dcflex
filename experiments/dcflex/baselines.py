"""DR baselines B1-B5, placebo accuracy, inflation and baseline-shopping tests (paper Section III.F)."""
import numpy as np

NAMES = {"b1": "High-5-of-10", "b2": "Rolling regression", "b3": "Committed regression", "b4": "Digital twin",
         "b5": "Physics-guided network"}


def _cell(M, d, h):
    return M[d + h // 24, h % 24]


def _win(M, d, hrs):
    return np.array([_cell(M, d, h) for h in hrs])


# Time-of-week-and-temperature (TOWT) regression (Mathieu et al. 2011): one indicator per hour of the week and a
# piecewise-linear temperature response. Knots suit a hot climate; beyond the last knot the response is linear.
TOWT_KNOTS = (20.0, 25.0, 30.0, 35.0, 40.0)


def _temp_segments(t, knots=TOWT_KNOTS):
    edges = (-np.inf,) + tuple(knots) + (np.inf,)
    seg = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_ = knots[0] if lo == -np.inf else lo
        seg.append(min(t, hi) if lo == -np.inf else float(np.clip(t - lo_, 0.0, hi - lo_)))
    return seg


def _reg_rows(Tm, d, hrs, dow):
    """Design rows for day d and hours hrs (which may run past midnight): 168 time-of-week indicators and
    len(TOWT_KNOTS) + 1 temperature segments."""
    X = []
    for h in hrs:
        row = np.zeros(168 + len(TOWT_KNOTS) + 1)
        dd = d + h // 24
        row[int(dow[dd]) * 24 + h % 24] = 1
        row[168:] = _temp_segments(_cell(Tm, d, h))
        X.append(row)
    return np.array(X)


def reg_fit(L, Tm, days, dow):
    X = np.vstack([_reg_rows(Tm, d, range(24), dow) for d in days])
    y = np.concatenate([L[d] for d in days])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def day_of_week(idx, nd):
    """Day of week (Monday = 0) of each of the nd days; one extra day so windows past midnight have a value."""
    dw = np.asarray(idx.dayofweek)[::24][:nd]
    return np.append(dw, (dw[-1] + 1) % 7)


def b_high(L, d, hrs, excluded):
    prior = [p for p in range(d - 1, -1, -1) if p not in excluded][:10]
    top = sorted(prior, key=lambda p: _win(L, p, hrs).mean(), reverse=True)[:5]
    base = np.mean([_win(L, p, hrs) for p in top], axis=0)
    # day-of adjustment over the four hours before the window, reaching into the previous day if needed
    adj = [h for h in range(hrs[0] - 4, hrs[0]) if min([d] + top) + h // 24 >= 0]
    if adj:
        ratio = np.mean([_cell(L, d, h) for h in adj]) / np.mean([_cell(L, p, h) for p in top for h in adj])
        base = base * np.clip(ratio, 0.8, 1.2)
    return base


def inflate(L, ev_days, peak_hour, g, G, pre_hours=4):
    """Strategic baseline inflation: load x (1 + g) on the G days before each event and in the
    pre_hours hours before its window (reaching into the previous day if needed). Hours covered by
    several events are inflated once, not compounded."""
    mask = np.zeros(L.shape, bool)
    for e in ev_days:
        mask[max(0, e - G):e] = True
        for h in range(peak_hour[e] - pre_hours, peak_hour[e]):
            if e + h // 24 >= 0:
                mask[e + h // 24, h % 24] = True
    return np.where(mask, L * (1 + g), L)


def estimate(L, Tm, TW, d, hrs, excluded, coef_committed, dow, B5=None):
    """Every baseline's estimate for day d, hours hrs. B5: day matrix of the committed learned baseline's
    hourly predictions (None leaves B5 out)."""
    prior = [p for p in range(d - 1, -1, -1) if p not in excluded][:30]
    out = {"b1": b_high(L, d, hrs, excluded),
           "b2": _reg_rows(Tm, d, hrs, dow) @ reg_fit(L, Tm, prior, dow),
           "b3": _reg_rows(Tm, d, hrs, dow) @ coef_committed,
           "b4": _win(TW, d, hrs)}
    if B5 is not None:
        out["b5"] = _win(B5, d, hrs)
    return out


def pre_season(idx, study):
    """Hour mask of the 60 days before the first summer month: the committed baselines' training window."""
    nd = len(idx) // 24
    summer = [d for d in range(nd) if idx[d * 24].month in study.summer_months]
    m = np.zeros(nd * 24, bool)
    m[max(0, summer[0] - 60) * 24: summer[0] * 24] = True
    return m


def experiment(load, temp, twin, idx, ev_days, peak_hour, study, r_true, rng, learned=None):
    """learned: optional callable load -> hourly predictions of a baseline trained only on the pre-season hours
    of that load (B5). It is refitted on the inflated load only if inflation reaches the pre-season window."""
    L, Tm, TW = load.reshape(-1, 24), temp.reshape(-1, 24), twin.reshape(-1, 24)
    names = [k for k in NAMES if k != "b5" or learned is not None]
    nd = L.shape[0]
    summer = [d for d in range(nd) if idx[d * 24].month in study.summer_months]
    start = summer[0]
    pre = list(range(max(0, start - 60), start))
    excluded = set(ev_days)
    H = study.event_hours
    # --- accuracy on placebo windows (true counterfactual known)
    cand = [d for d in summer if d not in excluded and d >= 31 and d + 1 < nd]
    placebo = rng.choice(cand, size=min(study.placebo_days, len(cand)), replace=False)
    dow = day_of_week(idx, nd)
    coef = reg_fit(L, Tm, pre, dow)
    B5 = learned(load).reshape(-1, 24) if learned is not None else None
    err = {k: [] for k in names}
    for d in placebo:
        hrs = list(range(peak_hour[d], peak_hour[d] + H))
        truth = _win(L, d, hrs)
        for k, b in estimate(L, Tm, TW, d, hrs, excluded, coef, dow, B5).items():
            err[k].append(b - truth)
    acc = {}
    for k, e in err.items():
        e = np.concatenate(e)
        m = np.mean([_win(L, d, range(peak_hour[d], peak_hour[d] + H)).mean() for d in placebo])
        acc[k] = dict(bias=float(e.mean()), nmae=float(100 * np.abs(e).mean() / m),
                      cvrmse=float(100 * np.sqrt((e ** 2).mean()) / m))
    # --- baseline inflation before events
    g, G = study.gaming_inflation, study.gaming_days
    Lg = inflate(L, ev_days, peak_hour, g, G)
    coef_g = reg_fit(Lg, Tm, pre, dow)     # committed before the season: unaffected if pre-season is clean
    B5g = None
    if learned is not None:
        pm = pre_season(idx, study)
        B5g = B5 if np.array_equal(Lg.ravel()[pm], L.ravel()[pm]) else learned(Lg.ravel()).reshape(-1, 24)
    over, credit_clean = {k: [] for k in names}, {k: [] for k in names}
    for e in ev_days:
        hrs = list(range(peak_hour[e], peak_hour[e] + H))
        metered = _win(L, e, hrs) - r_true
        true = r_true * H
        for k, b in estimate(Lg, Tm, TW, e, hrs, excluded, coef_g, dow, B5g).items():
            over[k].append(100 * ((b - metered).sum() - true) / true)
        for k, b in estimate(L, Tm, TW, e, hrs, excluded, coef, dow, B5).items():
            credit_clean[k].append((b - metered).sum())
    for k in names:
        acc[k]["overcredit"] = float(np.mean(over[k]))
    return acc, shopping(credit_clean, r_true * H)


def shopping(credits, true):
    """Over-crediting (% of the true reduction) if the method is chosen after the event: the largest
    credit across methods minus the true reduction, averaged over events (definition in Sec. III.F)."""
    n = len(next(iter(credits.values())))
    return float(np.mean([100 * (max(credits[k][i] for k in credits) - true) / true for i in range(n)]))


def inflation_example(load, temp, twin, idx, ev_days, peak_hour, study, r_true, net, learned_pred=None):
    """One event of the inflation experiment, for the figure: the event with the highest net-load peak.
    Returns the true and inflated hourly load over the G days before it and the event day, and each
    baseline's event-window estimate with and without inflation (same estimators as experiment()).
    learned_pred: B5's hourly predictions (trained on the clean pre-season, which inflation does not reach)."""
    L, Tm, TW = load.reshape(-1, 24), temp.reshape(-1, 24), twin.reshape(-1, 24)
    nd = L.shape[0]
    summer = [d for d in range(nd) if idx[d * 24].month in study.summer_months]
    pre = list(range(max(0, summer[0] - 60), summer[0]))
    excluded, H = set(ev_days), study.event_hours
    g, G = study.gaming_inflation, study.gaming_days
    Lg = inflate(L, ev_days, peak_hour, g, G)
    dow = day_of_week(idx, nd)
    coef, coef_g = reg_fit(L, Tm, pre, dow), reg_fit(Lg, Tm, pre, dow)
    dmax = np.asarray(net).reshape(-1, 24).max(1)
    e = max(ev_days, key=lambda d: dmax[d])
    hrs = list(range(peak_hour[e], peak_hour[e] + H))
    days = list(range(e - G, e + 1))
    truth = _win(L, e, hrs)
    B5 = None if learned_pred is None else np.asarray(learned_pred).reshape(-1, 24)
    return dict(event_day=e, date=str(idx[e * 24].date()), season_start=str(idx[summer[0] * 24].date()),
                coef=coef, features=Tm[e, hrs], days=days, hrs=hrs, G=G, g=g, r_true=r_true,
                load=L[days].ravel(), load_inflated=Lg[days].ravel(), truth=truth, metered=truth - r_true,
                clean=estimate(L, Tm, TW, e, hrs, excluded, coef, dow, B5),
                gamed=estimate(Lg, Tm, TW, e, hrs, excluded, coef_g, dow, B5))
