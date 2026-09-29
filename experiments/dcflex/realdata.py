"""Real metered data for validating the learned baselines out of sample.

BDG2, the Building Data Genome 2 (Miller et al., Sci. Data 2020; Zenodo v1.0, CC BY 4.0): hourly meter data for
1,636 non-residential buildings at 19 sites, 2016-2017, local time, with on-site weather. We use the three hottest
US sites with chilled-water meters as hot-climate analogues: Fox (Tempe/Phoenix, hot desert like Abu Dhabi),
Bull (Austin) and Panther (Orlando, hot and humid). Raw files go in data/raw/bdg2/ (see data/README.md).

The validation mirrors the paper's DR setting. A "season" runs from June to September 2017. A new facility has
only the 60 days before the season to train a committed baseline; an established facility also has the year
before (2016). Events are called on the hottest weekdays, which is when a baseline trained in spring must
extrapolate; the event window is 14:00-18:00 local time.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .data import dew_point

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
BDG2 = RAW / "bdg2"
HOT_SITES = ("Fox", "Bull", "Panther")
YEAR = 2017
SEASON = (6, 7, 8, 9)
PRE_DAYS = 60
EVENT_DAYS = 10
EVENT_HOURS = (14, 15, 16, 17)


def _cache(name):
    d = BDG2 / "cache"
    d.mkdir(exist_ok=True)
    return d / f"{name}.parquet"


def meter(kind):
    """Wide hourly table (timestamp x building) for 'electricity' or 'chilledwater', 2016-2017, float32."""
    c = _cache(kind)
    if c.exists():
        return pd.read_parquet(c)
    m = pd.read_csv(BDG2 / f"{kind}_cleaned.csv", parse_dates=["timestamp"]).set_index("timestamp")
    m = m.astype("float32")
    full = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    m = m[~m.index.duplicated()].reindex(full)
    m.index.name = "timestamp"
    m.to_parquet(c)
    return m


def weather(site):
    """Hourly dry-bulb and dew-point temperature (deg C) at a site; gaps of up to 6 h interpolated."""
    c = _cache("weather")
    if not c.exists():
        w = pd.read_csv(BDG2 / "weather.csv", parse_dates=["timestamp"])
        w[["timestamp", "site_id", "airTemperature", "dewTemperature"]].to_parquet(c)
    w = pd.read_parquet(c)
    w = w[w.site_id == site].drop_duplicates("timestamp").set_index("timestamp")
    full = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    w = w[["airTemperature", "dewTemperature"]].reindex(full).interpolate(limit=6, limit_area="inside")
    return w.rename(columns={"airTemperature": "temp", "dewTemperature": "dew"})


def metadata():
    return pd.read_csv(BDG2 / "metadata.csv").set_index("building_id")


def windows(idx, year=YEAR):
    """Boolean masks over idx: pre-season (60 days), season (Jun-Sep) and prior year."""
    t = pd.DatetimeIndex(idx)
    start = pd.Timestamp(f"{year}-{SEASON[0]:02d}-01")
    pre = (t >= start - pd.Timedelta(days=PRE_DAYS)) & (t < start)
    season = (t.year == year) & t.month.isin(SEASON)
    prior = t.year == year - 1
    return np.asarray(pre), np.asarray(season), np.asarray(prior)


def clean(y, max_run=24):
    """Meter outages to NaN: runs of more than max_run hours with a constant reading (zeros included)."""
    y = np.array(y, float)
    ch = np.flatnonzero(np.r_[True, np.diff(y) != 0, True])
    for a, b in zip(ch[:-1], ch[1:]):
        if b - a > max_run:
            y[a:b] = np.nan
    return y


def eligible(Y, masks, min_cover=0.97, max_zero=0.02):
    """Columns of Y (already cleaned) with coverage >= min_cover and at most max_zero zero readings in every
    window in masks, and a positive mean."""
    keep = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        ok = all(np.isfinite(y[w]).mean() >= min_cover and np.nanmean(y[w] == 0) <= max_zero for w in masks)
        if ok and np.nanmean(y[np.logical_or.reduce(masks)]) > 0:
            keep.append(j)
    return keep


def event_days(temp, idx, n=EVENT_DAYS):
    """The n hottest working weekdays of the season (daily maximum temperature; no US federal holidays),
    as day indices into idx."""
    t = pd.DatetimeIndex(idx)
    _, season, _ = windows(idx)
    tmax = pd.Series(temp, index=t).resample("D").max()
    days = tmax.index
    hol = USFederalHolidayCalendar().holidays(days[0], days[-1])
    ok = (np.asarray(days.dayofweek < 5) & np.asarray(days.year == YEAR) & np.asarray(days.month.isin(SEASON))
          & ~np.asarray(days.isin(hol)))
    cand = np.flatnonzero(ok & np.isfinite(tmax.to_numpy()))
    order = cand[np.argsort(-tmax.to_numpy()[cand], kind="stable")]
    return np.sort(order[:n])


def site_panel(site, kind, established=False, max_buildings=None, seed=0):
    """Everything one site-meter evaluation needs: loads [N, B] (outages cleaned), weather, calendar, masks and
    event days. established=True also requires a near-complete prior year (2016)."""
    M = meter(kind)
    cols = [c for c in M.columns if c.startswith(site + "_")]
    idx = M.index
    Y = np.column_stack([clean(M[c].to_numpy()) for c in cols]) if cols else np.zeros((len(idx), 0))
    pre, season, prior = windows(idx)
    keep = eligible(Y, [pre, season] + ([prior] if established else []))
    if max_buildings and len(keep) > max_buildings:
        keep = sorted(np.random.default_rng(seed).choice(keep, size=max_buildings, replace=False))
    W = weather(site)
    return dict(site=site, kind=kind, buildings=[cols[j] for j in keep], Y=Y[:, keep], idx=idx,
                hour=idx.hour.to_numpy(), dow=idx.dayofweek.to_numpy(), temp=W.temp.to_numpy(),
                dew=W.dew.to_numpy(), pre=pre, season=season, prior=prior,
                events=event_days(W.temp.to_numpy(), idx))


def pool(kind, exclude=(), max_per_site=None, seed=0):
    """Pretraining pool: every BDG2 building with a `kind` meter outside the sites in `exclude`, both years,
    outages cleaned; one panel per site (with `use` = all hours and `names`) for pretrain.pool_rows."""
    M = meter(kind)
    sites = sorted({c.split("_")[0] for c in M.columns} - set(exclude))
    rng = np.random.default_rng(seed)
    out = []
    for s in sites:
        cols = [c for c in M.columns if c.startswith(s + "_")]
        if max_per_site and len(cols) > max_per_site:
            cols = sorted(rng.choice(cols, size=max_per_site, replace=False))
        W = weather(s)
        if W.temp.isna().mean() > 0.5:
            continue
        idx = M.index
        out.append(dict(site=s, names=cols, Y=np.column_stack([clean(M[c].to_numpy()) for c in cols]),
                        hour=idx.hour.to_numpy(), dow=idx.dayofweek.to_numpy(), temp=W.temp.to_numpy(),
                        dew=W.dew.to_numpy(), use=np.ones(len(idx), bool)))
    return out


ESIF = RAW / "nlr_esif"


def esif():
    """NLR (formerly NREL) ESIF HPC data center, Golden, Colorado (NREL data catalog, DOI 10.7799/3015212):
    hourly means of IT power, cooling, HVAC, pump and plug-and-light power (kW), with outdoor dry-bulb (deg C)
    and dew point (deg C, Magnus formula from relative humidity). Timestamps are UTC (the daily temperature
    maximum falls at 21:00, i.e. 14:00 MST)."""
    c = ESIF / "esif_hourly.parquet"
    if c.exists():
        return pd.read_parquet(c)
    pue = pd.read_parquet(next(ESIF.glob("*PUE*.parquet")),
                          columns=["ts", "it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"])
    out = pd.read_parquet(next(ESIF.glob("*outside*.parquet")), columns=["ts", "outdoor_air_temp", "outdoor_air_humidity"])
    out = out[out.ts >= "2015-01-01"]
    out = out[(out.outdoor_air_temp > -40) & (out.outdoor_air_temp < 115) & (out.outdoor_air_humidity > 0)]
    h = pue.set_index("ts").sort_index().resample("h").mean()
    w = out.set_index("ts").sort_index().resample("h").mean()
    t = (w.outdoor_air_temp - 32) / 1.8
    df = h.join(pd.DataFrame({"temp": t, "dew": dew_point(t, w.outdoor_air_humidity)}, index=w.index), how="left")
    df.to_parquet(c)
    return df
