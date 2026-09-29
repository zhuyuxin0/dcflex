"""Data acquisition. Real fetchers cache to data/raw. The synthetic generators exist ONLY for
software tests (run_all.py --synthetic); their outputs are flagged as synthetic in the paper macros."""
import pathlib
import warnings
import numpy as np
import pandas as pd

HOURLY = "temperature_2m,relative_humidity_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation"
RENAME = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "shortwave_radiation": "ghi",
          "direct_normal_irradiance": "dni", "diffuse_radiation": "dhi"}


def hourly_index(year):
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")


RADIATION = ["ghi", "dni", "dhi"]


def to_hour_start(w):
    """Open-Meteo radiation is the mean over the PRECEDING hour, stamped at the hour's end, whereas the
    pipeline labels every hour by its start. Shift radiation back one hour so that label t holds the
    mean over [t, t+1); temperature and humidity are instantaneous and keep their label."""
    w = w.copy()
    w[RADIATION] = w[RADIATION].shift(-1)
    return w.ffill()


OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"


def weather_request(lat, lon, start, end, tz, model="era5"):
    """URL and query of the Open-Meteo historical API; model pins the dataset (ERA5 by default)."""
    return OPEN_METEO, dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=HOURLY,
                            models=model, cell_selection="land", timezone=tz)


def weather_cache_path(lat, lon, start, end, tz, model="era5", cache_dir="data/raw"):
    return pathlib.Path(cache_dir) / f"openmeteo_{model}_{lat}_{lon}_{start}_{end}_{tz.replace('/', '-')}.json"


def fetch_weather(lat, lon, start, end, tz, cache_dir="data/raw", model="era5", archive_dir="data/weather"):
    """Hourly weather from the Open-Meteo historical API (CC BY 4.0), labeled by hour start. The raw
    API response is cached as JSON, so a browser download of the same URL can be saved under the cache
    name instead; to_hour_start() is applied on every read. Responses archived in the repository
    (archive_dir, committed) are used first, so the paper's run does not depend on the API quota."""
    import json
    archived = weather_cache_path(lat, lon, start, end, tz, model, archive_dir)
    path = archived if archived.exists() else weather_cache_path(lat, lon, start, end, tz, model, cache_dir)
    if not path.exists():
        import requests
        url, params = weather_request(lat, lon, start, end, tz, model)
        r = requests.get(url, params=params, timeout=180)
        if not r.ok:
            full = requests.Request("GET", url, params=params).prepare().url
            raise RuntimeError(f"Open-Meteo returned {r.status_code}: {r.text[:200]}\n"
                               f"Download {full} in a browser and save it as {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    js = json.loads(path.read_text())
    if js.get("error"):
        raise RuntimeError(f"{path} holds an API error, not data: {js.get('reason')}")
    df = pd.DataFrame(js["hourly"]).rename(columns=RENAME)
    df.index = pd.to_datetime(df.pop("time"))
    df = df[~df.index.duplicated()].interpolate().bfill().ffill()
    return to_hour_start(df)


def dew_point(temp, rh):
    """Dew point (deg C) from dry-bulb temperature (deg C) and relative humidity (%), Magnus formula
    (Alduchov and Eskridge coefficients 17.62 and 243.12 deg C)."""
    t = np.asarray(temp, float)
    g = np.log(np.clip(np.asarray(rh, float), 1.0, 100.0) / 100.0) + 17.62 * t / (243.12 + t)
    return 243.12 * g / (17.62 - g)


def pv_output(w, lat, lon, mwp, tz):
    """AC output (MW) of a fixed, equator-facing array: pvlib if installed, else a PVWatts-style approximation."""
    ghi = w["ghi"].to_numpy(float)
    temp = w["temp"].to_numpy(float)
    try:
        import pvlib
    except ImportError:
        pvlib = None
        warnings.warn("pvlib not installed: using the PVWatts-style approximation, not the model in the paper")
    if pvlib is not None:
        idx = w.index.tz_localize(tz) if w.index.tz is None else w.index
        # irradiance is the mean over [t, t+1): take the sun's position at mid-hour
        sp = pvlib.solarposition.get_solarposition(idx + pd.Timedelta(minutes=30), lat, lon)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=round(abs(lat)), surface_azimuth=180 if lat >= 0 else 0,
            solar_zenith=sp["apparent_zenith"].to_numpy(), solar_azimuth=sp["azimuth"].to_numpy(),
            dni=w["dni"].to_numpy(float), ghi=ghi, dhi=w["dhi"].to_numpy(float))["poa_global"]
        poa = np.nan_to_num(np.asarray(poa, float))
        tcell = pvlib.temperature.faiman(poa, temp)
        p = pvlib.pvsystem.pvwatts_dc(poa, tcell, mwp, -0.0037) * 0.86
    else:
        tcell = temp + 0.03 * ghi
        p = mwp * ghi / 1000.0 * 0.80 * (1 - 0.0037 * (tcell - 25))
    return np.clip(np.nan_to_num(np.asarray(p, float)), 0, mwp)


def profile_from_csv(index, path, value_col=None):
    """Week-periodic hourly shape (mean 1) from a trace CSV: first column = timestamp (naive Gulf time, or
    offset-aware, which is converted to Gulf time), optional value column (e.g., tokens or GPU-hours).
    See README for how to prepare Azure/Alibaba/Google traces."""
    df = pd.read_csv(path)
    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        raise ValueError(f"{path}: first column must be a clock timestamp, not a number")
    ts = pd.to_datetime(df.iloc[:, 0])
    if ts.dt.tz is not None:                      # offset-aware stamps: read them on the Gulf clock
        ts = ts.dt.tz_convert("Asia/Dubai").dt.tz_localize(None)
    if value_col is None and not ts.dt.floor("h").duplicated().any():
        raise ValueError(f"{path} has one row per hour (already aggregated): pass its value column, "
                         "e.g. --inference-col requests or --deferrable-col gpu_hours")
    val = df[value_col].to_numpy(float) if value_col else np.ones(len(df))
    s = pd.Series(val, index=ts).resample("h").sum()
    prof = s.groupby([s.index.dayofweek, s.index.hour]).mean()
    out = np.array([prof.get((d, h), np.nan) for d, h in zip(index.dayofweek, index.hour)], float)
    out = np.where(np.isnan(out), np.nanmean(out), out)
    return out / out.mean()


def local_clock(idx_utc, tz):
    """Naive local wall-clock times for a naive UTC index (hours repeat or skip at DST changes)."""
    return idx_utc.tz_localize("UTC").tz_convert(tz).tz_localize(None)


def parametric_profile(index, amp, peak_hour, weekend_factor=1.0):
    """Fallback shape when no trace is supplied. ASSUMPTION; replace with trace-derived shapes."""
    h = index.hour.to_numpy()
    d = index.dayofweek.to_numpy()
    s = 1 + amp * np.cos(2 * np.pi * (h - peak_hour) / 24)
    s = s * np.where(d >= 5, weekend_factor, 1.0)   # UAE weekend: Saturday-Sunday
    return s / s.mean()


def chiller_rating_points(data_dir="data", lcwt=10):
    """Manufacturer rating points for the COP figure: Carrier 30XA median COP across 19 units at 30-45 C
    (data/chiller_carrier_30xa_si.csv) and the median EER of 14 Daikin EWAD-MZ units at 46 C
    (data/chiller_daikin_ewad_46c.csv; evaporator 12.2/6.7 C)."""
    d = pd.read_csv(pathlib.Path(data_dir) / "chiller_carrier_30xa_si.csv")
    d = d[d.lcwt_c == lcwt].assign(cop=lambda x: x.cap_kw / x.input_kw).groupby("t_air_c").cop.median()
    k = pd.read_csv(pathlib.Path(data_dir) / "chiller_daikin_ewad_46c.csv")
    return [(d.index.to_numpy(float), d.to_numpy(float), f"Carrier 30XA, {lcwt} °C water (median)", "o"),
            ([46.0], [float(k.eer.median())], "Daikin EWAD-MZ, 46 °C (median)", "s")]


def build_arrivals(index, fac, rng, inf_shape, def_shape):
    """Hourly IT work (MWh/h) by class, scaled so mean IT power = util_mean * capacity."""
    dyn = (fac.util_mean - fac.idle_frac) * fac.it_mw
    a = {"inf": dyn * fac.shares["inf"] * inf_shape}
    for c in ("trn", "bat"):
        noise = np.clip(rng.lognormal(0.0, 0.25, len(index)), 0.6, 1.4)
        s = def_shape * noise
        a[c] = dyn * fac.shares[c] * s / s.mean()
    p0 = fac.idle_frac * fac.it_mw + a["inf"]
    return a, p0


def load_ercot(path, n_hours=8760):
    """CSV with columns time (UTC), rt_price (USD/MWh), as_price (USD/MW-h); see README.
    Returns the hourly index (UTC, naive) and the most recent n_hours of prices."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)     # naive stamps are read as UTC; offsets are converted
    df = df.set_index("time").sort_index()
    df.index = df.index.tz_localize(None)
    df = df.resample("h").mean().interpolate().bfill().ffill()
    if len(df) < n_hours:
        raise ValueError(f"ERCOT file has {len(df)} hours, need {n_hours}")
    df = df.iloc[-n_hours:]                                # the most recent n_hours (Appendix F)
    return df.index, df["rt_price"].to_numpy(float), df["as_price"].to_numpy(float)


# ---------------- synthetic generators (software tests only) ----------------
def synthetic_weather(index, rng, lat=24.45, t_mean=28.0, t_seas=8.0, t_diur=5.5):
    doy = index.dayofyear.to_numpy()
    h = index.hour.to_numpy() + 0.5
    temp = (t_mean + t_seas * np.sin(2 * np.pi * (doy - 110) / 365)
            + t_diur * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.8, len(index)))
    decl = np.radians(23.45 * np.sin(np.radians(360 * (284 + doy) / 365)))
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(np.radians(15 * (h - 12)))
    ghi = 1050 * np.clip(cosz, 0, None) ** 1.15 * rng.uniform(0.85, 1.0, len(index))
    return pd.DataFrame({"temp": temp, "rh": 50.0, "ghi": ghi, "dni": 0.8 * ghi, "dhi": 0.2 * ghi}, index=index)


def synthetic_ercot(n, rng):
    rt = rng.lognormal(np.log(30), 0.5, n)
    spikes = rng.random(n) < 0.01
    rt[spikes] *= rng.uniform(5, 30, spikes.sum())
    return rt, rng.lognormal(np.log(4), 0.8, n)
