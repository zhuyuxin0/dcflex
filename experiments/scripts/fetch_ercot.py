"""Fetch 12 months of ERCOT hub real-time prices and day-ahead ECRS prices for the S4 benchmark.

Writes data/ercot_hb_west.csv with columns time, rt_price, as_price (read by dcflex.data.load_ercot).

Run from experiments/:
    python scripts/fetch_ercot.py [--start 2025-09-01] [--end 2026-08-31] [--hub HB_WEST] [--as ECRS]
                                  [--out data/ercot_hb_west.csv] [--raw-dir data/raw/ercot]

Sources (ERCOT Market Information System, public, no login or API key; accessed 2026-09-27)
  * NP6-785-ER "Historical RTM Load Zone and Hub Prices", reportTypeId 13061.
    One zip per calendar year, RTMLZHBSPP_<year>.zip, holding an .xlsx with one sheet per month.
    The current-year file is year to date and is re-posted weekly.
    https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER
  * NP4-181-ER "Historical DAM Clearing Prices for Capacity", reportTypeId 13091.
    One zip per calendar year, DAMASMCPC_<year>.zip, holding a .csv (REGDN, REGUP, RRS, NSPIN, ECRS).
    Year to date for the current year, re-posted weekly.
    https://www.ercot.com/mp/data-products/data-product-details?id=NP4-181-ER
  Document list: https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=<id>
  Download:      https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=<DocID>
  These are the public MIS endpoints the gridstatus library also uses. gridstatus 0.36.0 reads 13061
  (Ercot.get_rtm_spp) but has no reader for 13091; its get_as_prices and get_mcpc_dam read the daily
  report NP4-188-CD (reportTypeId 12329), which the MIS keeps for about one month only. So this script
  reads both annual files directly, with requests, and caches them.

Output columns (one row per hour)
  time      hour start in UTC, ISO 8601 with +00:00.
  rt_price  real-time settlement point price (SPP) at the hub, USD/MWh. ERCOT settles real time in
            15-minute intervals; the hourly value is the simple mean of the four 15-minute SPPs
            (equal-length intervals, so this equals the time-weighted mean).
  as_price  day-ahead market (DAM) market clearing price for capacity (MCPC) of ECRS,
            USD/MW per hour (one price per hour).

Ancillary-service product and market
  ECRS (ERCOT Contingency Reserve Service) is used because it is the reserve product that large
  flexible loads can provide and that fits a curtailment-type data-center offer. Its first DAM MCPC is
  for operating day 10 June 2023 (checked in DAMASMCPC_2023). If ECRS is blank for any hour in the
  window, the whole series switches to Non-Spin (NSPIN) and the script says so.
  We use the DAM MCPC for the whole window. Since RTC+B went live (first operating day 5 December 2025)
  ERCOT also publishes real-time AS clearing prices (NP6-331-CD, reportTypeId 24898, per 15-minute
  interval; NP6-332-CD, reportTypeId 24891, per SCED interval), but those do not exist before
  5 December 2025 and the MIS keeps them for about a week only. The DAM MCPC exists on the same
  basis before and after the change and is published in the annual historical file.

Time zone and DST
  ERCOT files are in Central Prevailing Time (US/Central) with hour ending 1-24, a 15-minute
  interval number (1-4, prices only) and a Repeated Hour Flag. Local interval start
  = delivery date + (hour ending - 1) h + (interval - 1) x 15 min. On the spring-forward day ERCOT
  omits hour ending 03 (23 hours). On the fall-back day hour ending 02 appears twice: flag N is the
  first pass (CDT, UTC-5), flag Y the repeated hour (CST, UTC-6). We localize with ambiguous =
  (flag == "N") and convert to UTC, so each UTC hour appears exactly once. The window runs from
  --start 00:00 to --end 23:00 Central time (hour starts), converted to UTC. If that gives fewer than
  8,760 hours, the end is extended by the shortfall (never the start) and the script says so. The
  script checks for duplicate intervals and hours, hours with fewer than four intervals, and gaps.
  Any gap is filled by linear interpolation and counted; more than --max-missing gaps is an error.

Caveats
  * All prices are nominal USD, as published by ERCOT; no currency or inflation adjustment.
  * The default window straddles RTC+B (real-time co-optimization of energy and AS plus battery
    changes, first operating day 5 December 2025, ERCOT news release of that date). Real-time price
    formation and AS procurement changed on that day; the script prints means before and after.
  * ECRS started on 10 June 2023, so ECRS series cannot start earlier.
  * The current-year files are year to date: a run shortly after a month ends may not yet cover it
    (ERCOT re-posts weekly). The script then stops with a coverage error; rerun after the next post.
  * Reading the .xlsx needs openpyxl (in requirements.txt).
"""
import argparse
import datetime
import io
import pathlib
import sys
import zipfile

import pandas as pd
import requests

MIS_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rtid}"
MIS_DOC = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc_id}"
RTM_RTID, RTM_NAME = 13061, "RTMLZHBSPP_{year}"   # NP6-785-ER
DAM_AS_RTID, DAM_AS_NAME = 13091, "DAMASMCPC_{year}"  # NP4-181-ER
TZ = "US/Central"
AS_FALLBACK = "NSPIN"
RTCB_DAY = "2025-12-05"
MIN_HOURS = 8760


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2025-09-01", help="first delivery day (Central time)")
    ap.add_argument("--end", default="2026-08-31", help="last delivery day (Central time)")
    ap.add_argument("--hub", default="HB_WEST")
    ap.add_argument("--as", dest="as_product", default="ECRS", choices=["ECRS", "NSPIN", "RRS", "REGUP", "REGDN"])
    ap.add_argument("--out", default="data/ercot_hb_west.csv")
    ap.add_argument("--raw-dir", default="data/raw/ercot")
    ap.add_argument("--max-missing", type=int, default=24, help="max hours to fill per series before failing")
    return ap.parse_args()


# ---------------- download and cache ----------------
def list_docs(rtid):
    r = requests.get(MIS_LIST.format(rtid=rtid), timeout=60)
    r.raise_for_status()
    return [d["Document"] for d in r.json()["ListDocsByRptTypeRes"]["DocumentList"]]


def fetch_annual_zip(rtid, friendly, raw_dir):
    """Newest posted version of one annual file, cached under ERCOT's constructed name (which
    contains the posting time). Downloads only if ERCOT has posted a version we do not have."""
    cached = sorted(raw_dir.glob(f"*.{friendly}.zip"))
    try:
        docs = [d for d in list_docs(rtid) if d["FriendlyName"] == friendly]
    except requests.RequestException as e:
        if cached:
            print(f"  warning: could not list reportTypeId {rtid} ({e}); using cached {cached[-1].name}")
            return cached[-1]
        raise
    if not docs:
        sys.exit(f"ERROR: ERCOT MIS lists no {friendly} for reportTypeId {rtid}")
    doc = max(docs, key=lambda d: pd.Timestamp(d["PublishDate"]))
    path = raw_dir / doc["ConstructedName"]
    if path.exists():
        print(f"  cached   {path.name}")
        return path
    print(f"  download {path.name} (posted {doc['PublishDate']})")
    r = requests.get(MIS_DOC.format(doc_id=doc["DocID"]), timeout=600)
    r.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        sys.exit(f"ERROR: download of {path.name} is not a zip file")
    tmp.rename(path)
    return path


# ---------------- parse ----------------
def read_rtm_hub(zip_path, hub):
    """15-minute SPPs of one settlement point from the annual xlsx; cached as a small CSV."""
    cache = zip_path.with_name(zip_path.stem + f".{hub}.csv")
    if cache.exists():
        return pd.read_csv(cache, dtype={"flag": str})
    import openpyxl
    with zipfile.ZipFile(zip_path) as z:
        wb = openpyxl.load_workbook(io.BytesIO(z.read(z.namelist()[0])), read_only=True)
    rows, names = [], set()
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        head = [str(c).strip() for c in next(it)]
        i = {k: head.index(k) for k in ("Delivery Date", "Delivery Hour", "Delivery Interval",
                                        "Repeated Hour Flag", "Settlement Point Name", "Settlement Point Price")}
        for r in it:
            name = r[i["Settlement Point Name"]]
            names.add(name)
            if name != hub:
                continue
            d = r[i["Delivery Date"]]
            d = d.strftime("%m/%d/%Y") if isinstance(d, datetime.datetime) else str(d).strip()
            rows.append((d, int(r[i["Delivery Hour"]]), int(r[i["Delivery Interval"]]),
                         str(r[i["Repeated Hour Flag"]]).strip(), float(r[i["Settlement Point Price"]])))
    wb.close()
    if not rows:
        sys.exit(f"ERROR: {hub} not in {zip_path.name}; available: {sorted(n for n in names if n)}")
    df = pd.DataFrame(rows, columns=["date", "hour_ending", "interval", "flag", "spp"])
    df.to_csv(cache, index=False)
    return df


def read_dam_as(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype={"Repeated Hour Flag": str})
    df.columns = [c.strip() for c in df.columns]
    df["hour_ending"] = df["Hour Ending"].str.split(":").str[0].astype(int)
    return df.rename(columns={"Delivery Date": "date", "Repeated Hour Flag": "flag"})


def to_utc(date, hour_ending, flag, minutes=0):
    """ERCOT delivery date + hour ending (1-24) [+ minutes] in Central Prevailing Time -> UTC start."""
    local = (pd.to_datetime(date, format="%m/%d/%Y") + pd.to_timedelta(hour_ending - 1, unit="h")
             + pd.to_timedelta(minutes, unit="min"))
    first_pass = (flag.str.strip().str.upper() == "N").to_numpy()  # N = CDT pass of the repeated hour
    return pd.DatetimeIndex(local).tz_localize(TZ, ambiguous=first_pass, nonexistent="raise").tz_convert("UTC")


def hourly_rt(spp):
    """Hourly mean of the four 15-minute SPPs, indexed by UTC hour start."""
    t = to_utc(spp["date"], spp["hour_ending"], spp["flag"], 15 * (spp["interval"] - 1))
    s = pd.Series(spp["spp"].to_numpy(float), index=t).sort_index()
    dup = int(s.index.duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicated 15-minute intervals after UTC conversion")
    g = s.groupby(s.index.floor("h"))
    return g.mean(), g.size()


def hourly_as(asp):
    t = to_utc(asp["date"], asp["hour_ending"], asp["flag"])
    df = asp.set_index(t).sort_index()
    dup = int(df.index.duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicated AS hours after UTC conversion")
    return df


# ---------------- window, checks, summary ----------------
def utc_window(start, end):
    t0 = pd.Timestamp(start).tz_localize(TZ).tz_convert("UTC")
    t1 = (pd.Timestamp(end) + pd.Timedelta(hours=23)).tz_localize(TZ).tz_convert("UTC")
    idx = pd.date_range(t0, t1, freq="h")
    extra = max(0, MIN_HOURS - len(idx))
    if extra:
        print(f"  window is {len(idx)} h; extending the end by {extra} h to reach {MIN_HOURS}")
        idx = pd.date_range(t0, t1 + pd.Timedelta(hours=extra), freq="h")
    return idx


def fill_gaps(s, name, max_missing):
    miss = s.index[s.isna()]
    print(f"  {name}: {len(miss)} missing hours" + (f" (first {list(miss[:5].strftime('%Y-%m-%d %H:%M'))})" if len(miss) else ""))
    if len(miss) > max_missing or pd.isna(s.iloc[0]) or pd.isna(s.iloc[-1]):
        sys.exit(f"ERROR: {name} does not cover the window ({len(miss)} hours missing, "
                 f"first {s.first_valid_index()}, last {s.last_valid_index()}). The ERCOT year-to-date "
                 f"file may not yet reach the end date; rerun after ERCOT's next weekly posting.")
    return s.interpolate(limit_direction="both"), len(miss)


def summarize(out, sp_count, as_col):
    loc = out.tz_convert(TZ)
    print("\nCoverage")
    print(f"  first hour (UTC): {out.index[0]}   last hour (UTC): {out.index[-1]}   hours: {len(out)}")
    print(f"  duplicated hours: {int(out.index.duplicated().sum())}   "
          f"gaps in hourly index: {int((out.index.to_series().diff().dropna() != pd.Timedelta('1h')).sum())}")
    bad = sp_count.reindex(out.index)
    print(f"  hours with != 4 fifteen-minute SPPs: {int((bad != 4).sum())}")
    print(f"\nMonthly summary (Central-time months; rt_price USD/MWh, as_price = DAM {as_col} MCPC USD/MW-h)")
    m = loc.groupby(loc.index.strftime("%Y-%m")).agg(
        hours=("rt_price", "size"), rt_mean=("rt_price", "mean"), rt_max=("rt_price", "max"),
        as_mean=("as_price", "mean"), as_max=("as_price", "max"))
    m.index.name = "month"
    print(m.to_string(float_format=lambda x: f"{x:9.2f}"))
    cut = pd.Timestamp(RTCB_DAY).tz_localize(TZ)
    pre, post = out[out.index < cut], out[out.index >= cut]
    print(f"\nBefore vs after RTC+B ({RTCB_DAY} 00:00 Central)")
    print(f"  before: {len(pre):5d} h  mean as_price {pre.as_price.mean():7.2f}  mean rt_price {pre.rt_price.mean():7.2f}")
    print(f"  after:  {len(post):5d} h  mean as_price {post.as_price.mean():7.2f}  mean rt_price {post.rt_price.mean():7.2f}")
    print(f"\nWhole window: mean rt_price {out.rt_price.mean():.2f} USD/MWh, mean as_price {out.as_price.mean():.2f} USD/MW-h")


def main():
    a = parse_args()
    raw_dir = pathlib.Path(a.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"ERCOT {a.hub} real-time SPP and DAM {a.as_product} MCPC, {a.start} to {a.end} (Central)")
    idx = utc_window(a.start, a.end)
    years = range(idx[0].tz_convert(TZ).year, idx[-1].tz_convert(TZ).year + 1)
    spp, asp = [], []
    for y in years:
        spp.append(read_rtm_hub(fetch_annual_zip(RTM_RTID, RTM_NAME.format(year=y), raw_dir), a.hub))
        asp.append(read_dam_as(fetch_annual_zip(DAM_AS_RTID, DAM_AS_NAME.format(year=y), raw_dir)))
    rt, sp_count = hourly_rt(pd.concat(spp, ignore_index=True))
    as_all = hourly_as(pd.concat(asp, ignore_index=True))

    in_win = as_all.loc[idx[0]:idx[-1]]
    as_col = a.as_product
    if in_win[as_col].isna().any():
        print(f"  NOTE: {as_col} is blank for {int(in_win[as_col].isna().sum())} hours in the window; "
              f"using {AS_FALLBACK} for the whole series instead")
        as_col = AS_FALLBACK

    print("\nChecks")
    print(f"  hours in window: {len(idx)}")
    rt_s, _ = fill_gaps(rt.reindex(idx), "rt_price", a.max_missing)
    as_s, _ = fill_gaps(pd.to_numeric(as_all[as_col], errors="coerce").reindex(idx), "as_price", a.max_missing)
    out = pd.DataFrame({"rt_price": rt_s.round(4), "as_price": as_s.round(2)}, index=idx)
    out.index.name = "time"
    if len(out) < MIN_HOURS:
        sys.exit(f"ERROR: only {len(out)} hours, need {MIN_HOURS}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out)
    summarize(out, sp_count, as_col)
    print(f"\nWrote {a.out} ({pathlib.Path(a.out).stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
