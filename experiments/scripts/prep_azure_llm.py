#!/usr/bin/env python3
"""Inference load shape from the Azure LLM inference traces -> data/azure_llm.csv.

Source
    Azure Public Dataset, "Azure LLM inference trace 2024" (github.com/Azure/AzurePublicDataset,
    AzureLLMInferenceDataset2024.md; Stojkovic et al., DynamoLLM, HPCA 2025). License: CC BY 4.0
    (stated on the dataset page). Two services, one week each, per-request timestamps in UTC:
        code          2024-05-10 00:00 to 2024-05-16 23:59 UTC   (~16.8 M requests)
        conversation  2024-05-12 00:00 to 2024-05-18 23:59 UTC   (~27.3 M requests)
    This is the longest Azure LLM trace; the 2023 trace (Splitwise) covers only about one hour.
    Raw files (~1.8 GB) are downloaded to data/raw/azure/ (git-ignored) and cached.

Output (data/azure_llm.csv, 168 rows = one representative week, hourly)
    time_gst          hour start, Gulf Standard Time (UTC+4, no daylight saving), on a nominal week
                      Mon 2024-05-13 00:00 .. Sun 2024-05-19 23:00; only day of week and hour matter
    requests          requests per hour, both services
    context_tokens, generated_tokens, tokens (= context + generated), both services
    requests_code, requests_conv
    Use with:  run_all.py --inference-trace data/azure_llm.csv --inference-col requests
    (dcflex.data.profile_from_csv sums the value column per hour; without --inference-col every row
    counts as one request and the shape would be flat.)

Time-zone mapping, trace time -> Gulf time (UTC+4)
    --mapping local (default): the trace's daily and weekly cycle is replayed at the same hours of
        the Gulf day. Timestamps are converted from UTC to US Eastern time (--source-tz, EDT = UTC-4
        in May 2024) and that wall-clock day and hour is used as the Gulf day and hour. ASSUMPTION: an
        Abu Dhabi campus mainly serves customers whose working day follows Gulf time, with the same
        daily shape as the trace. US Eastern is inferred from the trace itself: the code service is
        lowest at 08-09 UTC (04-05 EDT) and highest at 17-20 UTC (13-16 EDT). Where the customers of
        the services are located is not published [UNCERTAIN].
    --mapping utc: absolute-time mapping. Gulf time = UTC + 4 h, i.e. every request keeps the instant
        at which it arrived in the trace. This moves the code service's peak to 21-24 GST.
    Weekends: the trace weeks are Saturday-Sunday weekends (US); the UAE weekend is also
    Saturday-Sunday (since 2022), so day of week is kept as is under both mappings.

Construction
    For each service: hourly totals over its seven full UTC days (168 hours), shifted by the mapping,
    keyed by (day of week, hour). Each key occurs exactly once per service. The two services are
    added key by key. No scaling is applied here; run_all.py normalizes the shape to mean 1.

Usage (from experiments/):
    python scripts/prep_azure_llm.py                     # default mapping, writes data/azure_llm.csv
    python scripts/prep_azure_llm.py --mapping utc --out data/azure_llm_utc.csv
"""
import argparse
import pathlib

import pandas as pd
import requests

BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
FILES = {"code": "AzureLLMInferenceTrace_code_1week.csv", "conv": "AzureLLMInferenceTrace_conv_1week.csv"}
NOMINAL_MONDAY = pd.Timestamp("2024-05-13")
TARGET_TZ = "Asia/Dubai"


def download(name, raw_dir):
    path = raw_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(BASE + name, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    tmp.rename(path)
    return path


def hourly_utc(path):
    """Hourly request and token totals (UTC hour start), read in chunks; timestamps are sliced to
    the hour rather than parsed, which is much faster for ~20 M rows."""
    parts = []
    for ch in pd.read_csv(path, dtype={"TIMESTAMP": str, "ContextTokens": "int64", "GeneratedTokens": "int64"},
                          chunksize=4_000_000):
        ch["hour"] = ch["TIMESTAMP"].str.slice(0, 13)
        parts.append(ch.groupby("hour").agg(requests=("ContextTokens", "size"),
                                            context_tokens=("ContextTokens", "sum"),
                                            generated_tokens=("GeneratedTokens", "sum")))
    h = pd.concat(parts).groupby(level=0).sum()
    h.index = pd.to_datetime(h.index + ":00:00", utc=True)
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    if len(full) != 168 or len(h) != 168:
        raise ValueError(f"{path.name}: expected 168 consecutive hours, got {len(h)} over {len(full)}")
    return h


def to_gulf_clock(index, mapping, source_tz):
    """Wall-clock time on the Gulf clock for each UTC hour (see module docstring)."""
    if mapping == "utc":
        return index.tz_convert(TARGET_TZ).tz_localize(None)
    return index.tz_convert(source_tz).tz_localize(None)


def week_table(h, mapping, source_tz):
    clock = to_gulf_clock(h.index, mapping, source_tz)
    t = h.copy()
    t.index = pd.MultiIndex.from_arrays([clock.dayofweek, clock.hour], names=["dow", "hour"])
    if t.index.duplicated().any() or len(t) != 168:
        raise ValueError("mapping does not give one value per (day of week, hour)")
    return t.sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mapping", choices=("local", "utc"), default="local")
    ap.add_argument("--source-tz", default="America/New_York", help="users' time zone for --mapping local")
    ap.add_argument("--raw-dir", default="data/raw/azure")
    ap.add_argument("--out", default="data/azure_llm.csv")
    a = ap.parse_args()

    raw = pathlib.Path(a.raw_dir)
    tables = {}
    for svc, name in FILES.items():
        h = hourly_utc(download(name, raw))
        print(f"{svc}: {int(h['requests'].sum()):,} requests, {h.index[0]} to {h.index[-1]} (UTC)")
        tables[svc] = week_table(h, a.mapping, a.source_tz)

    wk = tables["code"] + tables["conv"]
    wk["tokens"] = wk["context_tokens"] + wk["generated_tokens"]
    wk["requests_code"] = tables["code"]["requests"]
    wk["requests_conv"] = tables["conv"]["requests"]
    days = wk.index.get_level_values("dow")
    hours = wk.index.get_level_values("hour")
    wk.index = NOMINAL_MONDAY + pd.to_timedelta(days * 24 + hours, unit="h")
    wk.index.name = "time_gst"
    cols = ["requests", "context_tokens", "generated_tokens", "tokens", "requests_code", "requests_conv"]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wk[cols].to_csv(out)

    shape = wk["requests"].groupby(wk.index.hour).mean()
    print(f"mapping={a.mapping}" + (f" (source tz {a.source_tz})" if a.mapping == "local" else ""))
    print("mean shape by Gulf hour (requests, mean 1):", (shape / shape.mean()).round(2).tolist())
    print(f"wrote {out} ({len(wk)} rows)")


if __name__ == "__main__":
    main()
