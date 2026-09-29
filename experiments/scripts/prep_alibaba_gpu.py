#!/usr/bin/env python3
"""Arrival shape of deferrable (training and batch) GPU work from the Alibaba PAI GPU trace
-> data/alibaba_jobs.csv.

Source
    Alibaba cluster-trace-gpu-v2020 (github.com/alibaba/clusterdata, cluster-trace-gpu-v2020/README.md;
    Weng et al., "MLaaS in the Wild", NSDI 2022): ~6,500 GPUs on ~1,800 machines, July-August 2020.
    License: the repository has no license file; its README permits use for research or study, and the
    trace README asks users to cite Weng et al. (NSDI 2022).
    Tables used: pai_job_table (job submission) and pai_task_table (instances, requested GPUs,
    task start and end). The primary download host (aliopentrace.oss-cn-beijing.aliyuncs.com) is
    unreachable from some networks, so the script falls back to the alternative data repository linked
    from the trace README
    (github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data), where archives are split in parts.
    Every archive and extracted CSV is checked against the SHA-256 sums published in the README.
    Raw files go to data/raw/alibaba/ (git-ignored).

Output (data/alibaba_jobs.csv, one row per hour of the analysis window)
    time_gst    hour start on the Gulf clock (UTC+4); dates are the trace's desensitized dates
                (1970), only day of week and hour of day are meaningful
    jobs        GPU jobs submitted in that hour
    gpu_hours   GPU-hours of the jobs submitted in that hour
    Use with:  run_all.py --deferrable-trace data/alibaba_jobs.csv --deferrable-col gpu_hours

GPU-hours
    Per task: inst_num x plan_gpu/100 x (end_time - start_time)/3600, summed over the tasks of a job
    and attributed to the job's submission hour (job start_time = submission time). Tasks without an
    end time (status Running or Waiting at the end of the trace) have no duration and are left out;
    jobs with zero GPU-hours (CPU-only) are dropped. All GPU jobs are kept regardless of framework
    or final status (Failed jobs also consumed GPU time).

Clock time (the trace uses relative timestamps)
    The README states that start_time and end_time are seconds shifted by a constant, and that when
    read as Unix time in UTC+8 (Asia/Shanghai) they have the original time of day and day of week,
    with fake dates. So: t_beijing = to_datetime(start_time, unit="s", utc=True) in Asia/Shanghai.
    --mapping local (default): the Beijing wall-clock day and hour are used as the Gulf day and hour
        (the daily submission cycle is replayed at the same hours of the Gulf day; ASSUMPTION, same
        convention as prep_azure_llm.py).
    --mapping utc: absolute time, Gulf = Beijing - 4 h.
    Window: the trace ramps up over its first week (11 to 962 jobs a day until the fake date
    1970-01-14). By default the script keeps the complete Monday-Sunday weeks that start after the
    first day with at least 25% of the median daily job count and end on or before the last full day
    (1970-01-19 to 1970-03-15, 8 weeks), so every day of the week is equally represented.

Usage (from experiments/):
    python scripts/prep_alibaba_gpu.py
    python scripts/prep_alibaba_gpu.py --mapping utc --out data/alibaba_jobs_utc.csv
"""
import argparse
import hashlib
import pathlib
import tarfile

import pandas as pd
import requests

PRIMARY = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/"
ALTERNATIVE = "https://raw.githubusercontent.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data/master/"
HEADER_URL = "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/"
# SHA-256 sums published in cluster-trace-gpu-v2020/README.md
SHA = {"pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
       "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
       "pai_job_table.header": "3ac33aefab9a4d81338794fa145fe280594a379444961a9c639f00181c508567",
       "pai_task_table.header": "978bbaabfc8695874c605c01c144b2977f611ceca73aeb72189988cdfbfb0a9c",
       "pai_job_table.csv": "379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c",
       "pai_task_table.csv": "6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de"}
TARGET_TZ = "Asia/Dubai"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path):
    if sha256(path) != SHA[path.name]:
        raise ValueError(f"checksum mismatch: {path}")
    return path


def fetch(url, path):
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        return False
    path.write_bytes(r.content)
    return True


def get_archive(name, raw):
    """Download name (e.g. pai_job_table.tar.gz) from the primary host, else from the alternative
    repository (single file or parts .partaa, .partab, ...)."""
    path = raw / name
    if path.exists() and sha256(path) == SHA[name]:
        return path
    try:
        if fetch(PRIMARY + name, path):
            return check(path)
    except requests.RequestException as e:
        print(f"primary host failed for {name}: {e}")
    if fetch(ALTERNATIVE + name, path):
        return check(path)
    parts, suffix = [], "aa"
    while fetch(ALTERNATIVE + f"{name}.part{suffix}", raw / f"{name}.part{suffix}"):
        parts.append(raw / f"{name}.part{suffix}")
        suffix = suffix[0] + chr(ord(suffix[1]) + 1)
    if not parts:
        raise RuntimeError(f"could not download {name}; save it by hand from {PRIMARY} or {ALTERNATIVE} into {raw}")
    with open(path, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
            p.unlink()
    return check(path)


def load_table(table, raw):
    csv = raw / f"{table}.csv"
    if not (csv.exists() and sha256(csv) == SHA[csv.name]):
        with tarfile.open(get_archive(f"{table}.tar.gz", raw)) as tf:
            tf.extractall(raw)
        check(csv)
    header = raw / f"{table}.header"
    if not header.exists():
        fetch(HEADER_URL + header.name, header) or fetch(ALTERNATIVE + header.name, header)
    names = check(header).read_text().strip().split(",")
    return pd.read_csv(csv, header=None, names=names)


def job_gpu_hours(job, task):
    task = task.dropna(subset=["start_time", "end_time"])
    gpu_h = task["inst_num"] * task["plan_gpu"].fillna(0) / 100 * (task["end_time"] - task["start_time"]) / 3600
    g = gpu_h.groupby(task["job_name"]).sum()
    j = job.set_index("job_name").join(g.rename("gpu_hours"))
    return j[j["gpu_hours"] > 0]


def gulf_clock(start_time, mapping):
    beijing = pd.to_datetime(start_time, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    if mapping == "utc":
        return beijing.dt.tz_convert(TARGET_TZ).dt.tz_localize(None)
    return beijing.dt.tz_localize(None)


def full_weeks(daily_jobs, min_frac=0.25):
    """First Monday after ramp-up to last Sunday on or before the last full day."""
    ok = daily_jobs[daily_jobs >= min_frac * daily_jobs.median()].index
    start, end = ok.min(), daily_jobs.index.max()
    start = start + pd.Timedelta(days=(7 - start.dayofweek) % 7)
    end = end - pd.Timedelta(days=(end.dayofweek + 1) % 7)
    return start, end + pd.Timedelta(hours=23)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mapping", choices=("local", "utc"), default="local")
    ap.add_argument("--raw-dir", default="data/raw/alibaba")
    ap.add_argument("--out", default="data/alibaba_jobs.csv")
    a = ap.parse_args()

    raw = pathlib.Path(a.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    job, task = load_table("pai_job_table", raw), load_table("pai_task_table", raw)
    j = job_gpu_hours(job, task)
    j["t"] = gulf_clock(j["start_time"], a.mapping)
    # ramp-up is judged on the Beijing clock, where the trace's days begin
    beijing_day = gulf_clock(job["start_time"], "local").dt.floor("D")
    start, end = full_weeks(beijing_day.value_counts().sort_index())
    if a.mapping == "utc":
        start, end = start - pd.Timedelta(hours=4), end - pd.Timedelta(hours=4)
    w = j[(j["t"] >= start) & (j["t"] <= end + pd.Timedelta(minutes=59, seconds=59))]
    hourly = w.set_index("t")["gpu_hours"].resample("h").agg(["size", "sum"])
    hourly = hourly.reindex(pd.date_range(start, end, freq="h"), fill_value=0)
    hourly.columns = ["jobs", "gpu_hours"]
    hourly.index.name = "time_gst"
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.round(4).to_csv(out)

    print(f"{len(job):,} jobs, {len(j):,} with GPU-hours > 0; window {start} to {end} "
          f"({len(hourly) // 24} days, {len(w):,} jobs, {w['gpu_hours'].sum():,.0f} GPU-h)")
    shape = hourly["gpu_hours"].groupby(hourly.index.hour).mean()
    print(f"mapping={a.mapping}; mean GPU-hour shape by Gulf hour (mean 1):",
          (shape / shape.mean()).round(2).tolist())
    print(f"wrote {out} ({len(hourly)} rows)")


if __name__ == "__main__":
    main()
