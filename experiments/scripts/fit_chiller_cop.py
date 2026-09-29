#!/usr/bin/env python3
"""Calibrate the chiller COP curve (Facility.cop_ref, Facility.cop_slope) to manufacturer data.

Source: Carrier, "AquaForce 30XA Air-Cooled Chillers" product data (30XA-7PD), SI ratings table
("30XA flooded cooler packaged air-cooled chiller ratings table - SI"): cooling capacity and total
input power of 19 unit sizes (080-350) at condenser entering-air temperatures of 30, 35, 40 and 45 C,
for leaving chilled-water temperatures (LCWT) of 5, 6, 7, 8 and 10 C.
The extracted table is data/chiller_carrier_30xa_si.csv; rebuild it with --pdf <30xa-7pd.pdf>.

For each unit size, COP = capacity / input is fitted linearly in the air temperature; the reported
curve is the median intercept at 29 C and the median slope across sizes, at the warmest LCWT in the
table (10 C, the closest to data-center chilled-water practice).

    python scripts/fit_chiller_cop.py [--lcwt 10] [--pdf 30xa-7pd.pdf]
"""
import argparse
import pathlib
import re
import subprocess

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
CSV = HERE / "data" / "chiller_carrier_30xa_si.csv"
T_AIR = (30, 35, 40, 45)
T_REF = 29.0


def parse_pdf(pdf):
    """Extract the SI ratings table from the product-data PDF (pdftotext -layout)."""
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    start = text.index("CHILLER RATINGS TABLE — SI")
    cont = text.index("CHILLER RATINGS TABLE — SI (cont)", start)
    end = text.index("LEGEND", cont)                      # the SI table ends with the second legend
    rows, block = [], None
    for line in text[start:end].splitlines():
        m = re.match(r"^\s*(?:(\d{1,2})\s+)?(\d{3})\s+((?:[\d.]+\s+){11}[\d.]+)\s*$", line)
        if not m:
            continue
        label, size, rest = m.group(1), m.group(2), [float(x) for x in m.group(3).split()]
        if size == "080" or block is None:
            block = {"lcwt": None, "rows": []}
            rows.append(block)
        if label:
            block["lcwt"] = int(label)
        block["rows"].append((size, rest))
    out = []
    for b in rows:
        for size, v in b["rows"]:
            for i, t in enumerate(T_AIR):
                out.append(dict(lcwt_c=b["lcwt"], unit_size=size, t_air_c=t, cap_kw=v[3 * i], input_kw=v[3 * i + 1]))
    return pd.DataFrame(out)


def fit(df, lcwt):
    d = df[df.lcwt_c == lcwt]
    ref, slope = [], []
    for _, g in d.groupby("unit_size"):
        k, a = np.polyfit(g.t_air_c.to_numpy(float), (g.cap_kw / g.input_kw).to_numpy(float), 1)
        ref.append(a + k * T_REF)
        slope.append(-k)
    return dict(lcwt_c=lcwt, n_sizes=len(ref), cop_ref=float(np.median(ref)), cop_slope=float(np.median(slope)),
                cop_ref_range=(float(min(ref)), float(max(ref))), slope_range=(float(min(slope)), float(max(slope))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcwt", type=int, default=10)
    ap.add_argument("--pdf", help="rebuild the CSV from the Carrier 30XA product-data PDF")
    a = ap.parse_args()
    if a.pdf:
        df = parse_pdf(a.pdf)
        df.to_csv(CSV, index=False)
        print(f"wrote {CSV} ({len(df)} rows)")
    df = pd.read_csv(CSV)
    for l in sorted(df.lcwt_c.unique()):
        r = fit(df, l)
        mark = "  <- used" if l == a.lcwt else ""
        print(f"LCWT {l:>2} C: COP({T_REF:.0f} C) = {r['cop_ref']:.3f} [{r['cop_ref_range'][0]:.2f}-{r['cop_ref_range'][1]:.2f}], "
              f"slope = {r['cop_slope']:.4f}/K [{r['slope_range'][0]:.3f}-{r['slope_range'][1]:.3f}], n = {r['n_sizes']}{mark}")


if __name__ == "__main__":
    main()
