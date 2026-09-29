#!/usr/bin/env python3
"""Download the ESIF data-center files used by run_ml.py into data/raw/nlr_esif/ and check their SHA-256 digests.

Source: NLR (formerly NREL) HPC Facility Power Usage Effectiveness (PUE) Data, Energy Systems Integration Facility,
Golden, Colorado (doi:10.7799/3015212; NREL data catalog). Credit: U.S. Department of Energy, National Laboratory of
the Rockies (formerly NREL). The digests are those of the files used for the paper.
"""
import hashlib, pathlib, sys, urllib.request

BASE = "https://data.nlr.gov/system/files/300/"
FILES = {
    "1757103411-esif.influx.buildingData.PUE.combined.parquet":
        "19cd12405dde9144b1a360e8c8418666c399a3d0d15a7f846880d71ab22f9dd4",
    "1757105566-esif.influx.buildingData.outside.combined_2.parquet":
        "97b424993fa77a15117fb2c4659a2c327fc83280f943fab47d9036260289a6a0",
}


def main(out="data/raw/nlr_esif"):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        f = out / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(BASE + name, f)
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != digest:
            sys.exit(f"{f}: SHA-256 {got} does not match {digest}")
        print("ok", f)


if __name__ == "__main__":
    main(*sys.argv[1:])
