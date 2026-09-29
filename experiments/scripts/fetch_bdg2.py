#!/usr/bin/env python3
"""Download the BDG2 files used by run_ml.py into data/raw/bdg2/ and check their SHA-256 digests.

Source: the Building Data Genome Project 2 repository (github.com/buds-lab/building-data-genome-project-2), whose
v1.0 release is archived on Zenodo (doi:10.5281/zenodo.3887306, CC BY 4.0). The digests are those of the files
used for the paper.
"""
import hashlib, pathlib, sys, urllib.request

BASE = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/"
FILES = {
    "meters/cleaned/electricity_cleaned.csv": "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3",
    "meters/cleaned/chilledwater_cleaned.csv": "8211aaf210379af50cbf7af87579d12414c3500d3e3f4dea725761c93a172bcd",
    "weather/weather.csv": "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd",
    "metadata/metadata.csv": "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02",
}


def main(out="data/raw/bdg2"):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for rel, digest in FILES.items():
        f = out / pathlib.Path(rel).name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            print("downloading", rel, flush=True)
            urllib.request.urlretrieve(BASE + rel, f)
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != digest:
            sys.exit(f"{f}: SHA-256 {got} does not match {digest}")
        print("ok", f)


if __name__ == "__main__":
    main(*sys.argv[1:])
