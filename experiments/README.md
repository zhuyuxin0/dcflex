# dcflex: experiments for "Digitized Demand Response" (YFEL 2026, Group 4)

Two commands reproduce every number, table and data figure in the paper. They write `results/` and `figures/`
one level up (the default `--root ..`), which is where the manuscript reads them.

```bash
pip install -r requirements.txt -r requirements-ml.txt
python concept_figs.py ../figures                  # conceptual figures (context, framework, protocol, B5)
python run_all.py --out out [trace options below]  # facility model, scenarios, baselines B1-B5
python run_ml.py all                               # learned baselines on real buildings, forecast-driven control
python -m pytest -q tests                          # unit tests
python run_all.py --synthetic --quick --root /tmp/test   # software test only: outputs are marked SYNTHETIC
```

`run_all.py` with the workload traces and the ERCOT benchmark, as used for the paper:
```bash
python run_all.py --out out \
  --inference-trace data/azure_llm.csv --inference-col requests \
  --deferrable-trace data/alibaba_jobs.csv --deferrable-col gpu_hours \
  --inference-trace-alt data/azure_llm_utc.csv --deferrable-trace-alt data/alibaba_jobs_utc.csv \
  --ercot-csv data/ercot_hb_west.csv
```
The `*-alt` traces use the absolute-time (UTC) mapping and feed the trace-mapping sensitivity (`\Rutc...` macros);
the base case replays each trace's working day on Gulf time. The run is deterministic (about 100 s); B5 trains on one CPU thread.

`run_ml.py` needs `out/series.npz` from `run_all.py` and runs four cached stages (`out/ml/`), which can also be run
one at a time: `pretrain` (cross-building pretraining, leave one site out), `realdata` (baselines on BDG2 meters,
new-facility and established-facility regimes; `--max-buildings`, default 40, caps the established regime only),
`mpc` (forecast-driven control and the ESIF workload stress test) and `export` (macros, tables, figures). On a
4-core machine the full run takes several hours, most of it pretraining and the real-building runs. Results are
seeded; being many-threaded, they can differ in the last digit across machines.

## Data preparation
Derived inputs live in `data/`; raw downloads go to `data/raw/` (git-ignored). Each file is rebuilt by a script in
`scripts/`, run from `experiments/`. Sources, licenses and caveats: `../DATA.md`.

| File | Script | Source | Used by |
|---|---|---|---|
| `data/weather/*.json` | none (archived API responses) | Open-Meteo: ERA5 reanalysis and ECMWF IFS day-ahead forecasts | `run_all.py`, `run_ml.py mpc` |
| `data/azure_llm*.csv` | `python scripts/prep_azure_llm.py` | Azure LLM inference traces 2024 (1 week per service, ~1.8 GB raw) | `--inference-trace data/azure_llm.csv --inference-col requests` |
| `data/alibaba_jobs*.csv` | `python scripts/prep_alibaba_gpu.py` | Alibaba cluster-trace-gpu-v2020 (job and task tables, ~0.1 GB compressed) | `--deferrable-trace data/alibaba_jobs.csv --deferrable-col gpu_hours` |
| `data/ercot_hb_west.csv` | `python scripts/fetch_ercot.py` | ERCOT public MIS reports (HB_WEST real-time SPP; DAM ECRS MCPC) | `--ercot-csv data/ercot_hb_west.csv` |
| `data/chiller_*.csv` | none (transcribed ratings) | Carrier 30XA and Daikin EWAD catalogues | `python scripts/fit_chiller_cop.py` (COP calibration) |
| `data/raw/bdg2/` | `python scripts/fetch_bdg2.py` | Building Data Genome Project 2, release 1.0 (CC BY 4.0) | `run_ml.py pretrain`, `realdata` |
| `data/raw/nlr_esif/` | `python scripts/fetch_esif.py` | ESIF HPC data center, National Laboratory of the Rockies (doi:10.7799/3015212) | `run_ml.py mpc` |

- **Always pass the value columns.** The trace files are already aggregated to hourly totals, and `profile_from_csv` counts one unit per row when no value column is given. Without `--inference-col requests`, the inference shape would be flat.
- **Time zones.** Every derived file is on the Gulf clock (UTC+4, no daylight saving), except the ERCOT file, which is in UTC.
  - The default (`--mapping local`) replays each trace's own local working day on Gulf time. Azure traces are read on the US Eastern clock; the Alibaba trace is read on the Beijing clock, which its README says keeps the original time of day and day of week.
  - `*_utc.csv` files use the absolute-time alternative (Gulf time = UTC + 4 h) for sensitivity.
  - This choice moves the inference peak between midday and evening. See the headers of the two prep scripts.
- **Weather.** The archived responses are read first; only a new location or year calls the Open-Meteo API. The free API has a daily request limit per IP; on HTTP 429 the pipeline stops and prints the URL. Retry later, or open the URL in a browser and save the response under the printed file name.
- Without trace files the pipeline falls back to parametric shapes (documented as ASSUMPTION); do not publish results from that mode without saying so.

## Where things are
- `dcflex/config.py`: every parameter with its source.
- `dcflex/model.py`: facility LP, business as usual and the rule-based strategy.
- `dcflex/signal.py`: synthetic 2030 Abu Dhabi net-load signal.
- `dcflex/analysis.py`: scenarios, flexibility envelope, DR contract, sensitivity.
- `dcflex/baselines.py`, `dcflex/protocol.py`: baselines B1-B5, manipulation tests, commit-reveal.
- `dcflex/learned.py`, `learned_batch.py`, `pretrain.py`: the physics-guided baseline B5 and cross-building pretraining.
- `dcflex/realdata.py`, `validate.py`: BDG2 and ESIF loaders, out-of-sample tests on metered buildings.
- `dcflex/mpc.py`, `workload_fc.py`: receding-horizon control, forecast correction, conformal margins, workload forecasts.
- `dcflex/export.py`, `export_ml.py`, `figures.py`, `figures_ml.py`, `style.py`: macros, tables and figures.
