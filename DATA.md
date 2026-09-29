# Data sources, licenses and attribution

## Included in this repository

| Files | Content | Source | License and attribution |
|---|---|---|---|
| `experiments/data/weather/openmeteo_era5_*.json` | Hourly ERA5 reanalysis for Abu Dhabi (2025) and West Texas (Sep 2025–Aug 2026), as returned by the API | Open-Meteo Historical Weather API (Zippenfenig, doi:10.5281/zenodo.7970649); ERA5 (Hersbach et al., doi:10.1002/qj.3803) | CC BY 4.0. Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. |
| `experiments/data/weather/openmeteo_previous_runs_ecmwf_ifs025_2025.json` | ECMWF IFS 0.25° forecasts for Abu Dhabi, 2025, each 24 h before its valid hour | Open-Meteo Previous Runs API | Provided by Open-Meteo under CC BY 4.0; forecast model: ECMWF IFS. |
| `experiments/data/azure_llm.csv`, `azure_llm_utc.csv` | Hourly request counts derived from the Azure LLM inference traces 2024 | Azure Public Dataset (github.com/Azure/AzurePublicDataset); Patel et al., "Splitwise," ISCA 2024 | CC BY 4.0 (dataset license). |
| `experiments/data/chiller_*.csv` | Chiller efficiency ratings transcribed from manufacturer catalogues (Carrier 30XA, Daikin EWAD), used to calibrate the COP model | Manufacturer product catalogues, cited in the paper | Numerical ratings included so that the calibration can be reproduced; the catalogues remain the property of their publishers. |
| `models/pretrain/*.pt` | Cross-building pretrained networks | Trained by this code on BDG2 meters | CC BY 4.0. Derived from the Building Data Genome Project 2 (Miller et al., doi:10.1038/s41597-020-00712-x; release 1.0, doi:10.5281/zenodo.3887306, CC BY 4.0). |
| `results/`, `figures/` | Every number, table and figure in the paper | This code | CC BY 4.0; please cite the paper. |
| `assets/logos.png` | Logos of Khalifa University of Science and Technology and the YFEL Programme | The programme | Used with the organizers' permission; not covered by this repository's licenses. |

## Downloaded by scripts (not redistributed here)

| Script | Data | Source | License and notes |
|---|---|---|---|
| `scripts/prep_alibaba_gpu.py` | Alibaba cluster-trace-gpu-v2020 (job and task tables) | github.com/alibaba/clusterdata; Weng et al., "MLaaS in the Wild," NSDI 2022 | Not redistributed: the repository has no license file; its README permits use for research or study. |
| `scripts/fetch_ercot.py` | ERCOT real-time settlement point prices (NP6-785-ER) and day-ahead ancillary service prices (NP4-181-ER) | ERCOT Market Information System | Public market data; not redistributed. ERCOT keeps only recent reports online. |
| `scripts/fetch_bdg2.py` | Building Data Genome Project 2: cleaned electricity and chilled-water meters, weather, metadata | github.com/buds-lab/building-data-genome-project-2; release 1.0 on Zenodo (doi:10.5281/zenodo.3887306) | CC BY 4.0. The script checks the SHA-256 digests of the files used for the paper. |
| `scripts/fetch_esif.py` | ESIF high-performance computing data center: IT, cooling and other power, and outdoor weather | NREL data catalog, "HPC Facility Power Usage Effectiveness (PUE) Data" (doi:10.7799/3015212) | Credit: U.S. Department of Energy, National Laboratory of the Rockies (formerly NREL). The script checks the SHA-256 digests of the files used for the paper. |
| (automatic, `run_ml.py mpc`) | Chronos-Bolt-small forecasting model (`amazon/chronos-bolt-small`) | Hugging Face; Ansari et al., "Chronos," arXiv:2403.07815 | Apache-2.0; downloaded on first use. |
| (automatic, only for a new location or year) | Open-Meteo weather | Open-Meteo APIs | CC BY 4.0, as above. |
