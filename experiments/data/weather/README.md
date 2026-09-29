# Archived weather inputs (ERA5 via Open-Meteo)

Raw responses of the Open-Meteo Historical Weather API (`models=era5`, `cell_selection=land`), saved under the
cache names that `dcflex.data.fetch_weather` looks for. Downloaded in a browser on 27 Sep 2026, because the
build machine had used up the API's daily quota; the requests are those given by `dcflex.data.weather_request`.

| File | Site | Period | ERA5 grid cell |
|---|---|---|---|
| `openmeteo_era5_24.45_54.38_2025-01-01_2025-12-31_Asia-Dubai.json` | Abu Dhabi | 2025, local time (UTC+4), 8,760 h | 24.50 N, 54.50 E |
| `openmeteo_era5_32.45_-99.73_2025-09-01_2026-09-01_UTC.json` | West Texas (ERCOT benchmark) | Sep 2025–Aug 2026, UTC, 8,784 h | 32.50 N, 99.75 W |
| `openmeteo_previous_runs_ecmwf_ifs025_2025.json` | Abu Dhabi, day-ahead forecasts (ECMWF IFS 0.25°, Open-Meteo Previous Runs API) | 2025, UTC, 8,760 h | 24.50 N, 54.50 E |

Licences: Open-Meteo data are CC BY 4.0. Contains modified Copernicus Climate Change Service information 2026.
Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus
information or data it contains. Cite Hersbach et al. (2020), doi:10.1002/qj.3803, and Open-Meteo, doi:10.5281/zenodo.7970649.

**Day-ahead forecasts.** `openmeteo_previous_runs_ecmwf_ifs025_2025.json` was downloaded on 28 Sep 2026 from
`https://previous-runs-api.open-meteo.com/v1/forecast?latitude=24.45&longitude=54.38&hourly=temperature_2m,temperature_2m_previous_day1,shortwave_radiation,shortwave_radiation_previous_day1&start_date=2025-01-01&end_date=2025-12-31&timezone=GMT&models=ecmwf_ifs025`
(SHA-256 f341ad834b4f74e4dd6c8de2587795abd50afdd39df04d6412ca1f86dfb4e0de). `*_previous_day1` is the value the
model predicted 24 h before each valid hour (a fixed lead, not one coherent issuance); the unsuffixed columns are
the stitched short-lead values. Data: ECMWF IFS via Open-Meteo, CC BY 4.0.
