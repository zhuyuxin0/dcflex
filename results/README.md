# Results

Every number, table and data figure in the paper is written here by the pipeline; nothing is edited by hand.
A rerun reproduces every file, except the machine-dependent timing `RcommitMicros`.

| File | Written by | Content |
|---|---|---|
| `numbers.csv` | both runners | every number printed in the paper, one row per LaTeX macro: `macro`, `value`, `written_by` |
| `macros.tex` | `run_all.py` | the same numbers as LaTeX macros, e.g. `\newcommand{\RshedFourHourMW}{35.7}`, which the manuscript reads |
| `macros-ml.tex` | `run_ml.py` | the numbers of the learned baselines and forecast-driven operation |
| `tab-*.tex` | `run_all.py` (`tab-ml-*`: `run_ml.py`) | table bodies, read by the manuscript |
| `commit-example.txt` | `run_all.py` | the commitment record and its SHA-256 hash (Appendix D) |
| `protocol-code.txt` | `run_all.py` | the source of the commit and verify functions (Appendix D) |

Macro names spell out the quantity and its unit. For example:
- `RshedFourHourMW` is the four-hour firm reduction F(4) in MW;
- `RsystemValueKWyr` is the system value in AED/kW-yr;
- `RrdTempeChwEvBfive` is B5's median event-day error on the Tempe chilled-water meters, in %.

An empty value in `numbers.csv` marks a quantity with no data. For example, no summer hour in Orlando was hotter than its training period, so the errors "beyond training" are undefined there.
