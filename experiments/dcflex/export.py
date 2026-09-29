"""Write results/macros.tex, results/numbers.csv, results/tab-*.tex and placeholder figures."""
import csv
import inspect
import json
import math
import pathlib
import re

MACROS = ["RbauEnergyGWh", "RbauPUE", "RbauPUEsummer", "RcoolShareSummer", "RbauPeakMW",
          "RshedOneHourMW", "RshedTwoHourMW", "RshedFourHourMW", "RshedFourHourPct",
          "RflexComputeMW", "RflexTesMW", "RflexBessMW", "RslaDelayPctl", "RslaDelayTrnPctl", "RslaDelayBatPctl",
          "RslaDelayBauPctl",
          "RsolarShareBau", "RsolarShareFlat", "RsolarShareSys", "RsurplusAbsorbedGWh", "RnetPeakRedMW",
          "RgrossPeakChangePct", "RcoolChangeFlatPct", "RcoolChangeSysPct", "RcoTwoAvgRed", "RcoTwoMargRed",
          "RbillSaveCommAED", "RbillSaveIndAED", "RprivateValueKWyr", "RprivateValueTouKWyr", "RprivateValueTransKWyr", "RsystemValueKWyr",
          "RercotValueKWyr", "RvalueGapKWyr", "RvalueGapTouKWyr", "RercotAsShare", "RercotNegHours", "RercotSimulHours", "RercotNegValuePct",
          "RdrFirmMW", "RdrTargetSharePct", "RdrInterimSharePct",
          "RcapPayBreakeven", "RbaselineBest", "RbaselineBestNMAE", "RbaselineHighNMAE",
          "RgamingOverpayHighPct", "RgamingOverpayCommitPct", "RgamingOverpayRollPct", "RbaselineCommitNMAE",
          "RshoppingOverpayPct", "RbaselineLearnedNMAE", "RbaselineLearnedBias", "RgamingOverpayLearnedPct",
          "RbfiveParams", "RbfiveKB", "RbfiveDigest", "RpreTmax", "RsummerTmax",
          "RcommitMicros", "RcommitBytes", "RtornadoTop", "RruleGapPct",
          "RutcShedFourHourMW", "RutcDrFirmMW", "RutcSystemValueKWyr",
          "RcoolPenaltyPts", "RsysMiddayMW", "RsysMiddayConstCopMW", "RsysShiftHourConstCop", "RsysShiftHour", "RflatShiftHour", "RnetPeakHour", "RsurplusHours",
          "RsurplusSummerHours", "RopenCycleHours", "RitShedThreePct", "RitShedThreeMedPct",
          "RpeakEventDate", "RpeakEventComputeMW", "RpeakEventTesMW", "RpeakEventBessMW", "RpeakEventRedMW",
          "RpeakEventPreRiseMW", "RinflEventDate", "RinflCreditHighMW", "RinflCreditHighCleanMW", "RinflCreditCommitMW", "RinflTrueMW"]

TABLE_COLS = {"tab-ablation": 4, "tab-scenarios": 7, "tab-value": 3, "tab-baselines": 5}
DATA_FIGS = ["fig-duck", "fig-flex", "fig-heatmap", "fig-value", "fig-baselines", "fig-tornado",
             "fig-dispatch", "fig-inflation"]


# Decimals per macro where one decimal is wrong (PUE needs two; counts, bytes and hours are whole numbers).
DECIMALS = {"RbauPUE": 2, "RbauPUEsummer": 2, "RcommitBytes": 0, "RbfiveParams": 0, "RbfiveKB": 0, "RcapPayBreakeven": 1,
            "RercotNegHours": 0, "RercotSimulHours": 0, "RsysShiftHour": 0, "RsysShiftHourConstCop": 0, "RflatShiftHour": 0, "RnetPeakHour": 0,
            "RsurplusHours": 0, "RsurplusSummerHours": 0, "RopenCycleHours": 0,
            "RslaDelayPctl": 0, "RslaDelayTrnPctl": 0, "RslaDelayBatPctl": 0, "RslaDelayBauPctl": 0,
            # the penalty is the difference of the two changes, so show all three at 2 decimals to keep them additive
            "RcoolChangeFlatPct": 2, "RcoolChangeSysPct": 2, "RcoolPenaltyPts": 2}


def fmt(v, nd=1):
    """Number as LaTeX text: thousands separators from 1,000, a proper minus sign, no negative zero."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if not math.isfinite(v):
        return None
    v = round(float(v), nd)
    s = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{nd}f}"
    if s.startswith("-"):
        s = s[1:] if float(s[1:].replace(",", "")) == 0 else "$-$" + s[1:]
    return s


def write_macros(values, results_dir, synthetic=False):
    lines = ["% Written by experiments/run_all.py. Do not edit by hand."]
    for name in MACROS:
        s = fmt(values.get(name), DECIMALS.get(name, 1))
        body = r"\tbd{%s}" % name[1:] if s is None else (r"\tbd{syn: %s}" % s if synthetic else s)
        lines.append(r"\newcommand{\%s}{%s}" % (name, body))
    pathlib.Path(results_dir, "macros.tex").write_text("\n".join(lines) + "\n")


def write_numbers_csv(results_dir):
    """results/numbers.csv: every number the paper prints, one row per macro of macros.tex and macros-ml.tex, for
    readers who do not use LaTeX. Values the paper does not define (placeholders) are left empty."""
    rows = []
    for fname, writer in (("macros.tex", "run_all.py"), ("macros-ml.tex", "run_ml.py")):
        path = pathlib.Path(results_dir, fname)
        if not path.exists():
            continue
        for name, v in re.findall(r"^\\newcommand\{\\(R\w+)\}\{(.*)\}$", path.read_text(), re.M):
            rows.append((name, "" if r"\tbd" in v else v.replace("$-$", "-"), writer))
    with open(pathlib.Path(results_dir, "numbers.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["macro", "value", "written_by"])
        w.writerows(rows)


def write_table(name, rows, results_dir, synthetic=False):
    n = TABLE_COLS[name]
    body = []
    if synthetic:
        body.append(r"\multicolumn{%d}{c}{\tbd{SYNTHETIC TEST DATA: not for publication}}\\" % n)
    for r in rows:
        body.append(" & ".join(x if isinstance(x, str) else (fmt(x) or r"\tbd{n/a}") for x in r) + r" \\")
    pathlib.Path(results_dir, name + ".tex").write_text("\n".join(body) + "\n")


def pending_figure(path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.text(0.5, 0.5, f"{name}\npending: generated by experiments/run_all.py", ha="center", va="center",
            fontsize=11, color="#6B7280")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#D1D5DB")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_placeholders(root):
    root = pathlib.Path(root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    write_macros({}, root / "results")
    for name, n in TABLE_COLS.items():
        pathlib.Path(root / "results", name + ".tex").write_text(
            " & ".join([r"\tbd{pending}"] * n) + r" \\" + "\n")
    for f in DATA_FIGS:                  # always reset, so a crash cannot leave figures from an earlier run
        pending_figure(root / "figures" / f"{f}.pdf", f)
    write_commit_example(None, None, root / "results")
    write_numbers_csv(root / "results")


def write_commit_example(digest, rec, results_dir, synthetic=False):
    """Appendix D: one real commitment record (pretty-printed JSON), its digest, and the source of the
    protocol functions, read by the manuscript with \\lstinputlisting."""
    from . import protocol
    rd = pathlib.Path(results_dir)
    if rec is None:
        body = '{"pending": "written by run_all.py"}'
    else:
        body = json.dumps(dict(rec, **({"note": "SYNTHETIC TEST DATA"} if synthetic else {})), indent=1)
        body += "\n\nC = SHA-256(canonical record) =\n" + digest
    (rd / "commit-example.txt").write_text(body + "\n")
    src = "\n\n".join(inspect.getsource(f) for f in (protocol.canonical, protocol.sha, protocol.commit, protocol.verify))
    (rd / "protocol-code.txt").write_text(src)
