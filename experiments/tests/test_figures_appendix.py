"""Tests for the dispatch decomposition (Fig. dispatch) and the Appendix D commitment record."""
import numpy as np
from dcflex import model, protocol, figures, export
from tests.test_core import _inp


def test_lever_parts_sum_to_change_in_import():
    fac, inp = _inp()
    s0 = model.bau(inp, fac)
    s3 = model.optimize(inp, fac, np.linspace(100, 300, 48))
    parts = figures._lever_parts(s0, s3, inp["cop"], fac, slice(None))
    assert np.allclose(sum(parts.values()), s3["pg"] - s0["pg"], atol=1e-4)


def test_storage_levels_are_returned_within_bounds():
    fac, inp = _inp()
    o = model.optimize(inp, fac, np.linspace(100, 300, 48))
    cap = fac.tes_hours * fac.peak_heat_mw
    assert len(o["soc"]) == 48 and np.all(o["soc"] >= -1e-6) and np.all(o["soc"] <= cap + 1e-6)
    assert np.all(o["soe"] <= fac.bess_mwh + 1e-6)


def test_commit_example_is_written_and_verifies(tmp_path):
    coef, feats = np.array([1.0, 2.0, 3.0]).tobytes(), np.array([40.0, 39.0]).tobytes()
    c, rec = protocol.commit("AD-DR-test", "2025-06-01T00:00:00+04:00", [1, 2], [100.0, 99.0], coef, feats, nonce="00" * 32)
    assert protocol.verify(c, rec, coef, feats)
    assert not protocol.verify(c, dict(rec, baseline_mw=[101.0, 99.0]), coef, feats)
    export.write_commit_example(c, rec, tmp_path)
    txt = (tmp_path / "commit-example.txt").read_text()
    assert c in txt and '"baseline_mw"' in txt
    assert "def verify" in (tmp_path / "protocol-code.txt").read_text()


def test_numbers_csv_lists_every_macro_as_plain_text(tmp_path):
    import csv
    (tmp_path / "macros.tex").write_text("% header\n\\newcommand{\\RaMW}{35.7}\n\\newcommand{\\RbNeg}{$-$0.65}\n"
                                         "\\newcommand{\\RcPending}{\\tbd{cPending}}\n\\newcommand{\\RdDate}{15 August}\n")
    (tmp_path / "macros-ml.tex").write_text("\\newcommand{\\ReSynthetic}{\\tbd{syn: 1.0}}\n")
    export.write_numbers_csv(tmp_path)
    rows = list(csv.reader(open(tmp_path / "numbers.csv")))
    assert rows == [["macro", "value", "written_by"], ["RaMW", "35.7", "run_all.py"], ["RbNeg", "-0.65", "run_all.py"],
                    ["RcPending", "", "run_all.py"], ["RdDate", "15 August", "run_all.py"],
                    ["ReSynthetic", "", "run_ml.py"]]
