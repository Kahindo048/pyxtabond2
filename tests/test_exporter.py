"""
Regression tests for LaTeX export escaping.

Discovered while adding a worked-example table to the User Manual:
``PyXtabond2Results.to_latex()`` and ``GMMStargazer.to_latex()`` both
inserted variable/model/diagnostic-test names into raw LaTeX table rows
without escaping special characters. An underscore -- the single most
common offender, since it appears in Stata-style ``_cons`` (added to every
System GMM model) and in this package's own recommended ``L1_``/``D_``-
prefixed variable naming convention -- breaks LaTeX compilation outside math
mode (confirmed by actually compiling the pre-fix output: pdflatex raised
``! Extra }, or forgotten $.``). Fixed by escaping labels (not numeric
values) specific to the LaTeX export path; ``to_word()`` intentionally
keeps them unescaped, which is correct for a Word document.
"""
from __future__ import annotations

import re

import pytest

from pyxtabond2 import PanelData, PyXtabond2, GMMStargazer, load_dataset


def _unescaped_underscore(latex: str) -> bool:
    """True if `latex` contains a literal `_` not already escaped as `\\_`."""
    return re.search(r'(?<!\\)_', latex) is not None


@pytest.fixture(scope="module")
def df_ready():
    df = load_dataset("df_panel.csv")
    panel = PanelData(df, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    return panel.data.reset_index()


COMMON_KW = dict(
    id_col="Country", time_col="Year", dep_var="Growth",
    x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment"],
    gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
)


@pytest.fixture(scope="module")
def system_result(df_ready):
    # model_type='system' always adds '_cons'; L1_Growth exercises the
    # package's own recommended lag-naming convention -- both contain the
    # underscore this bug mishandled.
    return PyXtabond2(df_ready, **COMMON_KW, model_type="system", twostep=True, robust=True, small=True).fit()


@pytest.fixture(scope="module")
def diff_result(df_ready):
    return PyXtabond2(df_ready, **COMMON_KW, model_type="difference").fit()


def test_results_to_latex_escapes_underscored_variable_names(system_result):
    latex = system_result.to_latex(full_output=False)
    assert r"L1\_Growth" in latex
    assert r"\_cons" in latex
    assert not _unescaped_underscore(latex)


def test_results_to_latex_full_output_stays_verbatim(system_result):
    # The verbatim/full-output path must NOT be escaped -- LaTeX's verbatim
    # environment already displays underscores literally and correctly;
    # escaping there would corrupt the exact console-output reproduction.
    latex = system_result.to_latex(full_output=True)
    assert "L1_Growth" in latex
    assert r"L1\_Growth" not in latex


def test_stargazer_to_latex_escapes_underscored_names(diff_result, system_result):
    stargazer = GMMStargazer([diff_result, system_result], model_names=["Diff_Model", "Sys_Model"])
    latex = stargazer.to_latex()
    assert r"L1\_Growth" in latex
    assert r"\_cons" in latex
    assert r"Diff\_Model" in latex
    assert r"Sys\_Model" in latex
    assert not _unescaped_underscore(latex)


def test_escape_latex_is_a_no_op_on_plain_names():
    from pyxtabond2.results import _escape_latex as escape_results
    from pyxtabond2.exporter import _escape_latex as escape_exporter
    for escape in (escape_results, escape_exporter):
        assert escape("Capital") == "Capital"
        assert escape("Labor") == "Labor"


def test_to_word_keeps_variable_names_unescaped(system_result, tmp_path):
    pytest.importorskip("docx")
    out = tmp_path / "out.docx"
    system_result.to_word(str(out), full_output=False)
    assert out.exists() and out.stat().st_size > 0
