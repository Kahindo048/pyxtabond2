"""
Tests for EM-PCA missing-cell imputation in the IFE-GMM path
(``numfac.em_pca_imputation``, shared by ``ife.py``'s one-shot panel-matrix
builders and the main defactoring loop's per-iteration rank-r reconstruction
fill).

``tests/test_reference_values.py``'s ``diff_ife_r2`` spec exercises this only
incidentally, via the universal first-year lag-truncation gap every dynamic
panel has. This module adds a direct correctness check of the imputation
mechanism itself, an end-to-end check with an extra missing *value*, and
(now that estimator.py's separate base-engine crash on any missing *row* is
fixed) an end-to-end check on a genuinely unbalanced panel too.
"""

from __future__ import annotations

import numpy as np

from pyxtabond2 import PanelData, PyXtabond2, load_dataset
from pyxtabond2.numfac import em_pca_imputation
from pyxtabond2.ife import _impute_gaps


def test_em_pca_imputation_recovers_exact_lowrank_structure():
    rng = np.random.default_rng(0)
    n_ids, n_times, r = 40, 15, 1
    loadings = rng.normal(size=(n_ids, r))
    factors = rng.normal(size=(n_times, r))
    true_mat = loadings @ factors.T  # exact rank-1, noiseless

    mat = true_mat.copy()
    missing = rng.random((n_ids, n_times)) < 0.15
    mat[missing] = np.nan

    # Tight tol/max_iter: checks the mechanism converges close to the true
    # values given enough iterations, not just "close enough" at the 1e-5
    # default stopping tolerance used operationally in ife.py/numfac.py.
    imputed = em_pca_imputation(mat, kmax=r, max_iter=500, tol=1e-10)

    assert np.all(np.isfinite(imputed))
    assert np.allclose(imputed[missing], true_mat[missing], atol=1e-6)
    assert np.array_equal(imputed[~missing], true_mat[~missing])


def test_impute_gaps_is_noop_without_missing_values():
    mat = np.arange(20.0).reshape(4, 5)
    out = _impute_gaps(mat, kmax=2)
    assert out is mat


def test_ife_gmm_runs_with_an_extra_missing_value(capsys):
    """A second, non-lag-truncation missing cell (e.g. one unrecorded X
    observation) -- distinct from the universal first-year lag truncation
    already covered by diff_ife_r2 -- must not crash the defactoring loop
    and must produce finite, well-posed inference.

    NOTE: this injects a NaN *value* into an existing (id, time) row, not a
    missing row -- see test_ife_gmm_runs_on_a_genuinely_unbalanced_panel
    below for that case, which the base GMM engine could not reach at all
    until a separate fix (estimator.py's per-group stacking crashed
    outright on any missing row, regardless of the IFE loop).
    """
    df = load_dataset("df_panel.csv")
    panel = PanelData(df, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    df_ready = panel.data.reset_index()

    # Blank out one country's mid-panel Capital value (row still present).
    country0 = df_ready["Country"].unique()[0]
    blank = (df_ready["Country"] == country0) & (df_ready["Year"] == 5)
    df_ready.loc[blank, "Capital"] = np.nan

    model = PyXtabond2(
        df_ready,
        id_col="Country", time_col="Year", dep_var="Growth",
        x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
        gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
        model_type="difference", twostep=False, r=2, ife_max_iter=30, ife_tol=1e-5,
    )
    result = model.fit()
    captured = capsys.readouterr()

    # Confirm both imputation sites actually fired, not just "didn't crash".
    assert "[xtnumfac] Missing values detected" in captured.out
    assert "no defined residual" in captured.out

    assert np.all(np.isfinite(result.beta))
    assert np.all(np.isfinite(result.se))
    assert np.all(result.se > 0)


def test_ife_gmm_runs_on_a_genuinely_unbalanced_panel(capsys):
    """A real (id, time) row absent from the input -- distinct from the
    value-level-only missingness above -- exercises the same imputation
    machinery for IFE-GMM's own residual/factor-extraction matrices, on top
    of estimator.py's base-engine crash fix (see
    tests/test_unbalanced_panel_pipeline.py for that fix's own dedicated
    coverage). L1_Growth is recomputed *after* dropping the row (not reused
    from a version precomputed on the full data), matching a real user's
    workflow -- see stata_validation/README.md's "Methodology pitfall" note
    for why the reverse order silently understates which rows a gap
    invalidates.
    """
    df = load_dataset("df_panel.csv")
    country0 = sorted(df["Country"].unique())[0]
    df_gap = df[~((df["Country"] == country0) & (df["Year"] == 5))].copy()

    panel = PanelData(df_gap, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    df_ready = panel.data.reset_index()

    model = PyXtabond2(
        df_ready,
        id_col="Country", time_col="Year", dep_var="Growth",
        x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
        gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
        model_type="difference", twostep=False, r=2, ife_max_iter=30, ife_tol=1e-5,
    )
    result = model.fit()
    captured = capsys.readouterr()

    assert "[xtnumfac] Missing values detected" in captured.out
    assert "no defined residual" in captured.out

    assert np.all(np.isfinite(result.beta))
    assert np.all(np.isfinite(result.se))
    assert np.all(result.se > 0)
