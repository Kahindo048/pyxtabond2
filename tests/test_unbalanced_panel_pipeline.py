"""
Full-pipeline (PyXtabond2.fit) regression tests for a genuinely unbalanced
panel -- a real (id, time) row absent from the input, not just a missing
value within a present row (which GMMEngine.__init__'s existing
``valid_rows`` already handled correctly; see tests/test_ife_missing_data.py
and the commit fixing ife.py's imputation for that distinct case).

Before this fix, dropping even a single row for a single individual (at a
panel boundary or mid-panel) crashed estimator.py::_fit_base with
``IndexError: boolean index did not match indexed array along axis 0`` for
both model types, with or without IFE-GMM -- confirmed by direct testing,
not a hypothetical. These tests lock that crash fixed, and pin down (by
hand-verified arithmetic, see each test's comment) exactly how many
observations survive so a future change can't silently alter the row-
selection logic without a test failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyxtabond2 import PanelData, PyXtabond2, load_dataset

COMMON_KW = dict(
    id_col="Country",
    time_col="Year",
    dep_var="Growth",
    x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
    gmm_vars=["Growth", "Capital"],
    iv_vars=["Ide"],
)


@pytest.fixture(scope="module")
def df_raw():
    """Raw panel, L1_Growth *not* precomputed -- each test recomputes it
    after dropping a row, matching a real user's workflow (drop/clean the
    raw data, *then* compute lags) and matching what Stata's own L.Growth
    does dynamically. Precomputing on the full data and dropping afterward
    would leave a stale, pre-drop value at the row right after a gap (it
    would silently keep referencing the now-absent period), understating
    which rows a gap actually invalidates -- exactly the discrepancy that
    surfaced comparing an earlier draft of this fixture against real Stata.
    """
    return load_dataset("df_panel.csv")


@pytest.fixture(scope="module")
def country0(df_raw):
    return sorted(df_raw["Country"].unique())[0]


def _prep(df_raw, country0=None, year=None):
    d = df_raw.copy()
    if country0 is not None:
        d = d[~((d["Country"] == country0) & (d["Year"] == year))].copy()
    panel = PanelData(d, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    return panel.data.reset_index()


@pytest.mark.parametrize("model_type", ["difference", "system"])
@pytest.mark.parametrize("dropped_year", [9, 5], ids=["boundary_truncation", "interior_gap"])
def test_dropped_row_no_longer_crashes(df_raw, country0, model_type, dropped_year):
    data = _prep(df_raw, country0, dropped_year)
    model = PyXtabond2(data, model_type=model_type, twostep=False, **COMMON_KW)
    result = model.fit()
    assert np.all(np.isfinite(result.beta))
    assert np.all(np.isfinite(result.se))
    assert np.all(result.se > 0)


def test_dropped_row_observation_count_matches_hand_derivation(df_raw, country0):
    """Difference GMM, full sample: 200 countries x 8 valid diff periods
    (t=2..9 survive L1_Growth's own double lag-truncation at t=0,1) = 1600.

    Dropping country0's *last* year (boundary truncation) removes exactly
    one of its own 8 candidate periods (t=9) -> 199*8 + 7 = 1599.

    Dropping country0's *year 5* (interior gap) removes three of its own 8
    candidate periods, not just one: t=5 itself (no longer a row at all),
    t=6 (D_Growth/D_L1_Growth need Growth[5], now absent), and t=7 (its own
    D_L1_Growth needs L1_Growth[6], itself undefined per the previous point
    -- the gap's "shadow" extends one period further for a regressor that
    is already once-lagged) -> only {2,3,4,8,9} survive: 199*8 + 5 = 1597.
    """
    model_full = PyXtabond2(_prep(df_raw), model_type="difference", twostep=False, **COMMON_KW)
    assert model_full.fit().engine.n_obs == 1600

    model_boundary = PyXtabond2(_prep(df_raw, country0, 9), model_type="difference", twostep=False, **COMMON_KW)
    assert model_boundary.fit().engine.n_obs == 1599

    model_gap = PyXtabond2(_prep(df_raw, country0, 5), model_type="difference", twostep=False, **COMMON_KW)
    assert model_gap.fit().engine.n_obs == 1597


def test_matches_real_stata_on_unbalanced_panel():
    """Direct cross-validation against a real Stata 17 (StataMP-64) run of
    stata_validation/validate_new_features.do cases 7-8 (System GMM, one-step,
    default h(3)) -- an interior gap and a late-starting individual,
    respectively. Coefficients and SE match Stata to 6 decimal places (see
    stata_validation/README.md for the full log comparison); this pins that
    match down permanently as a regression fixture using the same dataset.
    """
    import pandas as pd

    df_ready = pd.read_stata("stata_validation/pyxtabond2_validation.dta")
    df_raw = df_ready.drop(columns=["L1_Growth"])
    common = dict(
        id_col="country_id", time_col="Year", dep_var="Growth",
        x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
        gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
        model_type="system", twostep=False,
    )

    def prep(cid, year):
        d = df_raw[~((df_raw["country_id"] == cid) & (df_raw["Year"] == year))].copy()
        panel = PanelData(d, id_col="country_id", time_col="Year")
        panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
        return panel.data.reset_index()

    # Case 7: interior gap (country_id==1, Year==5 dropped). Stata: L1.=.7981863
    # (.0105558) Capital=1.362369 (.1077595) _cons=1.076923 (.4146986).
    r7 = PyXtabond2(prep(1, 5), **common).fit()
    beta7 = dict(zip(r7.x_names, r7.beta.flatten()))
    se7 = dict(zip(r7.x_names, r7.se.flatten()))
    assert beta7["L1_Growth"] == pytest.approx(0.7981863, abs=1e-6)
    assert se7["L1_Growth"] == pytest.approx(0.0105558, abs=1e-6)
    assert beta7["Capital"] == pytest.approx(1.362369, abs=1e-5)
    assert beta7["_cons"] == pytest.approx(1.076923, abs=1e-5)

    # Case 8: late start (country_id==1, Year==0 dropped). Stata: L1.=.7931669
    # (.0108189) Capital=1.386876 (.1075773) _cons=1.032555 (.4153134).
    r8 = PyXtabond2(prep(1, 0), **common).fit()
    beta8 = dict(zip(r8.x_names, r8.beta.flatten()))
    se8 = dict(zip(r8.x_names, r8.se.flatten()))
    assert beta8["L1_Growth"] == pytest.approx(0.7931669, abs=1e-6)
    assert se8["L1_Growth"] == pytest.approx(0.0108189, abs=1e-6)
    assert beta8["Capital"] == pytest.approx(1.386876, abs=1e-5)
    assert beta8["_cons"] == pytest.approx(1.032555, abs=1e-5)


@pytest.mark.parametrize("orthogonal", [False, True], ids=["diff", "fod"])
@pytest.mark.parametrize("arlevels", [False, True])
def test_ar_test_finite_with_gap_across_all_branches(df_raw, country0, orthogonal, arlevels):
    """Exercises all 3 compute_ar branches (default, orthogonal, arlevels)
    on a genuinely gapped panel -- each had its own positional-lag bug,
    fixed independently (see engine.py::compute_ar)."""
    if arlevels and not orthogonal:
        pytest.skip("arlevels forces model_type='system'; combine freely with orthogonal only")
    data = _prep(df_raw, country0, 5)
    model = PyXtabond2(
        data, model_type="system", twostep=False, robust=True,
        orthogonal=orthogonal, arlevels=arlevels, collapse=orthogonal,
        **COMMON_KW,
    )
    result = model.fit()
    ar1, ar2 = result.diag["ar"][1], result.diag["ar"][2]
    assert np.isfinite(ar1[0]) and np.isfinite(ar1[1])
    assert np.isfinite(ar2[0]) and np.isfinite(ar2[1])


def test_balanced_panel_ar_test_unchanged_by_compute_ar_rewrite(df_raw):
    """Hard no-op gate: on the (unmodified) balanced dataset, compute_ar's
    dense-reconstruction rewrite must reduce to the exact previous
    computation -- checked here against a value pinned from the pre-rewrite
    code (git commit 6cd45e1), not re-derived from the current code."""
    model = PyXtabond2(
        _prep(df_raw), model_type="system", twostep=False, robust=True,
        orthogonal=True, collapse=True,
        **COMMON_KW,
    )
    ar1, ar2 = model.fit().diag["ar"][1], model.fit().diag["ar"][2]
    assert ar1[0] == pytest.approx(-3.8752227439, abs=1e-9)
    assert ar2[0] == pytest.approx(0.8878427745, abs=1e-9)
