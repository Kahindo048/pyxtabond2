"""
Gap-awareness tests for PanelData's temporal operators on unbalanced panels.

A synthetic 4-individual panel isolates the distinct failure modes a naive
positional lag/diff implementation confuses:
  - id 0 "balanced": t=0..5, no gaps -- control.
  - id 1 "interior_gap": t=0,1,2,4,5 (missing t=3) -- the case a plain
    row-position shift gets wrong (mislabels a 2-period gap as a 1-period
    lag/diff).
  - id 2 "late_start": t=2,3,4,5 (missing t=0,1) -- joins the panel late,
    but is otherwise contiguous.
  - id 3 "early_attrition": t=0,1,2,3 (missing t=4,5) -- leaves the panel
    early, but is otherwise contiguous.

Values are simply `value = float(t)`, so lags/diffs are hand-verifiable: a
1-period lag/diff is always well-defined unless the immediately preceding
period is genuinely absent from the data, regardless of row position.

A brute-force ``{(id, t): value}`` dict oracle sidesteps the position-vs-
calendar-time question entirely by construction, so it's trustworthy without
needing a Stata comparison for this specific, purely mechanical property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyxtabond2 import PanelData
from pyxtabond2.data_utils import scatter_to_grid


def _build_panel_df() -> pd.DataFrame:
    rows = []
    for gid, times in {
        0: range(0, 6),        # balanced
        1: [0, 1, 2, 4, 5],    # interior gap at t=3
        2: [2, 3, 4, 5],       # late start (missing t=0,1)
        3: [0, 1, 2, 3],       # early attrition (missing t=4,5)
    }.items():
        for t in times:
            rows.append({"id": gid, "time": t, "value": float(t)})
    return pd.DataFrame(rows)


def _lag_oracle(df: pd.DataFrame, lags: int) -> dict:
    """{(id, t): value at t - lags, or NaN if that (id, t-lags) row is absent}."""
    lookup = {(r.id, r.time): r.value for r in df.itertuples()}
    return {(r.id, r.time): lookup.get((r.id, r.time - lags), np.nan) for r in df.itertuples()}


def _diff_oracle(df: pd.DataFrame) -> dict:
    """{(id, t): value(t) - value(t-1), or NaN if the (id, t-1) row is absent}."""
    lookup = {(r.id, r.time): r.value for r in df.itertuples()}
    return {
        (r.id, r.time): r.value - lookup[(r.id, r.time - 1)] if (r.id, r.time - 1) in lookup else np.nan
        for r in df.itertuples()
    }


@pytest.fixture(scope="module")
def panel() -> PanelData:
    return PanelData(_build_panel_df(), id_col="id", time_col="time")


def _assert_matches(result: pd.Series, oracle: dict):
    for (gid, t), expected in oracle.items():
        actual = result.loc[(gid, t)]
        if np.isnan(expected):
            assert np.isnan(actual), f"id={gid}, t={t}: expected NaN, got {actual}"
        else:
            assert actual == expected, f"id={gid}, t={t}: expected {expected}, got {actual}"


def test_get_first_difference_matches_oracle(panel):
    result = panel.get_first_difference("value")
    _assert_matches(result, _diff_oracle(panel.data.reset_index()))
    # The interior gap is the one case a naive positional diff gets wrong
    # (it would compute value(t=4) - value(t=2) = 2 and mislabel it as
    # D(t=4), instead of the correct NaN since t=3 is missing).
    assert np.isnan(result.loc[(1, 4)])


def test_get_lag_1_matches_oracle(panel):
    result = panel.get_lag("value", 1)
    _assert_matches(result, _lag_oracle(panel.data.reset_index(), lags=1))


def test_get_lag_2_matches_oracle(panel):
    """lags=2 is the case a fixed-row-position shift (rather than a true
    search by calendar offset) can miss a valid, more-distant source row --
    id 1's t=4 needs t=2 (present, value 2.0), which sits 2 *rows* back
    (not 2 *periods* back) once the t=3 gap has shifted row positions out of
    sync with calendar time.
    """
    result = panel.get_lag("value", 2)
    _assert_matches(result, _lag_oracle(panel.data.reset_index(), lags=2))
    assert result.loc[(1, 4)] == 2.0


def test_late_start_and_attrition_are_not_confused_with_gaps(panel):
    """Boundary truncation (late start / early attrition) has no interior
    gap, so every defined diff/lag should simply equal the arithmetic
    result -- these two ids exist to prove the fix doesn't *break* the
    already-correct boundary case while fixing the interior-gap one."""
    diffs = panel.get_first_difference("value")
    for gid in (2, 3):
        sub = panel.data.loc[gid]
        times = sub.index.to_numpy()
        for t in times[1:]:
            assert diffs.loc[(gid, t)] == 1.0


def test_rebuild_fast_preserves_time_offsets(panel):
    df_new = panel.data.reset_index().copy()
    df_new["value"] = df_new["value"] * 10.0
    rebuilt = PanelData.rebuild_fast(panel, df_new)
    assert np.array_equal(rebuilt._time_offsets, panel._time_offsets)


def test_scatter_to_grid_basic():
    grid = scatter_to_grid(np.array([10.0, 20.0, 30.0]), np.array([0, 2, 4]), T_span=6)
    expected = np.array([10.0, np.nan, 20.0, np.nan, 30.0, np.nan])
    assert np.array_equal(grid, expected, equal_nan=True)


def test_lag_and_diff_fast_path_matches_pandas_at_scale():
    """_lag_group/_diff_group short-circuit to a plain positional shift when
    a group has no interior gap (see its docstring) instead of paying for a
    searchsorted -- mathematically identical, but only exercised by the tiny
    4-id fixture above. Cross-check at a more realistic scale (30 ids x 20
    periods, fully balanced, the common case this fast path targets) against
    pandas' own independent groupby().shift()/diff() implementation, so this
    isn't just "the same code checked against itself".
    """
    rng = np.random.default_rng(0)
    n_ids, n_times = 30, 20
    rows = [
        {"id": i, "time": t, "value": rng.normal()}
        for i in range(n_ids) for t in range(n_times)
    ]
    df = pd.DataFrame(rows)
    panel = PanelData(df, id_col="id", time_col="time")

    lag1 = panel.get_lag("value", 1)
    lag3 = panel.get_lag("value", 3)
    diff1 = panel.get_first_difference("value")

    ref = df.set_index(["id", "time"]).sort_index()
    expected_lag1 = ref.groupby(level=0)["value"].shift(1)
    expected_lag3 = ref.groupby(level=0)["value"].shift(3)
    expected_diff1 = ref.groupby(level=0)["value"].diff(1)

    pd.testing.assert_series_equal(lag1.sort_index(), expected_lag1.rename(None), check_names=False)
    pd.testing.assert_series_equal(lag3.sort_index(), expected_lag3.rename(None), check_names=False)
    pd.testing.assert_series_equal(diff1.sort_index(), expected_diff1.rename(None), check_names=False)


def test_y_lvl_x_lvl_only_built_when_orthogonal():
    """y_lvl/X_lvl feed only compute_ar's orthogonal branch (estimator.py),
    so building them is skipped entirely when orthogonal=False -- this test
    pins that optimization (protects it from silently regressing back to
    always-on) without needing to touch beta/se, which never depend on it.
    """
    from pyxtabond2 import PyXtabond2, load_dataset

    df = load_dataset("df_panel.csv")
    panel = PanelData(df, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    df_ready = panel.data.reset_index()

    common = dict(
        id_col="Country", time_col="Year", dep_var="Growth",
        x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
        gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
        model_type="difference", twostep=False,
    )

    res_plain = PyXtabond2(df_ready, **common, orthogonal=False).fit()
    assert res_plain.engine.y_lvl == []
    assert res_plain.engine.X_lvl == []

    res_orth = PyXtabond2(df_ready, **common, orthogonal=True).fit()
    assert len(res_orth.engine.y_lvl) == res_orth.engine.N_groups
    assert len(res_orth.engine.X_lvl) == res_orth.engine.N_groups
