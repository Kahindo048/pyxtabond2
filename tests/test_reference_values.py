"""
Non-regression fixtures: freeze current numeric outputs so refactors/
optimizations cannot silently change results.

These values are *not* a claim of correctness vs Stata (see the methodology
notes and ``References/xtabond2.mata`` for that) — they only protect against
accidental regressions while the engine is restructured/optimized. When a
change is *intended* to alter numbers (e.g. a bugfix), update
``fixtures/reference_values.json`` deliberately via
``scripts/generate_reference_fixtures.py`` and explain the diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyxtabond2 import PanelData, PyXtabond2, load_dataset

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REFERENCE_PATH = FIXTURES_DIR / "reference_values.json"

COMMON_KW = dict(
    id_col="Country",
    time_col="Year",
    dep_var="Growth",
    x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
    gmm_vars=["Growth", "Capital"],
    iv_vars=["Ide"],
)

# Single source of truth for both this test module and
# scripts/generate_reference_fixtures.py.
SPECS = {
    "diff_1step": dict(model_type="difference", twostep=False),
    "sys_1step": dict(model_type="system", twostep=False),
    "sys_2step_robust_small": dict(model_type="system", twostep=True, robust=True, small=True),
    "sys_collapsed": dict(model_type="system", twostep=True, robust=True, small=True, collapse=True),
    "sys_fod_collapsed": dict(
        model_type="system", twostep=True, robust=True, small=True, collapse=True, orthogonal=True
    ),
    "sys_1step_robust_small": dict(model_type="system", twostep=False, robust=True, small=True),
    # NOTE: this fixture's numbers were deliberately re-baselined (2026-07) when
    # ife.py's panel-matrix imputation was fixed. L1_Growth is undefined for
    # each country's first year, so every IFE call was silently imputing this
    # cell (and the residual matrix's copy of it, every loop iteration) with a
    # single global scalar mean -- even on this fully balanced panel. It's now
    # imputed via numfac.em_pca_imputation (one-shot, for the pre-loop
    # matrices) and an evolving rank-r reconstruction reused from the previous
    # iteration (in the main loop, at ~zero extra cost). See tests/
    # test_ife_missing_data.py for direct correctness/regression coverage of
    # the new imputation path.
    "diff_ife_r2": dict(model_type="difference", twostep=False, r=2, ife_max_iter=30, ife_tol=1e-5),
}


def _prepare(df):
    panel = PanelData(df, id_col="Country", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    return panel.data.reset_index()


@pytest.fixture(scope="module")
def df_ready():
    return _prepare(load_dataset("df_panel.csv"))


@pytest.fixture(scope="module")
def reference_values():
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract(res):
    diag = res.diag
    return {
        "beta": res.beta.flatten().tolist(),
        "se": res.se.flatten().tolist(),
        "x_names": res.x_names,
        "n_obs": int(res.engine.n_obs),
        "n_groups": int(res.engine.N_groups),
        "n_instruments": int(res.engine.n_instruments),
        "ar": {str(lag): list(vals) for lag, vals in diag["ar"].items()},
        "sargan": list(diag["sargan"]),
        "hansen": list(diag["hansen"]),
        "wald": list(diag["wald"][:3]) + [diag["wald"][3]],
    }


@pytest.mark.parametrize("name", list(SPECS.keys()))
def test_matches_reference(df_ready, reference_values, name):
    """Current output must match the frozen reference within tight tolerance."""
    expected = reference_values[name]
    model = PyXtabond2(df_ready, **COMMON_KW, **SPECS[name])
    actual = _extract(model.fit())

    assert actual["x_names"] == expected["x_names"]
    assert actual["n_obs"] == expected["n_obs"]
    assert actual["n_groups"] == expected["n_groups"]
    assert actual["n_instruments"] == expected["n_instruments"]

    for key in ("beta", "se", "sargan", "hansen"):
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=f"{name}: {key} mismatch"
        )

    assert actual["ar"].keys() == expected["ar"].keys()
    for lag in actual["ar"]:
        np.testing.assert_allclose(
            actual["ar"][lag], expected["ar"][lag], rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg=f"{name}: ar({lag}) mismatch",
        )

    np.testing.assert_allclose(actual["wald"][:3], expected["wald"][:3], rtol=1e-6, atol=1e-9)
    assert actual["wald"][3] == expected["wald"][3]
