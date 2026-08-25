"""
Computes PyXtabond2 reference numbers for each case in validate_new_features.do,
using the exact dataset exported by export_dataset.py (same rows, same
numeric ids), so Stata's log can be diffed directly against these numbers.

Usage
-----
    python stata_validation/export_dataset.py   # once, to produce the .dta
    python stata_validation/compute_python_reference.py
"""

import pandas as pd

from pyxtabond2 import PanelData, PyXtabond2
from pyxtabond2.specs import GMMStyle

df_ready = pd.read_stata("stata_validation/pyxtabond2_validation.dta")

COMMON = dict(
    id_col="country_id", time_col="Year", dep_var="Growth",
    x_vars=["L1_Growth", "Capital", "Labor", "Wage", "Investment", "Ide"],
    gmm_vars=["Growth", "Capital"], iv_vars=["Ide"],
    model_type="system", twostep=False,
)


def show(title, res):
    print(f"\n=== {title} ===")
    for name, b, s in zip(res.x_names, res.beta.flatten(), res.se.flatten()):
        print(f"  {name:12s} beta={b: .6f}  se={s:.6f}")
    print(f"  n_obs={res.engine.n_obs}  n_groups={res.engine.N_groups}  "
          f"n_instruments={res.engine.n_instruments}  N_clusters={res.engine.N_clusters}")
    for lag, (stat, p) in res.diag["ar"].items():
        print(f"  AR({lag}): z={stat:.6f}  p={p:.6f}")
    print(f"  Sargan: chi2={res.diag['sargan'][0]:.6f}  p={res.diag['sargan'][1]:.6f}")
    if not pd.isna(res.diag["hansen"][0]):
        print(f"  Hansen: chi2={res.diag['hansen'][0]:.6f}  p={res.diag['hansen'][1]:.6f}")


# 1. h(#)
for h in (1, 2, 3):
    show(f"1. h={h}", PyXtabond2(df_ready, **COMMON, h=h).fit())

# 2. artests(#)
show("2. artests=4", PyXtabond2(df_ready, **COMMON, artests=4).fit())

# 3. arlevels
show("3. arlevels", PyXtabond2(df_ready, **COMMON, arlevels=True).fit())

# 4. Multi-group gmm() (per-variable lag/collapse)
show(
    "4. gmm(Growth, lag(2 4) collapse) + gmm(Capital, lag(2 .))",
    PyXtabond2(
        df_ready, **{k: v for k, v in COMMON.items() if k != "gmm_vars"},
        gmm=[GMMStyle("Growth", lag=(2, 4), collapse=True), GMMStyle("Capital", lag=(2, None))],
    ).fit(),
)

# 5. cluster()
show(
    "5. cluster(region_id), twostep robust small",
    PyXtabond2(
        df_ready, **{k: v for k, v in COMMON.items() if k != "twostep"},
        twostep=True, robust=True, small=True, cluster="region_id",
    ).fit(),
)

# 6. qc bugfix confirmation (one-step robust, small)
show("6. one-step robust small (qc bugfix)", PyXtabond2(df_ready, **COMMON, robust=True, small=True).fit())

# 7-8. Unbalanced panel cases: L1_Growth must be recomputed *after* dropping
# the row, not reused from df_ready (precomputed on the full data) -- a stale
# precomputed value would keep referencing the now-absent period at the row
# right after a gap, understating which rows the gap actually invalidates.
# This is also what Stata's own L.Growth does dynamically in the do-file.
df_raw = df_ready.drop(columns=["L1_Growth"])


def _prep(cid, year):
    d = df_raw[~((df_raw["country_id"] == cid) & (df_raw["Year"] == year))].copy()
    panel = PanelData(d, id_col="country_id", time_col="Year")
    panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
    return panel.data.reset_index()


# 7. Unbalanced: interior gap (country_id==1, Year==5 dropped)
show("7. unbalanced interior gap (country_id==1, Year==5 dropped)", PyXtabond2(_prep(1, 5), **COMMON).fit())

# 8. Unbalanced: late-starting individual (country_id==1, Year==0 dropped)
show("8. unbalanced late start (country_id==1, Year==0 dropped)", PyXtabond2(_prep(1, 0), **COMMON).fit())

# 9. Bug 4 closure: explicit cluster() at the same grouping as the default,
# away from Case 5's singular-S2 regime (200 clusters vs ~90 instruments
# here, always full rank). Beta should match exactly between 9a/9b; SE
# should differ by exactly sqrt(199/200) (9b smaller), per
# GMMEngine._cluster_qc_factor. See the matching note in validate_new_features.do.
show(
    "9a. No cluster() (default = panel id), twostep robust small",
    PyXtabond2(df_ready, **{k: v for k, v in COMMON.items() if k != "twostep"},
               twostep=True, robust=True, small=True).fit(),
)
show(
    "9b. cluster(country_id) explicit (same grouping as default), twostep robust small",
    PyXtabond2(df_ready, **{k: v for k, v in COMMON.items() if k != "twostep"},
               twostep=True, robust=True, small=True, cluster="country_id").fit(),
)
