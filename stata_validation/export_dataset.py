"""
Exports the bundled example dataset (plus a synthetic clustering variable
and a numeric panel id) to Stata .dta format, for side-by-side validation of
the new options (h(#), artests(#), arlevels, per-group gmm(), cluster())
against real Stata output. See validate_new_features.do in this folder.

Usage
-----
    python stata_validation/export_dataset.py
"""

import pandas as pd

from pyxtabond2 import PanelData, load_dataset

df = load_dataset("df_panel.csv")
panel = PanelData(df, id_col="Country", time_col="Year")
panel.data["L1_Growth"] = panel.get_lag("Growth", 1)
df_ready = panel.data.reset_index()

# Numeric panel id (Stata's xtset requires numeric) + a coarser synthetic
# "Region" grouping (10 countries each) to exercise cluster() with a
# different (coarser) clustering variable than the panel id.
countries = sorted(df_ready["Country"].unique())
id_map = {c: i + 1 for i, c in enumerate(countries)}
region_map = {c: (i // 10) + 1 for i, c in enumerate(countries)}
df_ready["country_id"] = df_ready["Country"].map(id_map)
df_ready["region_id"] = df_ready["Country"].map(region_map)

out_path = "stata_validation/pyxtabond2_validation.dta"
df_ready.to_stata(out_path, write_index=False, version=118)
print(f"Wrote {out_path} ({len(df_ready)} rows, {df_ready['country_id'].nunique()} countries, "
      f"{df_ready['region_id'].nunique()} regions)")
