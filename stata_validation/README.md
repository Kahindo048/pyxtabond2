# Stata validation kit

Validates 6 newly-added Stata-parity features that had no prior PyXtabond2
output to freeze as a regression fixture: `h(#)`, `artests(#)`, `arlevels`,
per-group `gmm()` (lag/collapse per variable), `cluster()`, and a re-check of
the one-step-robust small-sample "qc" bugfix.

## Steps

1. `python stata_validation/export_dataset.py` — writes `pyxtabond2_validation.dta`.
2. `python stata_validation/compute_python_reference.py` — writes
   `python_reference_output.txt` (same numbers already embedded as comments
   in the do-file, kept here in full for convenience).
3. In Stata (with `xtabond2` installed — the do-file runs
   `ssc install xtabond2` for you): run `validate_new_features.do` from this
   folder.
4. Compare Stata's log to the numbers in the do-file's comments / in
   `python_reference_output.txt`.

## Status

This do-file has been run against real Stata and the log compared back
against the Python reference numbers. Results:

- **h(2), h(3), artests(4), per-group gmm(), and the one-step-robust qc
  bugfix (cases 1b/1c/2/4/6): confirmed correct** — matched Stata to the
  reported precision on the first run.
- **h(1) and arlevels (cases 1a/3): initially wrong, now fixed and
  confirmed.** Coefficients matched from the start, but SE/AR/Sargan didn't;
  root cause was `engine.py`/`estimator.py`'s `sig2` (error-variance scale)
  hardcoding the diff-equation's residuals/count and a divisor of 2
  regardless of `h`, when `xtabond2.mata:201`'s `ErrorEq` and `_ARTests`'s
  `psit` term both specify different behavior for `h==1`/`arlevels`. Fixed;
  re-run `compute_python_reference.py` output now matches Stata exactly for
  both. See the updated comments in `validate_new_features.do` for the
  precise mechanism.
- **cluster()+twostep+robust with few clusters relative to instrument count
  (case 5): a real, root-caused, unresolved divergence in beta** — not a
  cluster() bug (confirmed cluster() reproduces the default-clustering case
  exactly outside this regime), but a fundamental difference between NumPy's
  Moore-Penrose pseudo-inverse and Stata's `invsym`-based generalized
  inverse when the moment covariance matrix is exactly rank-deficient (here,
  90 instruments vs 20 clusters forces rank <= 20). Both are valid
  generalized inverses of a singular matrix but not numerically the same
  one. See the detailed comment above case 5 in the do-file. Documented as a
  known limitation rather than chased further, since bit-exact replication
  of Mata's specific pivoting order is a deep, fragile undertaking
  disproportionate to this package's scope — and Stata's own warning
  message already flags this combination as an unusual, weakened-test
  regime.
- **A second, distinct bug in the same case 5's SE/AR tests (not beta) was
  found later and fixed, then independently confirmed via case 9**:
  `xtabond2.mata:564`'s `rows(clusts)` branch does not key off "was
  `cluster()` passed explicitly" — it keys off whether the clustering
  variable induces a genuinely *different partition* than the panel id. An
  earlier fix attempt read it as the former and dropped the small-sample
  `N_clusters/(N_clusters-1)` factor for *any* explicit `cluster()`,
  over-correcting whenever the named variable happened to coincide with the
  panel id. `GMMEngine._cluster_qc_factor` now checks partition equality
  (`_same_partition()`), not just whether `cluster_ids` was passed. Case 9
  (`cluster(country_id)` on a panel id'd by `country_id`, full-rank, avoiding
  case 5's singular regime) was added specifically to isolate this and
  **confirmed against two independent real Stata 17 runs**: 9a
  (no `cluster()`) and 9b (`cluster(country_id)`) are bit-identical on every
  coefficient and SE, matching the current code exactly
  (`python_reference_output.txt` regenerated accordingly).

If you re-run this do-file (e.g. after a future change), report any mismatch
beyond ~1e-4 relative difference on cases other than 5's beta (case 5's SE
now reflects the qc fix above, and its beta remains the known singular-regime
divergence) — that would point to a fresh regression.

## Unbalanced panel (cases 7-8)

Added after discovering `estimator.py::_fit_base` crashed outright
(`IndexError: boolean index did not match indexed array along axis 0`) the
instant any single individual was missing even one `(id, time)` row —
boundary or interior, either model type, with or without IFE-GMM. Fixed
across `data_utils.py` (gap-aware lag/diff), `estimator.py` (the crash
itself, by indexing the instrument block by absolute time offset instead of
local row position), and `engine.py::compute_ar` (the AR(l) test had its own,
independent positional-lag bug in all 3 branches).

Validated directly against real Stata 17 (StataMP-64, installed locally),
run in batch mode from this repo rather than a manual round-trip — both new
cases:

- **Case 7 (interior gap)**: drop `country_id==1`'s `Year==5` row.
- **Case 8 (late-starting individual)**: drop `country_id==1`'s `Year==0`
  row, specifically to settle an ambiguity the Mata source alone couldn't
  resolve — does System GMM's level equation exclude the panel's *global*
  first calendar period for every individual, or each individual's *own*
  first observed period? Every other validated case has every individual
  starting at the panel's global t_min, so the two readings were previously
  indistinguishable.

**Both match Stata to 6 decimal places on every coefficient and SE** (see the
CONFIRMED notes on each case in the do-file) — confirming both the crash fix
and, via case 8, that pyxtabond2's existing mask_level formula (the "global"
reading) is correct as-is; no code change was needed there.

**Methodology pitfall found and fixed while building these cases**: computing
`L1_Growth` once on the full dataset and only *afterward* dropping a row (as
`tests/test_ife_missing_data.py` and an earlier draft of the pytest
unbalanced-panel fixtures did) leaves a stale value at the row right after a
gap — it silently keeps referencing the now-absent period instead of
correctly becoming missing, understating which rows the gap actually
invalidates. `L1_Growth` must be recomputed via `panel.get_lag(...)` *after*
dropping the row, matching both a real user's actual workflow and what
Stata's own `L.Growth` does dynamically. `compute_python_reference.py` and
`tests/test_unbalanced_panel_pipeline.py` do this correctly; the mismatch
this caused during validation was in the test/reference-computation script,
not in `pyxtabond2` itself.
