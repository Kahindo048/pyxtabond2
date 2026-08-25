/*==============================================================================
Validation do-file for pyxtabond2 v0.4-in-progress

Purpose: confirm, against real Stata xtabond2, the 6 Stata-parity features
added in this pass that had no prior PyXtabond2 baseline to freeze as a
regression fixture: h(#), artests(#), arlevels, per-group gmm() (lag/collapse
per variable), cluster(), plus a re-check of the one-step-robust small-sample
"qc" bugfix (case 6). Cases 7-8 (added later) confirm the fix for a genuinely
unbalanced panel (estimator.py previously crashed outright on any missing
(id,time) row) against real Stata, using Stata 17 (StataMP-64) run directly
from this repo in batch mode rather than a manual round-trip.

STATUS (after a real Stata run of this exact do-file was compared back):
cases 1b/1c/2/3/4/6 matched Stata to the reported precision on first try.
Case 1a (h=1) initially matched on coefficients but NOT on SE/AR/Sargan --
root-caused to two stacked bugs in engine.py/estimator.py's sig2 (the
error-variance scale used for SE/Sargan/AR): the divisor was hardcoded to 2
regardless of h, and the residuals/count it sums were hardcoded to the
diff equation regardless of h. Per xtabond2.mata:201 (`ErrorEq`) and :417,
h==1 must use the *levels*-equation residuals/count with divisor 1, not the
diff-equation ones with divisor 2. Fixed; now matches exactly (see case 1a's
updated comment). Case 3 (arlevels)'s AR(2) had the same class of bug (the
psit cross-term wrongly reused the AR-test's forced-identity H instead of the
actual-h diff-style block per xtabond2.mata::_ARTests) -- also fixed and now
matches exactly. Case 5 (cluster+twostep+robust) still shows a real,
understood-but-unresolved divergence in beta -- see the note above case 5.
A second, distinct bug was later found in the same case's SE/AR (not beta):
the small-sample N_clusters/(N_clusters-1) factor was applied even when
cluster() was explicit, where xtabond2.mata:564 drops it -- fixed (see the
updated case 5 comment) but not yet re-confirmed against a fresh Stata run
(the SE shift is a mechanical, unit-verified consequence of the formula
change, not an independent Stata comparison).

Cases 7-8 (unbalanced panel): both match Stata to 6 decimal places -- see
each case's CONFIRMED note. This also resolves, empirically, an ambiguity
that couldn't be settled from the Mata source alone (case 8's comment):
pyxtabond2's mask_level formula (System GMM's level-equation row inclusion)
needed no change.

How to use
----------
1. Run `python stata_validation/export_dataset.py` to (re)generate
   pyxtabond2_validation.dta next to this file (2000 obs, 200 countries,
   20 regions, already includes L1_Growth precomputed the same way
   PanelData.get_lag() computes it -- L.Growth below should reproduce it).
2. Open this do-file in Stata from the `stata_validation/` folder (or edit
   the `cd`/`use` path below) and run it (requires xtabond2 installed:
   `ssc install xtabond2`).
3. Compare the coefficients/SE/AR/Sargan/Hansen from Stata's log against the
   PyXtabond2 numbers in the comments above each block (also saved in full in
   python_reference_output.txt, produced by compute_python_reference.py).
4. Report back any mismatch beyond ~1e-4 relative -- that pins down exactly
   which of the 6 features (if any) needs a fix, with a concrete numeric
   discrepancy to debug from.

Note: L1_Growth is passed as a precomputed column to PyXtabond2 (Python has
no direct equivalent of Stata's `L.` operator at estimation time), so the
do-file uses L.Growth as the regressor -- these should be numerically
identical since both are "lag of Growth within country, sorted by year".
==============================================================================*/

capture log close
log using "validate_new_features.log", replace text

capture ssc install xtabond2
use "pyxtabond2_validation.dta", clear
xtset country_id Year

* xtabond2's cluster() option requires Mata's speed-favoring mode; space-favoring
* raises r(198) ("cluster() not available in space-favoring mode."). Session-only
* (no `perm`), so this script does not alter the user's persistent Mata setting.
mata: mata set matafavor speed

*-------------------------------------------------------------------------
* 1. h(#) -- h=1 vs h=2 vs h=3 (System GMM, one-step, non-robust, default)
*-------------------------------------------------------------------------
* Python h=1: L1_Growth=0.655754 (se=0.020019)  Capital=1.206416 (se=0.092757)
*             AR(1) z=-19.290549 p=0.000000  AR(2) z=1.758433 p=0.078674
*             Sargan chi2=123.988643 p=0.002397
*             [CONFIRMED vs the Stata run of this case -- matches to reported
*             precision after the sig2/ErrorEq fix described above.]
di as txt _n "=== 1a. h(1) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) h(1)

* Python h=2: L1_Growth=0.650597 (se=0.013985)  Capital=1.236086 (se=0.098418)
*             AR(1) z=-16.027214 p=0.000000  AR(2) z=1.421811 p=0.155081
*             Sargan chi2=140.541324 p=0.000082
di as txt _n "=== 1b. h(2) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) h(2)

* Python h=3 (default): L1_Growth=0.797824 (se=0.010591)  Capital=1.363375 (se=0.108812)
*             AR(1) z=-12.099224 p=0.000000  AR(2) z=1.907831 p=0.056413
*             Sargan chi2=221.400409 p=0.000000
di as txt _n "=== 1c. h(3) (default) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) h(3)

*-------------------------------------------------------------------------
* 2. artests(#) -- generalized AR(l) up to lag 4 (same spec as h=3 above)
*-------------------------------------------------------------------------
* Python: AR(1) z=-12.099224 p=0.000000  AR(2) z=1.907831 p=0.056413
*         AR(3) z=-0.812647 p=0.416421   AR(4) z=0.474350 p=0.635250
di as txt _n "=== 2. artests(4) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) artests(4)

*-------------------------------------------------------------------------
* 3. arlevels -- AR(l) tests on the LEVELS-equation residuals (System GMM only)
*-------------------------------------------------------------------------
* Python: AR(1) z=nan (d<=0, non-positive variance estimate)
*         AR(2) z=5.915896 p=0.000000
*         [CONFIRMED vs the Stata run of this case: AR(1) is *also* missing
*         in Stata's own log (z=. Pr>z=.), i.e. the degenerate case is a real
*         shared feature of this specification, not a bug. AR(2) matches
*         exactly after fixing the psit cross-term (see note above).]
di as txt _n "=== 3. arlevels ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) arlevels

*-------------------------------------------------------------------------
* 4. Per-group gmm() -- different lag/collapse per variable
*-------------------------------------------------------------------------
* Python: L1_Growth=0.693329 (se=0.015821)  Capital=1.016930 (se=0.153057)
*         n_instruments=50  AR(1) z=-8.737335 p=0.000000  AR(2) z=1.851445 p=0.064106
*         Sargan chi2=76.809712 p=0.001162
di as txt _n "=== 4. gmm(Growth, lag(2 4) collapse) + gmm(Capital, lag(2 .)) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth, lag(2 4) collapse) gmm(Capital, lag(2 .)) iv(Ide)

*-------------------------------------------------------------------------
* 5. cluster() -- cluster by region_id (20 clusters) instead of country_id (200)
*-------------------------------------------------------------------------
* Python: L1_Growth=0.618184 (se=0.084294)  Capital=1.296848 (se=0.265196)
*         N_clusters=20  AR(1) z=-5.464193 p=0.000000  AR(2) z=1.101418 p=0.270715
*         Hansen chi2=10.062570 p=1.000000 (very high p is expected here --
*         90 instruments vs only 20 clusters is a severe over-instrumentation
*         ratio, a textbook case of the Hansen test being weakened by
*         instrument proliferation, per Roodman 2009)
*
*         SECOND BUG FOUND AND FIXED (SE/AR only, not beta -- the numbers
*         above already reflect the fix): xtabond2.mata:564 shows the
*         two-step/one-step-robust small-sample correction as
*         `onestepnonrobust ? tmp : (rows(clusts) ? (NObs-1)/(NObs-k) :
*         (NObs-1)/(NObs-k)*NGroups/(NGroups-1))` -- the NGroups/(NGroups-1)
*         cluster-count factor applies ONLY when clustering defaults to the
*         panel id (rows(clusts)==0); an explicit cluster() drops it
*         entirely. engine.py/estimator.py previously applied this factor
*         unconditionally (also when cluster() was explicit), over-correcting
*         the variance here by N_clusters/(N_clusters-1) = 20/19 (SE inflated
*         by sqrt(20/19) =~1.0260x). Fixed via GMMEngine._cluster_qc_factor
*         (returns 1.0 when cluster_ids is not None); the SE ratio old/new
*         matches sqrt(20/19) to the precision shown. Distinct from the
*         divergence below (beta-level, not SE, and still unresolved).
*
*         KNOWN, ROOT-CAUSED DIVERGENCE (still unresolved, beta-level): the
*         Stata log for this case prints "Two-step estimated covariance
*         matrix of moments is singular ... using a generalized inverse" --
*         with 90 instruments and only 20 clusters, the moment covariance
*         matrix S2 is exactly rank <= 20 (confirmed: its singular values
*         drop from ~656 to ~1e-12 between the 20th and 21st, an unambiguous
*         rank of 20). pyxtabond2 uses NumPy's Moore-Penrose pseudo-inverse
*         there; Stata's Mata `invsym` uses a pivoted-elimination generalized
*         inverse that drops specific collinear instrument coordinates
*         instead of projecting via SVD -- both are *valid* generalized
*         inverses of a singular matrix, but not the *same* one, so point
*         estimates (not just SE) can diverge here. Confirmed this is
*         specific to the singular regime, not a cluster() bug: cluster(country_id)
*         (=200 clusters, matching the default) reproduces the no-cluster-argument
*         case exactly (max abs diff = 0.0), and away from this singular case
*         S2 is always full rank. Mitigation (same one Stata's own warning
*         implies): collapse instruments or otherwise keep instrument count
*         <= cluster count when combining twostep/robust with a coarse
*         cluster(). Please still confirm the Stata numbers above match what
*         your run printed, so we know the diagnosis is exactly this and not
*         something else.
di as txt _n "=== 5. cluster(region_id), twostep robust small ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) twostep robust small cluster(region_id)

*-------------------------------------------------------------------------
* 6. One-step robust, small -- confirms the qc bugfix (api.py one-step-robust
*    branch: was N_obs/(N_obs-k), fixed to (N_obs-1)/(N_obs-k) per
*    xtabond2.mata:562-566)
*-------------------------------------------------------------------------
* Python: L1_Growth=0.797824 (se=0.015621)  Capital=1.363375 (se=0.116666)
*         AR(1) z=-9.587813 p=0.000000  AR(2) z=2.077840 p=0.037724
di as txt _n "=== 6. one-step robust, small (qc bugfix check) ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) robust small

*-------------------------------------------------------------------------
* 7. Unbalanced panel -- interior gap (drop country_id==1's Year==5 row)
*
* Confirms the estimator.py crash fix: before it, ANY missing (id,time) row
* (boundary or interior) raised "IndexError: boolean index did not match
* indexed array along axis 0" in _fit_base -- unconditionally, for both
* model types, with or without IFE-GMM. Also exercises the gap-aware
* _lag_group/_diff_group fix in data_utils.py (year 6's D_Growth/D_L1_Growth
* become undefined too, since they need year 5's now-absent level value).
*-------------------------------------------------------------------------
* Python: L1_Growth=0.798186 (se=0.010556)  Capital=1.362369 (se=0.107759)
*         _cons=1.076923 (se=0.414699)  AR(1) z=-12.173331 p=0.000000
*         AR(2) z=1.900417 p=0.057378  Sargan chi2=223.072381 p=0.000000
*         [CONFIRMED vs a real Stata run of this exact do-file: matches
*         Stata's L1.=.7981863 (.0105558) Capital=1.362369 (.1077595)
*         _cons=1.076923 (.4146986) to 6 decimal places, and AR/Sargan to the
*         precision Stata reports. NOTE: getting this match required
*         recomputing L1_Growth *after* dropping the row (not reusing the
*         precomputed column already in pyxtabond2_validation.dta) --
*         reusing the stale, pre-drop value (computed on the full data)
*         understates which rows the gap invalidates, since it silently
*         keeps referencing the now-absent Year==5 at Year==6's row. This
*         matches what Stata's L.Growth does dynamically above, computed
*         fresh after `drop if`+`xtset`. See compute_python_reference.py.]
di as txt _n "=== 7. Unbalanced: interior gap (country_id==1, Year==5 dropped) ==="
preserve
drop if country_id == 1 & Year == 5
xtset country_id Year
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide)
restore

*-------------------------------------------------------------------------
* 8. Unbalanced panel -- late-starting individual (drop country_id==1's
*    Year==0 row, so its own first observed period becomes Year==1)
*
* Settled an ambiguity the Mata source alone couldn't resolve (see the
* comment on mask_level in estimator.py::_fit_base): does the System GMM
* level equation exclude the panel's GLOBAL first calendar period for every
* individual (regardless of when that individual's own data starts), or
* each individual's OWN first observed period specifically? Every other
* validated case so far had every individual start at the panel's global
* t_min, so the two readings were indistinguishable there. pyxtabond2
* implements the GLOBAL reading (mask_level's t_off_g is computed relative
* to the panel's global t_min, exactly like mask_diff) -- deliberately left
* unchanged by the crash fix pending this exact test.
*
* RESOLVED: the coefficients below match a real Stata run of this case to
* 6 decimal places (see the CONFIRMED note), which would not happen if the
* level equation's row-inclusion differed between the two implementations --
* directly confirming the GLOBAL reading is correct. No code change needed.
*-------------------------------------------------------------------------
* Python: L1_Growth=0.793167 (se=0.010819)  Capital=1.386876 (se=0.107577)
*         _cons=1.032555 (se=0.415313)  AR(1) z=-12.050835 p=0.000000
*         AR(2) z=1.786445 p=0.074027  Sargan chi2=222.754517 p=0.000000
*         [CONFIRMED vs a real Stata run of this exact do-file: matches
*         Stata's L1.=.7931669 (.0108189) Capital=1.386876 (.1075773)
*         _cons=1.032555 (.4153134) to 6 decimal places, and AR/Sargan to
*         the precision Stata reports. Same L1_Growth-recompute-after-drop
*         note as case 7 applies here.]
di as txt _n "=== 8. Unbalanced: late start (country_id==1, Year==0 dropped) ==="
preserve
drop if country_id == 1 & Year == 0
xtset country_id Year
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide)
restore

*-------------------------------------------------------------------------
* 9. Bug 4 closure -- explicit cluster() at the SAME grouping as the
*    default (panel id), away from the singular-S2 regime of Case 5.
*
* Case 5 (cluster(region_id), 20 clusters vs 90 instruments) sits inside the
* singular-moment-matrix regime, so it cannot cleanly isolate the qc-factor
* fix (GMMEngine._cluster_qc_factor) from the separate, unresolved
* generalized-inverse divergence documented there. This case avoids that
* confound entirely: cluster(country_id) has 200 clusters against ~90
* instruments (same as the no-cluster() case below), so S2 is always full
* rank.
*
* ORIGINAL HYPOTHESIS (wrong): that xtabond2.mata:564's rows(clusts) branch
* keys off whether cluster() was passed at all, so 9b (explicit
* cluster(country_id)) would drop the N_clusters/(N_clusters-1) factor that
* 9a (default clustering) keeps, making 9b's SE smaller by sqrt(199/200).
*
* CONFIRMED vs a real Stata run of this exact do-file (twice, on two
* different machines): 9a and 9b are bit-identical on every coefficient AND
* every SE (both se(L1.)=.0226018, etc. -- see the log). rows(clusts) does
* not key off "was cluster() explicit"; it must key off whether the
* clustering variable induces a *different partition* than the panel id --
* an explicit cluster() naming the same variable as the panel id still gets
* the factor. GMMEngine._cluster_qc_factor implements exactly this
* (_same_partition() check, not a bare "is cluster_ids None" check) --
* re-running compute_python_reference.py after that fix reproduces Stata's
* 9a==9b exactly (se=0.022602 for L1_Growth in both; the previously
* committed python_reference_output.txt was stale, predating the
* _same_partition fix, and has been regenerated).
*
* (Stata's own Hansen/AR(1) test statistics do move a hair between 9a/9b --
* chi2=127.21 vs 126.58, z=-8.34 vs -8.33 -- so the two runs are not
* completely inert internally; only the reported coefficient/SE are pinned
* to be identical.)
*-------------------------------------------------------------------------
di as txt _n "=== 9a. No cluster() (default = panel id), twostep robust small ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) twostep robust small

di as txt _n "=== 9b. cluster(country_id) explicit (same grouping as default), twostep robust small ==="
xtabond2 Growth L.Growth Capital Labor Wage Investment Ide, ///
    gmm(Growth Capital, lag(2 .)) iv(Ide) twostep robust small cluster(country_id)

log close
