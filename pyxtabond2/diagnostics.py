"""
Difference-in-Sargan/Hansen instrument-exogeneity tests.

Re-estimates the model excluding one instrument subset at a time (e.g. the
System GMM "instruments for levels", or a standard ``iv()`` block) and
compares its overidentification statistic to the full-instrument-set
statistic, following Stata's ``xtabond2`` (Roodman, 2009).
"""

import numpy as np
import scipy.stats as stats

from .engine import GMMEngine, _accumulate_ZeZe


def compute_diff_sargan_tests(engine, *, model_type, iv_vars, twostep, robust, small, orthogonal,
                               stat_full_c, df_full_c, h=3):
    """
    Run the Diff-in-Sargan/Hansen tests for the "GMM instruments for levels"
    (System GMM only) and "iv(...)" instrument groups.

    Parameters
    ----------
    engine : GMMEngine
        The fitted engine for the full instrument set.
    stat_full_c : float
        Overidentification statistic (Sargan or Hansen, as applicable) for
        the full instrument set.
    df_full_c : int
        Its degrees of freedom (n_instruments - k_vars).

    Returns
    -------
    list of dict
        One entry per tested instrument group, with keys ``name``,
        ``stat_rest``, ``df_rest``, ``p_rest``, ``diff_stat``, ``diff_df``,
        ``diff_p``.
    """
    diff_sargan_results = []

    def compute_c_stat(exclude_mask, test_name):
        if not np.any(exclude_mask):
            return

        Z_rest = engine.Z[:, ~exclude_mask]
        if Z_rest.shape[1] < engine.k_vars:
            return

        try:
            engine_rest = GMMEngine(
                engine.y, engine.X, Z_rest,
                group_ids=engine.group_ids, is_level=engine.is_level,
                small=small, orthogonal=orthogonal, h=h,
                t_index=getattr(engine, 't_index', None), T_span=getattr(engine, 'T_span', None),
                y_lvl=getattr(engine, 'y_lvl', None), X_lvl=getattr(engine, 'X_lvl', None),
                cluster_ids=getattr(engine, 'cluster_ids', None),
                # engine.y/X/Z are already NaN-filtered, and Z_rest is a column
                # subset of the same rows, so the group structure carries over unchanged.
                group_structure=(engine._unique_groups, engine._group_masks),
                # The full model was already identification-checked; removing a
                # subset of instruments here only for a diagnostic refit isn't
                # worth an extra full-Z rank SVD (the column-count guard above
                # already catches the common under-identified case).
                check_rank=False,
            )
        except ValueError as e:
            if "under-identified" in str(e).lower():
                return
            raise e

        df_rest = engine_rest.n_instruments - engine_rest.k_vars
        if df_rest < 0:
            return

        if twostep or robust:
            S_rest = _accumulate_ZeZe(engine_rest.Z, engine.e1, engine_rest._cluster_masks)
            try:
                W2_rest = np.linalg.pinv(S_rest)
            except np.linalg.LinAlgError:
                return

            Zy_rest = engine_rest.Z.T @ engine_rest.y
            XZ_rest = engine_rest.X.T @ engine_rest.Z
            XZ_W2_ZX_rest = XZ_rest @ W2_rest @ XZ_rest.T
            V2_rest = np.linalg.pinv((XZ_W2_ZX_rest + XZ_W2_ZX_rest.T) / 2.0)
            beta2_rest = V2_rest @ XZ_rest @ W2_rest @ Zy_rest
            e2_rest = engine_rest.y - engine_rest.X @ beta2_rest
            stat_rest = (e2_rest.T @ engine_rest.Z @ W2_rest @ engine_rest.Z.T @ e2_rest)[0, 0]
        else:
            engine_rest.estimate_one_step()
            sig2_full = getattr(engine, 'sig2_v_1step', 1.0)
            stat_rest = (engine_rest.e1.T @ engine_rest.Z @ engine_rest.W1 @ engine_rest.Z.T @ engine_rest.e1)[0, 0] / sig2_full

        diff_stat = stat_full_c - stat_rest
        diff_df = df_full_c - df_rest
        diff_p = 1.0 - stats.chi2.cdf(diff_stat, diff_df) if diff_df > 0 else np.nan

        diff_sargan_results.append({
            'name': test_name,
            'stat_rest': stat_rest, 'df_rest': df_rest,
            'p_rest': 1.0 - stats.chi2.cdf(stat_rest, df_rest) if df_rest > 0 else np.nan,
            'diff_stat': diff_stat, 'diff_df': diff_df, 'diff_p': diff_p,
        })

    # --- Test 1: "GMM instruments for levels" (System only) ---
    if model_type == 'system':
        diff_mask = ~engine.is_level
        level_mask = engine.is_level
        is_level_inst = np.all(np.abs(engine.Z[diff_mask, :]) < 1e-10, axis=0)
        is_cons_inst = is_level_inst & np.all(np.abs(engine.Z[level_mask, :] - 1.0) < 1e-10, axis=0)
        is_gmm_level_inst = is_level_inst & ~is_cons_inst
        compute_c_stat(is_gmm_level_inst, 'GMM instruments for levels')

    # --- Test 2: "iv(x)" (standard instruments) ---
    if iv_vars:
        n_iv = len(iv_vars)
        iv_mask = np.zeros(engine.Z.shape[1], dtype=bool)

        # IV instruments are always appended contiguously at the end of the
        # Z matrix. In System GMM, they are followed by one more, auto-
        # appended '_cons' instrument column -- but ONLY if the user did not
        # already include '_cons' in iv_vars themselves (estimator.py skips
        # the auto-append in that case, exactly mirrored here). Assuming the
        # auto-column unconditionally would silently test the wrong columns
        # whenever a user instruments the constant explicitly (e.g. Stata's
        # iv(_cons)).
        auto_cons_appended = (model_type == 'system') and ('_cons' not in iv_vars)
        if auto_cons_appended:
            iv_mask[-(n_iv + 1):-1] = True
        else:
            iv_mask[-n_iv:] = True

        iv_names = " ".join(iv_vars)
        compute_c_stat(iv_mask, f'iv({iv_names})')

    return diff_sargan_results
