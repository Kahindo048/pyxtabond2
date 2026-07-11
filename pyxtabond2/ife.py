"""
Interactive Fixed Effects GMM (PCA-GMM).

Implements the iterative defactoring loop (residuals -> SVD -> factor space
projection -> re-estimate) and the split-panel jackknife bias correction
(Dhaene & Jochmans) layered on top of the base Arellano-Bond/Blundell-Bond
GMM engine. See :class:`~pyxtabond2.numfac` for the factor-count selection
criteria (Bai & Ng, 2002; Ahn & Horenstein, 2013).
"""

import contextlib
import os

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats as stats

from .numfac import estimate_num_factors, show_factor_selection, em_pca_imputation


def _residuals_from_beta(df: pd.DataFrame, dep_var: str, x_vars: list, x_names: list, beta: np.ndarray) -> np.ndarray:
    """Level residuals y - X*beta as a 1d numpy array."""
    resid = df[dep_var].to_numpy(dtype=np.float64, copy=True)
    beta_map = dict(zip(x_names, beta.flatten()))
    for x_name in x_vars:
        if x_name in beta_map:
            resid -= beta_map[x_name] * df[x_name].to_numpy(dtype=np.float64, copy=False)
    if '_cons' in beta_map:
        resid -= beta_map['_cons']
    return resid


def _codes_to_panel_matrix(id_codes: np.ndarray, time_codes: np.ndarray, values: np.ndarray,
                           n_ids: int, n_times: int) -> np.ndarray:
    """Scatter long panel values into an (N x T) matrix; unobserved cells are NaN."""
    mat = np.full((n_ids, n_times), np.nan, dtype=np.float64)
    mat[id_codes, time_codes] = values
    return mat


def _impute_gaps(mat: np.ndarray, kmax: int) -> np.ndarray:
    """One-shot EM-PCA fill of unobserved (NaN) panel cells; a no-op on balanced panels."""
    if not np.isnan(mat).any():
        return mat
    return em_pca_imputation(mat, kmax=kmax)


def _panel_matrix_to_codes(id_codes: np.ndarray, time_codes: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Extract long vector from (N x T) matrix using precomputed factor codes."""
    return mat[id_codes, time_codes]


class IFEMixin:
    """
    Adds PCA-GMM / IFE-GMM estimation to :class:`~pyxtabond2.estimator.PyXtabond2`.

    Expects the host class to provide ``_fit_base`` (classic GMM estimation)
    and the constructor attributes set by ``PyXtabond2.__init__``.
    """

    def _fit_ife(self, beta_warm_start=None, is_subpanel=False):
        """
        Iterative PCA-GMM engine: alternates defactoring (residuals -> SVD ->
        projection onto the factor-orthogonal space) with GMM re-estimation,
        plus an optional split-panel Jackknife bias correction (Dhaene & Jochmans).

        Starts from the plain (non-defactored) GMM fit rather than a
        consistent-under-factors initial estimator; see the ``r`` and
        ``bias_correction`` parameter docs on :class:`~pyxtabond2.estimator.PyXtabond2`
        for the convergence-guarantee and bias caveats this implies.
        """
        if not is_subpanel:
            print(f"\nStarting IFE-GMM estimation")

        # 1. Keep the raw data and a working copy to project in place.
        df_raw = self.df.copy()
        df_proj = self.df.copy()

        id_codes, _ = pd.factorize(df_raw[self.id_col], sort=False)
        time_codes, _ = pd.factorize(df_raw[self.time_col], sort=False)
        n_ids = int(id_codes.max()) + 1
        n_times = int(time_codes.max()) + 1

        # --- INITIALIZATION AND WARM START ---
        choix_r_utilisateur = self.r
        self.r = 0

        # res_iter is always needed for its structure and variable names, but
        # not (yet) for its diagnostics -- full_diagnostics=False skips the
        # Sargan/Hansen/AR/Wald/Diff-Sargan block, which would otherwise be
        # thrown away immediately if r>0 triggers the loop below.
        res_iter = self._fit_base(df_raw, self.dep_var, full_diagnostics=False)
        # df_proj is a value-only mutated copy of df_raw (same rows/order):
        # every subsequent _fit_base call in this loop can reuse the sort
        # order and group slices computed here instead of re-running pandas'
        # sort_values/dropna/groupby on every IFE iteration.
        panel_cache = self._last_panel

        if beta_warm_start is not None:
            # WARM START: inject the near-perfect solution directly.
            beta_old = beta_warm_start.copy()
        else:
            # COLD START: classic GMM.
            beta_old = res_iter.beta.flatten()

        # --- AUTOMATIC SELECTION OR CONVERSION OF THE FACTOR COUNT ---
        if str(choix_r_utilisateur).lower() == 'auto':
            resid_temp = _residuals_from_beta(df_raw, self.dep_var, self.x_vars, res_iter.x_names, beta_old)
            E_mat = _codes_to_panel_matrix(id_codes, time_codes, resid_temp, n_ids, n_times)
            self.fac_results = estimate_num_factors(E_mat, kmax=getattr(self, 'r_max', 5))
            self.r = self.fac_results['best_er']
            if not is_subpanel:
                print(f">> Optimal number of factors estimated: r = {self.r}")
        else:
            self.r = int(choix_r_utilisateur)

        if self.r == 0:
            if not is_subpanel:
                print(">> r=0 detected. Fallback to standard GMM.")
                # The light call above skipped diagnostics; this is the
                # result actually returned to the user, so recompute it once
                # with full_diagnostics=True (same df_raw -> identical beta).
                res_iter = self._fit_base(df_raw, self.dep_var, full_diagnostics=True)
            return res_iter

        if not is_subpanel: print(f">> IFE-GMM algorithm configured with r = {self.r}")

        # --- PRE-COMPUTE PANEL MATRICES ---
        # Built once, so any missing cell (a genuine unbalanced-panel gap, or
        # e.g. a lagged regressor's undefined first observation per id) gets
        # the full iterative EM-PCA fill here (a no-op when nothing is
        # missing). kmax=self.r: the model already assumes Y and every
        # defactored X/instrument share the same r-factor structure, since
        # they're all projected by the same M_F derived from Y's residuals.
        Y_mat_raw = _impute_gaps(
            _codes_to_panel_matrix(id_codes, time_codes, df_raw[self.dep_var].to_numpy(), n_ids, n_times),
            kmax=self.r,
        )

        X_mats_raw = {}
        for x_name in self.x_vars:
            mat = _codes_to_panel_matrix(id_codes, time_codes, df_raw[x_name].to_numpy(), n_ids, n_times)
            X_mats_raw[x_name] = _impute_gaps(mat, kmax=self.r)

        other_vars_mats = {}
        for v_name in set(self.gmm_vars + self.iv_vars):
            if v_name not in X_mats_raw and v_name != self.dep_var and v_name != '_cons':
                mat = _codes_to_panel_matrix(id_codes, time_codes, df_raw[v_name].to_numpy(), n_ids, n_times)
                other_vars_mats[v_name] = _impute_gaps(mat, kmax=self.r)

        diff = np.inf
        last_reconstruction = None
        warned_missing = False

        # --- PCA-GMM OPTIMIZATION LOOP ---
        for iteration in range(1, self.ife_max_iter + 1):
            # Step A: residuals (always based on the raw data).
            resid_ife = _residuals_from_beta(df_raw, self.dep_var, self.x_vars, res_iter.x_names, beta_old)
            E_mat = _codes_to_panel_matrix(id_codes, time_codes, resid_ife, n_ids, n_times)

            # nan_mask catches both a genuinely unbalanced panel (id x time
            # cells absent from df_raw) and e.g. a lagged regressor's
            # undefined first observation per id (present cell, undefined
            # residual) -- both leave E_mat with the same kind of hole, fixed
            # across iterations since it's driven by df_raw, not beta.
            nan_mask = np.isnan(E_mat)
            if nan_mask.any():
                if not warned_missing and not is_subpanel:
                    print(f">> {int(nan_mask.sum())} panel cell(s) per iteration have no defined "
                          f"residual (lag truncation and/or an unbalanced panel); filling via "
                          f"the evolving rank-{self.r} reconstruction (cross-sectional mean on "
                          f"the first iteration).")
                    warned_missing = True
                # EM-style fill: gaps get the previous iteration's rank-r
                # reconstruction (cross-sectional mean on the first iteration,
                # when none exists yet) instead of a nested EM-PCA sub-loop --
                # this outer loop already *is* the EM alternation.
                E_mat[nan_mask] = (
                    last_reconstruction[nan_mask] if last_reconstruction is not None
                    else np.nanmean(resid_ife)
                )

            # Step B: SVD factor extraction.
            U, S, Vt = scipy.linalg.svd(E_mat, full_matrices=False)
            F = Vt[:self.r, :].T * np.sqrt(n_times)

            if nan_mask.any():
                last_reconstruction = U[:, :self.r] @ np.diag(S[:self.r]) @ Vt[:self.r, :]

            # Step C: temporal projection matrix M_F (T x T).
            M_F = np.eye(n_times) - (F @ F.T) / n_times

            # Step D: projection (column replace; safe under pandas Copy-on-Write).
            Y_proj_mat = Y_mat_raw @ M_F
            df_proj[self.dep_var] = _panel_matrix_to_codes(id_codes, time_codes, Y_proj_mat)

            for x_name, mat in X_mats_raw.items():
                df_proj[x_name] = _panel_matrix_to_codes(id_codes, time_codes, mat @ M_F)

            for v_name, mat in other_vars_mats.items():
                df_proj[v_name] = _panel_matrix_to_codes(id_codes, time_codes, mat @ M_F)

            # Step E: GMM estimation (diagnostics deferred to after the loop --
            # only .beta is needed here to test convergence).
            res_iter = self._fit_base(df_proj, self.dep_var, panel_cache=panel_cache, full_diagnostics=False)
            beta_new = res_iter.beta.flatten()

            # Step F: convergence check.
            diff = np.sum(np.abs(beta_new - beta_old))

            if not is_subpanel:
                print(f"Iteration {iteration:02d} | Convergence delta: {diff:.7f}")

            if diff < self.ife_tol:
                if not is_subpanel: print(f"IFE-GMM convergence achieved at iteration {iteration}!")
                self.iteration = iteration
                break

            beta_old = beta_new

        if diff >= self.ife_tol and not is_subpanel:
            print(f"Warning: convergence not achieved after {self.ife_max_iter} iterations.")

        if not is_subpanel:
            # The loop ran entirely with full_diagnostics=False; df_proj still
            # holds the exact projection that produced this res_iter/beta_new
            # (nothing mutates it after the loop), so refitting on it here is
            # deterministic -- identical beta, now with diagnostics populated.
            # Subpanel calls (is_subpanel=True) skip this: their caller only
            # ever reads .beta.flatten().
            res_iter = self._fit_base(df_proj, self.dep_var, panel_cache=panel_cache, full_diagnostics=True)

        res_iter.dep_var = self.dep_var

        if not is_subpanel and self.bias_correction:
            print(">> Applying Split-Panel Jackknife bias correction...")

            df_backup = self.df
            r_backup = self.r

            try:
                # Access the time column directly, without assuming a MultiIndex.
                t_min = df_raw[self.time_col].min()
                t_max = df_raw[self.time_col].max()
                t_mid = t_min + (t_max - t_min) // 2

                mask_t1 = df_raw[self.time_col] <= t_mid
                mask_t2 = df_raw[self.time_col] > t_mid

                df_t1 = df_raw[mask_t1].copy()
                df_t2 = df_raw[mask_t2].copy()

                def _estimate_half(df_half):
                    self.df = df_half
                    self.r = r_backup
                    # Redirect stdout so the sub-estimation doesn't clutter the console.
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        res_half = self._fit_ife(beta_warm_start=beta_new, is_subpanel=True)
                    return res_half.beta.flatten()

                beta_t1 = _estimate_half(df_t1)
                beta_t2 = _estimate_half(df_t2)

                self.df = df_backup
                self.r = r_backup

                # Dhaene & Jochmans split-panel jackknife formula.
                beta_bc = 2.0 * beta_new - 0.5 * (beta_t1 + beta_t2)

                # --- FULL INFERENCE UPDATE ---
                res_iter.beta = beta_bc.reshape(-1, 1)
                res_iter.t_stats = res_iter.beta / res_iter.se

                if res_iter.small:
                    res_iter.p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(res_iter.t_stats), res_iter.df_resid))
                else:
                    res_iter.p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(res_iter.t_stats)))

                # Confidence-interval correction (shift only, SE unchanged).
                res_iter.ci_lower = res_iter.beta - res_iter.ci_crit * res_iter.se
                res_iter.ci_upper = res_iter.beta + res_iter.ci_crit * res_iter.se

                print(f">> Bias correction applied successfully.")

            except Exception as e:
                import traceback
                print(f">> Warning: Jackknife correction failed: {e}")
                traceback.print_exc()
                print(">> Returning uncorrected estimates.")
                self.df = df_backup
                self.r = r_backup

        return res_iter

    def plot_factor_selection(self, mode='graph'):
        """
        Displays the table or plot for factor selection criteria.
        Only functions if the model was estimated with r='auto'.
        """
        if not hasattr(self, 'fac_results'):
            print("Warning: cannot display criteria: IFE-GMM estimation with r='auto' has not been executed yet.")
            return

        return show_factor_selection(self.fac_results, mode=mode)
