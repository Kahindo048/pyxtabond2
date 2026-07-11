"""
GMM estimation engine for dynamic panel models.

Implements one-step and two-step GMM, the Windmeijer (2005) variance correction,
and diagnostic tests (Sargan, Hansen, Arellano-Bond AR) following the algorithm
in Stata's ``xtabond2`` (Roodman, 2009).
"""

import numpy as np
import scipy.stats as stats
import scipy.linalg

from .data_utils import scatter_to_grid


def _build_H_diff_block(T_span: int) -> np.ndarray:
    """Stata h=3 differenced-equation covariance block."""
    H_diff = np.eye(T_span, dtype=np.float64) * 2.0
    if T_span > 1:
        i = np.arange(T_span - 1)
        H_diff[i, i + 1] = -1.0
        H_diff[i + 1, i] = -1.0
    H_diff[0, 0] = 1.0
    return H_diff


def _build_H_full_system(T_span: int, orthogonal: bool) -> np.ndarray:
    """Full 2T x 2T H matrix for system GMM."""
    H_full = np.eye(2 * T_span, dtype=np.float64)
    if orthogonal:
        for t_d in range(1, T_span):
            t_source = t_d - 1
            future_count = T_span - 1 - t_source
            if future_count > 0:
                c_val = np.sqrt(future_count / (future_count + 1.0))
                H_full[t_d, T_span + t_source] = c_val
                H_full[T_span + t_source, t_d] = c_val
                inv_fc = -c_val / future_count
                for k in range(t_source + 1, T_span):
                    H_full[t_d, T_span + k] = inv_fc
                    H_full[T_span + k, t_d] = inv_fc
    else:
        H_diff = _build_H_diff_block(T_span)
        H_full[:T_span, :T_span] = H_diff
        i = np.arange(T_span)
        H_full[i, T_span + i] = 1.0
        H_full[T_span + i, i] = 1.0
        if T_span > 1:
            i2 = np.arange(T_span - 1)
            H_full[i2 + 1, T_span + i2] = -1.0
            H_full[T_span + i2, i2 + 1] = -1.0
    return H_full


def _build_H(h: int, T_span: int, orthogonal: bool) -> np.ndarray:
    """
    General h(#) weighting-matrix construction, matching Stata's
    ``xtabond2.mata::_H``. Always returns the full 2*T_span x 2*T_span
    matrix; for a Difference GMM model only the top-left T_span x T_span
    block is ever indexed into (via ``t_index``), so this single
    construction serves both model types -- exactly as the existing h=3
    default already did via :func:`_build_H_full_system`.

    h=3 (default) models the known cross-covariance between the
    differenced/orthogonal-deviations equation and the levels equation.
    h=2 assumes that covariance is zero (block-diagonal). h=1 assumes no
    serial-correlation structure at all (identity). h=2 and h=3 coincide
    for a pure Difference GMM model.
    """
    if h == 1:
        return np.eye(2 * T_span, dtype=np.float64)
    if h == 3:
        return _build_H_full_system(T_span, orthogonal)

    # h == 2: zero out the h=3 cross-covariance between the two blocks.
    top = np.eye(T_span, dtype=np.float64) if orthogonal else _build_H_diff_block(T_span)
    H_full = np.eye(2 * T_span, dtype=np.float64)
    H_full[:T_span, :T_span] = top
    return H_full


def _build_H_diff(n_diff: int) -> np.ndarray:
    H = np.eye(n_diff, dtype=np.float64) * 2.0
    if n_diff > 1:
        i = np.arange(n_diff - 1)
        H[i, i + 1] = -1.0
        H[i + 1, i] = -1.0
    return H


def _xform_matrix(xform_type: int, T: int) -> np.ndarray:
    M = np.eye(T, dtype=np.float64)
    if xform_type == 1:
        M_lag = np.zeros((T, T), dtype=np.float64)
        M_lag[:-1, :] = M[1:, :]
        M = M_lag
        for r in range(T - 1, 0, -1):
            M[r:, r] = -1.0 / (T - r)
        col_norms = np.sqrt(np.sum(M ** 2, axis=0))
        col_norms[col_norms == 0] = 1.0
        return M / col_norms
    M_lag = np.zeros((T, T), dtype=np.float64)
    M_lag[:-1, :] = M[1:, :]
    M = M - M_lag
    M[0, 0] = 0.0
    return M


def _inv_or_pinv(S: np.ndarray) -> np.ndarray:
    """
    Inverse of a matrix expected to be generically full-rank, falling back to
    the (much slower, SVD-based) Moore-Penrose pseudo-inverse if it isn't.

    ``np.linalg.inv`` (LU) is markedly cheaper than ``np.linalg.pinv`` (SVD)
    and gives the same result to machine precision for an invertible matrix.
    Only use this for matrices like ``S1`` in ``_compute_W1`` (a sum of
    per-group quadratic forms Z_g'HZ_g, generically full rank) -- NOT for
    matrices that are structurally rank-deficient whenever instruments
    outnumber groups/clusters (e.g. ``_accumulate_ZeZe`` outputs used in the
    two-step/robust/cluster() weighting matrices), where ``inv`` would just
    fail (wasting the LU attempt) before falling back to the same ``pinv``
    already used there directly.
    """
    try:
        return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S)


def _accumulate_ZeZe(Z: np.ndarray, e: np.ndarray, group_masks: list) -> np.ndarray:
    """
    Accumulate group-wise outer products Z_g' e_g e_g' Z_g.

    Uses the identity Z' e e' Z = (Z' e)(Z' e)' for numerical efficiency.
    Called internally by the engine and the high-level API.
    """
    n_instr = Z.shape[1]
    S = np.zeros((n_instr, n_instr), dtype=np.float64)
    e_flat = e.reshape(-1)
    for mask in group_masks:
        Ze = Z[mask].T @ e_flat[mask]
        S += np.outer(Ze, Ze)
    return S


class GMMEngine:
    """
    Generalized Method of Moments (GMM) estimation engine for dynamic panels.
    Replicates the matrix logic, Windmeijer finite-sample correction, and
    diagnostics of Stata's `xtabond2`.

    Parameters
    ----------
    check_rank : bool, optional
        If False, skips the (SVD-based) instrument-rank identification check
        and trusts the caller. Stata's own Diff-in-Sargan computation
        (``xtabond2.mata``) does not re-check identification when refitting
        on an instrument subset either; use this for secondary/diagnostic
        refits of an already-validated specification, not for primary estimation.
    group_structure : tuple(np.ndarray, list), optional
        Precomputed ``(unique_groups, group_masks)`` to reuse instead of
        recomputing them. Only valid when ``y``/``X``/``Z`` come from a prior
        :class:`GMMEngine` with the *same* ``group_ids`` and no additional
        NaN-driven row filtering occurs here (e.g. Z is a column subset of an
        already NaN-filtered Z).
    h : int, optional
        Stata's ``h()`` option (1, 2, or 3) selecting the one-step weighting
        matrix / GLS structure. Default 3 (Stata's default). Also threaded
        through the Sargan and Arellano-Bond AR(l) test statistics, matching
        ``xtabond2.mata``'s ``_H``/``_ARTests``.
    artests : int, optional
        Highest-order Arellano-Bond AR(l) serial-correlation test to compute
        (Stata's ``artests()`` option, default 2, i.e. AR(1) and AR(2)).
    arlevels : bool, optional
        If True, the AR(l) tests target the *levels* residuals instead of
        the differenced/orthogonal-deviations residuals (Stata's
        ``arlevels`` option), with the AR-test weighting forced to the
        identity regardless of ``h``. Only valid for System GMM
        (``is_level`` must contain at least one True).
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, Z: np.ndarray,
                 group_ids: np.ndarray = None, is_level: np.ndarray = None,
                 small: bool = False, orthogonal: bool = False,
                 t_index: np.ndarray = None, T_span: int = None,
                 y_lvl: list = None, X_lvl: list = None, r: int = 0,
                 check_rank: bool = True, group_structure: tuple = None,
                 h: int = 3, artests: int = 2, arlevels: bool = False,
                 cluster_ids: np.ndarray = None):

        if h not in (1, 2, 3):
            raise ValueError(f"h({h}) invalid: must be 1, 2, or 3.")
        if artests < 1:
            raise ValueError(f"artests({artests}) invalid: must be >= 1.")
        if arlevels and not (is_level is not None and np.any(is_level)):
            raise ValueError("arlevels is only valid for System GMM (invalid for Difference GMM).")

        self.small = small
        self.orthogonal = orthogonal
        self.T_span = T_span
        self.h = h
        self.artests = artests
        self.arlevels = arlevels
        self.y_lvl = y_lvl
        self.X_lvl = X_lvl
        self.r = r

        valid_rows = ~np.isnan(y).flatten() & ~np.isnan(X).any(axis=1) & ~np.isnan(Z).any(axis=1)
        
        self.y = y[valid_rows]
        self.X = X[valid_rows]
        self.Z = Z[valid_rows]
        
        if group_ids is not None:
            self.group_ids = group_ids[valid_rows]
        if is_level is not None:
            self.is_level = is_level[valid_rows]
        if t_index is not None:
            self.t_index = t_index[valid_rows]
        if cluster_ids is not None:
            cluster_ids = cluster_ids[valid_rows]
        self.cluster_ids = cluster_ids

        non_zero_cols = ~np.all(self.Z == 0, axis=0)
        if np.any(~non_zero_cols):
            self.Z = self.Z[:, non_zero_cols]
            
        self.n_obs, self.k_vars = self.X.shape
        self.n_instruments = self.Z.shape[1]

        if check_rank:
            # rank(Z) == rank(Z'Z); computing it via the (n_instruments x
            # n_instruments) Gram matrix is far cheaper than SVD-ing the full
            # (n_obs x n_instruments) Z when n_obs >> n_instruments (the
            # common case -- this rank check dominated profiled runtime on
            # large-T panels). hermitian=True uses eigh, faster and more
            # numerically stable than a generic SVD for a symmetric PSD
            # matrix. Empirically (synthetic sweep of the smallest singular
            # value from 1 to 1e-16) the two computations disagree only in an
            # extremely narrow band (~1e-7 to 1e-11 here), where the
            # Gram-matrix version is always the more conservative one (flags
            # rank-deficiency slightly earlier), never the more permissive one.
            self.z_rank = np.linalg.matrix_rank(self.Z.T @ self.Z, hermitian=True)
            if self.z_rank < self.k_vars:
                raise ValueError(f"Under-identified model: {self.z_rank} valid instruments for {self.k_vars} regressors.")
        else:
            self.z_rank = self.n_instruments

        if group_structure is not None:
            self._unique_groups, self._group_masks = group_structure
        else:
            self._init_group_structure()
        self.N_groups = len(self._unique_groups) if self.group_ids is not None else 1

        # Clustering for robust/two-step inference (meat matrix, Windmeijer D,
        # small-sample qc factors, Wald/F df). Defaults to the panel-id
        # grouping already used for instruments (Stata's default when no
        # cluster() is given) -- distinct from `_group_masks`/`N_groups` above,
        # which stay panel-id-based even when `cluster_ids` differs, since
        # they drive the instrument/H-matrix construction, not inference.
        if cluster_ids is not None:
            unique_clusters = np.unique(cluster_ids)
            self._cluster_masks = [(cluster_ids == c) for c in unique_clusters]
            self.N_clusters = len(unique_clusters)
        else:
            self._cluster_masks = self._group_masks
            self.N_clusters = self.N_groups

        if self.r > 0:
            self.df_pca = self.r * (self.N_groups + self.T_span - self.r)
        else:
            self.df_pca = 0

        self._H_full_system = None
        self._Ml_orth = None
        self._Mr_orth = None

        self.W1 = self._compute_W1()

    def _cluster_qc_factor(self) -> float:
        """
        Small-sample cluster-count factor N_clusters/(N_clusters-1) used in
        the two-step/one-step-robust variance and AR-test corrections.

        xtabond2.mata:564 applies this factor whenever the clustering
        variable induces the *same partition* of the sample as the default
        panel id -- whether that is because ``cluster()`` was left
        unspecified (Stata's ``rows(clusts)==0`` branch, so clustering
        defaults to the panel id) or because an explicit ``cluster()`` was
        passed but happens to match the panel id one-for-one (e.g.
        ``cluster(country_id)`` when the panel is already indexed by
        country). The factor is dropped only when the clustering variable
        genuinely coarsens the panel into a *different* set of groups
        (Stata's ``rows(clusts)!=0`` branch with ``clusts`` distinct from
        the panel id), leaving just the (NObs-1)/(NObs-k) piece.

        An earlier version of this method treated *any* explicit
        ``cluster()`` argument as sufficient to drop the factor, which
        over-corrected the variance whenever the user's ``cluster()``
        variable happened to coincide with the panel id. This was caught by
        a dedicated Stata validation case (``stata_validation`` cases
        9a/9b): ``cluster(country_id)`` on a panel id'd by ``country_id``
        reproduces Stata's no-``cluster()`` standard errors exactly, not
        the factor-free ones this method used to return.
        """
        group_ids = getattr(self, 'group_ids', None)
        if (self.cluster_ids is None or group_ids is None
                or self._same_partition(self.cluster_ids, group_ids)):
            return self.N_clusters / (self.N_clusters - 1.0)
        return 1.0

    @staticmethod
    def _same_partition(a, b) -> bool:
        """
        True iff two equal-length label arrays induce the same partition of
        the observations into groups (same equivalence classes), regardless
        of label naming or order. Used to detect whether an explicit
        ``cluster()`` grouping coincides with the panel id.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape[0] != b.shape[0]:
            return False
        n_a = len(np.unique(a))
        n_b = len(np.unique(b))
        if n_a != n_b:
            return False
        pairs = np.stack([a, b], axis=1)
        return len(np.unique(pairs, axis=0)) == n_a

    def _init_group_structure(self) -> None:
        if self.group_ids is None:
            self._unique_groups = np.array([0], dtype=np.int64)
            self._group_masks = [np.ones(self.n_obs, dtype=bool)]
            return
        self._unique_groups = np.unique(self.group_ids)
        self._group_masks = [(self.group_ids == g) for g in self._unique_groups]

    def _get_H_full_system(self) -> np.ndarray:
        if self._H_full_system is None:
            self._H_full_system = _build_H(self.h, self.T_span, self.orthogonal)
        return self._H_full_system

    def _get_orthogonal_xforms(self):
        if self._Ml_orth is None:
            T = self.T_span
            self._Ml_orth = _xform_matrix(0, T)
            self._Mr_orth = _xform_matrix(1, T)
        return self._Ml_orth, self._Mr_orth
        
    def _compute_W1(self):
        """
        Computes the weighting matrix A1 with surgical time alignment.
        Replicates Stata's underlying `glsaccum` logic.
        """
        S1 = np.zeros((self.n_instruments, self.n_instruments), dtype=np.float64)
        has_system = hasattr(self, 'is_level') and self.is_level is not None

        if has_system:
            H_full = self._get_H_full_system()
            has_tindex = hasattr(self, 't_index')
            for mask in self._group_masks:
                Z_g = self.Z[mask]
                if has_tindex:
                    t_idx_g = self.t_index[mask].astype(np.int64)
                    H_g = H_full[np.ix_(t_idx_g, t_idx_g)]
                else:
                    H_g = np.eye(len(Z_g), dtype=np.float64)
                S1 += Z_g.T @ H_g @ Z_g
        else:
            for mask in self._group_masks:
                Z_g = self.Z[mask]
                S1 += Z_g.T @ Z_g

        return _inv_or_pinv(S1)
    
    def estimate_one_step(self) -> np.ndarray:
        """
        Performs the 1-step (One-Step) GMM estimation.
        Saves vital matrices (W1, V1, e1) for step 2 and diagnostics.
        
        Returns
        -------
        np.ndarray
            The estimated coefficient vector (beta1).
        """
        XZ = self.X.T @ self.Z
        Zy = self.Z.T @ self.y
        
        XZ_W_ZX = XZ @ self.W1 @ XZ.T
        self.V1 = np.linalg.pinv((XZ_W_ZX + XZ_W_ZX.T) / 2.0)
        
        self.beta1 = self.V1 @ XZ @ self.W1 @ Zy
        self.e1 = self.y - self.X @ self.beta1
        
        return self.beta1

    def estimate_two_step_robust(self) -> tuple:
        """
        Performs the 2-step GMM estimation with a robust weight matrix.
        Applies the Windmeijer (2005) variance correction and the 'small' option.
        
        Returns
        -------
        tuple
            A tuple containing (beta2, se2_robust, V2_robust).
        """
        self.estimate_one_step()

        S2 = _accumulate_ZeZe(self.Z, self.e1, self._cluster_masks)
        # KNOWN LIMITATION: when instruments outnumber clusters (e.g. cluster()
        # on a coarser grouping than the panel id, combined with many
        # instruments), S2 is exactly rank-deficient (rank <= N_clusters) and
        # Stata prints "covariance matrix of moments is singular ... using a
        # generalized inverse." Our Moore-Penrose pinv and Mata's `invsym`-based
        # generalized inverse are both *valid* generalized inverses of a
        # singular matrix, but not the *same* one in general (invsym drops
        # specific pivoted coordinates rather than projecting via SVD), so
        # point estimates -- not just SE -- can diverge from Stata in this
        # regime. Confirmed empirically: away from this singular case (e.g.
        # cluster() on the panel id itself, or any case where instruments <=
        # clusters) results match Stata exactly (see stata_validation/). The
        # mitigation is the same one Stata's own warning implies: collapse
        # instruments or reduce their count relative to the number of clusters.
        self.W2 = np.linalg.pinv(S2)

        XZ = self.X.T @ self.Z
        Zy = self.Z.T @ self.y
        
        XZ_W2_ZX = XZ @ self.W2 @ XZ.T
        self.V2 = np.linalg.pinv((XZ_W2_ZX + XZ_W2_ZX.T) / 2.0)
        self.beta2 = self.V2 @ XZ @ self.W2 @ Zy
        self.e2 = self.y - self.X @ self.beta2

        VZXA1 = self.V1 @ XZ @ self.W1
        V1_robust_windmeijer = VZXA1 @ S2 @ VZXA1.T
        V1_robust_windmeijer = (V1_robust_windmeijer + V1_robust_windmeijer.T) / 2.0

        D_sum = np.zeros((self.n_instruments, self.k_vars), dtype=np.float64)
        A2Ze = self.W2 @ self.Z.T @ self.e2

        for mask in self._cluster_masks:
            Z_g = self.Z[mask]
            X_g = self.X[mask]
            e1_g = self.e1[mask]
            
            Ze1 = e1_g.T @ Z_g
            ZXi = Z_g.T @ X_g
            
            scalar_part = (Ze1 @ A2Ze)[0, 0]
            D_sum += scalar_part * ZXi + (Ze1.T @ A2Ze.T) @ ZXi
            
        VZXA2 = self.V2 @ XZ @ self.W2
        D = VZXA2 @ D_sum
        
        V2_robust = self.V2 + D @ V1_robust_windmeijer @ D.T + 2.0 * (D @ self.V2)
        self.V2_robust = (V2_robust + V2_robust.T) / 2.0
        
        if getattr(self, 'small', False):
            if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                N_obs_eff = np.sum(self.is_level)
            else:
                N_obs_eff = self.n_obs
                
            qc1 = (N_obs_eff - 1.0) / (N_obs_eff - self.k_vars - self.df_pca)
            qc2 = self._cluster_qc_factor()

            self.V2_robust = self.V2_robust * (qc1 * qc2)
            
        se2_robust = np.sqrt(np.diag(self.V2_robust)).reshape(-1, 1)
        return self.beta2, se2_robust, self.V2_robust

    def get_wald_test(self) -> tuple:
        """
        Computes the Wald test statistic for overall joint significance.
        Adapts automatically (One-Step or Two-Step, with or without a constant).
        
        Returns
        -------
        tuple
            (wald_stat, wald_df, p_val, test_type) where test_type is 'F' or 'chi2'.
        """
        if hasattr(self, 'is_level') and self.is_level is not None:
            is_const = np.all(self.X[self.is_level, -1] == 1) and np.all(self.X[~self.is_level, -1] == 0)
        else:
            is_const = np.all(self.X[:, -1] == self.X[0, -1])
            
        k_indep = self.k_vars - 1 if is_const else self.k_vars
        
        if k_indep <= 0:
            return np.nan, 0, np.nan, 'chi2'
            
        beta_to_use = getattr(self, 'beta2', getattr(self, 'beta1', None))
        
        V_to_use = getattr(self, 'V2_robust', getattr(self, 'V2', getattr(self, 'V1', None)))
        
        U, s, Vh = scipy.linalg.svd(V_to_use)
        tol = 1e-12 
        s_inv = np.zeros_like(s)
        valid_idx = s > (tol * s.max())
        s_inv[valid_idx] = 1.0 / s[valid_idx]
        
        V_inv = (Vh.T * s_inv) @ U.T
        
        wald_stat = (beta_to_use.T @ V_inv @ beta_to_use)[0, 0]
        wald_df = k_indep
        
        if getattr(self, 'small', False):
            f_stat = wald_stat / wald_df
            is_robust_or_2step = hasattr(self, 'e2') or hasattr(self, 'V2_robust')
            
            if is_robust_or_2step:
                df_resid = self.N_clusters - (1 if is_const else 0)
            else:
                df_resid = self.n_obs - self.k_vars - self.df_pca
                
            p_val = 1.0 - stats.f.cdf(f_stat, wald_df, df_resid)
            return f_stat, wald_df, p_val, 'F'
        else:
            p_val = 1.0 - stats.chi2.cdf(wald_stat, wald_df)
            return wald_stat, wald_df, p_val, 'chi2'

    def get_diagnostics(self) -> dict:
        """
        Generates all diagnostic tests (Sargan, Hansen, AR1, AR2, Wald).
        Dynamically adapts based on whether estimation is One-Step or Two-Step.
        
        Returns
        -------
        dict
            Dictionary containing test statistics, p-values, and degrees of freedom.
        """
        df_tests = self.n_instruments - self.k_vars

        if df_tests <= 0:
            sargan_stat, sargan_p = np.nan, np.nan
            hansen_stat, hansen_p = np.nan, np.nan
        
        has_two_step = hasattr(self, 'e2')
        e_resid = self.e2 if has_two_step else self.e1
        
        if has_two_step:
            hansen_stat = (self.e2.T @ self.Z @ self.W2 @ self.Z.T @ self.e2)[0, 0]
            hansen_p = 1.0 - stats.chi2.cdf(hansen_stat, df_tests) if df_tests > 0 else np.nan
        else:
            hansen_stat, hansen_p = np.nan, np.nan
            
        # xtabond2.mata:201 `ErrorEq = (h!=1\h==1)#J(T,1,1)`: sig2 comes from the
        # diff/transformed residuals when h>1, but from the *levels* residuals
        # when h==1 (h=1 assumes no differencing-induced structure at all). For
        # Difference GMM, is_level is all-False so h==1 falls back to n_obs.
        if self.h > 1:
            e1_diff = self.e1[~self.is_level]
            N_obs_diff = np.sum(~self.is_level)
        elif np.any(self.is_level):
            e1_diff = self.e1[self.is_level]
            N_obs_diff = np.sum(self.is_level)
        else:
            e1_diff = self.e1
            N_obs_diff = self.n_obs
        divisor = 1.0 if (getattr(self, 'orthogonal', False) or self.h == 1) else 2.0

        if N_obs_diff > 0:
            sig2 = np.sum(e1_diff**2) / N_obs_diff / divisor
            sargan_stat = (self.e1.T @ self.Z @ self.W1 @ self.Z.T @ self.e1)[0, 0] / sig2
            sargan_p = 1.0 - stats.chi2.cdf(sargan_stat, df_tests) if df_tests > 0 else np.nan
        else:
            sargan_stat, sargan_p = np.nan, np.nan

        def compute_ar(l):
            if l >= self.T_span:
                return np.nan, np.nan
                
            ew_total, wHw = 0.0, 0.0
            Xw = np.zeros((1, self.k_vars), dtype=np.float64)
            ZHw = np.zeros((1, self.n_instruments), dtype=np.float64)
            
            has_two_step = hasattr(self, 'beta2')
            V_for_ar = getattr(self, 'V2', self.V1).copy()
            has_robust = hasattr(self, 'V2_robust') or (not has_two_step and hasattr(self, 'V2_robust'))
            V_rob_for_ar = getattr(self, 'V2_robust', V_for_ar).copy()
            
            is_non_robust_1step = (not has_two_step and not hasattr(self, 'V2_robust'))
            sig2_v = getattr(self, 'sig2_v_1step', 1.0) if is_non_robust_1step else 1.0

            if getattr(self, 'small', False):
                if is_non_robust_1step:
                    N_obs = np.sum(self.is_level) if (hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level)) else self.n_obs
                    qc = N_obs / (N_obs - self.k_vars - self.df_pca)
                    V_for_ar = V_for_ar / qc
                    V_rob_for_ar = V_rob_for_ar / qc
                else:
                    qc = self._cluster_qc_factor()
                    V_for_ar = V_for_ar / qc
                    if getattr(self, 'orthogonal', False) and not has_robust:
                        if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                            N_obs = np.sum(self.is_level)
                        else:
                            N_obs = self.n_obs
                        qc_1step = (N_obs / (N_obs - self.k_vars - self.df_pca)) * qc
                        V_rob_for_ar = V_rob_for_ar * qc_1step
                    else:
                        if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                            N_obs_eff = np.sum(self.is_level)
                        else:
                            N_obs_eff = self.n_obs
                            
                        qc1 = (N_obs_eff - 1.0) / (N_obs_eff - self.k_vars - self.df_pca) if has_two_step else (N_obs_eff / (N_obs_eff - self.k_vars - self.df_pca))
                        qc2 = self._cluster_qc_factor()

                        V_rob_for_ar = V_rob_for_ar / (qc1 * qc2)

            if getattr(self, 'arlevels', False):
                # Test serial correlation of the *levels* residuals (System GMM
                # only), with H forced to the identity (Stata forces h=1 here
                # regardless of the model's own h()/orthogonal setting --
                # xtabond2.mata::_ARTests: H = _H(arlevels?1:h, 0, 0, 0, T)).
                for idx, mask_g in enumerate(self._group_masks):
                    e_g = e_resid[mask_g].flatten()
                    Z_g = self.Z[mask_g]

                    mask_level_g = mask_g & self.is_level
                    e_level_g = e_g[self.is_level[mask_g]]
                    X_level_g = self.X[mask_level_g]
                    Z_level_g = Z_g[self.is_level[mask_g]]
                    # Level rows carry t_index in [T_span, 2*T_span) (estimator.py
                    # offsets them by T_span to tell the two blocks apart) --
                    # subtract it back out to get the true calendar offset.
                    t_level_g = (self.t_index[mask_level_g] - self.T_span).astype(np.int64)

                    # Dense (T_span-long) reconstruction, indexed by true
                    # calendar offset: a positional lag-shift below is then a
                    # true calendar-time lag regardless of any gap/attrition
                    # in this group, mirroring Mata's own colshape-then-shift
                    # (xtabond2.mata::_ARTests) rather than assuming this
                    # group's own rows are calendar-contiguous.
                    e_dense = scatter_to_grid(e_level_g, t_level_g, self.T_span)
                    X_dense = np.full((self.T_span, X_level_g.shape[1]), np.nan)
                    X_dense[t_level_g] = X_level_g
                    Z_dense = np.full((self.T_span, Z_level_g.shape[1]), np.nan)
                    Z_dense[t_level_g] = Z_level_g

                    n_level = self.T_span
                    # wHw term: Mata forces H to h=1 (identity) when arlevels,
                    # regardless of the model's own h() -- xtabond2.mata::_ARTests,
                    # `H = _H(arlevels?1:h, ...)`.
                    H_g_level = np.eye(n_level, dtype=np.float64)
                    # psit/cross term with instruments: Mata does NOT force this one;
                    # it always uses the actual h() (`psit = _H(h, ...)`), only
                    # repositioned into the levels block. h=2/h=3 share the same
                    # diff-style block, so only h=1 needs special-casing here.
                    # (n_level = T_span, not this group's own row count, is what
                    # makes _build_H_diff's adjacent-period structure meaningful
                    # here -- it assumes its own row i/i+1 are calendar-adjacent.)
                    C_g = H_g_level if self.h == 1 else _build_H_diff(n_level)

                    w_i = np.full(self.T_span, np.nan)
                    if self.T_span > l:
                        w_i[l:] = e_dense[:-l]

                    valid_ar = ~np.isnan(w_i) & ~np.isnan(e_dense)

                    if np.sum(valid_ar) > 0:
                        ew_i = np.sum(w_i[valid_ar] * e_dense[valid_ar])
                        ew_total += ew_i
                        Xw += w_i[valid_ar].reshape(1, -1) @ X_dense[valid_ar]

                        if is_non_robust_1step:
                            H_scaled = H_g_level * sig2_v
                            C_scaled = C_g * sig2_v

                            w_val = w_i[valid_ar].reshape(-1, 1)
                            H_val = H_scaled[np.ix_(valid_ar, valid_ar)]

                            C_cross = C_scaled[valid_ar, :]

                            wHw += (w_val.T @ H_val @ w_val)[0, 0]
                            # Zero (not NaN), matching Mata's own touse2-zero
                            # convention: an invalid period still has a real
                            # (zero) theoretical cross-covariance column in
                            # C_cross, so its instrument contribution must be
                            # zero, not omitted/NaN, for the sum to be right.
                            ZHw += (w_val.T @ C_cross @ np.nan_to_num(Z_dense))
                        else:
                            wHw += ew_i**2
                            ZHw += ew_i * (e_g.reshape(1, -1) @ Z_g)

            elif getattr(self, 'orthogonal', False):
                T = self.T_span
                Ml, Mr = self._get_orthogonal_xforms()

                H_full = Ml.T @ Ml
                if is_non_robust_1step:
                    H_full = H_full * sig2_v

                beta_to_use = getattr(self, 'beta2', getattr(self, 'beta1', None))

                for idx, mask_g in enumerate(self._group_masks):
                    y_lvl_g = self.y_lvl[idx].flatten()
                    X_lvl_g = self.X_lvl[idx]
                    
                    if len(beta_to_use) > X_lvl_g.shape[1]:
                        e_lvl_g = y_lvl_g - X_lvl_g @ beta_to_use[:-1].flatten() - beta_to_use[-1]
                    else:
                        e_lvl_g = y_lvl_g - X_lvl_g @ beta_to_use.flatten()
                        
                    e_diff_full = np.zeros(T, dtype=np.float64)
                    diff_raw = e_lvl_g[1:] - e_lvl_g[:-1]
                    
                    n_raw = len(diff_raw)
                    if n_raw > 0:
                        e_diff_full[-n_raw:] = np.nan_to_num(diff_raw)
                    
                    touse_mask = np.zeros(T, dtype=bool)
                    mask_g_diff = mask_g & ~self.is_level if hasattr(self, 'is_level') and self.is_level is not None else mask_g
                    Z_diff_g = self.Z[mask_g_diff]
                    
                    if hasattr(self, 't_index'):
                        t_idx_g = self.t_index[mask_g_diff].astype(np.int64)
                        valid_z = t_idx_g < T
                        t_idx_g = t_idx_g[valid_z]
                        touse_mask[t_idx_g] = True
                        Z_diff_g_aligned = Z_diff_g[valid_z]
                    else:
                        n_valid = np.sum(mask_g_diff)
                        if n_valid > 0:
                            touse_mask[-n_valid:] = True
                        Z_diff_g_aligned = Z_diff_g
                            
                    w = e_diff_full * touse_mask
                    
                    wl = np.zeros(T, dtype=np.float64)
                    if T > l:
                        wl[l:] = e_diff_full[:-l]
                    wl = wl * touse_mask
                    
                    ew_i = np.sum(w * wl)
                    ew_total += ew_i
                    
                    X_diff_full = np.zeros((T, self.k_vars), dtype=np.float64)
                    X_lvl_diff_raw = X_lvl_g[1:] - X_lvl_g[:-1]
                    
                    n_raw_x = len(X_lvl_diff_raw)
                    if n_raw_x > 0:
                        if X_lvl_g.shape[1] == self.k_vars - 1:
                            X_diff_full[-n_raw_x:, :-1] = np.nan_to_num(X_lvl_diff_raw)
                        else:
                            X_diff_full[-n_raw_x:] = np.nan_to_num(X_lvl_diff_raw)
                        
                    Xw += wl.reshape(1, -1) @ X_diff_full
                    
                    if is_non_robust_1step:
                        wHw += (wl.T @ H_full @ wl)
                        
                        psiw = (Mr.T @ Ml) @ wl * sig2_v
                        
                        Z_diff_full = np.zeros((T, self.n_instruments), dtype=np.float64)
                        if hasattr(self, 't_index') and len(Z_diff_g_aligned) > 0:
                            Z_diff_full[t_idx_g] = Z_diff_g_aligned
                        elif len(Z_diff_g_aligned) > 0:
                            Z_diff_full[-len(Z_diff_g_aligned):] = Z_diff_g_aligned
                        
                        ZHw += (Z_diff_full.T @ psiw).reshape(1, -1)
                    else:
                        wHw += ew_i**2
                        
                        e_g_full = e_resid[mask_g].flatten()
                        Z_g_full = self.Z[mask_g]
                        
                        ZHw += ew_i * (e_g_full.reshape(1, -1) @ Z_g_full)

            else:
                for idx, mask_g in enumerate(self._group_masks):
                    e_g = e_resid[mask_g].flatten()
                    Z_g = self.Z[mask_g]

                    if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                        mask_diff_g = mask_g & ~self.is_level
                    else:
                        mask_diff_g = mask_g

                    e_diff_g = e_g[~self.is_level[mask_g]] if (hasattr(self, 'is_level') and self.is_level is not None) else e_g
                    X_diff_g = self.X[mask_diff_g]
                    Z_diff_g = Z_g[~self.is_level[mask_g]] if (hasattr(self, 'is_level') and self.is_level is not None) else Z_g
                    t_diff_g = self.t_index[mask_diff_g].astype(np.int64)

                    # Dense (T_span-long) reconstruction by true calendar
                    # offset (see the arlevels branch above for why).
                    e_dense = scatter_to_grid(e_diff_g, t_diff_g, self.T_span)
                    X_dense = np.full((self.T_span, X_diff_g.shape[1]), np.nan)
                    X_dense[t_diff_g] = X_diff_g
                    Z_dense = np.full((self.T_span, Z_diff_g.shape[1]), np.nan)
                    Z_dense[t_diff_g] = Z_diff_g

                    n_diff = self.T_span
                    # h=1 assumes no serial-correlation structure (H=I); h=2/h=3
                    # share the same diff-transform block -- xtabond2.mata::_H.
                    H_g_diff = np.eye(n_diff, dtype=np.float64) if self.h == 1 else _build_H_diff(n_diff)
                    C_g = H_g_diff

                    w_i = np.full(self.T_span, np.nan)
                    if self.T_span > l:
                        w_i[l:] = e_dense[:-l]

                    valid_ar = ~np.isnan(w_i) & ~np.isnan(e_dense)

                    if np.sum(valid_ar) > 0:
                        ew_i = np.sum(w_i[valid_ar] * e_dense[valid_ar])
                        ew_total += ew_i
                        Xw += w_i[valid_ar].reshape(1, -1) @ X_dense[valid_ar]

                        if is_non_robust_1step:
                            H_scaled = H_g_diff * sig2_v
                            C_scaled = C_g * sig2_v

                            w_val = w_i[valid_ar].reshape(-1, 1)
                            H_val = H_scaled[np.ix_(valid_ar, valid_ar)]

                            C_cross = C_scaled[valid_ar, :]

                            wHw += (w_val.T @ H_val @ w_val)[0, 0]
                            # Zero, not NaN -- see the identical comment in the
                            # arlevels branch above.
                            ZHw += (w_val.T @ C_cross @ np.nan_to_num(Z_dense))
                        else:
                            wHw += ew_i**2
                            ZHw += ew_i * (e_g.reshape(1, -1) @ Z_g)

            if is_non_robust_1step:
                W_mat_scaled = self.W1 / sig2_v
                m2VZXA = -2.0 * V_for_ar @ self.X.T @ self.Z @ W_mat_scaled
            else:
                m2VZXA = -2.0 * V_for_ar @ self.X.T @ self.Z @ getattr(self, 'W2', self.W1)
                
            d = wHw + Xw @ (m2VZXA @ ZHw.T + V_rob_for_ar @ Xw.T)
            
            if d[0, 0] <= 0:
                return np.nan, np.nan
            ar_stat = ew_total / np.sqrt(d[0, 0])
            ar_p = 2.0 * stats.norm.cdf(-np.abs(ar_stat))
            return ar_stat, ar_p
        
        ar_results = {l: compute_ar(l) for l in range(1, self.artests + 1)}
        wald_stat, wald_df, wald_p, wald_type = self.get_wald_test()

        return {
            'df': df_tests,
            'sargan': (sargan_stat, sargan_p),
            'hansen': (hansen_stat, hansen_p),
            'ar': ar_results,
            'wald': (wald_stat, wald_df, wald_p, wald_type)
        }
