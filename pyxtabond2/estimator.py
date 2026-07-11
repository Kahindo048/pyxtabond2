"""
High-level orchestration for PyXtabond2.

Defines :class:`PyXtabond2`, the main estimator class mirroring Stata's
``xtabond2`` options. It builds instruments and stacks the panel via
:class:`~pyxtabond2.gmm_builder.SystemGMMBuilder`, delegates the linear
algebra to :class:`~pyxtabond2.engine.GMMEngine`, adds PCA-GMM / IFE-GMM
support via :class:`~pyxtabond2.ife.IFEMixin`, and returns
:class:`~pyxtabond2.results.PyXtabond2Results`.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

from .data_utils import PanelData, scatter_to_grid
from .gmm_builder import SystemGMMBuilder
from .engine import GMMEngine, _accumulate_ZeZe
from .diagnostics import compute_diff_sargan_tests
from .ife import IFEMixin
from .results import PyXtabond2Results
from .specs import normalize_gmm_styles, normalize_iv_styles


class PyXtabond2(IFEMixin):
    """
    Main user interface for PyXtabond2, replicating the options of Stata's `xtabond2`.

    This class orchestrates data preparation, instrument building, and model
    estimation for both standard Dynamic Panel GMM and PCA-GMM (Interactive Fixed Effects).

    Parameters
    ----------
    df : pd.DataFrame
        The panel dataset.
    id_col : str
        The name of the column identifying the panel groups.
    time_col : str
        The name of the column identifying time periods.
    dep_var : str
        The dependent variable.
    x_vars : list of str
        The list of strictly exogenous variables to be included as standard regressors.
    gmm_vars : list of str
        The list of endogenous/predetermined variables to be instrumented GMM-style.
        Shortcut for a single :class:`~pyxtabond2.specs.GMMStyle` group sharing
        ``lag_limits_diff``/``collapse``; ignored if ``gmm`` is given.
    iv_vars : list of str
        The list of exogenous variables to be used as standard IV instruments.
        Shortcut for a single :class:`~pyxtabond2.specs.IVStyle` group;
        ignored if ``iv`` is given.
    model_type : str, optional
        'difference' for Arellano-Bond, or 'system' for Arellano-Bover/Blundell-Bond. Default is 'difference'.
    twostep : bool, optional
        If True, estimates the two-step GMM. Default is False (one-step).
    robust : bool, optional
        If True, computes robust standard errors (Windmeijer corrected for two-step). Default is False.
    lag_limits_diff : tuple, optional
        The lag limits for GMM instruments (e.g., `(2, None)` means lags 2 to maximum).
        Applies to all of ``gmm_vars``; for per-variable-group lag limits use ``gmm``.
    collapse : bool, optional
        If True, limits instrument proliferation by collapsing the instrument matrix.
        Applies to all of ``gmm_vars``; for per-variable-group collapse use ``gmm``.
    gmm : list of GMMStyle, optional
        Advanced, per-group control mirroring Stata's repeated ``gmm()``
        option: each :class:`~pyxtabond2.specs.GMMStyle` is one group of
        variables with its own ``lag``/``collapse``/``equation``. Takes
        precedence over ``gmm_vars``/``lag_limits_diff``/``collapse`` if given.
    iv : list of IVStyle, optional
        Advanced, per-group control mirroring Stata's repeated ``iv()``
        option: each :class:`~pyxtabond2.specs.IVStyle` is one group with its
        own ``equation``. Takes precedence over ``iv_vars`` if given.
    orthogonal : bool, optional
        If True, uses Forward Orthogonal Deviations (FOD) instead of first differences.
    small : bool, optional
        If True, applies small-sample degree-of-freedom corrections to covariance matrices and test statistics.
    r : int or str, optional
        The number of interactive unobserved factors for PCA-GMM. If 'auto', estimates the optimal number of factors. Default is 0.
        The iteration that defactors Y/X/instruments and re-estimates GMM
        (see :mod:`~pyxtabond2.ife`) starts from the plain (non-defactored)
        GMM fit. Hong, Su & Jiang (2022, "Profile GMM Estimation of Panel
        Data Models with Interactive Fixed Effects") prove fast contraction
        to the truth for this exact update formula, but only from a starting
        point already consistent in the presence of factors (they use a
        nuclear-norm-regularized estimator for that); absent that, per the
        iterative estimator they compare against (Jiang et al. 2021), there
        is no guarantee of convergence to the right solution when the
        starting point's factor-induced bias is large. Not implemented here
        given the added complexity; treat convergence as a good empirical
        sign, not a proof of correctness, especially with dominant factors.
    r_max : int, optional
        Maximum number of factors to test if `r='auto'`. Default is 5.
    ife_max_iter : int, optional
        Maximum number of iterations for the PCA-GMM convergence loop. Default is 30.
    ife_tol : float, optional
        Convergence tolerance limit for PCA-GMM. Default is 1e-5.
    bias_correction : bool, optional
        If True, applies the Dhaene & Jochmans (2015) split-panel jackknife
        (splitting on time, matching their theory of an O(1/T) incidental-
        parameter-style bias) on top of the IFE-GMM fit. Default is False.
        Recommended for serious inference (hypothesis tests, confidence
        intervals): Hong, Su & Jiang (2022)'s asymptotic theory for this
        same iterative estimator shows the *variance* used here (plain GMM
        at the estimated factors) is already correct to leading order, but
        the *point estimate* carries asymptotic bias terms analogous to
        Bai (2009)'s, vanishing only under special cases (strictly exogenous
        instruments, no cross-sectional heteroskedasticity, no serial
        correlation/heteroskedasticity in the idiosyncratic errors) that do
        not hold in general. Left off by default to match prior behavior and
        because it roughly doubles computation (two half-panel refits).
    h : int, optional
        Stata's ``h()`` option (1, 2, or 3) selecting the one-step weighting
        matrix structure. Default is 3 (Stata's default: models the known
        cross-covariance between the transformed and levels equations).
        h=2 assumes no such covariance; h=1 assumes no serial-correlation
        structure at all. h=1/h=2 are rarely needed in practice.
    artests : int, optional
        Highest-order Arellano-Bond AR(l) serial-correlation test to report
        (Stata's ``artests()`` option). Default is 2 (AR(1) and AR(2)).
    arlevels : bool, optional
        If True, the AR(l) tests target the levels-equation residuals
        instead of the differenced/orthogonal-deviations residuals (Stata's
        ``arlevels`` option). Only valid for ``model_type='system'``.
        Default is False.
    cluster : str, optional
        Name of a column to cluster the robust/two-step variance matrix by,
        instead of the default panel id (Stata's ``cluster()`` option).
        Only a single clustering variable is supported (not Stata's
        multi-way ``cluster(var1 var2)`` combinatorial clustering).
        Default is None (cluster by the panel id, i.e. ``robust``'s default).
        Note: if the number of clusters is smaller than the number of
        instruments (a coarser ``cluster()`` combined with ``twostep``/
        ``robust``), the moment covariance matrix becomes exactly
        rank-deficient; see the comment on ``GMMEngine.estimate_two_step_robust``
        for why point estimates can then diverge from Stata in that regime.

    Raises
    ------
    ValueError
        If `model_type` is not 'difference' or 'system'.
    """
    def __init__(self, df: pd.DataFrame, id_col: str, time_col: str,
                 dep_var: str, x_vars: list, gmm_vars: list = None, iv_vars: list = None,
                 model_type: str = 'difference',  # 'difference' or 'system'
                 twostep: bool = False,        # False = One-step, True = Two-step
                 robust: bool = False,         # Robust variance matrix
                 lag_limits_diff: tuple = (2, None),
                 collapse: bool = False,
                 gmm: list = None,
                 iv: list = None,
                 orthogonal: bool = False,
                 small: bool = False,
                 r: int = 0,
                 r_max: int = 5,
                 ife_max_iter: int = 30,
                 ife_tol: float = 1e-5,
                 bias_correction: bool = False,
                 h: int = 3,
                 artests: int = 2,
                 arlevels: bool = False,
                 cluster: str = None):

        if model_type not in ['difference', 'system']:
            raise ValueError("model_type must be 'difference' or 'system'")
        if h not in (1, 2, 3):
            raise ValueError("h must be 1, 2, or 3 (Stata's h() option; default 3).")
        if artests < 1:
            raise ValueError("artests must be >= 1 (Stata's artests() option; default 2).")
        if arlevels and model_type != 'system':
            raise ValueError("arlevels is only valid for model_type='system' (invalid for Difference GMM).")
        if cluster is not None and cluster not in df.columns:
            raise ValueError(f"cluster column {cluster!r} not found in df.")

        self.df = df.copy()
        self.id_col = id_col
        self.time_col = time_col
        self.dep_var = dep_var
        self.x_vars = x_vars.copy()

        self._gmm_styles = normalize_gmm_styles(gmm_vars, lag_limits_diff, collapse, gmm)
        self._iv_styles = normalize_iv_styles(iv_vars, iv)
        # Flattened variable lists, kept for internal reuse (IFE defactoring
        # in ife.py) and for display (PyXtabond2Results.summary()).
        self.gmm_vars = [v for grp in self._gmm_styles for v in grp.variables]
        self.iv_vars = [v for grp in self._iv_styles for v in grp.variables]
        self.model_type = model_type
        self.twostep = twostep
        self.robust = robust
        self.lag_limits_diff = lag_limits_diff
        self.collapse = collapse
        self.orthogonal = orthogonal
        self.small = small
        self.r = r
        self.r_max = r_max
        self.ife_max_iter = ife_max_iter
        self.ife_tol = ife_tol
        self.bias_correction = bias_correction
        self.h = h
        self.artests = artests
        self.arlevels = arlevels
        self.cluster = cluster
        self.iteration = 0

    def fit(self):
        """
        Main method: Routes between Classic GMM and GMM with Interactive Fixed Effects (IFE).

        Returns
        -------
        PyXtabond2Results
            The structured estimation results.
        """
        if self.r == 0:
            # Classic GMM (Direct call to the base engine)
            return self._fit_base(self.df, self.dep_var)
        else:
            # GMM with Interactive Fixed Effects (PCA-GMM)
            return self._fit_ife()

    def _fit_base(self, df_custom, dep_var_custom, panel_cache=None, full_diagnostics=True):
        """
        Prepares data, builds instruments according to model type, and estimates.

        ``panel_cache``, if given, is a previously-built :class:`PanelData`
        whose sort order and group slices are reused via
        :meth:`PanelData.rebuild_fast` -- valid only when ``df_custom`` has
        the same rows in the same order (see the IFE-GMM loop in ``ife.py``).

        ``full_diagnostics=False`` skips the Hansen/Sargan/AR/Wald/Diff-Sargan
        diagnostics (and, in the one-step+robust branch, the extra Hansen-only
        "Roodman's trick" refit) and returns early with an empty ``diag``.
        ``beta``/``se`` are unaffected -- they never depend on this block.
        Meant for the intermediate iterations of the IFE-GMM loop (``ife.py``),
        which only reads ``.beta`` from the result until convergence.
        """
        if panel_cache is not None:
            panel = PanelData.rebuild_fast(panel_cache, df_custom)
        else:
            panel = PanelData(df_custom, self.id_col, self.time_col)
        self._last_panel = panel
        df_clean = panel.data
        t_min = df_clean.index.get_level_values(self.time_col).min()
        t_max = df_clean.index.get_level_values(self.time_col).max()
        T_span = t_max - t_min + 1

        df_clean['_cons'] = 1.0
        df_clean['D__cons'] = 0.0
        if self.orthogonal: df_clean['FOD__cons'] = 0.0

        trans_prefix = 'FOD_' if self.orthogonal else 'D_'
        all_vars = set([dep_var_custom] + self.x_vars + self.gmm_vars + self.iv_vars)
        if '_cons' in all_vars: all_vars.remove('_cons')

        for v in all_vars:
            if trans_prefix + v not in df_clean.columns:
                df_clean[trans_prefix + v] = panel.get_fod(v) if self.orthogonal else panel.get_first_difference(v)
            if 'D_' + v not in df_clean.columns:
                df_clean['D_' + v] = panel.get_first_difference(v)

        sys_builder = SystemGMMBuilder(panel)
        Z_list = []

        # 1. Building Instrument Matrices (per-group lag/collapse/equation, see specs.py)
        for grp in self._gmm_styles:
            # Number of "level" (extra Blundell-Bond) instrument columns this
            # group produces -- depends only on collapse, not on equation.
            n_cols_level = 1 if grp.collapse else (T_span - 1)

            for gvar in grp.variables:
                Z_full = sys_builder.build_system_instruments(gvar, grp.lag, grp.collapse)
                if self.model_type == 'difference':
                    # Difference GMM has no levels equation: keep diff-instrument columns only.
                    Z_list.append(Z_full[:, :-n_cols_level])
                elif grp.equation == 'diff':
                    Z_list.append(Z_full[:, :-n_cols_level])
                elif grp.equation == 'level':
                    Z_list.append(Z_full[:, -n_cols_level:])
                else:  # 'both'
                    Z_list.append(Z_full)

        for grp in self._iv_styles:
            include_diff = grp.equation in ('both', 'diff')
            include_lvl = (self.model_type == 'system') and grp.equation in ('both', 'level')

            for ivar in grp.variables:
                Z_iv = sys_builder.build_iv_instruments(
                    ivar, include_in_diff=include_diff, include_in_level=include_lvl, orthogonal=self.orthogonal
                )

                # --- XTABOND2 SECRET FOR STANDARD IVs ---
                # In System GMM, iv() combines the difference and the level into A SINGLE column!
                if include_lvl and include_diff and Z_iv.shape[1] > 1:
                    Z_iv = np.sum(Z_iv, axis=1, keepdims=True)

                Z_list.append(Z_iv)

        if '_cons' not in self.iv_vars and self.model_type == 'system':
            Z_list.append(sys_builder.build_iv_instruments('_cons', include_in_diff=False, include_in_level=True, orthogonal=self.orthogonal))

        Z_sys_full = np.hstack(Z_list)

        # 2. Stacking Y and X (numpy slices — no per-group pandas .loc)
        Y_stacked, X_stacked, Z_stacked = [], [], []
        Y_lvl_list, X_lvl_list = [], []
        group_ids_list, is_level_list, t_idx_list = [], [], []
        cluster_ids_list = [] if self.cluster is not None else None
        if self.cluster is not None:
            # The clustering variable may be a regular column, or the panel
            # id/time itself (moved into the MultiIndex by PanelData).
            if self.cluster in df_clean.columns:
                cluster_raw = df_clean[self.cluster]
            else:
                cluster_raw = df_clean.index.get_level_values(self.cluster)
            # Factorize to integer codes so it can be numeric or string-valued.
            cluster_codes, _ = pd.factorize(cluster_raw, sort=False)

        x_names_final = self.x_vars + ['_cons'] if self.model_type == 'system' else self.x_vars
        time_vals = df_clean.index.get_level_values(self.time_col).to_numpy(dtype=np.int64)
        time_off = time_vals - t_min
        group_vals = df_clean.index.get_level_values(0).to_numpy()
        # Skip non-numeric columns (e.g. a string cluster() variable riding
        # along) -- only named lookups below matter, and those are all numeric.
        col_data = {}
        for c in df_clean.columns:
            try:
                col_data[c] = df_clean[c].to_numpy(dtype=np.float64, copy=False)
            except (ValueError, TypeError):
                continue
        dep_trans = trans_prefix + dep_var_custom
        n_x = len(self.x_vars)

        for gi, sl in enumerate(panel._group_slices):
            g = group_vals[sl.start]
            t_off_g = time_off[sl]

            # A group's own row count can be < T_span (a missing (id,time)
            # row, not just a missing value within a present row) -- t_off_g
            # then no longer runs 0..T_span-1, so masks must be indexed into
            # Z_block by absolute offset (t_off_g[mask]), never by local
            # position (mask alone), or the two go out of sync silently or
            # (when lengths differ) raise an IndexError.
            mask_diff = t_off_g >= 1
            # A standard IV variable's own missing value drops the whole
            # equation-row (xtabond2.mata:260, rowmissing(Z_IV)) -- distinct
            # from a GMM-style instrument's missing value, which only zeros
            # that one instrument cell (already handled by gmm_builder.py's
            # scatter-then-guard construction, untouched here).
            for iv_grp in self._iv_styles:
                if iv_grp.equation in ('both', 'diff'):
                    for ivar in iv_grp.variables:
                        mask_diff &= ~np.isnan(col_data[trans_prefix + ivar][sl])
            cluster_off_g = cluster_codes[sl] if self.cluster is not None else None

            y_diff = col_data[dep_trans][sl][mask_diff].reshape(-1, 1)
            t_diff = t_off_g[mask_diff]

            # y_lvl/X_lvl feed only engine.py::compute_ar's orthogonal-model
            # branch, which re-differences them itself for the AR-test
            # (diagnostics.py's Diff-Sargan refit forwards them but never
            # calls compute_ar, so it never reads them either). Skip building
            # them entirely when orthogonal=False (the default, and what
            # most specs use): otherwise this dense scatter_to_grid per
            # group per x_var/dep_var runs on every IFE iteration for values
            # nothing downstream will ever read. Built dense (T_span-long,
            # NaN at genuinely absent periods) so that a positional diff in
            # compute_ar is a true calendar-time diff regardless of
            # gaps/attrition -- otherwise identical row selection to before
            # (mask_diff-restricted for Difference GMM, unrestricted for
            # System GMM).
            if self.orthogonal:
                if self.model_type == 'difference':
                    y_lvl_g = scatter_to_grid(col_data[dep_var_custom][sl][mask_diff], t_off_g[mask_diff], T_span).reshape(-1, 1)
                    X_lvl_g = np.column_stack(
                        [scatter_to_grid(col_data[x][sl][mask_diff], t_off_g[mask_diff], T_span) for x in self.x_vars]
                    ) if n_x else np.zeros((T_span, 0))
                else:
                    y_lvl_g = scatter_to_grid(col_data[dep_var_custom][sl], t_off_g, T_span).reshape(-1, 1)
                    X_lvl_g = np.column_stack(
                        [scatter_to_grid(col_data[x][sl], t_off_g, T_span) for x in self.x_vars]
                    ) if n_x else np.zeros((T_span, 0))
                Y_lvl_list.append(y_lvl_g)
                X_lvl_list.append(X_lvl_g)

            z_base = gi * 2 * T_span
            Z_block = Z_sys_full[z_base: z_base + 2 * T_span]

            if self.model_type == 'system':
                n_d = int(np.sum(mask_diff))
                if n_x:
                    X_diff = np.column_stack([col_data[trans_prefix + x][sl][mask_diff] for x in self.x_vars] + [np.zeros(n_d)])
                else:
                    X_diff = np.zeros((n_d, 1))

                # Same base formula as mask_diff (t_off_g >= 1) -- confirmed
                # by stata_validation against real Stata (arlevels included)
                # -- but its own, level-relevant standard-IV extension, since
                # a diff-only iv() shouldn't gate the level equation or vice
                # versa. Reduces to exactly mask_diff on every spec validated
                # so far (no iv_var is ever missing there), so this is a
                # no-op on all existing behavior.
                mask_level = t_off_g >= 1
                for iv_grp in self._iv_styles:
                    if iv_grp.equation in ('both', 'level'):
                        for ivar in iv_grp.variables:
                            mask_level &= ~np.isnan(col_data[ivar][sl])
                y_level = col_data[dep_var_custom][sl][mask_level].reshape(-1, 1)
                n_l = int(np.sum(mask_level))
                if n_x:
                    X_level = np.column_stack([col_data[x][sl][mask_level] for x in self.x_vars] + [np.ones(n_l)])
                else:
                    X_level = np.ones((n_l, 1))
                t_level = t_off_g[mask_level]

                Z_g_diff = Z_block[:T_span][t_off_g[mask_diff]]
                Z_g_level = Z_block[T_span:][t_off_g[mask_level]]

                Y_stacked.append(np.vstack([y_diff, y_level]))
                X_stacked.append(np.vstack([X_diff, X_level]))
                Z_stacked.append(np.vstack([Z_g_diff, Z_g_level]))

                group_ids_list.extend([g] * (n_d + n_l))
                is_level_list.extend([False] * n_d + [True] * n_l)
                t_idx_list.extend(t_diff)
                t_idx_list.extend(t_level + T_span)
                if self.cluster is not None:
                    cluster_ids_list.extend(cluster_off_g[mask_diff])
                    cluster_ids_list.extend(cluster_off_g[mask_level])
            else:
                X_diff = np.column_stack([col_data[trans_prefix + x][sl][mask_diff] for x in self.x_vars]) if n_x else np.zeros((len(y_diff), 0))
                Z_g_diff = Z_block[:T_span][t_off_g[mask_diff]]
                n_d = len(y_diff)
                Y_stacked.append(y_diff)
                X_stacked.append(X_diff)
                Z_stacked.append(Z_g_diff)
                group_ids_list.extend([g] * n_d)
                is_level_list.extend([False] * n_d)
                t_idx_list.extend(t_diff)
                if self.cluster is not None:
                    cluster_ids_list.extend(cluster_off_g[mask_diff])

        Y_sys = np.vstack(Y_stacked)
        X_sys = np.vstack(X_stacked)
        Z_sys = np.vstack(Z_stacked)

        group_ids = np.array(group_ids_list)
        is_level = np.array(is_level_list)
        t_index = np.array(t_idx_list)
        cluster_ids = np.array(cluster_ids_list) if self.cluster is not None else None

        # 3. Launching the Engine with correct options
        engine = GMMEngine(
            Y_sys, X_sys, Z_sys,
            group_ids=group_ids, is_level=is_level,
            small=self.small, orthogonal=self.orthogonal,
            t_index=t_index, T_span=T_span,
            y_lvl=Y_lvl_list, X_lvl=X_lvl_list,
            r=self.r, h=self.h, artests=self.artests, arlevels=self.arlevels,
            cluster_ids=cluster_ids,
        )

        if self.twostep:
            if self.robust:
                beta, se, _ = engine.estimate_two_step_robust()
            else:
                # Non-robust Two-step (Rarely used, but Stata allows it)
                engine.estimate_two_step_robust() # Called to populate beta2
                beta = engine.beta2
                se = np.sqrt(np.diag(engine.V2)).reshape(-1, 1)
                delattr(engine, 'V2_robust')
        else:
            # One-step
            beta = engine.estimate_one_step()
            if self.robust:
                S1_robust = _accumulate_ZeZe(engine.Z, engine.e1, engine._cluster_masks)

                XZ = engine.X.T @ engine.Z
                meat = XZ @ engine.W1 @ S1_robust @ engine.W1 @ XZ.T
                V1_rob = engine.V1 @ meat @ engine.V1

                if self.small:
                    # xtabond2.mata:562-566 -- one-step-robust (like two-step) uses
                    # (NObs-1)/(NObs-k), not NObs/(NObs-k) (that's the one-step
                    # *non-robust* case only, handled in the `else` branch below).
                    # The N_clusters/(N_clusters-1) factor is itself dropped when
                    # an explicit cluster() is given (xtabond2.mata:564) -- see
                    # GMMEngine._cluster_qc_factor.
                    N_obs = np.sum(engine.is_level) if self.model_type == 'system' else len(engine.y)
                    qc = ((N_obs - 1.0) / (N_obs - engine.k_vars)) * engine._cluster_qc_factor()
                    V1_rob *= qc
                se = np.sqrt(np.diag(V1_rob)).reshape(-1, 1)
                engine.V2_robust = V1_rob # Trick for Wald test to use it

                if full_diagnostics:
                    # --- ROODMAN'S TRICK (xtabond2.ado) ---
                    # Silent execution of Step 2 just to compute the Hansen test.
                    # Diagnostic-only (never feeds beta/se above) -- skipped
                    # entirely when full_diagnostics=False.
                    W2_hansen = np.linalg.pinv(S1_robust)
                    Zy = engine.Z.T @ engine.y
                    XZ_W2_ZX = XZ @ W2_hansen @ XZ.T
                    V2_hansen = np.linalg.pinv((XZ_W2_ZX + XZ_W2_ZX.T) / 2.0)
                    beta2_hansen = V2_hansen @ XZ @ W2_hansen @ Zy
                    e2_hansen = engine.y - engine.X @ beta2_hansen

                    hansen_stat = (e2_hansen.T @ engine.Z @ W2_hansen @ engine.Z.T @ e2_hansen)[0, 0]
                    df_tests = engine.n_instruments - engine.k_vars
                    hansen_p = 1.0 - stats.chi2.cdf(hansen_stat, df_tests) if df_tests > 0 else np.nan
                    engine.hansen_1step_robust = (hansen_stat, hansen_p)
            else:
                # Non-Robust One-Step (Homoscedastic)
                # h=1 assumes no serial-correlation structure (H=I, diagonal 1),
                # unlike h=2/h=3's diff-transform diagonal of 2 -- xtabond2.mata::_H.
                divisor = 1.0 if (self.orthogonal or self.h == 1) else 2.0

                # --- STATA CORRECTION: sig2_v based exclusively on differences ---
                if self.model_type == 'system':
                    # xtabond2.mata:201 `ErrorEq = (h!=1\h==1)#J(T,1,1)` -- "dummy
                    # for equation whose errors to use for sig2 ... transformed
                    # eq unless h=1": sig2 comes from the diff/transformed
                    # residuals when h>1, but from the *levels* residuals when
                    # h==1 (h=1 assumes no differencing-induced structure at all).
                    if self.h == 1:
                        e1_for_sig2 = engine.e1[engine.is_level]
                        N_obs = np.sum(engine.is_level)
                    else:
                        e1_for_sig2 = engine.e1[~engine.is_level]
                        N_obs = np.sum(~engine.is_level)
                    sum_sq = np.sum(e1_for_sig2**2)
                else:
                    sum_sq = np.sum(engine.e1**2)
                    N_obs = engine.n_obs

                # --- MODIFIED: PCA penalty on the denominator ---
                df_pca_penalty = self.r * (engine.N_groups + T_span - self.r) if self.r > 0 else 0
                denom = (N_obs - engine.k_vars - df_pca_penalty) if self.small else N_obs
                sig2_v = sum_sq / denom / divisor

                # Crucial save of this exact scale for AR() tests
                engine.sig2_v_1step = sig2_v

                engine.V1 = engine.V1 * sig2_v
                se = np.sqrt(np.diag(engine.V1)).reshape(-1, 1)

        m_name = "system GMM" if self.model_type == 'system' else "difference GMM"
        s_name = "two-step" if self.twostep else "one-step"

        if not full_diagnostics:
            # PyXtabond2Results.__init__ only reads engine/beta/se to build
            # coefficients, t-stats, p-values, and CIs -- diag is stored
            # unread until summary()/export are called, so an empty dict is
            # safe here (the IFE loop only ever reads .beta from this result).
            return PyXtabond2Results(
                beta, se, engine, {}, x_names_final, dep_var_custom, m_name, s_name,
                gmm_vars=self.gmm_vars,
                iv_vars=self.iv_vars,
                lag_limits=self.lag_limits_diff,
                collapse=self.collapse,
                orthogonal=self.orthogonal,
                id_col=self.id_col,
                time_col=self.time_col
            )

        # --- DIFFERENCE-IN-SARGAN / HANSEN TESTS ---
        df_full_c = engine.n_instruments - engine.k_vars

        if self.twostep:
            stat_full_c = (engine.e2.T @ engine.Z @ engine.W2 @ engine.Z.T @ engine.e2)[0, 0]
        elif self.robust:
            stat_full_c = engine.hansen_1step_robust[0]
        else:
            if self.h == 1:
                e1_diff_c = engine.e1[engine.is_level]
                N_obs_diff_c = np.sum(engine.is_level)
            else:
                e1_diff_c = engine.e1[~engine.is_level]
                N_obs_diff_c = np.sum(~engine.is_level)
            divisor_c = 1.0 if (self.orthogonal or self.h == 1) else 2.0
            sig2_c = np.sum(e1_diff_c ** 2) / N_obs_diff_c / divisor_c if N_obs_diff_c > 0 else 1.0
            stat_full_c = (engine.e1.T @ engine.Z @ engine.W1 @ engine.Z.T @ engine.e1)[0, 0] / sig2_c

        diff_sargan_results = compute_diff_sargan_tests(
            engine,
            model_type=self.model_type, iv_vars=self.iv_vars,
            twostep=self.twostep, robust=self.robust, small=self.small, orthogonal=self.orthogonal,
            stat_full_c=stat_full_c, df_full_c=df_full_c, h=self.h,
        )

        diag = engine.get_diagnostics()
        diag['diff_sargan'] = diff_sargan_results

        return PyXtabond2Results(
            beta, se, engine, diag, x_names_final, dep_var_custom, m_name, s_name,
            gmm_vars=self.gmm_vars,
            iv_vars=self.iv_vars,
            lag_limits=self.lag_limits_diff,
            collapse=self.collapse,
            orthogonal=self.orthogonal,
            id_col=self.id_col,
            time_col=self.time_col
        )
