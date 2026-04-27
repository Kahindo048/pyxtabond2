import numpy as np
import scipy.stats as stats
import scipy.linalg

class GMMEngine:
    """
    Generalized Method of Moments (GMM) estimation engine for dynamic panels.
    Replicates the matrix logic, Windmeijer finite-sample correction, and 
    diagnostics of Stata's `xtabond2`.
    """
    
    def __init__(self, y: np.ndarray, X: np.ndarray, Z: np.ndarray, 
                 group_ids: np.ndarray = None, is_level: np.ndarray = None, 
                 small: bool = False, orthogonal: bool = False,
                 t_index: np.ndarray = None, T_span: int = None,
                 y_lvl: list = None, X_lvl: list = None, r: int = 0):
        
        self.small = small
        self.orthogonal = orthogonal
        self.T_span = T_span # Addition of T_span
        self.y_lvl = y_lvl # <-- NEW
        self.X_lvl = X_lvl # <-- NEW
        self.r = r

        # --- 1. ROBUST ROW CLEANING (Casewise Deletion) ---
        valid_rows = ~np.isnan(y).flatten() & ~np.isnan(X).any(axis=1) & ~np.isnan(Z).any(axis=1)
        
        self.y = y[valid_rows]
        self.X = X[valid_rows]
        self.Z = Z[valid_rows]
        
        if group_ids is not None:
            self.group_ids = group_ids[valid_rows]
        if is_level is not None:
            self.is_level = is_level[valid_rows]
        if t_index is not None:
            self.t_index = t_index[valid_rows] # Synchronized cleaning of t_index

        non_zero_cols = ~np.all(self.Z == 0, axis=0)
        dropped_cols = np.sum(~non_zero_cols)
        if dropped_cols > 0:
            self.Z = self.Z[:, non_zero_cols]
            
        self.n_obs, self.k_vars = self.X.shape
        self.n_instruments = self.Z.shape[1] 
        self.z_rank = np.linalg.matrix_rank(self.Z)
        
        if self.z_rank < self.k_vars:
            raise ValueError(f"Under-identified model: {self.z_rank} valid instruments for {self.k_vars} regressors.")
            
        self.N_groups = len(np.unique(self.group_ids)) if self.group_ids is not None else 1
        
        # --- NEW: Calculation of the PCA penalty ---
        if self.r > 0:
            self.df_pca = self.r * (self.N_groups + self.T_span - self.r)
        else:
            self.df_pca = 0
            
        self.W1 = self._compute_W1()
        
    def _compute_W1(self):
        """
        Computes the weighting matrix A1 with surgical time alignment.
        Replicates Stata's underlying `glsaccum` logic.
        """
        S1 = np.zeros((self.n_instruments, self.n_instruments))
        
        # 1. Construction of the global theoretical matrix H_full
        H_full = np.eye(2 * self.T_span)
        if hasattr(self, 'is_level') and self.is_level is not None:
            if self.orthogonal:
                # --- NEW: Exact H for System GMM + Forward Orthogonal Deviations ---
                for t_d in range(1, self.T_span):
                    t_source = t_d - 1 # Source period of the deviation
                    future_count = self.T_span - 1 - t_source
                    if future_count > 0:
                        c_val = np.sqrt(future_count / (future_count + 1.0))
                        
                        # Cross diagonal: Cov(FOD_t, Level_t)
                        H_full[t_d, self.T_span + t_source] = c_val
                        H_full[self.T_span + t_source, t_d] = c_val
                        
                        # Cross row: Cov(FOD_t, Level_{t+k})
                        for k in range(t_source + 1, self.T_span):
                            H_full[t_d, self.T_span + k] = -c_val / future_count
                            H_full[self.T_span + k, t_d] = -c_val / future_count
            else:
                # --- Stata h=3 default: Accounting for global covariance ---
                H_diff = np.eye(self.T_span) * 2.0
                for i in range(self.T_span - 1):
                    H_diff[i, i+1] = -1.0
                    H_diff[i+1, i] = -1.0
                H_diff[0, 0] = 1.0 # First observation lost, but mathematically correct
                
                H_full[:self.T_span, :self.T_span] = H_diff
                
                for i in range(self.T_span):
                    H_full[i, self.T_span + i] = 1.0
                    H_full[self.T_span + i, i] = 1.0
                    if i + 1 < self.T_span:
                        H_full[i+1, self.T_span + i] = -1.0
                        H_full[self.T_span + i, i+1] = -1.0
                    
        # 2. Dynamic extraction for each group
        for g in np.unique(self.group_ids):
            mask_g = (self.group_ids == g)
            Z_g = self.Z[mask_g]
            
            if hasattr(self, 't_index'):
                t_idx_g = self.t_index[mask_g].astype(int)
                # NumPy magic: extracts exactly the surviving rows/columns!
                H_g = H_full[np.ix_(t_idx_g, t_idx_g)] 
            else:
                # If t_index is missing (failsafe), use a simple diagonal block
                n_rows = len(Z_g)
                H_g = np.eye(n_rows)

            S1 += Z_g.T @ H_g @ Z_g
                
        return np.linalg.pinv(S1)
    
    def estimate_one_step(self) -> np.ndarray:
        """
        Performs the 1-step (One-Step) GMM estimation.
        Saves vital matrices (W1, V1, e1) for step 2 and diagnostics.
        
        Returns
        -------
        np.ndarray
            The estimated coefficients vector (beta1).
        """
        self.W1 = self._compute_W1()
        
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
        
        # 1. Calculation of W2 (A2 in Stata) based on One-Step residuals
        S2 = np.zeros((self.n_instruments, self.n_instruments))
        for g in np.unique(self.group_ids):
            mask_g = (self.group_ids == g)
            Z_g = self.Z[mask_g]
            e1_g = self.e1[mask_g]
            S2 += Z_g.T @ (e1_g @ e1_g.T) @ Z_g
            
        # A2 in Roodman's original code
        self.W2 = np.linalg.pinv(S2)
        
        # 2. Estimation of Beta2 and V2 (non-robust)
        XZ = self.X.T @ self.Z
        Zy = self.Z.T @ self.y
        
        XZ_W2_ZX = XZ @ self.W2 @ XZ.T
        self.V2 = np.linalg.pinv((XZ_W2_ZX + XZ_W2_ZX.T) / 2.0)
        self.beta2 = self.V2 @ XZ @ self.W2 @ Zy
        self.e2 = self.y - self.X @ self.beta2

        # --- V1 Robust used for Windmeijer ---
        # Stata: mat VZXA = e(V) * ZX * A1 (e(V) is V1 here)
        # Stata: mat V1robust = VZXA * A2 * VZXA'
        VZXA1 = self.V1 @ XZ @ self.W1
        V1_robust_windmeijer = VZXA1 @ S2 @ VZXA1.T
        V1_robust_windmeijer = (V1_robust_windmeijer + V1_robust_windmeijer.T) / 2.0

        # 3. Windmeijer (2005) Correction - Faithful DPD reproduction
        D_sum = np.zeros((self.n_instruments, self.k_vars))
        A2Ze = self.W2 @ self.Z.T @ self.e2
        
        for g in np.unique(self.group_ids):
            mask_g = (self.group_ids == g)
            Z_g = self.Z[mask_g]
            X_g = self.X[mask_g]
            e1_g = self.e1[mask_g]
            
            # Stata: mat Ze1 = e1' * Zi
            Ze1 = e1_g.T @ Z_g
            # Stata: mat ZXi = Zi' * Xi
            ZXi = Z_g.T @ X_g
            
            # SCALAR EXTRACTION FOR NUMPY
            # Stata: D = D + (Ze1 * A2Ze) * ZXi + (Ze1' * A2Ze') * ZXi
            scalar_part = (Ze1 @ A2Ze)[0, 0]
            term1 = scalar_part * ZXi               # Scalar multiplication
            term2 = (Ze1.T @ A2Ze.T) @ ZXi          # Pure matrix product
            
            D_sum += term1 + term2
            
        VZXA2 = self.V2 @ XZ @ self.W2
        D = VZXA2 @ D_sum
        
        # Stata: V2robust = e(V) + D * V1robust * D' + 2 * D * e(V)
        V2_robust = self.V2 + D @ V1_robust_windmeijer @ D.T + 2.0 * (D @ self.V2)
        self.V2_robust = (V2_robust + V2_robust.T) / 2.0
        
        # 4. Small-Sample Correction ('small' option)
        if getattr(self, 'small', False):
            # The exact multiplier hidden in xtabond2.mata includes (N-1)/(N-k)
            if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level) :
                N_obs_eff = np.sum(self.is_level)
            else:
                N_obs_eff = self.n_obs
                
            qc1 = (N_obs_eff - 1.0) / (N_obs_eff - self.k_vars - self.df_pca)
            qc2 = self.N_groups / (self.N_groups - 1.0)
            
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
        import scipy.linalg
        
        # In pure Difference GMM, there is no constant to exclude.
        # We detect if the last column is a constant (zero variance)
        # Detects the constant: 0 in difference, 1 in level
        if hasattr(self, 'is_level') and self.is_level is not None:
            # We use 'and' instead of '&' because vectors don't have the same size (e.g., 400 vs 350)
            is_const = np.all(self.X[self.is_level, -1] == 1) and np.all(self.X[~self.is_level, -1] == 0)
        else:
            is_const = np.all(self.X[:, -1] == self.X[0, -1])
            
        k_indep = self.k_vars - 1 if is_const else self.k_vars
        
        if k_indep <= 0:
            return np.nan, 0, np.nan, 'chi2'
            
        # Smart fallback: use step 2 if available, otherwise step 1
        beta_to_use = getattr(self, 'beta2', getattr(self, 'beta1', None))
        beta_indep = beta_to_use[:k_indep]
        
        V_to_use = getattr(self, 'V2_robust', getattr(self, 'V2', getattr(self, 'V1', None)))
        V_indep = V_to_use[:k_indep, :k_indep]
        
        U, s, Vh = scipy.linalg.svd(V_to_use)
        tol = 1e-12 
        s_inv = np.zeros_like(s)
        valid_idx = s > (tol * s.max())
        s_inv[valid_idx] = 1.0 / s[valid_idx]
        
        V_inv = (Vh.T * s_inv) @ U.T
        
        # Compute wald_stat on the full beta_to_use
        wald_stat = (beta_to_use.T @ V_inv @ beta_to_use)[0, 0]
        wald_df = k_indep # But we keep k_indep for degrees of freedom!
        
        if getattr(self, 'small', False):
            f_stat = wald_stat / wald_df
            is_robust_or_2step = hasattr(self, 'e2') or hasattr(self, 'V2_robust')
            
            if is_robust_or_2step:
                df_resid = self.N_groups - (1 if is_const else 0)
            else:
                df_resid = self.n_obs - self.k_vars
                
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
        
        # Detecting the estimation level
        has_two_step = hasattr(self, 'e2')
        e_resid = self.e2 if has_two_step else self.e1
        W_mat = self.W2 if has_two_step else self.W1
        
        # 1. HANSEN TEST - Only computable if W2 (Two-Step or Robust) exists
        if has_two_step:
            hansen_stat = (self.e2.T @ self.Z @ self.W2 @ self.Z.T @ self.e2)[0, 0]
            hansen_p = 1.0 - stats.chi2.cdf(hansen_stat, df_tests) if df_tests > 0 else np.nan
        else:
            hansen_stat, hansen_p = np.nan, np.nan
            
        # 2. SARGAN TEST (Non-Robust) - Always computed on Step 1
        e1_diff = self.e1[~self.is_level]
        N_obs_diff = np.sum(~self.is_level)
        divisor = 1.0 if getattr(self, 'orthogonal', False) else 2.0
        
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
            Xw = np.zeros((1, self.k_vars))
            ZHw = np.zeros((1, self.n_instruments))
            
            has_two_step = hasattr(self, 'beta2')
            V_for_ar = getattr(self, 'V2', self.V1).copy()
            has_robust = hasattr(self, 'V2_robust') or (not has_two_step and hasattr(self, 'V2_robust'))
            V_rob_for_ar = getattr(self, 'V2_robust', V_for_ar).copy()
            
            is_non_robust_1step = (not has_two_step and not hasattr(self, 'V2_robust'))
            sig2_v = getattr(self, 'sig2_v_1step', 1.0) if is_non_robust_1step else 1.0

            if getattr(self, 'small', False):
                if is_non_robust_1step:
                    N_obs = np.sum(self.is_level) if (hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level)) else self.n_obs
                    qc = N_obs / (N_obs - self.k_vars)
                    V_for_ar = V_for_ar / qc
                    V_rob_for_ar = V_rob_for_ar / qc
                else:
                    qc = self.N_groups / (self.N_groups - 1)
                    V_for_ar = V_for_ar / qc
                    if getattr(self, 'orthogonal', False) and not has_robust:
                        if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                            N_obs = np.sum(self.is_level)
                        else:
                            N_obs = self.n_obs
                        qc_1step = (N_obs / (N_obs - self.k_vars)) * qc
                        V_rob_for_ar = V_rob_for_ar * qc_1step
                    else:
                        # --- EXACT CORRECTION: Reversing the Small-sample scaling for the AR Test ---
                        if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                            N_obs_eff = np.sum(self.is_level)
                        else:
                            N_obs_eff = self.n_obs
                            
                        # --- MODIFIED: We subtract self.df_pca ---
                        qc1 = (N_obs_eff - 1.0) / (N_obs_eff - self.k_vars - self.df_pca) if has_two_step else (N_obs_eff / (N_obs_eff - self.k_vars - self.df_pca))
                        qc2 = self.N_groups / (self.N_groups - 1.0)
                        
                        # We ONLY divide the robust matrix. 
                        # V_for_ar (standard V1 or V2) is never touched because it was never scaled.
                        V_rob_for_ar = V_rob_for_ar / (qc1 * qc2)

            if getattr(self, 'orthogonal', False):
                def _xform(xform_type, T):
                    M = np.eye(T)
                    if xform_type == 1: 
                        M_lag = np.zeros((T, T))
                        M_lag[:-1, :] = M[1:, :]
                        M = M_lag
                        for r in range(T-1, 0, -1):
                            M[r:, r] = -1.0 / (T - r)
                        col_norms = np.sqrt(np.sum(M**2, axis=0))
                        col_norms[col_norms == 0] = 1.0
                        return M / col_norms
                    else: 
                        M_lag = np.zeros((T, T))
                        M_lag[:-1, :] = M[1:, :]
                        M = M - M_lag
                        M[0, 0] = 0.0
                        return M

                T = self.T_span
                Ml = _xform(0, T)
                Mr = _xform(1, T)
                    
                H_full = Ml.T @ Ml
                if is_non_robust_1step:
                    H_full *= sig2_v

                for idx, g in enumerate(np.unique(self.group_ids)):
                    mask_g = (self.group_ids == g)
                    
                    y_lvl_g = self.y_lvl[idx].flatten()
                    X_lvl_g = self.X_lvl[idx]
                    
                    beta_to_use = getattr(self, 'beta2', getattr(self, 'beta1', None))
                    if len(beta_to_use) > X_lvl_g.shape[1]:
                        e_lvl_g = y_lvl_g - X_lvl_g @ beta_to_use[:-1].flatten() - beta_to_use[-1]
                    else:
                        e_lvl_g = y_lvl_g - X_lvl_g @ beta_to_use.flatten()
                        
                    e_diff_full = np.zeros(T)
                    diff_raw = e_lvl_g[1:] - e_lvl_g[:-1]
                    
                    n_raw = len(diff_raw)
                    if n_raw > 0:
                        e_diff_full[-n_raw:] = np.nan_to_num(diff_raw)
                    
                    touse_mask = np.zeros(T, dtype=bool)
                    mask_g_diff = mask_g & ~self.is_level if hasattr(self, 'is_level') and self.is_level is not None else mask_g
                    Z_diff_g = self.Z[mask_g_diff]
                    
                    if hasattr(self, 't_index'):
                        t_idx_g = self.t_index[mask_g_diff].astype(int)
                        valid_z = t_idx_g < T
                        t_idx_g = t_idx_g[valid_z]
                        touse_mask[t_idx_g] = True
                        Z_diff_g_aligned = Z_diff_g[valid_z]
                    else:
                        n_valid = np.sum(mask_g_diff)
                        if n_valid > 0:
                            touse_mask[-n_valid:] = True
                        Z_diff_g_aligned = Z_diff_g
                            
                    # KEY CORRECTION: We strictly mask e_diff to keep only the valid sample
                    w = e_diff_full * touse_mask
                    
                    wl = np.zeros(T)
                    if T > l:
                        wl[l:] = e_diff_full[:-l]
                    wl = wl * touse_mask
                    
                    ew_i = np.sum(w * wl)
                    ew_total += ew_i
                    
                    X_diff_full = np.zeros((T, self.k_vars))
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
                        
                        Z_diff_full = np.zeros((T, self.n_instruments))
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
                for idx, g in enumerate(np.unique(self.group_ids)):
                    mask_g = (self.group_ids == g)
                    e_g = e_resid[mask_g].flatten()
                    Z_g = self.Z[mask_g]
                    
                    if hasattr(self, 'is_level') and self.is_level is not None and np.any(self.is_level):
                        mask_diff_g = mask_g & ~self.is_level
                    else:
                        mask_diff_g = mask_g
                        
                    e_diff_g = e_g[~self.is_level[mask_g]] if (hasattr(self, 'is_level') and self.is_level is not None) else e_g
                    X_diff_g = self.X[mask_diff_g]
                    
                    n_diff = len(e_diff_g)
                    H_g_diff = np.eye(n_diff) * 2.0
                    for i in range(n_diff - 1):
                        H_g_diff[i, i+1] = -1.0
                        H_g_diff[i+1, i] = -1.0
                    
                    C_g = H_g_diff 
                        
                    w_i = np.full_like(e_diff_g, np.nan)
                    if len(e_diff_g) > l:
                        w_i[l:] = e_diff_g[:-l]
                        
                    valid_ar = ~np.isnan(w_i) & ~np.isnan(e_diff_g)
                    
                    if np.sum(valid_ar) > 0:
                        ew_i = np.sum(w_i[valid_ar] * e_diff_g[valid_ar])
                        ew_total += ew_i
                        Xw += w_i[valid_ar].reshape(1, -1) @ X_diff_g[valid_ar]
                        
                        if is_non_robust_1step:
                            H_scaled = H_g_diff * sig2_v
                            C_scaled = C_g * sig2_v
                            
                            Z_diff_g = Z_g[~self.is_level[mask_g]] if (hasattr(self, 'is_level') and self.is_level is not None) else Z_g
                            
                            w_val = w_i[valid_ar].reshape(-1, 1)
                            H_val = H_scaled[np.ix_(valid_ar, valid_ar)]
                            
                            C_cross = C_scaled[valid_ar, :]
                            
                            wHw += (w_val.T @ H_val @ w_val)[0, 0]
                            ZHw += (w_val.T @ C_cross @ Z_diff_g)
                        else:
                            wHw += ew_i**2
                            ZHw += ew_i * (e_g.reshape(1, -1) @ Z_g)

            if is_non_robust_1step:
                W_mat_scaled = self.W1 / sig2_v
                m2VZXA = -2.0 * V_for_ar @ self.X.T @ self.Z @ W_mat_scaled
            else:
                m2VZXA = -2.0 * V_for_ar @ self.X.T @ self.Z @ getattr(self, 'W2', self.W1)
                
            d = wHw + Xw @ (m2VZXA @ ZHw.T + V_rob_for_ar @ Xw.T)
            
            if d[0, 0] <= 0: return np.nan, np.nan
            ar_stat = ew_total / np.sqrt(d[0, 0])
            ar_p = 2.0 * stats.norm.cdf(-np.abs(ar_stat))
            return ar_stat, ar_p
        
        ar1_stat, ar1_p = compute_ar(1)
        ar2_stat, ar2_p = compute_ar(2)
        wald_stat, wald_df, wald_p, wald_type = self.get_wald_test()
        
        return {
            'df': df_tests,
            'sargan': (sargan_stat, sargan_p),
            'hansen': (hansen_stat, hansen_p),
            'ar1': (ar1_stat, ar1_p),
            'ar2': (ar2_stat, ar2_p),
            'wald': (wald_stat, wald_df, wald_p, wald_type)
        }