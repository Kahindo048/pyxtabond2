import pandas as pd
import numpy as np
from .data_utils import PanelData

class SystemGMMBuilder:
    """
    Builder for the GMM instrument matrix (the Z matrix).
    
    This class handles the complex task of generating the block-diagonal (staircase) 
    instrument matrices for dynamic panel data models. It accurately replicates the 
    matrix creation logic found in the Mata source code of Stata's 'xtabond2'.

    Parameters
    ----------
    panel : PanelData
        An instance of the PanelData class containing the cleaned, sorted, 
        and MultiIndexed panel dataset.
    """
    
    def __init__(self, panel: PanelData):
        self.panel = panel

    def build_system_instruments(self, var_name: str, lag_limits_diff: tuple = (2, None), collapse: bool = False) -> np.ndarray:
        """
        Builds "Arellano-Bond" style GMM instruments for a System GMM model.
        
        This method generates lagged instruments in levels for the differenced equation, 
        and lagged differenced instruments for the levels equation. It effectively 
        creates the classic block-diagonal instrument matrix.

        Parameters
        ----------
        var_name : str
            The name of the endogenous or predetermined variable (e.g., 'lagged_y').
        lag_limits_diff : tuple, optional
            A tuple `(min_lag, max_lag)` specifying which lags to use as instruments 
            for the differenced equation. Default is (2, None), meaning lag 2 to the maximum available.
        collapse : bool, optional
            If True, "collapses" the instrument matrix. Instead of creating a new column 
            for each time period and lag, it creates one column per lag distance. This is 
            highly recommended to avoid instrument proliferation (the "too many instruments" problem).
            Default is False.

        Returns
        -------
        np.ndarray
            The stacked Z_gmm matrix (2*T rows x varying instrument columns) for all groups.

        Examples
        --------
        >>> builder = SystemGMMBuilder(panel_data)
        >>> # Standard expanding instrument matrix (lags 2 to end)
        >>> Z_gmm_std = builder.build_system_instruments('gdp', lag_limits_diff=(2, None))
        >>> # Collapsed instrument matrix to save degrees of freedom
        >>> Z_gmm_collapsed = builder.build_system_instruments('gdp', lag_limits_diff=(2, 4), collapse=True)
        """
        l_min, l_max = lag_limits_diff
        
        t_min = self.panel.data.index.get_level_values(self.panel.time_col).min()
        t_max = self.panel.data.index.get_level_values(self.panel.time_col).max()
        T_span = t_max - t_min + 1
        
        # If no maximum lag is specified, use all possible available lags
        if l_max is None:
            l_max = T_span - 1
            
        unique_groups = self.panel.data.index.get_level_values(0).unique()
        Z_sys_list = []
        
        # --- 1. STRICT COLUMN COUNT CALCULATION (Memory Pre-allocation) ---
        if collapse:
            # In collapse mode: 1 column per lag distance (diff eq) + 1 global column (level eq)
            n_cols_diff = l_max - l_min + 1
            n_cols_level = 1
        else:
            # Standard mode: calculate the exact dimensions of the "staircase" matrix (capped at l_max)
            n_cols_diff = sum(max(0, min(l_max, t - t_min) - l_min + 1) for t in range(t_min + 1, t_max + 1))
            n_cols_level = T_span - 1
            
        n_cols_total = n_cols_diff + n_cols_level

        # --- 2. BLOCK-BY-BLOCK CONSTRUCTION FOR EACH GROUP ---
        for g in unique_groups:
            Z_g = np.zeros((2 * T_span, n_cols_total))
            mask_g = (self.panel.data.index.get_level_values(0) == g)
            df_g = self.panel.data[mask_g].copy()
            
            # BLOCK A: Instruments in Levels (for the Differenced / FOD equation)
            col_idx = 0
            if collapse:
                for lag in range(l_min, l_max + 1):
                    for t in range(t_min + 1, t_max + 1):
                        row_idx = t - t_min
                        t_source = t - lag
                        if t_source >= t_min:
                            val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t_source, var_name]
                            if not val.empty and not pd.isna(val.iloc[0]):
                                Z_g[row_idx, col_idx] = val.iloc[0]
                    col_idx += 1
            else:
                for t in range(t_min + 1, t_max + 1):
                    row_idx = t - t_min
                    max_lag_for_t = t - t_min
                    actual_l_max = min(l_max, max_lag_for_t)
                    
                    for lag in range(l_min, actual_l_max + 1):
                        t_source = t - lag
                        val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t_source, var_name]
                        if not val.empty and not pd.isna(val.iloc[0]):
                            Z_g[row_idx, col_idx] = val.iloc[0]
                        col_idx += 1
                        
            # BLOCK B: Instruments in First Differences (for the Level equation)
            col_idx_level = n_cols_diff
            diff_col_name = f'D_{var_name}'
            
            # Generate difference if it wasn't pre-computed
            if diff_col_name not in df_g.columns:
                df_g[diff_col_name] = df_g[var_name].diff()
                
            # Calculate the lag shift (dlag) exactly as implemented in xtabond2
            dlag = l_min - 1 
                
            if collapse:
                for t in range(t_min + 1, t_max + 1):
                    row_idx = T_span + (t - t_min)
                    t_source = t - dlag  # Dynamic application of the lag
                    val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t_source, diff_col_name]
                    if not val.empty and not pd.isna(val.iloc[0]):
                        Z_g[row_idx, col_idx_level] = val.iloc[0]
            else:
                for t in range(t_min + 1, t_max + 1):
                    row_idx = T_span + (t - t_min)
                    t_source = t - dlag  
                    val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t_source, diff_col_name]
                    if not val.empty and not pd.isna(val.iloc[0]):
                        Z_g[row_idx, col_idx_level] = val.iloc[0]
                    col_idx_level += 1
                    
            Z_sys_list.append(Z_g)

        return np.vstack(Z_sys_list)

    def build_iv_instruments(self, var_name: str, include_in_diff: bool = True, include_in_level: bool = True, orthogonal: bool = False) -> np.ndarray:
        """
        Builds standard Instrumental Variable (IV) matrices for strictly exogenous regressors.
        
        Unlike Arellano-Bond instruments, standard IVs typically instrument themselves. 
        This method formats the exogenous variables correctly for the stacked System GMM equations.

        Parameters
        ----------
        var_name : str
            The name of the exogenous variable (use '_cons' for the intercept).
        include_in_diff : bool, optional
            Whether to include the instrument in the transformed (Differenced/FOD) equation. Default is True.
        include_in_level : bool, optional
            Whether to include the instrument in the level equation. Default is True.
        orthogonal : bool, optional
            If True, adapts the transformation to use Forward Orthogonal Deviations (FOD). Default is False.

        Returns
        -------
        np.ndarray
            The stacked Z_iv matrix for all groups.

        Raises
        ------
        ValueError
            If the instrument is not included in either the difference or level equation, or if 
            the required FOD transformation has not been pre-calculated.

        Examples
        --------
        >>> builder = SystemGMMBuilder(panel_data)
        >>> # Standard IV for an exogenous control variable
        >>> Z_iv_control = builder.build_iv_instruments('inflation', include_in_diff=True, include_in_level=True)
        >>> # IV for the constant term (only applicable to the levels equation in System GMM)
        >>> Z_iv_cons = builder.build_iv_instruments('_cons', include_in_diff=False, include_in_level=True)
        """
        t_min = self.panel.data.index.get_level_values(self.panel.time_col).min()
        t_max = self.panel.data.index.get_level_values(self.panel.time_col).max()
        T_span = t_max - t_min + 1
        
        unique_groups = self.panel.data.index.get_level_values(0).unique()
        Z_iv_list = []
        
        n_cols = int(include_in_diff) + int(include_in_level)
        if n_cols == 0:
            raise ValueError("The instrument must be included in at least one equation (diff or level).")
            
        for g in unique_groups:
            Z_g = np.zeros((2 * T_span, n_cols))
            mask_g = (self.panel.data.index.get_level_values(0) == g)
            df_g = self.panel.data[mask_g].copy()
            
            col_idx = 0
            
            # --- BLOCK A: Transformed Equation (First Differences or FOD) ---
            if include_in_diff:
                z_diff = np.zeros(T_span)
                if var_name == '_cons':
                    z_diff[:] = 0.0  # The constant drops out during differencing
                else:
                    diff_col = f'FOD_{var_name}' if orthogonal else f'D_{var_name}'
                    if diff_col not in df_g.columns:
                        if orthogonal:
                            raise ValueError(f"Please calculate {diff_col} using panel.get_fod('{var_name}') prior to estimation.")
                        else:
                            df_g[diff_col] = df_g[var_name].diff()
                    
                    for t in range(t_min + 1, t_max + 1):
                        val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t, diff_col]
                        if not val.empty and not pd.isna(val.iloc[0]):
                            z_diff[t - t_min] = val.iloc[0]
                Z_g[:T_span, col_idx] = z_diff
                col_idx += 1
            
            # --- BLOCK B: Level Equation ---
            if include_in_level:
                z_level = np.zeros(T_span)
                if var_name == '_cons':
                    z_level[:] = 1.0  # The constant remains intact in the level equation
                else:
                    for t in range(t_min + 1, t_max + 1):
                        val = df_g.loc[df_g.index.get_level_values(self.panel.time_col) == t, var_name]
                        if not val.empty and not pd.isna(val.iloc[0]):
                            z_level[t - t_min] = val.iloc[0]
                Z_g[T_span:, col_idx] = z_level
                
            Z_iv_list.append(Z_g)
            
        return np.vstack(Z_iv_list)