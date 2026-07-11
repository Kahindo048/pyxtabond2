"""
Construction of GMM instrument matrices (block-diagonal Z).

Generates Arellano-Bond style lag instruments and standard IV columns for
difference and system GMM, including collapsed specifications.
"""

import numpy as np
from .data_utils import PanelData, scatter_to_grid

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _numba_disk_cache() -> bool:
    name = globals().get("__name__", "")
    return bool(__package__) and "<" not in name and "dynamic" not in name.lower()


_NUMBA_CACHE = _numba_disk_cache()


def _n_cols_diff_standard(T_span: int, l_min: int, l_max: int) -> int:
    return sum(max(0, min(l_max, row) - l_min + 1) for row in range(1, T_span))


if _HAS_NUMBA:
    @njit(cache=_NUMBA_CACHE)
    def _fill_block_a_collapse(Z_g, level_grid, l_min, l_max):
        col_idx = 0
        T_span = len(level_grid)
        for lag in range(l_min, l_max + 1):
            for row_idx in range(1, T_span):
                t_source_idx = row_idx - lag
                if t_source_idx >= 0:
                    v = level_grid[t_source_idx]
                    if not np.isnan(v):
                        Z_g[row_idx, col_idx] = v
            col_idx += 1

    @njit(cache=_NUMBA_CACHE)
    def _fill_block_a_standard(Z_g, level_grid, l_min, l_max):
        T_span = len(level_grid)
        col_idx = 0
        for row_idx in range(1, T_span):
            actual_l_max = min(l_max, row_idx)
            for lag in range(l_min, actual_l_max + 1):
                t_source_idx = row_idx - lag
                if t_source_idx >= 0:
                    v = level_grid[t_source_idx]
                    if not np.isnan(v):
                        Z_g[row_idx, col_idx] = v
                col_idx += 1

    @njit(cache=_NUMBA_CACHE)
    def _fill_block_b_collapse(Z_g, diff_grid, T_span, n_cols_diff, dlag):
        col_idx = n_cols_diff
        for row_idx in range(1, T_span):
            z_row = T_span + row_idx
            t_source_idx = row_idx - dlag
            if t_source_idx >= 0:
                v = diff_grid[t_source_idx]
                if not np.isnan(v):
                    Z_g[z_row, col_idx] = v

    @njit(cache=_NUMBA_CACHE)
    def _fill_block_b_standard(Z_g, diff_grid, T_span, n_cols_diff, dlag):
        col_idx = n_cols_diff
        for row_idx in range(1, T_span):
            z_row = T_span + row_idx
            t_source_idx = row_idx - dlag
            if t_source_idx >= 0:
                v = diff_grid[t_source_idx]
                if not np.isnan(v):
                    Z_g[z_row, col_idx] = v
            col_idx += 1

    @njit(cache=_NUMBA_CACHE)
    def _fill_iv_diff(z_diff, diff_grid):
        for row_idx in range(1, len(z_diff)):
            v = diff_grid[row_idx]
            if not np.isnan(v):
                z_diff[row_idx] = v

    @njit(cache=_NUMBA_CACHE)
    def _fill_iv_level(z_level, level_grid):
        for row_idx in range(1, len(z_level)):
            v = level_grid[row_idx]
            if not np.isnan(v):
                z_level[row_idx] = v


def _fill_block_a_collapse_py(Z_g, level_grid, l_min, l_max):
    col_idx = 0
    T_span = len(level_grid)
    for lag in range(l_min, l_max + 1):
        for row_idx in range(1, T_span):
            t_source_idx = row_idx - lag
            if t_source_idx >= 0:
                v = level_grid[t_source_idx]
                if not np.isnan(v):
                    Z_g[row_idx, col_idx] = v
        col_idx += 1


def _fill_block_a_standard_py(Z_g, level_grid, l_min, l_max):
    T_span = len(level_grid)
    col_idx = 0
    for row_idx in range(1, T_span):
        actual_l_max = min(l_max, row_idx)
        for lag in range(l_min, actual_l_max + 1):
            t_source_idx = row_idx - lag
            if t_source_idx >= 0:
                v = level_grid[t_source_idx]
                if not np.isnan(v):
                    Z_g[row_idx, col_idx] = v
            col_idx += 1


def _fill_block_b_collapse_py(Z_g, diff_grid, T_span, n_cols_diff, dlag):
    col_idx = n_cols_diff
    for row_idx in range(1, T_span):
        z_row = T_span + row_idx
        t_source_idx = row_idx - dlag
        if t_source_idx >= 0:
            v = diff_grid[t_source_idx]
            if not np.isnan(v):
                Z_g[z_row, col_idx] = v


def _fill_block_b_standard_py(Z_g, diff_grid, T_span, n_cols_diff, dlag):
    col_idx = n_cols_diff
    for row_idx in range(1, T_span):
        z_row = T_span + row_idx
        t_source_idx = row_idx - dlag
        if t_source_idx >= 0:
            v = diff_grid[t_source_idx]
            if not np.isnan(v):
                Z_g[z_row, col_idx] = v
        col_idx += 1


def _fill_iv_diff_py(z_diff, diff_grid):
    for row_idx in range(1, len(z_diff)):
        v = diff_grid[row_idx]
        if not np.isnan(v):
            z_diff[row_idx] = v


def _fill_iv_level_py(z_level, level_grid):
    for row_idx in range(1, len(z_level)):
        v = level_grid[row_idx]
        if not np.isnan(v):
            z_level[row_idx] = v


if not _HAS_NUMBA:
    _fill_block_a_collapse = _fill_block_a_collapse_py
    _fill_block_a_standard = _fill_block_a_standard_py
    _fill_block_b_collapse = _fill_block_b_collapse_py
    _fill_block_b_standard = _fill_block_b_standard_py
    _fill_iv_diff = _fill_iv_diff_py
    _fill_iv_level = _fill_iv_level_py


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
        times = panel.data.index.get_level_values(panel.time_col).to_numpy()
        self.t_min = int(times.min())
        self.t_max = int(times.max())
        self.T_span = self.t_max - self.t_min + 1
        self._time_offsets = (times - self.t_min).astype(np.int64)
        self._group_slices = panel._group_slices
        # Skip columns that aren't numeric (e.g. a string cluster() variable
        # riding along in the dataframe) -- they're never looked up by name
        # below, so a failed conversion just means "not used here".
        self._column_arrays = {}
        for col in panel.data.columns:
            try:
                self._column_arrays[col] = panel.data[col].to_numpy(dtype=np.float64, copy=False)
            except (ValueError, TypeError):
                continue

    def _time_grid(self, col_name: str, group_slice: slice) -> np.ndarray:
        """Map group observations onto the global time grid."""
        offsets = self._time_offsets[group_slice]
        return scatter_to_grid(self._column_arrays[col_name][group_slice], offsets, self.T_span)

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
        if l_max is None:
            l_max = self.T_span - 1

        if collapse:
            n_cols_diff = l_max - l_min + 1
            n_cols_level = 1
        else:
            n_cols_diff = _n_cols_diff_standard(self.T_span, l_min, l_max)
            n_cols_level = self.T_span - 1

        n_cols_total = n_cols_diff + n_cols_level
        diff_col_name = f"D_{var_name}"
        dlag = l_min - 1

        Z_sys_list = []
        for sl in self._group_slices:
            Z_g = np.zeros((2 * self.T_span, n_cols_total), dtype=np.float64)
            level_grid = self._time_grid(var_name, sl)

            if diff_col_name in self._column_arrays:
                diff_grid = self._time_grid(diff_col_name, sl)
            else:
                offsets = self._time_offsets[sl]
                vals = self._column_arrays[var_name][sl]
                diff_grid = np.full(self.T_span, np.nan, dtype=np.float64)
                if len(vals) > 1:
                    diff_offsets = offsets[1:]
                    diff_grid[diff_offsets] = vals[1:] - vals[:-1]

            if collapse:
                _fill_block_a_collapse(Z_g, level_grid, l_min, l_max)
                _fill_block_b_collapse(Z_g, diff_grid, self.T_span, n_cols_diff, dlag)
            else:
                _fill_block_a_standard(Z_g, level_grid, l_min, l_max)
                _fill_block_b_standard(Z_g, diff_grid, self.T_span, n_cols_diff, dlag)

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
        n_cols = int(include_in_diff) + int(include_in_level)
        if n_cols == 0:
            raise ValueError("The instrument must be included in at least one equation (diff or level).")

        diff_col = f"FOD_{var_name}" if orthogonal else f"D_{var_name}"
        Z_iv_list = []

        for sl in self._group_slices:
            Z_g = np.zeros((2 * self.T_span, n_cols), dtype=np.float64)
            col_idx = 0

            if include_in_diff:
                z_diff = np.zeros(self.T_span, dtype=np.float64)
                if var_name != "_cons":
                    if diff_col not in self._column_arrays:
                        if orthogonal:
                            raise ValueError(
                                f"Please calculate {diff_col} using panel.get_fod('{var_name}') prior to estimation."
                            )
                        offsets = self._time_offsets[sl]
                        vals = self._column_arrays[var_name][sl]
                        diff_grid = np.full(self.T_span, np.nan, dtype=np.float64)
                        if len(vals) > 1:
                            diff_grid[offsets[1:]] = vals[1:] - vals[:-1]
                    else:
                        diff_grid = self._time_grid(diff_col, sl)
                    _fill_iv_diff(z_diff, diff_grid)
                Z_g[: self.T_span, col_idx] = z_diff
                col_idx += 1

            if include_in_level:
                z_level = np.zeros(self.T_span, dtype=np.float64)
                if var_name == "_cons":
                    z_level[:] = 1.0
                else:
                    level_grid = self._time_grid(var_name, sl)
                    _fill_iv_level(z_level, level_grid)
                Z_g[self.T_span :, col_idx] = z_level

            Z_iv_list.append(Z_g)

        return np.vstack(Z_iv_list)
