"""
Panel data preparation for dynamic GMM estimation.

:class:`PanelData` replicates Stata's ``tsset`` logic: sorting, indexing, and
within-group temporal operators (lags, first differences, forward orthogonal
deviations). Heavy operators use vectorized NumPy and Numba where appropriate.
"""

import pandas as pd
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _numba_disk_cache() -> bool:
    """Persist Numba JIT only for stable package imports (not dynamic test loaders)."""
    name = globals().get("__name__", "")
    return bool(__package__) and "<" not in name and "dynamic" not in name.lower()


if _HAS_NUMBA:
    @njit(cache=_numba_disk_cache())
    def _fod_group_numba(vals: np.ndarray) -> np.ndarray:
        """Forward orthogonal deviations with casewise NaN handling (per group)."""
        T = len(vals)
        fod_vals = np.empty(T, dtype=np.float64)
        for i in range(T):
            fod_vals[i] = np.nan
        for i in range(T - 1):
            s = 0.0
            count = 0
            for j in range(i + 1, T):
                if not np.isnan(vals[j]):
                    s += vals[j]
                    count += 1
            if count > 0:
                c_t = np.sqrt(count / (count + 1.0))
                vi = vals[i]
                if not np.isnan(vi):
                    fod_vals[i] = c_t * (vi - s / count)
        return fod_vals


def _fod_group_vectorized(vals: np.ndarray) -> np.ndarray:
    """Vectorized FOD for groups without missing values."""
    T = len(vals)
    fod_vals = np.full(T, np.nan, dtype=np.float64)
    if T <= 1:
        return fod_vals
    total = np.cumsum(vals)
    n_future = np.arange(T - 1, 0, -1, dtype=np.float64)
    suffix_sum = total[-1] - total[: T - 1]
    mean_future = suffix_sum / n_future
    c_t = np.sqrt(n_future / (n_future + 1.0))
    fod_vals[: T - 1] = c_t * (vals[: T - 1] - mean_future)
    return fod_vals


def _fod_group(vals: np.ndarray) -> np.ndarray:
    if _HAS_NUMBA and np.any(np.isnan(vals)):
        return _fod_group_numba(vals)
    if np.any(np.isnan(vals)):
        return _fod_group_python(vals)
    return _fod_group_vectorized(vals)


def _fod_group_python(vals: np.ndarray) -> np.ndarray:
    """Fallback when Numba is unavailable."""
    T = len(vals)
    fod_vals = np.full(T, np.nan, dtype=np.float64)
    for i in range(T - 1):
        future_vals = vals[i + 1 :]
        valid_future = future_vals[~np.isnan(future_vals)]
        T_it = len(valid_future)
        if T_it > 0 and not np.isnan(vals[i]):
            c_t = np.sqrt(T_it / (T_it + 1.0))
            fod_vals[i] = c_t * (vals[i] - np.mean(valid_future))
    return fod_vals


def scatter_to_grid(vals: np.ndarray, t_off: np.ndarray, T_span: int) -> np.ndarray:
    """Scatter a group's sparse values onto a dense length-``T_span`` array,
    indexed by absolute time offset; unobserved cells are NaN. Mirrors
    Stata xtabond2's internal ``Fill``/``tsfill``-replacement grid
    (``xtabond2.mata:118-172``): position in the returned array always
    equals absolute calendar time, gap or not.
    """
    grid = np.full(T_span, np.nan, dtype=np.float64)
    grid[t_off] = vals
    return grid


def _lag_group(vals: np.ndarray, t_off: np.ndarray, lags: int) -> np.ndarray:
    """Lag by exactly `lags` calendar periods, wherever that source row sits
    in this (possibly gapped) group's own row list -- not just `lags` rows
    earlier positionally, which only equals true calendar adjacency when the
    group has no gap before the target row. A plain fixed-offset shift would
    miss a valid source that a gap has pushed further back in the row list
    than `lags` positions (e.g. lags=2 with one interior gap before the
    target); searchsorted finds it regardless of how many rows back it sits.
    `t_off` is strictly increasing (PanelData sorts by time), so a single
    per-group binary search suffices.

    Fast path: if this group has no gap anywhere (t_off advances by exactly
    1 every row, e.g. a balanced panel, or a group that starts late/ends
    early but is contiguous in between), row position and calendar offset
    move in lockstep, so a plain positional shift already *is* the exact
    calendar-time lag -- mathematically identical to the general path below,
    just without paying for a search. This is the common case (most panels
    have no interior gaps), and it's what every IFE-GMM iteration re-triggers
    for D_/L1_-style columns, so it's worth short-circuiting.
    """
    n = len(t_off)
    if n > 0 and t_off[-1] - t_off[0] == n - 1:
        out = np.full_like(vals, np.nan, dtype=np.float64)
        if lags < n:
            out[lags:] = vals[:-lags]
        return out
    target = t_off - lags
    idx = np.clip(np.searchsorted(t_off, target), 0, len(t_off) - 1)
    found = t_off[idx] == target
    out = np.full_like(vals, np.nan, dtype=np.float64)
    out[found] = vals[idx[found]]
    return out


def _diff_group(vals: np.ndarray, t_off: np.ndarray) -> np.ndarray:
    return vals - _lag_group(vals, t_off, 1)


class PanelData:
    """
    A foundational class to manage and structure panel data for dynamic GMM estimation.
    
    This class replicates the core functionality of Stata's `tsset` command, ensuring 
    that data is properly sorted, indexed, and that temporal operations (like lags and 
    differences) strictly respect group boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame containing the panel data.
    id_col : str
        The name of the column containing the cross-sectional group identifiers (e.g., 'country_id', 'firm_id').
    time_col : str
        The name of the column containing the time variable (e.g., 'year', 'quarter').

    Attributes
    ----------
    data : pd.DataFrame
        The cleaned, sorted, and MultiIndexed DataFrame ready for temporal operations.

    Raises
    ------
    ValueError
        If the DataFrame is empty after dropping missing identifiers or time periods.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyxtabond2.core.data_utils import PanelData
    >>> df = pd.DataFrame({
    ...     'id': [1, 1, 1, 2, 2, 2], 
    ...     'year': [2000, 2001, 2002, 2000, 2001, 2002], 
    ...     'gdp': [5.5, 5.8, 6.1, 3.2, 3.1, 3.4]
    ... })
    >>> panel = PanelData(df, id_col='id', time_col='year')
    >>> print(panel.data.index.names)
    ['id', 'year']
    """
    
    def __init__(self, df: pd.DataFrame, id_col: str, time_col: str):
        self.id_col = id_col
        self.time_col = time_col

        # Track each row's original position through sort/dropna/reindex so a
        # later value-only mutated copy of the same dataframe can be rebuilt
        # via rebuild_fast() without repeating this pandas processing.
        df = df.copy()
        df['__pyxtabond2_pos__'] = np.arange(len(df))

        # Sort values strictly by group and time to prevent cross-contamination of lags
        df = df.sort_values(by=[self.id_col, self.time_col])

        # Clean data: Equivalent to Stata's "count if `t' >= ."
        # Drops rows where the identifier or the time variable is missing
        initial_obs = len(df)
        df_clean = df.dropna(subset=[self.time_col, self.id_col]).copy()

        if len(df_clean) == 0:
            raise ValueError("No valid observations remain after filtering missing IDs and time periods.")
        elif len(df_clean) < initial_obs:
            print(f"Warning: {initial_obs - len(df_clean)} observations were dropped due to missing ID or time values.")

        # Create the MultiIndex (equivalent to 'tsset') and enforce strict sorting
        df_clean.set_index([self.id_col, self.time_col], inplace=True)
        df_clean.sort_index(level=[0, 1], inplace=True)

        self._sort_perm = df_clean.pop('__pyxtabond2_pos__').to_numpy()
        self.data = df_clean
        self._init_group_slices()

        # Absolute time offset of every row (0 = the panel's global t_min),
        # used by _lag_group/_diff_group to verify true calendar adjacency
        # instead of assuming a group's own rows are gap-free.
        times = self.data.index.get_level_values(self.time_col).to_numpy()
        self._time_offsets = (times - times.min()).astype(np.int64)

    @classmethod
    def rebuild_fast(cls, reference: "PanelData", df_new: pd.DataFrame) -> "PanelData":
        """
        Rebuild for ``df_new``, reusing the row selection/sort order and
        group slices already computed for ``reference`` instead of repeating
        the ``sort_values``/``dropna``/``groupby`` cost.

        Only valid when ``df_new`` has the same length and row order as the
        dataframe ``reference`` was built from, with identical ``id_col``/
        ``time_col`` values (i.e. only *other* columns were mutated) --
        exactly the case in the IFE-GMM defactoring loop, where the panel
        structure never changes across iterations, only the (defactored)
        variable values do.
        """
        obj = cls.__new__(cls)
        obj.id_col = reference.id_col
        obj.time_col = reference.time_col
        df_sorted = df_new.iloc[reference._sort_perm].drop(columns=[reference.id_col, reference.time_col])
        df_sorted.index = reference.data.index
        obj.data = df_sorted
        obj._sort_perm = reference._sort_perm
        obj._group_slices = reference._group_slices
        obj._time_offsets = reference._time_offsets
        return obj

    def _init_group_slices(self) -> None:
        """Precompute contiguous row slices per panel group for vectorized operators."""
        sizes = self.data.groupby(level=0, sort=False).size()
        ends = sizes.cumsum().to_numpy(dtype=np.int64)
        starts = ends - sizes.to_numpy(dtype=np.int64)
        self._group_slices = [slice(int(s), int(e)) for s, e in zip(starts, ends)]

    def _transform_per_group(self, var_name: str, func) -> pd.Series:
        vals = self.data[var_name].to_numpy(dtype=np.float64, copy=False)
        out = np.empty_like(vals)
        for sl in self._group_slices:
            out[sl] = func(vals[sl], self._time_offsets[sl])
        return pd.Series(out, index=self.data.index, dtype=float)

    def get_lag(self, var_name: str, lags: int = 1) -> pd.Series:
        """
        Generates the temporal lag of a variable within its group.
        
        This method is the equivalent of Stata's `L.var` or `L2.var` operators. 
        It uses a group-by operation to guarantee that lags do not cross over 
        from one individual/group to the next.

        Parameters
        ----------
        var_name : str
            The name of the variable to lag.
        lags : int, optional
            The number of periods to lag. Default is 1.

        Returns
        -------
        pd.Series
            A Series containing the lagged values, retaining the original MultiIndex.

        Raises
        ------
        KeyError
            If the specified variable does not exist in the panel data.

        Examples
        --------
        >>> panel.data['lagged_gdp'] = panel.get_lag('gdp', lags=1)
        """
        if var_name not in self.data.columns:
            raise KeyError(f"The variable '{var_name}' does not exist in the panel.")
        return self._transform_per_group(var_name, lambda v, t: _lag_group(v, t, lags))

    def get_first_difference(self, var_name: str) -> pd.Series:
        """
        Generates the first temporal difference of a variable within its group.
        
        This method is the equivalent of Stata's `D.var` operator.

        Parameters
        ----------
        var_name : str
            The name of the variable to difference.

        Returns
        -------
        pd.Series
            A Series containing the first-differenced values, retaining the original MultiIndex.

        Raises
        ------
        KeyError
            If the specified variable does not exist in the panel data.

        Examples
        --------
        >>> panel.data['diff_gdp'] = panel.get_first_difference('gdp')
        """
        if var_name not in self.data.columns:
            raise KeyError(f"The variable '{var_name}' does not exist in the panel.")
        return self._transform_per_group(var_name, _diff_group)

    def get_fod(self, var_name: str) -> pd.Series:
        """
        Calculates the Forward Orthogonal Deviations (FOD) for a variable.
        
        This implements the Arellano-Bover (1995) transformation used by the 'orthogonal' 
        option in `xtabond2`. Unlike first differences, which subtract the previous observation, 
        FOD subtracts the average of all available future observations. This minimizes data 
        loss in panels with gaps.

        Parameters
        ----------
        var_name : str
            The name of the variable to transform.

        Returns
        -------
        pd.Series
            A Series containing the orthogonally deviated values.

        Raises
        ------
        KeyError
            If the specified variable does not exist in the panel data.

        Examples
        --------
        >>> panel.data['fod_gdp'] = panel.get_fod('gdp')
        """
        if var_name not in self.data.columns:
            raise KeyError(f"The variable '{var_name}' does not exist in the panel.")
        # _fod_group only counts how many future values remain, never their
        # calendar spacing, so it needs no time-offset argument (see its
        # docstring) -- ignore it here to keep _transform_per_group uniform.
        return self._transform_per_group(var_name, lambda v, t: _fod_group(v))
