"""
Panel data manager for dynamic GMM estimation.
Reproduces Stata's 'tsset' logic and temporal operators.
"""

import pandas as pd
import numpy as np

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
        
        # Sort values strictly by group and time to prevent cross-contamination of lags
        df = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
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
        
        self.data = df_clean

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
        return self.data.groupby(level=0)[var_name].shift(lags)

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
        return self.data.groupby(level=0)[var_name].diff(periods=1)

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
            
        out = pd.Series(index=self.data.index, dtype=float)
        
        for g, df_g in self.data.groupby(level=0):
            vals = df_g[var_name].values
            fod_vals = np.full(len(vals), np.nan)
            
            for i in range(len(vals) - 1):
                future_vals = vals[i+1:]
                
                # Strict handling of missing data in future observations (Casewise robust)
                valid_future = future_vals[~np.isnan(future_vals)]
                T_it = len(valid_future)
                
                if T_it > 0:
                    c_t = np.sqrt(T_it / (T_it + 1))
                    fod_vals[i] = c_t * (vals[i] - np.mean(valid_future))
                    
            out.loc[g] = fod_vals
            
        return out