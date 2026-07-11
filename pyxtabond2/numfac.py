"""
Factor-number selection for PCA-GMM (interactive fixed effects).

Implements the ER criterion (Ahn & Horenstein, 2013) and ICp2 criterion
(Bai & Ng, 2002), following Stata's ``xtnumfac`` logic. Includes an EM-PCA
routine to handle missing values (NaNs) in unbalanced panels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

def em_pca_imputation(E_mat, kmax, max_iter=100, tol=1e-5):
    """
    Expectation-Maximization PCA (EM-PCA) to impute missing panel cells via a
    rank-``kmax`` reconstruction. Based on the iterative principles for
    unbalanced panels (e.g., Bai and Ng, 2021). Shared by the factor-count
    selection in this module and the one-shot panel-matrix builders in
    :mod:`~pyxtabond2.ife` (Y/X/instrument levels and the initial ``r='auto'``
    residual matrix) -- not just residuals, hence the observed-mean init below
    rather than a hardcoded 0.0.
    """
    print(f">> [xtnumfac] Missing values detected. Running EM-PCA imputation (max_iter={max_iter})...")

    # Create a mask for NaN values
    mask = np.isnan(E_mat)

    # Initialize missing values at the observed-cell mean (this is ~0 for a
    # residual matrix, and the right anchor for a raw-level matrix too).
    E_filled = np.where(mask, np.nanmean(E_mat), E_mat)
    
    for i in range(max_iter):
        # Step E: Extract factors using SVD on the filled matrix
        U, S, Vt = svd(E_filled, full_matrices=False)
        
        # Keep only the top kmax components
        S_k = S[:kmax]
        U_k = U[:, :kmax]
        Vt_k = Vt[:kmax, :]
        
        # Step M: Reconstruct the common component matrix
        C = U_k @ np.diag(S_k) @ Vt_k
        
        # Update the filled matrix
        E_new = E_filled.copy()
        E_new[mask] = C[mask]
        
        # Check convergence
        diff = np.linalg.norm(E_new[mask] - E_filled[mask]) / (np.linalg.norm(E_new[mask]) + 1e-10)
        E_filled = E_new
        
        if diff < tol:
            print(f">> [xtnumfac] EM-PCA converged in {i+1} iterations.")
            return E_filled
            
    print(f">> [xtnumfac] Warning: EM-PCA reached max_iter ({max_iter}) without full convergence.")
    return E_filled

def estimate_num_factors(E_mat, kmax=8):
    """
    Exact translation of the xtnumfac algorithm (Stata/Mata).
    
    Implements the Bai & Ng (2002) information criteria and the Ahn & Horenstein (2013) 
    eigenvalue ratio criteria to determine the optimal number of interactive unobserved factors.
    Handles NaNs automatically using EM-PCA.

    Parameters
    ----------
    E_mat : np.ndarray
        The matrix of residuals (typically N x T or T x N) from which to extract factors.
    kmax : int, optional
        The maximum number of factors to test. Default is 8.

    Returns
    -------
    dict
        A comprehensive dictionary containing the optimal number of factors according 
        to each criterion, along with their historical values across all tested k.
    """
    T, N = E_mat.shape
    minNT = min(N, T)
    
    if minNT < (kmax + 5):
        kmax = max(1, minNT - 5)

    # Impute missing values if any
    if np.isnan(E_mat).any():
        E_mat = em_pca_imputation(E_mat, kmax=kmax)

    U, S, Vt = svd(E_mat, full_matrices=False)
    mus = (S**2) / (N * T)
    V0 = np.mean(E_mat**2)
    
    # --- IC_p2 Criterion (Bai & Ng) ---
    IC_p2 = np.zeros(kmax + 1)
    IC_p2[0] = np.log(V0)
    penalty_IC2 = ((N + T) / (N * T)) * np.log(minNT)
    suffix_var = np.cumsum(mus[::-1])[::-1]  # suffix_var[k] = sum(mus[k:minNT])
    V_val = np.zeros(kmax + 2)
    V_val[0] = V0
    V_val[1 : kmax + 1] = suffix_var[1 : kmax + 1]
    safe_V = np.maximum(V_val[1 : kmax + 1], 1e-16)
    IC_p2[1 : kmax + 1] = np.log(safe_V) + np.arange(1, kmax + 1) * penalty_IC2

    # --- ER Criterion (Ahn & Horenstein) ---
    ER = np.zeros(kmax + 1)
    mockEV = V0 / np.log(minNT)
    ER[0] = mockEV / mus[0]
    
    for k in range(1, kmax + 1):
        denominator = mus[k] if mus[k] > 0 else 1e-16
        ER[k] = mus[k-1] / denominator

    # argmax over the *whole* vector (index 0 = the mockEV/mus[0] term, Ahn &
    # Horenstein's stand-in for a "zero-th" eigenvalue) so that r=0 is a
    # reachable outcome, exactly like xtnumfac.ado's bestnum_ic_int: r=0 is
    # correctly selected when there is no factor structure at all -- a
    # defining feature of the ER estimator versus Bai & Ng's IC criteria,
    # which cannot conclude "0 factors" without this device.
    best_er = int(np.argmax(ER))
    best_ic2 = int(np.argmin(IC_p2[1:]) + 1)
    
    print(f">> [xtnumfac] ER Criterion (Ahn/Horenstein) : r = {best_er}")
    print(f">> [xtnumfac] ICp2 Criterion (Bai/Ng)      : r = {best_ic2}")
    
    # Return a comprehensive dictionary with the history of criteria
    return {
        'best_er': best_er,
        'best_ic2': best_ic2,
        'k_values': np.arange(1, kmax + 1),
        'ER': ER[1:],
        'IC2': IC_p2[1:]
    }

def show_factor_selection(fac_results, mode='graph'):
    """
    Displays the factor number selection criteria as a formatted table or a graph.
    
    Parameters
    ----------
    fac_results : dict
        The dictionary returned by the `estimate_num_factors` function.
    mode : str, optional
        'graph' for a side-by-side plot, or 'table' for a Pandas DataFrame console output. 
        Default is 'graph'.

    Returns
    -------
    pd.DataFrame or matplotlib.figure.Figure
        The generated DataFrame if mode is 'table', or the Figure object if mode is 'graph'.
    """
    k_vals = fac_results['k_values']
    er_vals = fac_results['ER']
    ic2_vals = fac_results['IC2']
    best_er = fac_results['best_er']
    best_ic2 = fac_results['best_ic2']
    
    if mode == 'table':
        df = pd.DataFrame({
            'k (Factors)': k_vals,
            'ER (Ahn & Horenstein Ratio)': er_vals,
            'ICp2 (Bai & Ng Criterion)': ic2_vals
        })
        df.set_index('k (Factors)', inplace=True)
        print("\n=== FACTOR NUMBER SELECTION (r) ===")
        print(df.to_string(float_format="%.4f"))
        print(f"\n>> ER Conclusion   : Maximum reached at k = {best_er}")
        print(f">> ICp2 Conclusion : Minimum reached at k = {best_ic2}\n")
        return df
        
    elif mode == 'graph':
        # Create the figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Graph 1 : ER Criterion ---
        ax1.plot(k_vals, er_vals, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
        ax1.axvline(x=best_er, color='#e74c3c', linestyle='--', alpha=0.5) 
        ax1.scatter(best_er, er_vals[best_er-1], color='#e74c3c', s=150, zorder=5, label=f'Max (r={best_er})')
        
        ax1.set_title('ER Criterion (Ahn & Horenstein)', fontsize=12, pad=10, fontweight='bold')
        ax1.set_xlabel('Number of factors (k)')
        ax1.set_ylabel('Eigenvalue Ratio')
        ax1.set_xticks(k_vals)
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend()
        
        # --- Graph 2 : ICp2 Criterion ---
        ax2.plot(k_vals, ic2_vals, marker='s', linestyle='-', color='#2980b9', linewidth=2)
        ax2.axvline(x=best_ic2, color='#e74c3c', linestyle='--', alpha=0.5)
        ax2.scatter(best_ic2, ic2_vals[best_ic2-1], color='#e74c3c', s=150, zorder=5, label=f'Min (r={best_ic2})')
        
        ax2.set_title('ICp2 Criterion (Bai & Ng)', fontsize=12, pad=10, fontweight='bold')
        ax2.set_xlabel('Number of factors (k)')
        ax2.set_ylabel("Information Criterion Value")
        ax2.set_xticks(k_vals)
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.legend()
        
        plt.suptitle('Selection of the optimal number of interactive factors', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()
        return fig
        
    else:
        raise ValueError("The 'mode' parameter must be 'graph' or 'table'.")