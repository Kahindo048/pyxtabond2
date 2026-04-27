import numpy as np

class VarianceEstimator:
    """
    Variance-Covariance Matrix (VCE) estimator for two-step GMM models.
    
    Standard two-step GMM standard errors are known to be severely downward biased 
    in finite samples. This class implements the exact finite-sample correction 
    derived by Windmeijer (2005), adhering strictly to the algorithmic implementation 
    found in Stata's 'xtabond2' by David Roodman.

    Parameters
    ----------
    engine : GMMEngine
        The core engine instance containing the data matrices (y, X, Z) and group identifiers.
    beta1 : np.ndarray
        The estimated coefficient vector from the one-step GMM.
    beta2 : np.ndarray
        The estimated coefficient vector from the two-step GMM.
    W1 : np.ndarray
        The initial weighting matrix used in the first step.
    W2 : np.ndarray
        The robust weighting matrix (inverse of the cross-moments) used in the second step.

    Examples
    --------
    >>> # Internally called within GMMEngine.estimate_two_step_robust()
    >>> estimator = VarianceEstimator(engine, beta1, beta2, W1, W2)
    >>> VCE_robust = estimator.compute_windmeijer_vce()
    >>> standard_errors = np.sqrt(np.diag(VCE_robust))
    """
    def __init__(self, engine, beta1, beta2, W1, W2):
        self.y = engine.y
        self.X = engine.X
        self.Z = engine.Z
        self.group_ids = engine.group_ids
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.W1 = W1
        self.W2 = W2
        
        self.n_obs, self.k_vars = self.X.shape
        self.n_instruments = self.Z.shape[1]
        
        # Residuals from step 1 and step 2
        self.e1 = self.y - self.X @ self.beta1
        self.e2 = self.y - self.X @ self.beta2

    def compute_windmeijer_vce(self) -> np.ndarray:
        """
        Calculates the Windmeijer-corrected Variance-Covariance matrix.
        
        The correction involves approximating the derivative of the second-step 
        estimator with respect to the first-step estimator. It requires reconstructing 
        the robust sandwich variance of the first step and accumulating partial derivatives 
        across all panel groups.

        Returns
        -------
        np.ndarray
            The fully corrected, symmetric, robust Variance-Covariance matrix (VCE).
        """
        # Compute X'Z
        XZ = self.X.T @ self.Z
        
        # 1. Calculate V1_naive and V1_robust (The missing link in standard texts)
        V1_naive = np.linalg.pinv(XZ @ self.W1 @ XZ.T)
        VZXA_1 = V1_naive @ XZ @ self.W1
        
        # Matrix 'S': Accumulator of cross-moments (inverse of W2)
        S = np.zeros((self.n_instruments, self.n_instruments))
        unique_groups = np.unique(self.group_ids)
        for g in unique_groups:
            mask = (self.group_ids == g)
            Z_g = self.Z[mask]
            e1_g = self.e1[mask]
            S += Z_g.T @ e1_g @ e1_g.T @ Z_g
            
        # First-step robust sandwich variance
        V1_robust = VZXA_1 @ S @ VZXA_1.T
        
        # 2. Calculate V2_naive and the VZXA_2 projector
        V2_naive = np.linalg.pinv(XZ @ self.W2 @ XZ.T)
        VZXA_2 = V2_naive @ XZ @ self.W2
        
        # 3. The derivative matrix D (Initialized to 0, size j x k)
        D = np.zeros((self.n_instruments, self.k_vars))
        
        # A2Ze: Common vector for step 2 (W2 * Z' * e2)
        A2Ze = self.W2 @ self.Z.T @ self.e2 
        
        for g in unique_groups:
            mask = (self.group_ids == g)
            Z_g = self.Z[mask]
            X_g = self.X[mask]
            e1_g = self.e1[mask]
            
            # Ze1 (1 x j) and ZXi (j x k)
            Ze1 = e1_g.T @ Z_g
            ZXi = Z_g.T @ X_g
            
            # Term 1: Scalar * ZXi
            scalar_part = (Ze1 @ A2Ze)[0, 0]
            
            # Term 2: Matrix (j x j) * ZXi
            # Transposes: Ze1.T is (j x 1), A2Ze.T is (1 x j)
            matrix_part = Ze1.T @ A2Ze.T
            
            # Accumulation (equivalent to Stata xtabond2 internal math)
            D += (scalar_part * ZXi) + (matrix_part @ ZXi)
            
        # 4. Finalization of D matrix projection
        D = VZXA_2 @ D
        
        # 5. Final assembly of the Windmeijer VCE
        # According to Roodman: V2robust = V2_naive + D*V1robust*D' + 2*D*V2_naive
        V_windmeijer = V2_naive + (D @ V1_robust @ D.T) + (D @ V2_naive) + (V2_naive @ D.T)
        
        # Safety symmetrization to handle floating point imprecision
        V_windmeijer = (V_windmeijer + V_windmeijer.T) / 2.0
        
        # Save both for potential debugging or uncorrected reporting
        self.V2_naive = V2_naive
        self.V_windmeijer = V_windmeijer
        
        return V_windmeijer