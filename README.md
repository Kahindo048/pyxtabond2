# PyXtabond2: Dynamic Panel Data Estimation in Python

`pyxtabond2` is a comprehensive Python package for estimating dynamic panel data models using the Generalized Method of Moments (GMM). It aims to faithfully replicate the functionality, matrix algebra, and robust diagnostics of Stata's highly popular `xtabond2` command developed by David Roodman.

Ideal for applied econometrics and macroeconomic research, this package bridges the gap between Python's data science ecosystem and advanced dynamic panel methodologies.

## 🌟 Key Features

* **Difference GMM (Arellano-Bond 1991):** Standard first-differenced GMM estimation.
* **System GMM (Blundell-Bond 1998):** Combined level and differenced equations for highly persistent series.
* **Interactive Fixed Effects (PCA-GMM):** Automatically detect and purge unobserved common factors using Bai & Ng (2002) and Ahn & Horenstein (2013) criteria.
* **Windmeijer (2005) Correction:** Exact finite-sample correction for two-step GMM standard errors.
* **Forward Orthogonal Deviations (FOD):** Arellano-Bover (1995) transformation, maximizing sample size in unbalanced panels with gaps.
* **Instrument Collapsing:** Prevents instrument proliferation and the weakening of overidentification tests.
* **Comprehensive Diagnostics:** Arellano-Bond AR(1)/AR(2) tests, Sargan/Hansen J-tests, and Difference-in-Hansen tests for instrument exogeneity.
* **Direct Export:** Export publication-ready tables directly to LaTeX or Microsoft Word.

## 📦 Installation

You can install `pyxtabond2` directly from PyPI using pip:

```bash
pip install pyxtabond2

🚀 Quick Start
pyxtabond2 comes with integrated example datasets so you can start experimenting immediately.

import pandas as pd
import pyxtabond2

# 1. Load the integrated example data
# Use pyxtabond2.list_datasets() to see all available integrated datasets
df = pyxtabond2.load_dataset('macro_panel.csv')

# 2. Estimate a basic Difference GMM model
model_diff = pyxtabond2.PyXtabond2(
    df=df, 
    id_col='country_id', time_col='year', dep_var='gdp_growth', 
    x_vars=['inflation', 'fdi'], 
    gmm_vars=['gdp_growth', 'inflation'], 
    iv_vars=['fdi'],
    model_type='difference', 
    twostep=False
)
res_diff = model_diff.fit()
res_diff.summary()

# 3. Estimate an advanced System GMM model with Interactive Fixed Effects
model_sys = pyxtabond2.PyXtabond2(
    df=df, 
    id_col='country_id', time_col='year', dep_var='gdp_growth', 
    x_vars=['inflation', 'fdi'], 
    gmm_vars=['gdp_growth', 'inflation'], 
    iv_vars=['fdi'],
    model_type='system',
    twostep=True,      # Two-step estimation
    robust=True,       # Windmeijer correction
    small=True,        # Small sample adjustments
    collapse=True,     # Prevent instrument proliferation
    orthogonal=True,   # Forward Orthogonal Deviations (FOD)
    r='auto'           # Auto-detect interactive fixed effects (PCA-GMM)
)
res_sys = model_sys.fit()

# Print the detailed Stata-style summary to the console
res_sys.summary()

# 4. Export results for publication
res_sys.to_latex("gmm_results.tex", full_output=False)
res_sys.to_word("gmm_results.docx", full_output=False)

```
📖 References & Methodology
This package implements the algorithms and corrections outlined in the following seminal papers:

Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. The Review of Economic Studies.

Arellano, M., & Bover, O. (1995). Another look at the instrumental variable estimation of error-components models. Journal of Econometrics.

Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. Journal of Econometrics.

Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. Journal of Econometrics.

Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. The Stata Journal.

Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica.

Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. Econometrica.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page on the GitHub repository.