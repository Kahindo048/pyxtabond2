# PyXtabond2: Dynamic Panel Data Estimation in Python

`pyxtabond2` is a Python package for estimating dynamic panel data models using the Generalized Method of Moments (GMM). It replicates the matrix algebra, options, and diagnostics of Stata's `xtabond2` (David Roodman), and adds an original extension for panels with unobserved interactive fixed effects: **IFE-GMM**.

Ideal for applied econometrics and macroeconomic research, this package bridges the gap between Python's data science ecosystem and dynamic panel methodologies.

📖 **Full documentation**: [`docs/latex/user_manual.pdf`](https://github.com/Kahindo048/pyxtabond2/blob/main/docs/latex/user_manual.pdf) (every option, Stata mapping, diagnostics) and [`docs/latex/methodological_note.pdf`](https://github.com/Kahindo048/pyxtabond2/blob/main/docs/latex/methodological_note.pdf) (full econometric derivations and validation record).

---
## 🌟 Key Features

* **Difference GMM (Arellano-Bond 1991)** and **System GMM (Blundell-Bond 1998)**, one-step and two-step, with all four robust/non-robust combinations Stata supports.
* **Windmeijer (2005) correction:** exact finite-sample correction for two-step robust standard errors.
* **Forward Orthogonal Deviations (FOD):** Arellano-Bover (1995) transformation, maximizing sample size in unbalanced panels with gaps.
* **Instrument collapsing** and **per-variable-group instrument suboptions** (`lag()`, `collapse`, `eq()` via `GMMStyle`/`IVStyle`) to control instrument proliferation.
* **Generalized `h()`, `artests()`, `arlevels`, and `cluster()`** options, matching `xtabond2.mata`.
* **Comprehensive diagnostics:** Arellano-Bond AR(ℓ) tests, Sargan/Hansen J-tests, and Difference-in-Sargan/Hansen tests for instrument exogeneity.
* **IFE-GMM:** an original iterative PCA-defactoring GMM estimator for panels with interactive fixed effects (Bai 2009), with automatic factor-count selection (Bai & Ng 2002; Ahn & Horenstein 2013) and a split-panel jackknife bias correction (Dhaene & Jochmans 2015).
* **Direct export:** publication-ready tables to LaTeX or Microsoft Word, including multi-model comparison tables (`GMMStargazer`).

---
## 📦 Installation

```bash
pip install pyxtabond2
```

For Word export:

```bash
pip install "pyxtabond2[export]"
```

Development install from source:

```bash
git clone https://github.com/Kahindo048/pyxtabond2.git
cd pyxtabond2
pip install -e ".[dev,export]"
```

## 🚀 Quick Start

`pyxtabond2` comes with integrated example datasets so you can start experimenting immediately.

```python
from pyxtabond2 import PanelData, PyXtabond2, load_dataset

# 1. Loading the data
df = load_dataset('df_panel.csv')

# 2. Data preparation
panel = PanelData(df, id_col='Country', time_col='Year')
panel.data['L1_Growth'] = panel.get_lag('Growth', 1)
df_ready = panel.data.reset_index()

modele = PyXtabond2(df_ready,
                    id_col='Country',       # Group identifier (country, firm)
                    time_col='Year',        # Time identifier
                    dep_var='Growth',       # Dependent variable
                    x_vars=['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment'],
                    gmm_vars=['Growth', 'Capital'],  # Arellano-Bond instruments
                    iv_vars=['Ide'],                 # Standard IV instruments
                    model_type='system', twostep=True, robust=True, small=True)

# 3. Estimation
result = modele.fit()
result.summary()

# 4. Export results for publication
result.to_latex("gmm_results.tex", full_output=False)
result.to_word("gmm_results.docx", full_output=False)
```

See `examples/examples.py` for a full workflow with four specifications and comparative export (and `examples/ife_exemples.py` for the IFE-GMM companion workflow), and the User Manual for the complete option reference (including IFE-GMM).

## Performance

`pyxtabond2` vectorizes the panel transforms, instrument construction, and GMM engine with NumPy/Numba, and caches panel structure across the IFE-GMM defactoring loop instead of re-running pandas on every iteration. On a realistic Monte Carlo design (System GMM, N=200, T=46, `collapse=True`, comparing standard GMM / IFE-GMM / IFE-GMM with jackknife bias correction per replication), the current release fits that three-estimator trio roughly **2.4× faster** than before the most recent optimization pass alone (gating diagnostics during IFE-GMM iterations, cheaper rank/inverse computations), and **substantially faster still** relative to earlier releases that re-ran the full data pipeline every IFE-GMM iteration. Always prefer `collapse=True` for simulation-heavy designs (see the User Manual's performance notes for parallelization guidance).

## Known limitations

Weights (`pweight`/`aweight`/`fweight`) and Stata's multi-way `cluster(var1 var2)` are not implemented; a few narrow numerical edge cases (documented in the Methodological Note) can diverge from Stata. See the User Manual's "Known Limitations" section for the complete, current list.

```bash
pytest tests/
```

---
## 📖 References & Methodology

This package implements the algorithms and corrections outlined in the following seminal papers (see `docs/latex/methodological_note.pdf` for the complete bibliography and full derivations):

Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *The Review of Economic Studies*.

Arellano, M., & Bover, O. (1995). Another look at the instrumental variable estimation of error-components models. *Journal of Econometrics*.

Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*.

Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. *Journal of Econometrics*.

Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. *The Stata Journal*.

Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica*.

Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*.

Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. *Econometrica*.

Dhaene, G., & Jochmans, K. (2015). Split-panel jackknife estimation of fixed-effect models. *Review of Economic Studies*.

---
## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page on the GitHub repository.

## License

MIT License. See `LICENSE`.
