"""
PyXtabond2 — Dynamic panel GMM estimation for Python.

This package replicates the core econometric functionality of Stata's
``xtabond2`` (Roodman, 2009), including:

* Difference and System GMM (Arellano-Bond / Blundell-Bond)
* Two-step estimation with Windmeijer (2005) finite-sample correction
* Forward orthogonal deviations (Arellano & Bover, 1995)
* Collapsed instruments, PCA-GMM with interactive fixed effects
* Publication-ready tables via :class:`~pyxtabond2.exporter.GMMStargazer`

References
----------
.. [1] Roodman, D. (2009). How to do xtabond2. *Stata Journal*, 9(1), 86-136.
.. [2] Windmeijer, F. (2005). A finite sample correction for the variance
       of linear efficient two-step GMM estimators. *Journal of Econometrics*,
       126(1), 25-51.
.. [3] Arellano, M., & Bover, O. (1995). Another look at the instrumental
       variable estimation of error-components models. *Journal of Econometrics*,
       68(1), 29-51.
"""

from .estimator import PyXtabond2
from .results import PyXtabond2Results
from .data_utils import PanelData
from .exporter import GMMStargazer
from .load_data import list_datasets, load_dataset
from .specs import GMMStyle, IVStyle

__version__ = "0.0.3"
__author__ = "Frederick Kahindo"

__all__ = [
    "PyXtabond2",
    "PyXtabond2Results",
    "PanelData",
    "GMMStargazer",
    "GMMStyle",
    "IVStyle",
    "load_dataset",
    "list_datasets",
    "__version__",
]
