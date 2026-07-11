"""
Per-variable-group instrument specifications.

Mirrors Stata's ``xtabond2`` repeated ``gmm(varlist, ...)``/``iv(varlist,
...)`` option syntax -- each call is a *group* with its own suboptions --
instead of one global ``lag_limits_diff``/``collapse`` setting applied
uniformly to every GMM-style variable.

Known limitation: Stata's further ``passthru``, ``mz``, and ``split``
suboptions are not implemented here.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

_VALID_EQUATIONS = ("both", "diff", "level")


@dataclass
class GMMStyle:
    """
    One Arellano-Bond/Blundell-Bond GMM-style instrument group
    (Stata's ``gmm(varlist, lag() collapse eq())``).

    Parameters
    ----------
    variables : str or list of str
        Variable(s) instrumented GMM-style in this group.
    lag : tuple(int, int or None), optional
        ``(min_lag, max_lag)`` for the differenced-equation instruments.
        ``max_lag=None`` means "as many as available". Default ``(2, None)``.
    collapse : bool, optional
        Collapse this group's instruments into one column per lag distance
        instead of one column per time period x lag. Default False.
    equation : {"both", "diff", "level"}, optional
        Which equation(s) this group instruments. Only meaningful for System
        GMM (``model_type='system'``); Difference GMM only ever has a
        differenced equation. Default "both".
    """

    variables: Union[str, List[str]]
    lag: Tuple[int, Optional[int]] = (2, None)
    collapse: bool = False
    equation: str = "both"

    def __post_init__(self):
        if isinstance(self.variables, str):
            self.variables = [self.variables]
        if self.equation not in _VALID_EQUATIONS:
            raise ValueError(f"equation must be one of {_VALID_EQUATIONS}, got {self.equation!r}")


@dataclass
class IVStyle:
    """
    One standard-instrument group (Stata's ``iv(varlist, eq())``).

    Parameters
    ----------
    variables : str or list of str
        Variable(s) used as their own instruments in this group.
    equation : {"both", "diff", "level"}, optional
        Which equation(s) this group instruments. Only meaningful for System
        GMM. Default "both".
    """

    variables: Union[str, List[str]]
    equation: str = "both"

    def __post_init__(self):
        if isinstance(self.variables, str):
            self.variables = [self.variables]
        if self.equation not in _VALID_EQUATIONS:
            raise ValueError(f"equation must be one of {_VALID_EQUATIONS}, got {self.equation!r}")


def normalize_gmm_styles(gmm_vars, lag_limits_diff, collapse, gmm_styles) -> List[GMMStyle]:
    """
    Build the normalized list of :class:`GMMStyle` groups to instrument.

    If ``gmm_styles`` is given, it takes precedence (advanced, per-group
    control). Otherwise a single default group is built from the flat
    ``gmm_vars``/``lag_limits_diff``/``collapse`` shortcut.
    """
    if gmm_styles is not None:
        return list(gmm_styles)
    if not gmm_vars:
        return []
    return [GMMStyle(variables=list(gmm_vars), lag=lag_limits_diff, collapse=collapse)]


def normalize_iv_styles(iv_vars, iv_styles) -> List[IVStyle]:
    """
    Build the normalized list of :class:`IVStyle` groups to instrument.

    If ``iv_styles`` is given, it takes precedence. Otherwise a single
    default group (``equation="both"``) is built from the flat ``iv_vars``
    shortcut.
    """
    if iv_styles is not None:
        return list(iv_styles)
    if not iv_vars:
        return []
    return [IVStyle(variables=list(iv_vars))]
