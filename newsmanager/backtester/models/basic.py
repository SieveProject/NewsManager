"""Compatibilidade: nomes usados pelos TOMLs da seção 12 (commit 54c5095).
Modelos novos: um módulo por modelo neste pacote (ver __init__.py)."""

from . import min_variance
from .base import describe  # noqa: F401
from .covariance import STATISTICAL as ESTIMATORS  # noqa: F401
from .equal_weight import EqualWeight  # noqa: F401
from .inverse_volatility import InverseVolatility  # noqa: F401


class MinVariance(min_variance.MinVariance):
    """MinVariance(estimator=...) -> MinVariance(cov=...)."""

    def __init__(self, estimator: str = "LW fator único"):
        super().__init__(cov=estimator)
