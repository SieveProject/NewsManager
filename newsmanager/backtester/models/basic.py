"""Modelos de referência que só usam retornos.

  EqualWeight       1/N (o teto de peso nunca morde quando N >= 1/max_weight)
  InverseVolatility peso ∝ 1/σ_i
  MinVariance       min w'Σ̂w s.t. Σw = 1 e os limites de peso, com Σ̂ de um
                    estimador de newsmanager/portfolio/estimators.py
"""

from __future__ import annotations

import numpy as np

from ...portfolio import estimators as est
from ...portfolio.optimize import min_variance
from ..plugin import Context, Model

ESTIMATORS = {
    "amostral": est.sample_cov,
    "LW identidade": est.lw_identity,
    "LW fator único": est.lw_single_factor,
    "LW corr. constante": est.lw_const_corr,
    "LW não linear": est.lw_nonlinear,
    "RMT": est.rmt_clip,
    "fator + resíduo diagonal": lambda X: est.factor_residual(X, "identity"),
    "fator + resíduo corr. constante": lambda X: est.factor_residual(X, "cc"),
}


def describe(ctx: Context, S: np.ndarray, w: np.ndarray) -> None:
    """Diagnósticos que todo modelo de Σ deveria registrar: número de condição
    da correlação estimada (λmax/λmin, López de Prado) e vol. ex ante anual."""
    sd = np.sqrt(np.diag(S))
    ev = np.linalg.eigvalsh(S / np.outer(sd, sd))
    ctx.info["cond"] = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    ctx.info["ex_ante_vol"] = float(np.sqrt(max(w @ S @ w, 0) * 252))


class EqualWeight(Model):
    def weights(self, ctx: Context) -> np.ndarray:
        return np.full(ctx.p, 1 / ctx.p)


class InverseVolatility(Model):
    def weights(self, ctx: Context) -> np.ndarray:
        sd = np.nanstd(ctx.returns, axis=0, ddof=1)
        w = 1 / sd
        return w / w.sum()


class MinVariance(Model):
    """Variância mínima com o estimador `estimator` (nome em ESTIMATORS)."""

    def __init__(self, estimator: str = "LW fator único"):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator {estimator!r}: use um de {list(ESTIMATORS)}")
        self.estimator = estimator

    def covariance(self, ctx: Context) -> np.ndarray:
        out = ESTIMATORS[self.estimator](ctx.returns)
        return out[0] if isinstance(out, tuple) else out

    def weights(self, ctx: Context) -> np.ndarray:
        S = self.covariance(ctx)
        w = min_variance(S, ctx.min_weight, ctx.max_weight)
        describe(ctx, S, w)
        return w
