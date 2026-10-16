"""Base dos plugins que otimizam sobre uma Σ̂: escolhe o estimador (`cov`,
ver covariance.py), registra os diagnósticos comuns e delega a otimização."""

from __future__ import annotations

import numpy as np

from ..plugin import Context, Env, Model
from . import covariance


def describe(ctx: Context, S: np.ndarray, w: np.ndarray) -> None:
    """Diagnósticos que todo modelo de Σ deveria registrar: número de condição
    da correlação estimada (λmax/λmin, López de Prado) e vol. ex ante anual."""
    sd = np.sqrt(np.diag(S))
    ev = np.linalg.eigvalsh(S / np.outer(sd, sd))
    ctx.info["cond"] = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    ctx.info["ex_ante_vol"] = float(np.sqrt(max(w @ S @ w, 0) * 252))


class CovModel(Model):
    """Subclasses implementam optimize(S, ctx) -> pesos dentro dos limites."""

    def __init__(self, cov: str = "LW fator único", cov_params: dict | None = None):
        self.cov = covariance.make(cov, cov_params)

    def setup(self, env: Env) -> None:
        self.cov.setup(env)

    def optimize(self, S: np.ndarray, ctx: Context) -> np.ndarray:
        raise NotImplementedError

    def weights(self, ctx: Context) -> np.ndarray:
        S = self.cov(ctx)
        w = self.optimize(S, ctx)
        describe(ctx, S, w)
        return w
