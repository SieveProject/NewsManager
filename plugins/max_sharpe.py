"""Plugin `plugins/max_sharpe.py:MaxSharpe`.

Máximo Sharpe (carteira tangente, rf = 0): max w'μ / √(w'Σ̂w) s.t. Σw = 1 e
limites, com μ de Bayes-Stein (Jorion, 1986). Se nenhuma carteira viável tem
w'μ > 0, devolve a de variância mínima.

    params = { cov = "LW fator único" }
"""

import numpy as np

from newsmanager.portfolio.optimize import bayes_stein_mean, max_ratio
from newsmanager.backtester.models.base import CovModel


class MaxSharpe(CovModel):
    def optimize(self, S, ctx):
        mu, phi = bayes_stein_mean(ctx.returns, S)
        w, r = max_ratio(S, mu, ctx.min_weight, ctx.max_weight)
        ctx.info.update({"bs_phi": phi, "ex_ante_sharpe": float(r * np.sqrt(252))})
        return w
