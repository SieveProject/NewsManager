"""Plugin `plugins/max_diversification.py:MaxDiversification`.

Máxima diversificação (Choueifaty e Coignard, 2008): max w'σ / √(w'Σ̂w)
s.t. Σw = 1 e limites, σ = √diag(Σ̂).

    params = { cov = "LW fator único" }
"""

import numpy as np

from newsmanager.portfolio.optimize import max_ratio
from newsmanager.backtester.models.base import CovModel


class MaxDiversification(CovModel):
    def optimize(self, S, ctx):
        w, r = max_ratio(S, np.sqrt(np.diag(S)), ctx.min_weight, ctx.max_weight)
        ctx.info["diversification_ratio"] = r
        return w
