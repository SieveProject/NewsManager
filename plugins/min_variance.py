"""Plugin `plugins/min_variance.py:MinVariance`.

Variância mínima: min w'Σ̂w s.t. Σw = 1, min_weight <= w <= max_weight.

    params = { cov = "fator + resíduo setor + notícias", cov_params = { placebo = false } }
"""

from newsmanager.portfolio.optimize import min_variance
from newsmanager.backtester.models.base import CovModel


class MinVariance(CovModel):
    def optimize(self, S, ctx):
        return min_variance(S, ctx.min_weight, ctx.max_weight)
