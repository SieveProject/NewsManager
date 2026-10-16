"""Plugin `plugins/mean_variance.py:MeanVariance`.

Markowitz: max w'μ - (γ/2) w'Σ̂w s.t. Σw = 1 e limites.

μ é a média de Bayes-Stein (Jorion, 1986) dos retornos da janela, encolhida
para a média da carteira GMV de Σ̂; γ = aversão a risco (padrão 5).

    params = { cov = "LW fator único", gamma = 5 }
"""

from newsmanager.portfolio.optimize import bayes_stein_mean, mean_variance
from newsmanager.backtester.models.base import CovModel


class MeanVariance(CovModel):
    def __init__(self, gamma: float = 5.0, **kw):
        super().__init__(**kw)
        self.gamma = gamma

    def optimize(self, S, ctx):
        mu, phi = bayes_stein_mean(ctx.returns, S)
        ctx.info["bs_phi"] = phi
        return mean_variance(S, mu, self.gamma, ctx.min_weight, ctx.max_weight)
