"""Plugin `plugins/inverse_volatility.py:InverseVolatility`.

Peso ∝ 1/σ_i (desvio-padrão amostral na janela); teto imposto por projeção."""

import numpy as np

from newsmanager.portfolio.optimize import project_box_simplex
from newsmanager.backtester.plugin import Context, Model


class InverseVolatility(Model):
    def weights(self, ctx: Context) -> np.ndarray:
        w = 1 / np.nanstd(ctx.returns, axis=0, ddof=1)
        return project_box_simplex(w / w.sum(), ctx.min_weight, ctx.max_weight)
