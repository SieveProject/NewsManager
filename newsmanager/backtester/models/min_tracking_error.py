"""Tracking error mínimo contra o benchmark do backtest (precisa de benchmark =
<ticker>): min w'Σ̂w - 2 w'c s.t. Σw = 1 e limites, com c = Cov(R, b) amostral
na janela. Σ̂ é a do estimador escolhido; c e σ²_b não dependem dele.

    params = { cov = "LW fator único" }
"""

import numpy as np

from ...portfolio.estimators import demeaned
from ...portfolio.optimize import min_tracking_error
from .base import CovModel


class MinTrackingError(CovModel):
    def optimize(self, S, ctx):
        b = ctx.benchmark_returns
        if b is None:
            raise ValueError("MinTrackingError precisa de benchmark = <ticker>")
        ok = np.isfinite(b)
        Z = demeaned(ctx.returns[ok])
        bb = b[ok] - b[ok].mean()
        c = Z.T @ bb / (len(bb) - 1)
        w = min_tracking_error(S, c, ctx.min_weight, ctx.max_weight)
        te2 = w @ S @ w - 2 * w @ c + bb @ bb / (len(bb) - 1)
        ctx.info["ex_ante_te"] = float(np.sqrt(max(te2, 0) * 252))
        return w
