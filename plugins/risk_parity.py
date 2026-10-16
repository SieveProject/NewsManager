"""Plugin `plugins/risk_parity.py:RiskParity`: paridade de risco ingênua com covariância de Ledoit-Wolf.

Use num TOML:

    [model]
    plugin = "plugins/risk_parity.py:RiskParity"
    params = { lookback = 126, iters = 500 }

Contrato (ver newsmanager/backtester/plugin.py):
  - weights(ctx) recebe só dados ANTERIORES a ctx.date e devolve um vetor
    alinhado a ctx.tickers (ou um dict ticker -> peso) que soma 1;
  - limites de peso em ctx.min_weight / ctx.max_weight (o motor projeta o que
    passar, ou para a run com on_violation = "error");
  - números em ctx.info viram colunas em rebalances.parquet e gráficos no notebook.
"""

import numpy as np

from newsmanager.backtester import Model
from newsmanager.backtester.models.base import describe
from newsmanager.portfolio import estimators as est
from newsmanager.portfolio.optimize import project_box_simplex


class RiskParity(Model):
    def __init__(self, lookback: int = 126, iters: int = 500):
        self.lookback, self.iters = lookback, iters

    def weights(self, ctx):
        X = ctx.returns[-self.lookback:]               # pregões x ativos, NaN onde falta
        S = est.lw_single_factor(X)
        # contribuições de risco iguais: w_i (Σw)_i = constante (iteração de ponto fixo)
        w = np.full(ctx.p, 1 / ctx.p)
        for _ in range(self.iters):
            w = 1 / (S @ w)
            w /= w.sum()
        w = project_box_simplex(w, ctx.min_weight, ctx.max_weight)
        describe(ctx, S, w)                            # número de condição e vol. ex ante
        rc = w * (S @ w)
        ctx.info["rc_dispersion"] = float(rc.std() / rc.mean())
        return w
