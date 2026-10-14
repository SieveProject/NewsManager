"""Peças numéricas da etapa portfolio, sem dados: correlação com faltantes,
Newey-West e Fama-MacBeth contra cálculos diretos.

    python tests/test_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager.portfolio import data, h1  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 6)) @ rng.standard_normal((6, 6))
    C, N = data.pairwise_corr(X)
    assert np.allclose(C, np.corrcoef(X, rowvar=False)) and (N == 300).all()

    # faltantes: cada par usa só os dias em comum
    Y = X.copy()
    Y[rng.random(Y.shape) < 0.2] = np.nan
    C, N = data.pairwise_corr(Y)
    for i in range(6):
        for j in range(6):
            m = ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])
            assert N[i, j] == m.sum()
            if i != j:
                assert np.isclose(C[i, j], np.corrcoef(Y[m, i], Y[m, j])[0, 1]), (i, j)
    # poucos dias em comum -> NaN; coluna constante -> NaN
    Y[:, 0] = np.nan
    Y[:5, 0] = 1.0
    Y[:, 1] = 2.0
    C, _ = data.pairwise_corr(Y, min_obs=10)
    assert np.isnan(C[0, 2]) and np.isnan(C[1, 2]) and C[1, 1] == 1.0

    # Newey-West com 0 defasagens = t clássico com variância populacional
    x = rng.standard_normal(200) + 0.3
    assert np.isclose(h1._nw_t(x, lags=0), x.mean() / (x.std() / np.sqrt(len(x))))

    # Fama-MacBeth recupera o coeficiente padronizado de um painel sintético
    d, a, f, y = [], [], [], []
    for t in range(40):
        g = rng.standard_normal(500)
        d += [t] * 500
        f.append(g)
        y.append(0.5 * g + rng.standard_normal(500) * 0.1)
    panel = {"date": np.array(d), "z_fut": np.concatenate(y), "g": np.concatenate(f)}
    r = h1.fama_macbeth(panel, ["g"])
    assert abs(r["coef"]["g"] - 0.5) < 0.02 and r["t"]["g"] > 50, r
    print("[test_portfolio] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
