"""Peças numéricas da etapa portfolio, sem dados: correlação com faltantes,
Newey-West e Fama-MacBeth contra cálculos diretos.

    python tests/test_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsmanager.portfolio import backtest, data, estimators as est, h1, optimize  # noqa: E402


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
    # --- estimadores -------------------------------------------------------
    p, T = 30, 120
    A = rng.standard_normal((p, 3))
    Sigma = A @ A.T * 1e-4 + np.diag(rng.uniform(1, 3, p)) * 1e-4
    Xr = rng.multivariate_normal(np.zeros(p), Sigma, size=T)

    # porte literal de covCor.m (Ledoit e Wolf) para conferir shrink_to_corr
    def cov_cor_ref(x):
        t, n = x.shape
        x = x - x.mean(axis=0)
        sample = x.T @ x / t
        var = np.diag(sample)
        sv = np.sqrt(var)
        rbar = ((sample / np.outer(sv, sv)).sum() - n) / (n * (n - 1))
        prior = rbar * np.outer(sv, sv)
        np.fill_diagonal(prior, var)
        y = x ** 2
        phiMat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
        phi = phiMat.sum()
        term1 = (x ** 3).T @ x / t
        hp = x.T @ x / t
        hd = np.diag(hp)
        # repmat(helpDiag,1,n), repmat(var,1,n): o valor da linha i em toda a linha
        theta = term1 - np.tile(hd[:, None], (1, n)) * sample - hp * np.tile(var[:, None], (1, n)) \
            + np.tile(var[:, None], (1, n)) * sample
        np.fill_diagonal(theta, 0)
        # conferência pela definição: θ_ii,ij = média de (x_i² - var_i)(x_i x_j - s_ij)
        i, j = 3, 7
        assert np.isclose(theta[i, j], np.mean((x[:, i] ** 2 - var[i]) * (x[:, i] * x[:, j] - sample[i, j])))
        rho = np.trace(phiMat) + rbar * (np.outer(1 / sv, sv) * theta).sum()
        gamma = np.linalg.norm(sample - prior, "fro") ** 2
        d = max(0, min(1, (phi - rho) / gamma / t))
        return d * prior + (1 - d) * sample, d

    S_ref, d_ref = cov_cor_ref(Xr)
    S_cc, d_cc = est.lw_const_corr(Xr, return_delta=True)
    assert np.allclose(S_cc, S_ref) and np.isclose(d_cc, d_ref), (d_cc, d_ref)

    # todos: simétricos, PD, diagonal positiva; δ em [0, 1]; perda menor que a amostral
    loss = lambda M: np.linalg.norm(M - Sigma, "fro")  # noqa: E731
    S0 = est.sample_cov(Xr)
    for name, f in [("id", est.lw_identity), ("cc", est.lw_const_corr), ("1f", est.lw_single_factor),
                    ("nl", est.lw_nonlinear), ("rmt", est.rmt_clip)]:
        M = f(Xr)
        assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0, name
        if name in ("id", "cc", "1f"):
            assert 0 <= f(Xr, return_delta=True)[1] <= 1, name
        assert loss(M) < loss(S0), (name, loss(M), loss(S0))
    # alvo verdadeiro: shrinkage para a correlação certa é melhor que tudo
    sd = np.sqrt(np.diag(Sigma))
    S_true, d_true = est.shrink_to_corr(Xr, Sigma / np.outer(sd, sd), return_delta=True)
    assert loss(S_true) < loss(est.lw_const_corr(Xr)) and d_true > d_cc
    # faltantes não quebram PSD
    Xm = Xr.copy()
    Xm[rng.random(Xm.shape) < 0.05] = np.nan
    assert np.linalg.eigvalsh(est.lw_nonlinear(Xm)).min() > 0

    # nearest_corr: PSD, diagonal 1, intacta se já era válida
    R = np.full((5, 5), 0.9)
    R[0, 1] = R[1, 0] = -0.9
    np.fill_diagonal(R, 1)
    Rp = est.nearest_corr(R)
    assert np.linalg.eigvalsh(Rp).min() > 0 and np.allclose(np.diag(Rp), 1)
    C = np.corrcoef(Xr, rowvar=False)
    assert np.allclose(est.nearest_corr(C.copy()), C)

    # --- otimização ------------------------------------------------------------
    w = optimize.gmv(Sigma)
    assert np.isclose(w.sum(), 1)
    g = Sigma @ w
    assert np.allclose(g, g.mean())                          # KKT sem restrição
    wl = optimize.gmv_long_only(Sigma)
    assert np.isclose(wl.sum(), 1) and (wl >= -1e-12).all()
    g = 2 * Sigma @ wl                                       # KKT com w >= 0
    mu = g[wl > 1e-8].mean()
    assert np.allclose(g[wl > 1e-8], mu, rtol=1e-4) and (g[wl <= 1e-8] >= mu * (1 - 1e-4)).all()
    assert wl @ Sigma @ wl >= w @ Sigma @ w - 1e-15
    if (w > 0).all():
        assert np.allclose(w, wl, atol=1e-6)

    # --- treino dos alvos: somas acumuladas == MQO direto nos meses anteriores --
    dd = np.repeat(np.arange(10), 50)
    rng.shuffle(dd)
    P = {"date": dd, "z_fut": rng.standard_normal(500), "a": rng.standard_normal(500), "b": rng.standard_normal(500)}
    tr = backtest._Trainer(P, {"m": ["a", "b"]})
    m = P["date"] < 6
    Xa, ya, da = np.column_stack([P["a"][m], P["b"][m]]), P["z_fut"][m].copy(), P["date"][m]
    for u in np.unique(da):
        k = da == u
        Xa[k] -= Xa[k].mean(0)
        ya[k] -= ya[k].mean()
    assert np.allclose(tr.coef("m", 6), np.linalg.lstsq(Xa, ya, rcond=None)[0])

    # carteira comprada e mantida: pesos derivam, retorno = variação do valor
    w0 = np.array([0.5, 0.5])
    r, w_end = backtest._hold(w0, np.array([[0.10, 0.0], [0.0, 0.10]]))
    # dia 1: ativo 1 +10% -> valor 1,05; dia 2: ativo 2 +10% -> 1,10; pesos voltam a 50/50
    assert np.allclose(r, [0.05, 1.10 / 1.05 - 1]) and np.allclose(w_end, [0.5, 0.5])

    # --- fator + resíduo ---------------------------------------------------
    T2, p2 = 500, 8
    m = rng.standard_normal(T2) * 0.01
    btrue = rng.uniform(0.5, 1.5, p2)
    Xf = np.outer(m, btrue) + rng.standard_normal((T2, p2)) * 0.005
    mm = data.market_factor(Xf)
    a, bb = data.market_betas(Xf, mm)
    E = data.residuals(Xf, bb, mm, a)
    assert np.allclose((E * (mm - mm.mean())[:, None]).mean(axis=0), 0, atol=1e-12)   # ortogonal ao mercado
    Sf = est.factor_residual(Xf, "identity")
    assert np.allclose(Sf, mm.var(ddof=1) * np.outer(bb, bb) + np.diag(np.diag(est.sample_cov(E))))
    for prior in ("cc", np.eye(p2)):
        Sx = est.factor_residual(Xf, prior)
        assert np.allclose(Sx, Sx.T) and np.linalg.eigvalsh(Sx).min() > 0

    print("[test_portfolio] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
