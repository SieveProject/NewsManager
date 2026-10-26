"""Estimadores da matriz de covariância (ver docs/metodo_covariancia.md, 4.3 e 5.2).

Todos recebem X (T x p, retornos diários in-sample, NaN onde falta) e devolvem
Σ (p x p). Os faltantes (no máximo 5% por ação, pelo filtro do universo) viram
0 depois de tirar a média da coluna: mantém a matriz positiva semidefinida, o
que a correlação par a par não garante.

Ledoit e Wolf lineares são portes diretos dos códigos MATLAB dos autores
(covCor.m, covMarket.m, cov1Para.m); o não linear é o "analytical nonlinear
shrinkage" (Ledoit e Wolf 2020, Annals of Statistics), também portado.

`shrink_to_corr(X, R)` generaliza o alvo de correlação constante: o alvo é
T = D^½ R D^½ com R FIXO (vindo das notícias, do setor...) e a intensidade
ótima sai da mesma fórmula de covCor.m com r̄ trocado por r_ij. Com R = r̄
constante ele reproduz covCor.m exatamente (tests/test_portfolio.py).
"""

from __future__ import annotations

import numpy as np


def demeaned(X: np.ndarray) -> np.ndarray:
    Z = X - np.nanmean(X, axis=0)
    return np.nan_to_num(Z)


def sample_cov(X: np.ndarray) -> np.ndarray:
    Z = demeaned(X)
    return Z.T @ Z / (len(Z) - 1)


def _corr_parts(Z: np.ndarray):
    t, n = Z.shape
    sample = Z.T @ Z / t
    var = np.diag(sample).copy()
    sqrtvar = np.sqrt(var)
    return t, n, sample, var, sqrtvar


def _phi_theta(Z, sample, var, t, n):
    y = Z ** 2
    phi_mat = y.T @ y / t - 2 * (Z.T @ Z) * sample / t + sample ** 2
    term1 = (Z ** 3).T @ Z / t
    help_ = Z.T @ Z / t
    help_diag = np.diag(help_)
    # covCor.m: repmat(var,1,n) tem var_i na linha i. θ_ii,ij = E[x_i³x_j] - var_i s_ij
    theta = term1 - help_diag[:, None] * sample - help_ * var[:, None] + var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    return phi_mat, theta


def shrink_to_corr(X: np.ndarray, R: np.ndarray, return_delta: bool = False):
    """Shrinkage linear ótimo para o alvo T = D^½ R D^½, R fixo (diagonal 1)."""
    Z = demeaned(X)
    t, n, sample, var, sqrtvar = _corr_parts(Z)
    prior = R * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)
    phi_mat, theta = _phi_theta(Z, sample, var, t, n)
    phi = phi_mat.sum()
    ratio = np.outer(1 / sqrtvar, sqrtvar)          # ratio[i,j] = sqrt(var_j / var_i)
    rho = np.trace(phi_mat) + (R * ratio * theta).sum()
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    delta = float(np.clip((phi - rho) / gamma / t, 0, 1)) if gamma > 0 else 1.0
    S = delta * prior + (1 - delta) * sample
    return (S, delta) if return_delta else S


def lw_const_corr(X: np.ndarray, return_delta: bool = False):
    """Ledoit e Wolf (2004), "Honey, I shrunk...": alvo de correlação constante."""
    Z = demeaned(X)
    _, n, sample, _, sqrtvar = _corr_parts(Z)
    rbar = ((sample / np.outer(sqrtvar, sqrtvar)).sum() - n) / (n * (n - 1))
    R = np.full((n, n), rbar)
    np.fill_diagonal(R, 1.0)
    return shrink_to_corr(X, R, return_delta)


def lw_identity(X: np.ndarray, return_delta: bool = False):
    """Ledoit e Wolf (2004, JMVA): alvo = identidade x variância média (cov1Para.m)."""
    Z = demeaned(X)
    t, n = Z.shape
    sample = Z.T @ Z / t
    meanvar = np.trace(sample) / n
    prior = meanvar * np.eye(n)
    y = Z ** 2
    phi = (y.T @ y / t - sample ** 2).sum()
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    delta = float(np.clip(phi / gamma / t, 0, 1))
    S = delta * prior + (1 - delta) * sample
    return (S, delta) if return_delta else S


def lw_single_factor(X: np.ndarray, return_delta: bool = False):
    """Ledoit e Wolf (2003, JEF): alvo de fator único, mercado = média igual (covMarket.m)."""
    Z = demeaned(X)
    t, n = Z.shape
    xmkt = Z.mean(axis=1)
    sample = np.cov(np.column_stack([Z, xmkt]), rowvar=False, bias=True)
    covmkt = sample[:n, n]
    varmkt = sample[n, n]
    sample = sample[:n, :n]
    prior = np.outer(covmkt, covmkt) / varmkt
    np.fill_diagonal(prior, np.diag(sample))
    y = Z ** 2
    phi = (y.T @ y / t - sample ** 2).sum()
    c = np.linalg.norm(sample - prior, "fro") ** 2
    r_diag = (y ** 2).sum() / t - (np.diag(sample) ** 2).sum()
    z = Z * xmkt[:, None]
    # covMarket.m: covmkt(:,ones(1,n)) tem covmkt_i na linha i; o transposto, na coluna j
    v1 = (y.T @ z) / t - covmkt[:, None] * sample
    roff1 = (v1 * covmkt[None, :]).sum() / varmkt - (np.diag(v1) * covmkt).sum() / varmkt
    v3 = (z.T @ z) / t - varmkt * sample
    roff3 = (v3 * np.outer(covmkt, covmkt)).sum() / varmkt ** 2 - (np.diag(v3) * covmkt ** 2).sum() / varmkt ** 2
    rho = r_diag + 2 * roff1 - roff3
    delta = float(np.clip((phi - rho) / c / t, 0, 1))
    S = delta * prior + (1 - delta) * sample
    return (S, delta) if return_delta else S


def lw_nonlinear(X: np.ndarray) -> np.ndarray:
    """Ledoit e Wolf (2020): shrinkage não linear analítico (analytical_shrinkage.m)."""
    Z = demeaned(X)
    n_obs, p = Z.shape
    n = n_obs - 1                                   # média estimada
    sample = Z.T @ Z / n
    lam, u = np.linalg.eigh(sample)
    lam = np.maximum(lam, 0)
    lam_nz = lam[max(0, p - n):]
    m = len(lam_nz)
    L = np.repeat(lam_nz[:, None], m, axis=1)
    h = n ** (-1 / 3)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4 / np.sqrt(5)) * np.mean(np.maximum(1 - x ** 2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = (-3 / 10 / np.pi) * x + (3 / 4 / np.sqrt(5) / np.pi) * (1 - x ** 2 / 5) * \
            np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    edge = np.isclose(np.abs(x), np.sqrt(5))
    hf[edge] = (-3 / 10 / np.pi) * x[edge]
    hftilde = np.mean(hf / H, axis=1)
    if p <= n:
        c = p / n
        dtilde = lam_nz / ((np.pi * c * lam_nz * ftilde) ** 2 + (1 - c - np.pi * c * lam_nz * hftilde) ** 2)
    else:
        hf0 = (1 / np.pi) * (3 / 10 / h ** 2 + 3 / 4 / np.sqrt(5) / h * (1 - 1 / 5 / h ** 2)
                             * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam_nz)
        d0 = 1 / (np.pi * (p - n) / n * hf0)
        d1 = lam_nz / (np.pi ** 2 * lam_nz ** 2 * (ftilde ** 2 + hftilde ** 2))
        dtilde = np.concatenate([np.full(p - n, d0), d1])
    return (u * dtilde) @ u.T


def rmt_clip(X: np.ndarray) -> np.ndarray:
    """Limpeza por matrizes aleatórias (Laloux et al. 1999): autovalores da
    correlação abaixo do limite de Marchenko-Pastur viram a média deles
    (traço preservado); volatilidades amostrais de volta."""
    S = sample_cov(X)
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    t, p = X.shape
    lam, u = np.linalg.eigh(C)
    lmax = (1 + np.sqrt(p / t)) ** 2
    noise = lam < lmax
    if noise.any():
        lam = lam.copy()
        lam[noise] = lam[noise].mean()
    Cc = (u * lam) @ u.T
    d = np.sqrt(np.diag(Cc))
    Cc = Cc / np.outer(d, d)
    return Cc * np.outer(sd, sd)


def nearest_corr(R: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Projeta uma matriz simétrica de "correlações" em PSD com diagonal 1
    (corte de autovalores + renormalização; aproximação de Higham 2002)."""
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    lam, u = np.linalg.eigh(R)
    if lam.min() >= eps:
        return R
    R = (u * np.maximum(lam, eps)) @ u.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def black_litterman_cov(S0: np.ndarray, i: np.ndarray, j: np.ndarray, q: np.ndarray, omega: np.ndarray,
                        T: int, rtol: float = 1e-10) -> tuple[np.ndarray, dict]:
    """Black-Litterman com visões sobre elementos de Σ (docs/metodo_covariancia.md, seção 16).

    Prior vec Σ ~ (vec S0, V), V_(ij),(kl) = (s_ik s_jl + s_il s_jk)/T (Wishart);
    visões σ_ij = q + ε, ε ~ N(0, diag ω), nos pares i < j. Posterior (Kalman):

        vec Σ = vec S0 + V P' α,   (P V P' + Ω) α = q - P vec S0

    só fora da diagonal (as variâncias ficam as de S0). V P' α = vec(S0 A S0)/T
    com A simétrica, A_ij = A_ji = α_ij: nem V nem P V P' são formados, e o
    sistema é resolvido por gradiente conjugado com precondicionador de Jacobi.
    Ao fim, projeção para PSD na escala de correlação (nearest_corr)."""
    from scipy.sparse.linalg import LinearOperator, cg
    p, K = len(S0), len(q)
    sd = np.sqrt(np.diag(S0))
    info = {"n_views": K, "cg_iter": 0, "psd_projected": 0.0}
    if K == 0:
        return S0.copy(), info

    def vp(x):                                   # V P' x como matriz p x p
        A = np.zeros((p, p))
        A[i, j] = x
        A[j, i] = x
        return S0 @ A @ S0 / T

    def mv(x):
        return vp(x)[i, j] + omega * x

    diag = (S0[i, i] * S0[j, j] + S0[i, j] ** 2) / T + omega
    it = [0]

    def count(_):
        it[0] += 1

    alpha, code = cg(LinearOperator((K, K), matvec=mv, dtype=float), q - S0[i, j], rtol=rtol, atol=0.0,
                     maxiter=10 * K, M=LinearOperator((K, K), matvec=lambda x: x / diag, dtype=float),
                     callback=count)
    if code != 0:
        raise RuntimeError(f"gradiente conjugado não convergiu ({code}, {K} visões)")
    S = S0 + vp(alpha)
    np.fill_diagonal(S, np.diag(S0))
    R = S / np.outer(sd, sd)
    R = (R + R.T) / 2
    info["cg_iter"] = it[0]
    info["psd_projected"] = float(np.linalg.eigvalsh(R)[0] < 1e-4)
    return nearest_corr(R) * np.outer(sd, sd), info


def multi_factor(X: np.ndarray, G: np.ndarray, resid: str = "diag") -> np.ndarray:
    """Σ̂ = B' Σ_g B + Σ̂_e com fatores observados G (T x k, mesmos dias de X).

    B = MQO de Z (retornos centrados, faltante -> 0) em G centrado, comum a
    todos os ativos; Σ_g = covariância amostral dos fatores. Fator sem
    variância na janela sai. Com Σ̂_e = covariância amostral dos resíduos o
    resultado seria S exatamente (resíduos ortogonais aos fatores); o ajuste
    vem de estruturar o resíduo:
      "diag" -> Σ̂_e = diag dos resíduos (modelo de fatores clássico)
      "cc"   -> Ledoit-Wolf correlação constante nos resíduos
      "none" -> só a parte comum B'Σ_g B: posto = nº de fatores, singular
                quando há menos fatores que ativos (só para diagnóstico)
    """
    Z = demeaned(X)
    Gc = G - G.mean(axis=0)
    Gc = Gc[:, Gc.var(axis=0) > 1e-12]
    n = len(Z) - 1
    B, *_ = np.linalg.lstsq(Gc, Z, rcond=None)                 # k x p
    E = Z - Gc @ B
    common = B.T @ (Gc.T @ Gc / n) @ B
    if resid == "none":
        return common
    if resid == "diag":
        Se = np.diag((E * E).sum(axis=0) / n)
    elif resid == "cc":
        Se = lw_const_corr(E)
    else:
        raise ValueError(f"resid {resid!r}: use 'diag', 'cc' ou 'none'")
    return common + Se


def factor_residual(X: np.ndarray, R_prior: np.ndarray | str = "identity", return_delta: bool = False):
    """Σ̂ = σ²_m b b' + Σ̂_e (docs/metodo_covariancia.md, seção 10).

    m = média igual do universo; b por MQO na janela; E = resíduos. Σ̂_e é o
    shrinkage ótimo da covariância residual para D_e^½ R D_e^½:
      "identity" -> R = I (fator único puro, Σ̂_e diagonal: δ forçado a 1)
      "cc"       -> correlação residual constante (LW covCor nos resíduos)
      matriz     -> R_prior fixo (setor, notícias...)
    """
    from .data import market_betas, market_factor, residuals
    m = market_factor(X)
    a, b = market_betas(X, m)
    E = residuals(X, b, m, a)
    var_m = m.var(ddof=1)
    if isinstance(R_prior, str) and R_prior == "identity":
        Se, delta = np.diag(np.diag(sample_cov(E))), 1.0
    elif isinstance(R_prior, str) and R_prior == "cc":
        Se, delta = lw_const_corr(E, return_delta=True)
    else:
        Se, delta = shrink_to_corr(E, R_prior, return_delta=True)
    S = var_m * np.outer(b, b) + Se
    return (S, delta) if return_delta else S
