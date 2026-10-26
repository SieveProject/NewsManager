"""Estimadores de Σ e carteiras da seção 18 que não estão em
newsmanager/portfolio/estimators.py. Todos recebem X (T x p, retornos diários
in-sample, NaN onde falta) e devolvem Σ̂ (p x p).

  block_target      B5: alvo de correlação constante por bloco de setores (Elton e Gruber)
  observable_factor B7: mercado + indústrias + tamanho + momentum, resíduo diagonal
  poet              B8/M3: PCA-k (Bai-Ng) + limiar adaptativo no resíduo (Fan, Liao e
                    Mincheva 2013); com `support`, mantém o resíduo só nos pares dados
  glasso            B9/M4: graphical lasso com penalidade por elemento (ADMM)
  ewma              B10: média móvel exponencial (RiskMetrics)
  ridge_factor      M1: fatores observáveis com β ridge centrado num prior
  spectral_factor   M5: cargas = autovetores do Laplaciano do grafo (+ mercado)
  pca_factor        PCA-k (+ cargas extras) com resíduo diagonal
  louvain           M8: comunidades (Blondel et al. 2008)
  hrp_order, hrp_communities, erc: carteiras
"""

from __future__ import annotations

import numpy as np

from ..portfolio import estimators as est
from ..portfolio.optimize import _single_linkage_order


def corr_of(S: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    return C


def sample_corr(X: np.ndarray) -> np.ndarray:
    return corr_of(est.sample_cov(X))


# ------------------------------------------------------------------ B5

def block_target(X: np.ndarray, groups: np.ndarray, return_delta: bool = False):
    """R_ij = correlação amostral média do bloco (g_i, g_j), i != j; δ da
    fórmula da seção 4.3 (estimators.shrink_to_corr). Sem setor (-1) é um grupo."""
    C = sample_corr(X)
    g = np.asarray(groups)
    u, gi = np.unique(g, return_inverse=True)
    K = len(u)
    off = ~np.eye(len(C), dtype=bool)
    num, den = np.zeros((K, K)), np.zeros((K, K))
    np.add.at(num, (gi[:, None].repeat(len(g), 1)[off], gi[None, :].repeat(len(g), 0)[off]), C[off])
    np.add.at(den, (gi[:, None].repeat(len(g), 1)[off], gi[None, :].repeat(len(g), 0)[off]), 1.0)
    rbar = (C[off]).mean()
    mean = np.where(den > 0, num / np.maximum(den, 1), rbar)
    R = mean[np.ix_(gi, gi)]
    np.fill_diagonal(R, 1.0)
    R = est.nearest_corr(R)
    return est.shrink_to_corr(X, R, return_delta)


# ------------------------------------------------------------------ B7

def style_factors(X: np.ndarray, industry: np.ndarray, size: np.ndarray, mom: np.ndarray) -> np.ndarray:
    """T x k: mercado (média igual), indústria - mercado (>= 3 ações), tamanho
    (tercil inferior - superior de `size`) e momentum (superior - inferior)."""
    Z = np.nan_to_num(X)
    m = Z.mean(axis=1)
    cols = [m]
    for s in np.unique(industry):
        ix = industry == s
        if s >= 0 and ix.sum() >= 3:
            cols.append(Z[:, ix].mean(axis=1) - m)
    for v, sgn in ((np.nan_to_num(size, nan=np.nanmedian(size)), -1.0), (mom, 1.0)):
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        cols.append(sgn * (Z[:, v >= hi].mean(axis=1) - Z[:, v <= lo].mean(axis=1)))
    return np.column_stack(cols)


def observable_factor(X, industry, size, mom) -> np.ndarray:
    return est.multi_factor(X, style_factors(X, industry, size, mom), "diag")


# ------------------------------------------------------------------ B8 / M3

def bai_ng_k(Z: np.ndarray, kmax: int = 10) -> int:
    """Número de fatores pelo critério IC2 de Bai e Ng (2002)."""
    T, p = Z.shape
    lam, V = np.linalg.eigh(Z.T @ Z / T)
    lam, V = lam[::-1], V[:, ::-1]
    tot = (Z * Z).sum()
    best, kbest = np.inf, 0
    for k in range(0, kmax + 1):
        resid = tot - T * lam[:k].sum()
        ic = np.log(max(resid, 1e-300) / (p * T)) + k * (p + T) / (p * T) * np.log(min(p, T))
        if ic < best:
            best, kbest = ic, k
    return kbest


def poet(X: np.ndarray, C: float = 0.5, kmax: int = 10, support: np.ndarray | None = None,
         k: int | None = None) -> tuple[np.ndarray, dict]:
    """POET. Sem `support`: limiar adaptativo (Cai e Liu 2011) suave no
    resíduo, τ_ij = C ω_T √θ_ij, ω_T = √(log p / T) + 1/√p; C sobe x1,25 até
    Σ̂ ser PD. Com `support` (p x p bool): resíduo amostral só nesses pares
    (M3), projetado para PSD."""
    Z = est.demeaned(X)
    T, p = Z.shape
    S = Z.T @ Z / T
    k = bai_ng_k(Z, kmax) if k is None else k
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    low = (V * lam) @ V.T
    U = Z - Z @ V @ V.T
    R = U.T @ U / T
    info = {"k": k}
    if support is not None:
        Rs = np.where(support | np.eye(p, dtype=bool), R, 0.0)
        d = np.sqrt(np.diag(Rs))
        Rc = est.nearest_corr(Rs / np.outer(d, d))
        info["density"] = float(support[np.triu_indices(p, 1)].mean())
        return low + Rc * np.outer(d, d), info
    theta = (U * U).T @ (U * U) / T - R ** 2
    omega = np.sqrt(np.log(p) / T) + 1 / np.sqrt(p)
    off = ~np.eye(p, dtype=bool)
    for _ in range(30):
        tau = C * omega * np.sqrt(np.maximum(theta, 0))
        Rt = np.where(off, np.sign(R) * np.maximum(np.abs(R) - tau, 0), R)
        Sig = low + Rt
        if np.linalg.eigvalsh(Sig)[0] > 1e-12 * np.trace(Sig) / p:
            break
        C *= 1.25
    info.update({"C": C, "density": float((Rt[np.triu_indices(p, 1)] != 0).mean())})
    info["n_pairs"] = int((Rt[np.triu_indices(p, 1)] != 0).sum())
    return Sig, info


# ------------------------------------------------------------------ B9 / M4

def glasso(Sc: np.ndarray, Lam: np.ndarray, rho: float = 1.0, tol: float = 1e-4, max_iter: int = 1000,
           init: tuple | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """min -log det Θ + tr(SΘ) + Σ_{i≠j} Λ_ij |Θ_ij| por ADMM (Boyd et al.
    2011, 6.5) com ρ adaptativo. Devolve (Θ PD, Z esparsa, info); `init` =
    (Z, U, ρ) de uma solução vizinha (partida a quente)."""
    p = len(Sc)
    L = Lam.copy()
    np.fill_diagonal(L, 0.0)
    Z, U = (np.eye(p), np.zeros((p, p))) if init is None else (init[0].copy(), init[1].copy())
    if init is not None:
        rho = init[2]
    for it in range(1, max_iter + 1):
        lam, Q = np.linalg.eigh(rho * (Z - U) - Sc)
        th = (lam + np.sqrt(lam * lam + 4 * rho)) / (2 * rho)
        Th = (Q * th) @ Q.T
        Zo = Z
        A = Th + U
        Z = np.sign(A) * np.maximum(np.abs(A) - L / rho, 0)
        U = U + Th - Z
        r = np.linalg.norm(Th - Z)
        s = rho * np.linalg.norm(Z - Zo)
        eps_p = p * 1e-6 + tol * max(np.linalg.norm(Th), np.linalg.norm(Z))
        eps_d = p * 1e-6 + tol * rho * np.linalg.norm(U)
        if r < eps_p and s < eps_d:
            break
        if r > 10 * s:
            rho, U = rho * 2, U / 2
        elif s > 10 * r:
            rho, U = rho / 2, U * 2
    return Th, Z, {"iter": it, "rho": rho, "state": (Z, U, rho)}


def factor_glasso(X: np.ndarray, Lam: np.ndarray, init=None) -> tuple[np.ndarray, dict]:
    """Σ̂ = σ²_m b b' + D_e^½ Θ⁻¹ D_e^½: fator de mercado e glasso na
    correlação residual (seção 18.5, B9/M4)."""
    from ..portfolio.data import market_betas, market_factor, residuals
    m = market_factor(X)
    a, b = market_betas(X, m)
    E = residuals(X, b, m, a)
    Se = est.sample_cov(E)
    sd = np.sqrt(np.diag(Se))
    Th, Zs, info = glasso(Se / np.outer(sd, sd), Lam, init=init)
    Ce = np.linalg.inv(Th)
    d = np.sqrt(np.diag(Ce))
    Ce = Ce / np.outer(d, d)
    info["support"] = Zs != 0
    info["Theta"] = Th
    return m.var(ddof=1) * np.outer(b, b) + Ce * np.outer(sd, sd), info


# ------------------------------------------------------------------ B10

def ewma(X: np.ndarray, lam: float = 0.97) -> np.ndarray:
    Z = np.nan_to_num(X)
    T = len(Z)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    return (Z * w[:, None]).T @ Z


# ------------------------------------------------------------------ M1 / B7'

def ridge_factor(X: np.ndarray, F: np.ndarray, prior: np.ndarray | None = None, lam: float = 0.0,
                 free: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Σ̂ = B' Σ_f B + diag(resíduo). F: T x K (as `free` primeiras colunas
    sem penalidade, p. ex. o mercado). β_i = argmin ||r_i - α - Fβ||² +
    Σ_k λ_k (β_k - prior_ik)², λ_k = lam·T·Var(f_k). Devolve (Σ̂, B K x p)."""
    Z = est.demeaned(X)
    Fc = np.nan_to_num(F) - np.nan_to_num(F).mean(axis=0)
    T, K = Fc.shape
    v = Fc.var(axis=0)
    pen = np.where(np.arange(K) < free, 0.0, lam * T * v)
    P = np.zeros((K, Z.shape[1])) if prior is None else prior
    A = Fc.T @ Fc + np.diag(pen)
    B = np.linalg.solve(A, Fc.T @ Z + pen[:, None] * P)
    E = Z - Fc @ B
    Sf = Fc.T @ Fc / (T - 1)
    return B.T @ Sf @ B + np.diag((E * E).sum(axis=0) / (T - 1)), B


# ------------------------------------------------------------------ M5

def laplacian_vectors(Cg: np.ndarray, q: int) -> np.ndarray:
    """q autovetores de menor autovalor do Laplaciano normalizado assinado
    L = I - D̄^-½ A D̄^-½, A = C^G - I, D̄ = diag(Σ|a|) (Kunegis et al. 2010).
    Ação isolada (grau 0) entra com linha 0 em A."""
    A = Cg - np.diag(np.diag(Cg))
    deg = np.abs(A).sum(axis=1)
    inv = np.divide(1, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    L = np.eye(len(A)) - inv[:, None] * A * inv[None, :]
    lam, U = np.linalg.eigh((L + L.T) / 2)
    return U[:, :q]


def loadings_factor(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Σ̂ = B F̂ B' + D̂ com fatores por regressão transversal diária f_t = (B'B)⁻¹B' z_t."""
    Z = est.demeaned(X)
    Bp = np.linalg.pinv(B)
    Fh = Z @ Bp.T
    E = Z - Fh @ B.T
    T = len(Z)
    return B @ (Fh.T @ Fh / (T - 1)) @ B.T + np.diag((E * E).sum(axis=0) / (T - 1))


def spectral_factor(X: np.ndarray, Cg: np.ndarray, q: int) -> np.ndarray:
    p = X.shape[1]
    return loadings_factor(X, np.column_stack([np.ones(p) / np.sqrt(p), laplacian_vectors(Cg, q)]))


def pca_vectors(X: np.ndarray, k: int) -> np.ndarray:
    S = est.sample_cov(X)
    lam, V = np.linalg.eigh(S)
    return V[:, ::-1][:, :k]


def pca_factor(X: np.ndarray, k: int, extra: np.ndarray | None = None) -> np.ndarray:
    B = pca_vectors(X, k)
    if extra is not None:
        B = np.column_stack([B, extra])
    return loadings_factor(X, B)


def principal_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """cos θ_ℓ = valores singulares de Q_U' Q_V (bases ortonormalizadas)."""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return np.linalg.svd(Qu.T @ Qv, compute_uv=False)


# ------------------------------------------------------------------ M8

def louvain(W: np.ndarray, seed: int = 7, resolution: float = 1.0, max_pass: int = 50) -> np.ndarray:
    """Comunidades por Louvain (Blondel et al. 2008) num grafo de pesos >= 0
    (diagonal ignorada). Devolve rótulos 0..K-1; nós sem aresta ficam sozinhos."""
    rng = np.random.default_rng(seed)
    A = np.maximum(W, 0).astype(float)
    np.fill_diagonal(A, 0.0)
    n0 = len(A)
    labels = np.arange(n0)
    while True:
        n = len(A)
        k = A.sum(axis=1)                       # inclui o laço (diagonal) dos nós agregados
        m2 = A.sum()
        if m2 <= 0:
            break
        comm = np.arange(n)
        tot = k.copy()
        moved_any = False
        for _ in range(max_pass):
            moved = False
            for i in rng.permutation(n):
                ci = comm[i]
                tot[ci] -= k[i]
                # peso de i para cada comunidade
                w_to = np.bincount(comm, weights=A[i], minlength=n)
                w_to[ci] -= A[i, i]
                gain = w_to - resolution * tot * k[i] / m2
                best = int(np.argmax(gain))
                if gain[best] <= gain[ci] + 1e-12:
                    best = ci
                comm[i] = best
                tot[best] += k[i]
                if best != ci:
                    moved = moved_any = True
            if not moved:
                break
        u, comm = np.unique(comm, return_inverse=True)
        labels = comm[labels]
        if not moved_any or len(u) == n:
            break
        Agg = np.zeros((len(u), len(u)))
        np.add.at(Agg, (comm[:, None].repeat(n, 1), comm[None, :].repeat(n, 0)), A)
        A = Agg                                 # diagonal = 2 x peso interno (soma do bloco)
    return np.unique(labels, return_inverse=True)[1]


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    return M


def ari(a, b) -> float:
    """Adjusted Rand index (Hubert e Arabie 1985)."""
    M = contingency(np.asarray(a), np.asarray(b))
    c2 = lambda x: x * (x - 1) / 2  # noqa: E731
    n = M.sum()
    s_ij = c2(M).sum()
    s_a, s_b = c2(M.sum(axis=1)).sum(), c2(M.sum(axis=0)).sum()
    exp = s_a * s_b / c2(n)
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


def nmi(a, b) -> float:
    """Informação mútua normalizada (média aritmética das entropias)."""
    M = contingency(np.asarray(a), np.asarray(b))
    P = M / M.sum()
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb)) if ha + hb > 0 else 1.0


def merge_small(labels: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Comunidades com < min_size ações viram um grupo só ("resto")."""
    u, c = np.unique(labels, return_counts=True)
    small = set(u[c < min_size])
    out = np.array([-1 if x in small else x for x in labels])
    return np.unique(out, return_inverse=True)[1]


def block_factor(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Modelo em blocos: mercado + (média do grupo - mercado), resíduo diagonal."""
    Z = np.nan_to_num(X)
    m = Z.mean(axis=1)
    cols = [m] + [Z[:, labels == g].mean(axis=1) - m for g in np.unique(labels) if (labels == g).sum() >= 3]
    return est.multi_factor(X, np.column_stack(cols), "diag")


# ------------------------------------------------------------------ carteiras

def _cluster_var(S, idx):
    sub = S[np.ix_(idx, idx)]
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return ivp @ sub @ ivp


def hrp_order(S: np.ndarray, order: list[int]) -> np.ndarray:
    """Bissecção recursiva de López de Prado (2016) numa ordem dada."""
    w = np.ones(len(S))
    stack = [list(order)]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        l, r = items[:half], items[half:]
        vl, vr = _cluster_var(S, l), _cluster_var(S, r)
        alpha = 1 - vl / (vl + vr)
        w[l] *= alpha
        w[r] *= 1 - alpha
        stack += [l, r]
    return w / w.sum()


def hrp_dist(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """HRP com a árvore da correlação C (distância √(½(1-C))) e variâncias de S."""
    order = _single_linkage_order(np.sqrt(np.clip(0.5 * (1 - C), 0, None)))
    return hrp_order(S, order)


def hrp_communities(S: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1º nível = comunidades (pesos ∝ 1/variância do grupo com IVP); dentro
    de cada uma, HRP com a correlação de S."""
    w = np.zeros(len(S))
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    iv = np.array([1 / _cluster_var(S, g) for g in groups])
    iv /= iv.sum()
    C = corr_of(S)
    for g, a in zip(groups, iv):
        if len(g) == 1:
            w[g] = a
            continue
        w[g] = a * hrp_dist(S[np.ix_(g, g)], C[np.ix_(g, g)])
    return w


def erc(S: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Contribuições de risco iguais: min ½ y'Σy - Σ log y (Newton com busca
    linear); w = y/Σy. Exige Σ positiva definida."""
    p = len(S)
    y = 1 / np.sqrt(np.diag(S))
    y /= np.sqrt(y @ S @ y)

    def f(z):
        return 0.5 * z @ S @ z - np.log(z).sum()

    for _ in range(max_iter):
        g = S @ y - 1 / y
        H = S + np.diag(1 / y ** 2)
        dy = -np.linalg.solve(H, g)
        if -(g @ dy) / 2 < tol:
            break
        s = 1.0
        while np.any(y + s * dy <= 0) or f(y + s * dy) > f(y) + 0.25 * s * (g @ dy):
            s /= 2
            if s < 1e-12:
                break
        y = y + s * dy
    return y / y.sum()
