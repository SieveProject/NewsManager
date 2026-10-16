"""Carteiras que só dependem de Σ: variância mínima global (GMV) com e sem
venda a descoberto. Nenhuma usa retorno esperado, para isolar o efeito de Σ."""

from __future__ import annotations

import numpy as np


def gmv(S: np.ndarray) -> np.ndarray:
    """w = Σ⁻¹1 / 1'Σ⁻¹1 (vendas a descoberto permitidas)."""
    x = np.linalg.solve(S, np.ones(len(S)))
    return x / x.sum()


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Projeção euclidiana em {w >= 0, Σw = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0)


def gmv_long_only(S: np.ndarray, tol: float = 1e-10, max_iter: int = 20000) -> np.ndarray:
    """min w'Σw s.t. w >= 0, Σw = 1 por gradiente projetado acelerado (FISTA).

    Convexo: converge ao ótimo. Parada pela variação de w; o teste confere as
    condições de KKT.
    """
    L = 2 * np.linalg.eigvalsh(S)[-1]
    w = np.full(len(S), 1 / len(S))
    y, t = w.copy(), 1.0
    for _ in range(max_iter):
        w_new = project_simplex(y - (2 * S @ y) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def project_box_simplex(v: np.ndarray, lo: float | np.ndarray = 0.0, hi: float | np.ndarray = np.inf,
                        total: float = 1.0) -> np.ndarray:
    """Projeção euclidiana em {Σw = total, lo <= w <= hi} (limites podem ser ±inf).

    w = clip(v - τ, lo, hi), com g(τ) = Σw linear por partes e não crescente
    entre os pontos de quebra v - hi e v - lo. g é avaliada em todos os pontos
    de quebra de uma vez (ordenação + somas acumuladas, O(p log p)) e τ sai
    por interpolação no intervalo que cruza `total`: exata, sem bisseção.
    """
    v = np.asarray(v, float)
    n = len(v)
    lo = np.broadcast_to(np.asarray(lo, float), (n,))
    hi = np.broadcast_to(np.asarray(hi, float), (n,))
    if (lo > hi).any() or lo.sum() > total + 1e-12 or hi.sum() < total - 1e-12:
        raise ValueError(f"limites inviáveis: Σmin={lo.sum():.4g}, Σmax={hi.sum():.4g}, total={total}")
    a, b = v - hi, v - lo                      # w_i = hi se τ <= a_i; lo se τ >= b_i
    oa, ob = np.argsort(a), np.argsort(b)
    sa, sb = a[oa], b[ob]
    # somas acumuladas pelos maiores a (cauda) e menores b (cabeça); ±inf nunca
    # entram na parte usada porque τ é finito
    hi_tail = np.concatenate([np.cumsum(np.where(np.isfinite(hi[oa]), hi[oa], 0)[::-1])[::-1], [0.0]])
    va_tail = np.concatenate([np.cumsum(v[oa][::-1])[::-1], [0.0]])
    lo_head = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(lo[ob]), lo[ob], 0))])
    vb_head = np.concatenate([[0.0], np.cumsum(v[ob])])
    vsum = v.sum()

    def g(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ia = np.searchsorted(sa, tau, side="left")   # a_i >= τ: posições ia..n-1
        ib = np.searchsorted(sb, tau, side="right")  # b_i <= τ: posições 0..ib-1
        k_hi, k_lo = n - ia, ib
        free = n - k_hi - k_lo
        s = hi_tail[ia] + lo_head[ib] + (vsum - va_tail[ia] - vb_head[ib]) - tau * free
        return s, free

    bp = np.unique(np.concatenate([sa[np.isfinite(sa)], sb[np.isfinite(sb)]]))
    if len(bp) == 0:                           # sem limites: translação pura
        return v - (vsum - total) / n
    gb, _ = g(bp)
    # g não cresce: o primeiro ponto com g <= total fecha o intervalo
    j = int(np.searchsorted(-gb, -total, side="left"))
    if j < len(bp) and np.isclose(gb[j], total, rtol=0, atol=1e-14):
        tau = bp[j]
    else:
        left = bp[j - 1] if j > 0 else bp[0] - 1.0
        right = bp[j] if j < len(bp) else bp[-1] + 1.0
        g0, free = g(np.array([(left + right) / 2]))
        free = int(free[0])
        if free == 0:                          # patamar: qualquer τ do intervalo serve
            tau = (left + right) / 2
        else:                                  # g(τ) = g(mid) - free (τ - mid)
            tau = (left + right) / 2 + (g0[0] - total) / free
    return np.clip(v - tau, lo, hi)


def min_variance(S: np.ndarray, lo: float = 0.0, hi: float = np.inf, tol: float = 1e-10,
                 max_iter: int = 20000) -> np.ndarray:
    """min w'Σw s.t. Σw = 1, lo <= w <= hi. Sem limites é a GMV fechada; com
    limites, FISTA com projeção em project_box_simplex (convexo: converge)."""
    if np.isinf(lo) and np.isinf(hi):
        return gmv(S)
    p = len(S)
    L = 2 * np.linalg.eigvalsh(S)[-1]
    w = project_box_simplex(np.full(p, 1 / p), lo, hi)
    y, t = w.copy(), 1.0
    for _ in range(max_iter):
        w_new = project_box_simplex(y - (2 * S @ y) / L, lo, hi)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def qp_box_simplex(A: np.ndarray, q: np.ndarray, lo: float = 0.0, hi: float = np.inf, tol: float = 1e-10,
                   max_iter: int = 20000, w0: np.ndarray | None = None) -> np.ndarray:
    """min ½ w'Aw - q'w s.t. Σw = 1, lo <= w <= hi (A psd), por FISTA com
    projeção em project_box_simplex. min_variance é o caso A = 2Σ, q = 0."""
    p = len(A)
    L = max(np.linalg.eigvalsh(A)[-1], 1e-18)
    w = project_box_simplex(np.full(p, 1 / p) if w0 is None else w0, lo, hi)
    y, t = w.copy(), 1.0
    for _ in range(max_iter):
        w_new = project_box_simplex(y - (A @ y - q) / L, lo, hi)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def mean_variance(S: np.ndarray, mu: np.ndarray, gamma: float, lo: float = 0.0, hi: float = np.inf) -> np.ndarray:
    """Markowitz: max w'μ - (γ/2) w'Σw s.t. Σw = 1 e limites."""
    return qp_box_simplex(gamma * S, mu, lo, hi)


def max_ratio(S: np.ndarray, a: np.ndarray, lo: float = 0.0, hi: float = np.inf,
              grid: int = 25, refine: int = 40) -> tuple[np.ndarray, float]:
    """max a'w / sqrt(w'Σw) s.t. Σw = 1 e limites (máximo Sharpe com a = μ;
    máxima diversificação com a = σ).

    O máximo está na fronteira {argmin w'Σw - λ a'w : λ >= 0}. Busca em λ:
    grade log-espaçada (escalada por a e Σ) e seção áurea entre os vizinhos do
    melhor ponto. Devolve (w, razão). Se nenhum a'w > 0 é atingível, devolve a
    GMV (λ = 0)."""
    def solve(lam, w0=None):
        return qp_box_simplex(2 * S, lam * a, lo, hi, w0=w0)

    def ratio(w):
        v = w @ S @ w
        return (a @ w) / np.sqrt(v) if v > 0 else -np.inf

    w_gmv = min_variance(S, lo, hi)
    scale = np.sqrt(w_gmv @ S @ w_gmv) / max(np.abs(a).max(), 1e-18)
    lams = np.concatenate([[0.0], scale * np.logspace(-3, 3, grid)])
    ws, rs, prev = [], [], w_gmv
    for lam in lams:
        prev = w_gmv if lam == 0 else solve(lam, prev)
        ws.append(prev)
        rs.append(ratio(prev))
    k = int(np.argmax(rs))
    if rs[k] <= 0:
        return w_gmv, float(rs[0])
    a_, b_ = lams[max(k - 1, 0)], lams[min(k + 1, len(lams) - 1)]
    g = (np.sqrt(5) - 1) / 2
    best_w, best_r = ws[k], rs[k]
    c, d = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
    wc, wd = solve(c, best_w), solve(d, best_w)
    rc, rd = ratio(wc), ratio(wd)
    for _ in range(refine):
        if rc > rd:
            b_, d, wd, rd = d, c, wc, rc
            c = b_ - g * (b_ - a_)
            wc = solve(c, wd)
            rc = ratio(wc)
        else:
            a_, c, wc, rc = c, d, wd, rd
            d = a_ + g * (b_ - a_)
            wd = solve(d, wc)
            rd = ratio(wd)
        for w_, r_ in ((wc, rc), (wd, rd)):
            if r_ > best_r:
                best_w, best_r = w_, r_
        if b_ - a_ < 1e-6 * max(b_, 1e-18):
            break
    return best_w, float(best_r)


def min_tracking_error(S: np.ndarray, c: np.ndarray, lo: float = 0.0, hi: float = np.inf) -> np.ndarray:
    """min Var(R'w - b) = w'Σw - 2 w'c + σ²_b, c = Cov(R, b), s.t. Σw = 1 e limites."""
    return qp_box_simplex(2 * S, 2 * c, lo, hi)


def _single_linkage_order(D: np.ndarray) -> list[int]:
    """Ordem das folhas do dendrograma de ligação simples (quasi-diagonalização
    de López de Prado, 2016). Aglomerativo O(p³), suficiente para p <= ~700."""
    p = len(D)
    clusters = {i: [i] for i in range(p)}
    dist = D.astype(float).copy()
    np.fill_diagonal(dist, np.inf)
    active = list(range(p))
    nxt = p
    while len(active) > 1:
        sub = dist[np.ix_(active, active)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = active[i], active[j]
        if a > b:
            a, b = b, a
        merged = clusters.pop(a) + clusters.pop(b)
        # ligação simples: distância do novo grupo = mínimo das duas
        row = np.minimum(dist[a], dist[b])
        dist = np.pad(dist, ((0, 1), (0, 1)), constant_values=np.inf)
        dist[nxt, :-1] = row
        dist[:-1, nxt] = row
        dist[nxt, nxt] = np.inf
        dist[a, :] = dist[:, a] = dist[b, :] = dist[:, b] = np.inf
        clusters[nxt] = merged
        active = [x for x in active if x not in (a, b)] + [nxt]
        nxt += 1
    return clusters[active[0]]


def hrp(S: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado, 2016): ligação simples na
    distância √(½(1-ρ)), quasi-diagonalização e bissecção recursiva com pesos
    de variância inversa dentro de cada metade. Long-only, sem inverter Σ."""
    sd = np.sqrt(np.diag(S))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    order = _single_linkage_order(np.sqrt(0.5 * (1 - C)))
    w = np.ones(len(S))

    def cluster_var(idx):
        sub = S[np.ix_(idx, idx)]
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return ivp @ sub @ ivp

    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        l, r = items[:half], items[half:]
        vl, vr = cluster_var(l), cluster_var(r)
        alpha = 1 - vl / (vl + vr)
        w[l] *= alpha
        w[r] *= 1 - alpha
        stack += [l, r]
    return w / w.sum()


def bayes_stein_mean(X: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float]:
    """Média de Bayes-Stein (Jorion, 1986): encolhe a média amostral para a
    média da carteira GMV, μ_BS = (1-φ) x̄ + φ μ_0 1, com
    φ = (p+2) / ((p+2) + T (x̄-μ_0 1)' Σ⁻¹ (x̄-μ_0 1)). Devolve (μ_BS, φ)."""
    T, p = X.shape
    xbar = np.nanmean(X, axis=0)
    Si = np.linalg.inv(S)
    one = np.ones(p)
    mu0 = (one @ Si @ xbar) / (one @ Si @ one)
    d = xbar - mu0
    phi = (p + 2) / ((p + 2) + T * d @ Si @ d)
    return (1 - phi) * xbar + phi * mu0, float(phi)
