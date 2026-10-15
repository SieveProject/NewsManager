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
