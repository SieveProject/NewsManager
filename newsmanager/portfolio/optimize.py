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
