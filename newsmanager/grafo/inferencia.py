"""Inferência da seção 18 (seção 9 do roteiro).

  nw_lrv, nw_t, nw_cov     variância de longo prazo de Newey-West (1987)
  dm_hln                   Diebold-Mariano com a correção de Harvey, Leybourne e Newbold (1997)
  pw_block                 bloco ótimo do bootstrap estacionário (Politis e White 2004,
                           com a correção de Patton, Politis e White 2009)
  stationary_idx           índices do bootstrap estacionário (Politis e Romano 1994)
  circular_idx             índices do bootstrap circular em blocos fixos (seções 12-17)
  var_ratio                razão de variâncias com p bicaudal por bootstrap pareado
  mcs                      Model Confidence Set, estatística T_max (Hansen, Lunde e Nason 2011)
  romano_wolf              stepdown com p ajustados (Romano e Wolf 2005)
  spa                      SPA de Hansen (2005) e Reality Check de White (2000)
  deflated_sharpe          Bailey e López de Prado (2014)
  holm, bh                 Holm (1979); Benjamini-Hochberg (1995)
  cmh                      Cochran-Mantel-Haenszel para tabelas 2x2 por mês
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def nw_lrv(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(s)


def nw_t(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    v = nw_lrv(x, lags)
    return float(x.mean() / np.sqrt(v / len(x))) if v > 0 else float("nan")


def nw_cov(X: np.ndarray, lags: int) -> np.ndarray:
    """Covariância (k x k) da média das linhas de X (n x k), Newey-West."""
    X = np.asarray(X, float)
    n = len(X)
    E = X - X.mean(axis=0)
    S = E.T @ E / n
    for L in range(1, min(lags, n - 1) + 1):
        G = E[L:].T @ E[:-L] / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    return S / n


def dm_hln(d: np.ndarray, h: int = 1, lags: int | None = None) -> dict:
    """d_t = perda_A - perda_B. DM* com t_{n-1}; p bicaudal e unicaudal (B melhor: d̄ > 0)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lags = h - 1 if lags is None else lags
    v = nw_lrv(d, max(lags, 0))
    dm = d.mean() / np.sqrt(v / n) if v > 0 else float("nan")
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = dm * k
    return {"mean": float(d.mean()), "stat": float(t), "p_two": float(2 * stats.t.sf(abs(t), n - 1)),
            "p_b_better": float(stats.t.sf(t, n - 1)), "n": n}


def pw_block(x: np.ndarray) -> float:
    """Bloco médio ótimo do bootstrap estacionário (Politis-White/PPW 2009)."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = int(np.ceil(min(3 * np.sqrt(n), n / 3)))
    c = 2 * np.sqrt(np.log10(n) / n)
    acv = np.array([e[k:] @ e[:n - k] / n for k in range(mmax + 1)])
    rho = acv[1:] / acv[0]
    m = mmax
    for j in range(1, mmax - kn + 2):
        if np.all(np.abs(rho[j - 1:j - 1 + kn]) < c):
            m = j - 1
            break
    M = min(2 * max(m, 1), mmax)
    ks = np.arange(-M, M + 1)
    lam = np.where(np.abs(ks / M) <= 0.5, 1.0, 2 * (1 - np.abs(ks / M)))
    R = acv[np.abs(ks)]
    G = (lam * np.abs(ks) * R).sum()
    g0 = (lam * R).sum()
    D = 2 * g0 ** 2
    b = (2 * G ** 2 / D) ** (1 / 3) * n ** (1 / 3) if D > 0 else 1.0
    return float(min(max(b, 1.0), bmax))


def stationary_idx(n: int, b: float, rng) -> np.ndarray:
    """Blocos de tamanho geométrico (média b), circulares."""
    p = 1 / b
    starts = rng.integers(0, n, n)
    new = rng.random(n) < p
    new[0] = True
    idx = np.empty(n, np.int64)
    cur = 0
    for t in range(n):
        cur = starts[t] if new[t] else (cur + 1) % n
        idx[t] = cur
    return idx


def circular_idx(n: int, block: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    return ((rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def var_ratio(a: np.ndarray, b: np.ndarray, rng, reps: int = 2000, block: int | None = 10,
              stationary: bool = False) -> dict:
    """σ²_a/σ²_b com p bicaudal e IC95; dias pareados. `stationary`: bloco
    geométrico com média de Politis-White (calculada em log(a²/b²))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    stat = np.log(a.var(ddof=1) / b.var(ddof=1))
    bl = pw_block(np.log((a - a.mean()) ** 2 + 1e-30) - np.log((b - b.mean()) ** 2 + 1e-30)) if stationary else block
    boots = np.empty(reps)
    for i in range(reps):
        idx = stationary_idx(n, bl, rng) if stationary else circular_idx(n, block, rng)
        boots[i] = np.log(a[idx].var(ddof=1) / b[idx].var(ddof=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"var_ratio": float(np.exp(stat)), "ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "p": float(np.mean(np.abs(boots - stat) >= abs(stat))), "block": float(bl)}


def _boot_means(L: np.ndarray, B: int, rng, block: float | None) -> np.ndarray:
    n = len(L)
    b = block or np.median([pw_block(L[:, k] - L.mean(axis=1)) for k in range(L.shape[1])])
    out = np.empty((B, L.shape[1]))
    for i in range(B):
        out[i] = L[stationary_idx(n, b, rng)].mean(axis=0)
    return out


def mcs(L: np.ndarray, names: list[str], B: int = 2000, rng=None, block: float | None = None) -> dict:
    """p-valores do MCS (T_max). L: n x M perdas (menor = melhor). Um modelo
    está no MCS de nível 1-α se o seu p > α."""
    rng = rng or np.random.default_rng(7)
    L = np.asarray(L, float)
    Lbar = L.mean(axis=0)
    Bm = _boot_means(L, B, rng, block)
    alive = list(range(L.shape[1]))
    pvals = {}
    pmax = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d = Lbar[a] - Lbar[a].mean()
        db = Bm[:, a] - Bm[:, a].mean(axis=1, keepdims=True)
        var = ((db - d) ** 2).mean(axis=0)
        t = d / np.sqrt(np.maximum(var, 1e-300))
        tb = ((db - d) / np.sqrt(np.maximum(var, 1e-300))).max(axis=1)
        p = float(np.mean(tb >= t.max()))
        pmax = max(pmax, p)
        worst = a[int(np.argmax(t))]
        pvals[names[worst]] = pmax
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    return pvals


def romano_wolf(stat: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """p ajustados (H1: stat > 0) a partir de estatísticas t e de réplicas
    centradas t*_b = (θ*_b - θ̂)/se (B x k)."""
    stat = np.asarray(stat, float)
    order = np.argsort(-stat)
    adj = np.empty(len(stat))
    prev = 0.0
    for r, j in enumerate(order):
        rest = order[r:]
        mx = boot[:, rest].max(axis=1)
        p = float(np.mean(mx >= stat[j]))
        prev = max(prev, p)
        adj[j] = prev
    return adj


def spa(L0: np.ndarray, L: np.ndarray, B: int = 2000, rng=None, block: float | None = None) -> dict:
    """H0: nenhum dos M modelos é melhor que o benchmark (perdas L0 n, L n x M).
    Devolve p do SPA consistente (Hansen 2005) e do Reality Check (White 2000)."""
    rng = rng or np.random.default_rng(7)
    D = L0[:, None] - np.asarray(L, float)          # > 0: modelo melhor
    n = len(D)
    dbar = D.mean(axis=0)
    Bm = _boot_means(D, B, rng, block)
    omega = np.sqrt(n * ((Bm - dbar) ** 2).mean(axis=0))
    tstat = np.sqrt(n) * dbar / np.maximum(omega, 1e-300)
    T_spa = max(0.0, tstat.max())
    mu = np.where(tstat >= -np.sqrt(2 * np.log(np.log(n))), dbar, 0.0)
    Ts = np.maximum(0.0, (np.sqrt(n) * (Bm - mu) / np.maximum(omega, 1e-300)).max(axis=1))
    T_rc = np.sqrt(n) * dbar.max()
    Trc = (np.sqrt(n) * (Bm - dbar)).max(axis=1)
    return {"spa_p": float(np.mean(Ts >= T_spa)), "rc_p": float(np.mean(Trc >= T_rc)),
            "best": int(np.argmax(tstat))}


def deflated_sharpe(r: np.ndarray, n_trials: int, var_trials: float) -> dict:
    """DSR de Bailey e López de Prado (2014), Sharpe por período (sem anualizar).
    var_trials = variância dos Sharpe (por período) das configurações testadas."""
    r = np.asarray(r, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    em = 0.5772156649
    N = max(n_trials, 2)
    sr0 = np.sqrt(var_trials) * ((1 - em) * stats.norm.ppf(1 - 1 / N) + em * stats.norm.ppf(1 - 1 / (N * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"sr": float(sr), "sr0": float(sr0), "dsr": float(stats.norm.cdf(z))}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for r, j in enumerate(order):
        prev = max(prev, min(1.0, (m - r) * p[j]))
        adj[j] = prev
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for r in range(m - 1, -1, -1):
        j = order[r]
        prev = min(prev, p[j] * m / (r + 1))
        adj[j] = prev
    return adj


def cmh(tables: list[np.ndarray]) -> dict:
    """Tabelas 2x2 [[a, b], [c, d]] por estrato; H1: odds ratio comum > 1."""
    num, var, mh_n, mh_d = 0.0, 0.0, 0.0, 0.0
    for T in tables:
        (a, b), (c, d) = T
        n = a + b + c + d
        if n < 2:
            continue
        r1, c1 = a + b, a + c
        num += a - r1 * c1 / n
        var += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
        mh_n += a * d / n
        mh_d += b * c / n
    z = num / np.sqrt(var) if var > 0 else float("nan")
    return {"z": float(z), "p_one": float(stats.norm.sf(z)), "or_mh": float(mh_n / mh_d) if mh_d > 0 else float("inf")}
