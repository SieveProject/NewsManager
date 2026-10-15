"""Medidas de desempenho a partir de séries diárias (usadas pelo motor, pelo
`compare` e pelo notebook). Retornos simples; 252 pregões por ano."""

from __future__ import annotations

import numpy as np

A = 252


def drawdown(r: np.ndarray) -> np.ndarray:
    nav = np.cumprod(1 + r)
    return nav / np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:] - 1


def _max_dd_duration(r: np.ndarray) -> int:
    """Maior número de pregões abaixo do pico anterior."""
    under = drawdown(r) < 0
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def summary(r: np.ndarray, bench: np.ndarray | None = None, rf: float = 0.0,
            turnover: np.ndarray | None = None, cost: np.ndarray | None = None,
            years: float | None = None) -> dict:
    """Medidas da série diária `r` (líquida de custos) e, se houver, contra `bench`."""
    r = np.asarray(r, float)
    n = len(r)
    years = years or n / A
    nav = np.cumprod(1 + r)
    rf_d = (1 + rf) ** (1 / A) - 1
    ex = r - rf_d
    vol = r.std(ddof=1) * np.sqrt(A)
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(A)
    cagr = nav[-1] ** (1 / years) - 1
    mdd = drawdown(r).min()
    q05 = np.quantile(r, 0.05)
    out = {
        "days": n, "first_nav": 1.0, "final_nav": float(nav[-1]),
        "total_return": float(nav[-1] - 1), "cagr": float(cagr),
        "ann_return": float(r.mean() * A), "ann_vol": float(vol),
        "sharpe": float(ex.mean() * A / vol) if vol > 0 else None,
        "sortino": float(ex.mean() * A / down) if down > 0 else None,
        "max_drawdown": float(mdd), "max_dd_days": _max_dd_duration(r),
        "calmar": float(cagr / -mdd) if mdd < 0 else None,
        "skew": float(((r - r.mean()) ** 3).mean() / r.std() ** 3) if r.std() > 0 else None,
        "excess_kurtosis": float(((r - r.mean()) ** 4).mean() / r.std() ** 4 - 3) if r.std() > 0 else None,
        "var95_1d": float(-q05), "cvar95_1d": float(-r[r <= q05].mean()),
        "hit_rate": float(np.mean(r > 0)), "best_day": float(r.max()), "worst_day": float(r.min()),
    }
    if turnover is not None:
        t = np.asarray(turnover, float)
        out["turnover_per_rebalance"] = float(t.mean()) if len(t) else 0.0
        out["turnover_per_year"] = float(t.sum() / years)
    if cost is not None:
        out["total_cost"] = float(np.sum(cost))
        out["cost_per_year"] = float(np.sum(cost) / years)
    if bench is not None:
        b = np.asarray(bench, float)
        act = r - b
        te = act.std(ddof=1) * np.sqrt(A)
        vb = b.var(ddof=1)
        beta = float(np.cov(r, b)[0, 1] / vb) if vb > 0 else None
        up, dn = b > 0, b < 0
        out.update({
            "bench_cagr": float(np.prod(1 + b) ** (1 / years) - 1),
            "bench_vol": float(b.std(ddof=1) * np.sqrt(A)),
            "beta": beta,
            "alpha_ann": float((r.mean() - rf_d - beta * (b.mean() - rf_d)) * A) if beta is not None else None,
            "correlation": float(np.corrcoef(r, b)[0, 1]),
            "tracking_error": float(te),
            "information_ratio": float(act.mean() * A / te) if te > 0 else None,
            "up_capture": float(r[up].mean() / b[up].mean()) if up.any() else None,
            "down_capture": float(r[dn].mean() / b[dn].mean()) if dn.any() else None,
        })
    return out


def variance_ratio(a: np.ndarray, b: np.ndarray, rng=None, block: int = 10, reps: int = 2000) -> dict:
    """σ²_a/σ²_b com IC95 e p por bootstrap circular em blocos (dias pareados)."""
    from ..portfolio.backtest import variance_test
    return variance_test(np.asarray(a, float), np.asarray(b, float),
                         rng or np.random.default_rng(7), block=block, reps=reps)


def sharpe_diff(a: np.ndarray, b: np.ndarray, rng=None, block: int = 10, reps: int = 2000) -> dict:
    """Sharpe_a - Sharpe_b com IC95 e p por bootstrap circular em blocos."""
    rng = rng or np.random.default_rng(7)
    a, b = np.asarray(a, float), np.asarray(b, float)

    def sr(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(A) if s > 0 else 0.0
    n = len(a)
    stat = sr(a) - sr(b)
    nb = int(np.ceil(n / block))
    boots = np.empty(reps)
    for i in range(reps):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        boots[i] = sr(a[idx]) - sr(b[idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": round(float(stat), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(float(np.mean(np.abs(boots - stat) >= abs(stat))), 4)}
