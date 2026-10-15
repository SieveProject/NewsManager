"""Matrizes de retornos e correlação para um universo numa janela de pregões."""

from __future__ import annotations

import warnings
from datetime import date

import duckdb
import numpy as np


def trading_window(con: duckdb.DuckDBPyConnection, asof: date | str, before: int = 0,
                   after: int = 0) -> tuple[date, date] | None:
    """Primeiro e último pregão de `before` pregões antes de `asof` (exclusivo) ou
    de `after` pregões a partir de `asof` (inclusivo)."""
    i = con.execute("SELECT min(cal_i) FROM news.trading_calendar WHERE date >= CAST(? AS DATE)",
                    [str(asof)]).fetchone()[0]
    if i is None:
        return None
    lo, hi = (i - before, i - 1) if before else (i, i + after - 1)
    r = con.execute("SELECT min(date), max(date) FROM news.trading_calendar WHERE cal_i BETWEEN ? AND ?",
                    [lo, hi]).fetchone()
    return None if r[0] is None else (r[0], r[1])


def returns_matrix(con: duckdb.DuckDBPyConnection, tickers: list[str], start: date, end: date) -> np.ndarray:
    """T x p de retornos válidos entre start e end (inclusive), NaN onde não há.

    As linhas são os pregões do calendário, não só os dias com algum retorno:
    um dia inválido para todos (emenda) aparece como linha de NaN.
    """
    dates = [d for (d,) in con.execute(
        "SELECT date FROM news.trading_calendar WHERE date BETWEEN ? AND ? ORDER BY date", [start, end]).fetchall()]
    pos = {d: k for k, d in enumerate(dates)}
    col = {t: j for j, t in enumerate(tickers)}
    X = np.full((len(dates), len(tickers)), np.nan)
    rows = con.execute("SELECT ticker, date, ret FROM news.returns WHERE date BETWEEN ? AND ? "
                       "AND ticker IN (SELECT unnest(?))", [start, end, tickers]).fetchall()
    for t, d, r in rows:
        X[pos[d], col[t]] = r
    return X


def pairwise_corr(X: np.ndarray, min_obs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Correlação de Pearson par a par com dados faltantes (pairwise complete).

    Devolve (C, N): C[i,j] usa só os dias em que i e j têm retorno; NaN se
    N[i,j] < min_obs ou variância zero.
    """
    M = (~np.isnan(X)).astype(float)
    Z = np.nan_to_num(X)
    N = M.T @ M
    Sx = Z.T @ M                       # soma de x_i nos dias em que j existe
    Sxx = (Z * Z).T @ M
    Sxy = Z.T @ Z
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sx.T / N
        vx = Sxx - Sx * Sx / N
        C = cov / np.sqrt(vx * vx.T)
    C[(N < min_obs) | ~np.isfinite(C)] = np.nan
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1, 1), N


def market_factor(X: np.ndarray) -> np.ndarray:
    """m_t = média igual dos retornos do universo no dia (o fator do LW fator único)."""
    with warnings.catch_warnings():            # dia de emenda: linha toda NaN -> 0
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(X, axis=1)
    return np.nan_to_num(m)


def market_betas(X: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) de x_it = a_i + b_i m_t + e_it por MQO, só nos dias com retorno."""
    M = ~np.isnan(X)
    a, b = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x, mm = X[M[:, j], j], m[M[:, j]]
        v = mm.var()
        b[j] = ((x - x.mean()) * (mm - mm.mean())).mean() / v if v > 0 else 0.0
        a[j] = x.mean() - b[j] * mm.mean()
    return a, b


def residuals(X: np.ndarray, b: np.ndarray, m: np.ndarray, a: np.ndarray | None = None) -> np.ndarray:
    """e = x - a - b m (NaN onde x é NaN). Fora da amostra: b da janela
    in-sample e m do próprio período; a omitido (não muda a correlação)."""
    return X - (0 if a is None else a) - np.outer(m, b)
