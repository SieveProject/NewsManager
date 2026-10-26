"""Dados comuns da seção 18: universo S&P 500 point-in-time, retornos,
volume financeiro, setores, datas de rebalanceamento e séries externas.

Universo U_t: membros do S&P 500 no último registro estritamente antes de t
(data/reference/sp500, fja05680 "(Updated)") com retorno válido do FNSPID em
>= 95% dos 252 pregões antes de t. Nada do período de carteira entra na
seleção (deslistagem no mês conta retorno 0, como no motor).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ..config import Config

SP500_CSV = ("reference", "sp500", "sp500_historical_components.csv")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
INSAMPLE = 252
MIN_COVERAGE = 0.95


def _clean(sym: str) -> str:
    """'AAL-199702' (ticker antigo, reutilizado depois) -> 'AAL'; 'BRK.B' -> 'BRK-B'."""
    head, _, tail = sym.rpartition("-")
    s = head if head and tail.isdigit() else sym
    return s.replace(".", "-")


def sp500_composition(cfg: Config) -> tuple[np.ndarray, list[frozenset]]:
    """(datas, membros) do arquivo de composição histórica."""
    import csv
    path = cfg.root.joinpath(*SP500_CSV)
    dates, members = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            dates.append(np.datetime64(row["date"], "D"))
            members.append(frozenset(_clean(x) for x in row["tickers"].split(",") if x))
    order = np.argsort(dates, kind="stable")
    return np.array(dates)[order], [members[k] for k in order]


def sp500_members(comp: tuple[np.ndarray, list[frozenset]], t) -> frozenset:
    """Membros no último registro estritamente antes de t."""
    dates, members = comp
    k = int(np.searchsorted(dates, np.datetime64(t, "D"), side="left")) - 1
    if k < 0:
        raise ValueError(f"composição do S&P 500 começa em {dates[0]}: nada antes de {t}")
    return members[k]


def sp500_tickers(cfg: Config, first, last, con=None) -> list[str]:
    """Todos os que foram membros entre first e last (inclusive) e têm retorno no FNSPID."""
    dates, members = sp500_composition(cfg)
    lo = int(np.searchsorted(dates, np.datetime64(first, "D"), side="left")) - 1
    hi = int(np.searchsorted(dates, np.datetime64(last, "D"), side="right"))
    tk = sorted(set().union(*members[max(lo, 0):hi]))
    if con is None:
        return tk
    have = {t for (t,) in con.execute("SELECT DISTINCT ticker FROM news.returns").fetchall()}
    return [t for t in tk if t in have]


def month_starts(days: list[date], first: str, last: str) -> list[int]:
    """Índices (em `days`) do 1º pregão de cada mês entre first e last."""
    out = []
    for i, d in enumerate(days):
        if str(d) < first or str(d) > last:
            continue
        if i == 0 or (d.year, d.month) != (days[i - 1].year, days[i - 1].month):
            out.append(i)
    return out


@dataclass
class Sectors:
    ff12: dict[str, int]
    ff49: dict[str, int]
    sic2: dict[str, int]

    def arrays(self, tickers: list[str]) -> dict[str, np.ndarray]:
        """Códigos por ação; -1 sem setor (nunca conta como "mesmo setor")."""
        return {k: np.array([getattr(self, k).get(t, -1) for t in tickers]) for k in ("ff12", "ff49", "sic2")}


def sectors(con) -> Sectors:
    rows = con.execute("SELECT ticker, ff12, ff49, sic2 FROM news.sectors WHERE sic IS NOT NULL").fetchall()
    return Sectors({t: int(a) for t, a, _, _ in rows}, {t: int(b) for t, _, b, _ in rows},
                   {t: int(c) for t, _, _, c in rows if c is not None})


class Panel:
    """Retornos e volume financeiro (pregões x tickers) do S&P 500 + ETFs de
    referência, carregados uma vez. NaN onde falta."""

    EXTRA = ("SPY", "USO", "FXI", "UUP", "GLD", "TLT", "LQD")

    def __init__(self, cfg: Config, con, first: str = "2009-06-01", last: str = "2023-12-31",
                 tickers: list[str] | None = None):
        self.cfg = cfg
        self.comp = sp500_composition(cfg)
        tk = tickers if tickers is not None else sp500_tickers(cfg, "2010-01-01", last, con)
        self.tickers = sorted(set(tk) | set(self.EXTRA))
        self.col = {t: j for j, t in enumerate(self.tickers)}
        self.days = [d for (d,) in con.execute(
            "SELECT date FROM news.trading_calendar WHERE date BETWEEN ? AND ? ORDER BY date", [first, last]).fetchall()]
        self.pos = {d: i for i, d in enumerate(self.days)}
        lit = ", ".join("'" + t.replace("'", "''") + "'" for t in self.tickers)
        self.R = np.full((len(self.days), len(self.tickers)), np.nan)
        self.DV = np.full((len(self.days), len(self.tickers)), np.nan)
        res = con.execute(f"SELECT ticker, date, ret FROM news.returns WHERE date BETWEEN ? AND ? "
                          f"AND ticker IN ({lit})", [first, last]).fetchnumpy()
        rows = [self.pos[d] for d in res["date"].astype("datetime64[D]").astype(object)]
        self.R[rows, [self.col[t] for t in res["ticker"]]] = res["ret"]
        res = con.execute(f"SELECT ticker, date, dollar_volume FROM news.prices WHERE date BETWEEN ? AND ? "
                          f"AND ticker IN ({lit}) AND dollar_volume > 0", [first, last]).fetchnumpy()
        rows = [self.pos.get(d, -1) for d in res["date"].astype("datetime64[D]").astype(object)]
        ok = np.array(rows) >= 0
        self.DV[np.array(rows)[ok], np.array([self.col[t] for t in res["ticker"]])[ok]] = res["dollar_volume"][ok]
        self.sectors = sectors(con)

    def universe(self, i: int, kind: str = "sp500", insample: int = INSAMPLE,
                 min_coverage: float = MIN_COVERAGE, extra: dict | None = None) -> list[str]:
        """U_t no pregão de índice i: membros em days[i] com cobertura na janela [i-insample, i).

        kind: "sp500" (principal); "sp100" (as 100 de maior volume financeiro
        mediano na janela, aproximação do S&P 100); "exfin" (sem SIC 60-67);
        "lista" (extra["tickers"], p. ex. news.universe n=100 na data).
        """
        t = self.days[i]
        if kind == "lista":
            cand = [x for x in extra["tickers"] if x in self.col]
        else:
            cand = [x for x in sp500_members(self.comp, t) if x in self.col and x not in self.EXTRA]
        cols = np.array([self.col[x] for x in cand], int)
        cov = (~np.isnan(self.R[i - insample:i, cols])).mean(axis=0)
        U = [x for x, c in zip(cand, cov) if c >= min_coverage]
        if kind == "sp100":
            dv = self.size(i, U, insample)
            U = [U[k] for k in np.argsort(-dv, kind="stable")[:100]]
            U.sort()
        elif kind == "exfin":
            U = [x for x in U if not (60 <= self.sectors.sic2.get(x, -1) <= 67)]
        return U

    def window(self, i: int, tickers: list[str], before: int = INSAMPLE, after: int = 0) -> np.ndarray:
        cols = [self.col[x] for x in tickers]
        return self.R[i - before:i + after][:, cols]

    def size(self, i: int, tickers: list[str], insample: int = INSAMPLE) -> np.ndarray:
        """log do volume financeiro mediano na janela (proxy de tamanho; sem market cap)."""
        cols = [self.col[x] for x in tickers]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(self.DV[i - insample:i, cols], axis=0)
        return np.log(np.where(np.isfinite(m) & (m > 0), m, np.nan))

    def momentum(self, i: int, tickers: list[str]) -> np.ndarray:
        """Retorno acumulado de t-252 a t-21 (12-1), faltante = 0."""
        X = np.nan_to_num(self.window(i - 21, tickers, before=231))
        return np.prod(1 + X, axis=0) - 1


# ------------------------------------------------------------------ séries externas

def fred(cfg: Config, sid: str) -> dict[date, float]:
    """Série diária do FRED, guardada em data/reference/fred/<sid>.csv na 1ª leitura."""
    path = cfg.root / "reference" / "fred" / f"{sid}.csv"
    if not path.exists():
        import httpx
        r = httpx.get(FRED_URL.format(sid=sid), timeout=60, follow_redirects=True)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    out = {}
    for line in io.StringIO(path.read_text()).readlines()[1:]:
        d, _, v = line.strip().partition(",")
        if v and v != ".":
            out[date.fromisoformat(d)] = float(v)
    return out


def fred_changes(cfg: Config, sid: str, days: list[date]) -> np.ndarray:
    """Variação diária da série nos pregões `days` (NaN sem observação nos dois dias)."""
    s = fred(cfg, sid)
    lv = np.array([s.get(d, np.nan) for d in days])
    out = np.full(len(days), np.nan)
    out[1:] = lv[1:] - lv[:-1]
    return out


def fomc_dates(cfg: Config) -> list[date]:
    """1º dia das reuniões do FOMC (data/reference/fomc_dates.csv, montado do calendário do Fed)."""
    path = cfg.root / "reference" / "fomc_dates.csv"
    return [date.fromisoformat(x.strip()) for x in path.read_text().splitlines()[1:] if x.strip()]


def days_between(a: date, b: date) -> int:
    return (b - a).days


def shift_days(d: date, n: int) -> date:
    return d + timedelta(days=n)


def out_dir(cfg: Config) -> Path:
    return cfg.marts / "grafo"
