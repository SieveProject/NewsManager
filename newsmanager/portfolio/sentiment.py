"""Modelo de fatores de sentimento: betas das ações ao sentimento de agentes.

Fator de cada agente k (Fed, dólar, China...) por pregão s:

    f_k,s = Σ w_e  sobre os eventos limpos do grafo que envolvem k (como
            origem ou destino), com data em [pregão anterior a s, s)
    w_e   = ±força (sinal = direção da tupla)

isto é, as notícias publicadas depois do pregão anterior e antes do pregão s
(o dia útil anterior mais o fim de semana/feriado). Como published_at não tem
hora, o retorno de s só vê notícias de datas < s: o fator entra defasado.

Agentes em cada pregão t: Fed, dólar, China e S&P 500 sempre, mais os
`n_rolling` nós sem ticker de tipo macro (MACRO_KINDS) com mais eventos
atribuídos aos `window` pregões antes de t (empate: node_id). A escolha também
só usa dados < t. Ficam de fora empresas sem preço, pessoas, fundos, setores
e os índices de ADR (texto padrão dos boletins de mercado da nasdaq.com).

Para cada ação do universo point-in-time (news.universe n, cobertura >= 95%
na janela, as mesmas regras do motor), MQO nos pregões s em [t - window, t)
com retorno válido:

    r_i,s = α_i + Σ_k β_ik f_k,s + ε_i,s

Fator sem variância na janela fica de fora da regressão daquele dia (β = NaN).

Saída em data/marts/sentiment/n=<n>_w=<window>/:
  factors.parquet   date, agent, f, n_events (pregão x agente, os que entraram em algum dia)
  agents.parquet    date, slot, agent, n_window (os agentes escolhidos em cada t)
  fits.parquet      date, ticker, n_obs, alpha, r2
  betas.parquet     date, ticker, slot, agent, beta, se, t
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..config import Config

FIXED = ("FEDERAL_RESERVE", "US_DOLLAR", "CHINA", "S_AND_P_500")
MACRO_KINDS = ("central_bank", "government", "country", "macro", "currency", "commodity", "index")
ADR = r"\bADRs?\b"
# candidatos: um nó com menos eventos que isso no grafo inteiro nunca fica
# entre os mais frequentes de uma janela (conferido em run)
MIN_TOTAL_EVENTS = 50


def out_dir(cfg: Config, n: int, window: int) -> Path:
    return cfg.marts / "sentiment" / f"n={n}_w={window}"


class Factors:
    """Fatores de sentimento nos pregões `days` (contíguos no calendário) e a
    escolha point-in-time dos agentes. F[s] usa só eventos de data < days[s]."""

    def __init__(self, con, days: list):
        self.agents, self.F, self.N = factors(con, days)
        col = {a: k for k, a in enumerate(self.agents)}
        self.fixed = [col[a] for a in FIXED]
        self.cN = np.concatenate([np.zeros((1, len(self.agents))), np.cumsum(self.N, axis=0)])
        self.others = np.flatnonzero(~np.isin(np.arange(len(self.agents)), self.fixed))
        self.names = np.array(self.agents, dtype=object)

    def choose(self, i: int, window: int, n_rolling: int = 6) -> tuple[list[int], np.ndarray]:
        """Colunas dos agentes do pregão i: os fixos + os n_rolling com mais
        eventos nos pregões [i - window, i) (empate: node_id); e as contagens."""
        cnt = self.cN[i] - self.cN[i - window]
        c = self.others
        top = c[np.lexsort((self.names[c], -cnt[c]))][:n_rolling]
        return self.fixed + list(top), cnt


def factors(con, days: list) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(agentes, F, N): pregões x agentes sem ticker de sentimento líquido e nº
    de eventos. O evento de data d entra no primeiro pregão do calendário
    estritamente depois de d; eventos de antes de days[0] ficam de fora, e não
    acumulados no primeiro dia."""
    rows = con.execute(f"""
        WITH ag AS (SELECT node_id FROM news.graph_nodes
                    WHERE ticker IS NULL AND n_events >= {MIN_TOTAL_EVENTS}
                      AND kind IN ({", ".join(repr(k) for k in MACRO_KINDS)})
                      AND NOT regexp_matches(label, '{ADR}')),
             e AS (SELECT dt, src AS a, w FROM news.graph_events WHERE src <> dst
                   UNION ALL SELECT dt, dst, w FROM news.graph_events WHERE src <> dst)
        SELECT dt, a, sum(w), count(*) FROM e WHERE a IN (SELECT node_id FROM ag) GROUP BY 1, 2""").fetchall()
    agents = sorted({a for _, a, _, _ in rows})
    col = {a: k for k, a in enumerate(agents)}
    cal = np.array([d for (d,) in con.execute("SELECT date FROM news.trading_calendar ORDER BY date").fetchall()],
                   dtype="datetime64[D]")
    i0 = int(np.searchsorted(cal, np.datetime64(days[0], "D")))
    F, N = np.zeros((len(days), len(agents))), np.zeros((len(days), len(agents)))
    for d, a, w, c in rows:
        s = int(np.searchsorted(cal, np.datetime64(d, "D"), side="right")) - i0
        if 0 <= s < len(days):
            F[s, col[a]] += w
            N[s, col[a]] += c
    return agents, F, N


def run(cfg: Config, n: int = 100, window: int = 252, start: str = "2011-01-01", end: str = "2023-12-31",
        n_rolling: int = 6, min_coverage: float = 0.95) -> Path:
    from ..backtester.engine import Market, Universe
    t0 = time.time()
    con = query.connect(cfg)
    uni = Universe(con, SimpleNamespace(universe={"n": n}))
    cal = [d for (d,) in con.execute("SELECT date FROM news.trading_calendar ORDER BY date").fetchall()]
    idx = [i for i, d in enumerate(cal) if start <= str(d) <= end]
    mk = Market(con, uni.all_tickers(cal[idx[0]], cal[idx[-1]]), cal[idx[0] - window], cal[idx[-1]])
    days = mk.days
    fx = Factors(con, days)
    con.close()
    agents, F, NE = fx.agents, fx.F, fx.N
    print(f"[sentiment] {len(days)} pregões ({days[0]} a {days[-1]}), {len(mk.tickers)} tickers, "
          f"{len(agents)} agentes candidatos ({time.time() - t0:.0f}s)", flush=True)

    fit = {k: [] for k in ("date", "ticker", "n_obs", "alpha", "r2")}
    bet = {k: [] for k in ("date", "ticker", "slot", "agent", "beta", "se", "t")}
    chosen = {k: [] for k in ("date", "slot", "agent", "n_window")}
    used: set[int] = set(fx.fixed)
    worst_kth = np.inf
    for i in range(window, len(days)):
        t = days[i]
        ks, cnt = fx.choose(i, window, n_rolling)
        worst_kth = min(worst_kth, cnt[ks[-1]])
        used.update(ks)
        for slot, k in enumerate(ks, start=1):
            chosen["date"].append(t)
            chosen["slot"].append(slot)
            chosen["agent"].append(agents[k])
            chosen["n_window"].append(int(cnt[k]))

        U = [x for x in uni.at(t) if x in mk.col]
        cols = np.array([mk.col[x] for x in U], int)
        Y = mk.R[i - window:i][:, cols]
        M = np.isfinite(Y)
        keep_t = M.mean(axis=0) >= min_coverage
        cols, Y, M = cols[keep_t], Y[:, keep_t], M[:, keep_t]
        if len(cols) == 0:
            continue
        Fw = F[i - window:i][:, ks]
        has_var = Fw.var(axis=0) > 1e-12
        X = np.column_stack([np.ones(window), Fw[:, has_var]])
        Yz, Mf = np.where(M, Y, 0.0), M.astype(float)
        A = np.einsum("tk,tp,tl->pkl", X, Mf, X)                          # X' diag(m_p) X
        b = np.einsum("tk,tp->pk", X, Yz)
        Ainv = np.linalg.pinv(A)
        beta = np.einsum("pij,pj->pi", Ainv, b)
        nn = Mf.sum(axis=0)
        yy = (Yz * Yz).sum(axis=0)
        sse = np.maximum(yy - np.einsum("pi,pi->p", beta, b), 0)
        dof = nn - X.shape[1]
        s2 = np.where(dof > 0, sse / np.maximum(dof, 1), np.nan)
        se = np.sqrt(np.maximum(np.einsum("pii->pi", Ainv), 0) * s2[:, None])
        sst = yy - b[:, 0] ** 2 / nn
        tick = [mk.tickers[c] for c in cols]
        fit["date"] += [t] * len(cols)
        fit["ticker"] += tick
        fit["n_obs"].append(nn.astype(np.int32))
        fit["alpha"].append(beta[:, 0])
        fit["r2"].append(np.where(sst > 0, 1 - sse / sst, np.nan))
        full_b = np.full((len(cols), len(ks)), np.nan)
        full_s = np.full((len(cols), len(ks)), np.nan)
        full_b[:, has_var], full_s[:, has_var] = beta[:, 1:], se[:, 1:]
        for slot, k in enumerate(ks, start=1):
            bet["date"] += [t] * len(cols)
            bet["ticker"] += tick
            bet["slot"].append(np.full(len(cols), slot, np.int8))
            bet["agent"] += [agents[k]] * len(cols)
            bet["beta"].append(full_b[:, slot - 1])
            bet["se"].append(full_s[:, slot - 1])
            bet["t"].append(full_b[:, slot - 1] / full_s[:, slot - 1])
    if worst_kth < MIN_TOTAL_EVENTS:
        raise RuntimeError(f"o {n_rolling}º agente de alguma janela tem {worst_kth:.0f} eventos < "
                           f"{MIN_TOTAL_EVENTS}: baixe MIN_TOTAL_EVENTS (candidatos podem ter ficado de fora)")

    d = out_dir(cfg, n, window)
    d.mkdir(parents=True, exist_ok=True)

    def write(name, cols, dates="date"):
        tab = {k: (pa.array(v, pa.date32()) if k == dates else
                   (np.concatenate(v) if v and isinstance(v[0], np.ndarray) else pa.array(v)))
               for k, v in cols.items()}
        pq.write_table(pa.table(tab), d / name, compression="zstd")

    write("fits.parquet", fit)
    write("betas.parquet", bet)
    write("agents.parquet", chosen)
    ks = sorted(used, key=lambda k: agents[k])
    fac = {"date": [x for _ in ks for x in days], "agent": [agents[k] for k in ks for _ in days],
           "f": [F[:, k] for k in ks], "n_events": [NE[:, k].astype(np.int32) for k in ks]}
    write("factors.parquet", fac)
    print(f"[sentiment] {sum(len(x) for x in fit['n_obs']):,} regressões (pregão x ação), "
          f"{len(used)} agentes usados; 6º agente mais fraco de uma janela: {worst_kth:.0f} eventos; "
          f"{d} ({time.time() - t0:.0f}s)")
    return d
