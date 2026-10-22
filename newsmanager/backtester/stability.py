"""Estabilidade diária de Σ̂: número de condição λmax/λmin por estimador.

Para cada pregão t entre `start` e `end`:
  1. universo point-in-time em t (news.universe n, rebalance_date <= t), menos
     os ativos com menos de `min_coverage` de retornos válidos nos `window`
     pregões antes de t (as mesmas regras do motor, engine.py);
  2. X = esses `window` retornos (só datas < t);
  3. notícias: eventos do grafo com data em [1º pregão da janela, t), isto é,
     só as tuplas dos mesmos `window` pregões (meia-vida de 90 dias dentro da
     janela). O β que traduz atributos em correlação vem do painel mensal da
     H1 residual, só com meses cujo período de carteira terminou antes de t;
  4. cada estimador de covariance.py dá Σ̂; grava λmax/λmin de Σ̂ e da
     correlação implícita D^-½ Σ̂ D^-½ (a medida de López de Prado das seções
     11 e 13 do método).

Os atributos de par são calculados uma vez por dia e servem aos quatro alvos
com setor/notícias (o placebo embaralha as notícias entre as ações).

Modelos de fatores de sentimento (newsmanager/portfolio/sentiment.py): os 10
agentes point-in-time do pregão (Fed, dólar, China, S&P 500 + 6 macro mais
frequentes na janela), fator f_k,s = Σ ±força das notícias com data em
[pregão anterior a s, s). Σ̂ = B'Σ_g B + Σ̂_e (estimators.multi_factor) com
g = sentimento, ou g = [mercado, sentimento]; resíduo diagonal ou LW corr.
constante. Placebo: os dias dos fatores embaralhados dentro da janela.

Só a parte comum (FACTOR_ONLY, Σ̂ = B'Σ_g B, sem resíduo): posto k = nº de
fatores com variância, então λmin = 0 e o κ usual é infinito. Para essas, o κ
gravado é λ₁/λₖ nos k autovalores não nulos (coluna `rank`), uma medida da
estabilidade da estrutura comum, não comparável com o κ de posto completo.

Saída: data/marts/stability/n=<n>_w=<window>/year=<ano>.parquet, uma linha
por (date, method). Cada ano é gravado ao terminar: rodar de novo pula os
anos prontos (--force refaz).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import multiprocessing as mp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..config import Config, ConfigError
from ..portfolio import data, estimators as est
from ..portfolio.sentiment import Factors
from .engine import Market, Universe
from .models import covariance
from .plugin import Context, Env

RAW = "amostral"
NEWS_METHODS = {
    "fator + resíduo setor": {},
    "fator + resíduo notícias": {},
    "fator + resíduo setor + notícias": {},
    "placebo (setor + notícias embaralhadas)": {"placebo": True},
}
SENT_METHODS = (
    "sentimento + resíduo diagonal",
    "mercado + sentimento + resíduo diagonal",
    "mercado + sentimento + resíduo corr. constante",
    "placebo (sentimento embaralhado no tempo)",
)
METHODS = (*covariance.STATISTICAL, *NEWS_METHODS, *SENT_METHODS)
FACTOR_ONLY = (
    "só fatores: sentimento",
    "só fatores: mercado + sentimento",
    "só fatores: placebo (sentimento embaralhado no tempo)",
)


def out_dir(cfg: Config, n: int, window: int) -> Path:
    return cfg.marts / "stability" / f"n={n}_w={window}"


def cond(S: np.ndarray, rank: int | None = None) -> tuple[float, float]:
    """λmax/λmin da covariância e da correlação implícita (inf se λmin <= 0).
    Com `rank` = k, usa só os k maiores autovalores: λ₁/λₖ (matriz de posto k)."""
    def ratio(A):
        ev = np.linalg.eigvalsh((A + A.T) / 2)
        lo = ev[-rank] if rank else ev[0]
        return float(ev[-1] / lo) if lo > 0 else float("inf")
    sd = np.sqrt(np.diag(S))
    return ratio(S), ratio(S / np.outer(sd, sd))


def _year(cfg: Config, n: int, window: int, min_coverage: float, first: date, last: date, path: Path) -> str:
    t_start = time.time()
    con = query.connect(cfg)
    uni = Universe(con, SimpleNamespace(universe={"n": n}))
    cal = [d for (d,) in con.execute("SELECT date FROM news.trading_calendar ORDER BY date").fetchall()]
    idx = [i for i, d in enumerate(cal) if first <= d <= last]
    if not idx:
        return f"{first.year}: sem pregões"
    if idx[0] < window:
        raise ConfigError(f"{cal[idx[0]]}: o calendário não tem {window} pregões antes")
    mk = Market(con, uni.all_tickers(cal[idx[0]], cal[idx[-1]]), cal[idx[0] - window], cal[idx[-1]])
    env = Env(cfg=cfg, con=con, bt=None, rebalance_dates=[], universe_n=n)
    stat = {m: covariance.make(m) for m in covariance.STATISTICAL}
    news = {}
    for m, params in NEWS_METHODS.items():
        news[m] = covariance.make("fator + resíduo setor + notícias" if params.get("placebo") else m, params)
        news[m].setup(env)
    feat = news["fator + resíduo setor + notícias"]
    fx = Factors(con, mk.days)
    rng = np.random.default_rng(first.year)

    cols = {k: [] for k in ("date", "method", "cond_cov", "cond_corr", "p", "rank", "delta", "news_days")}
    for i in range(window, len(mk.days)):
        t = mk.days[i]
        U = [x for x in uni.at(t) if x in mk.col]
        c = np.array([mk.col[x] for x in U], int)
        W = mk.R[i - window:i][:, c]
        keep = (~np.isnan(W)).mean(axis=0) >= min_coverage
        names = [x for x, k in zip(U, keep) if k]
        dates = mk.days[i - window:i]
        ctx = Context(date=t, tickers=names, returns=W[:, keep].copy(), dates=dates, prev_weights=np.zeros(len(names)),
                      min_weight=0.0, max_weight=1.0, benchmark_returns=None, env=env)
        news_days = (t - dates[0]).days            # tuplas com data em [dates[0], t)
        F = feat.features(ctx, news_days)
        X = ctx.returns
        ks, _ = fx.choose(i, window)
        G = fx.F[i - window:i][:, ks]
        mG = np.column_stack([data.market_factor(X), G])
        sent = {
            "sentimento + resíduo diagonal": lambda: est.multi_factor(X, G, "diag"),
            "mercado + sentimento + resíduo diagonal": lambda: est.multi_factor(X, mG, "diag"),
            "mercado + sentimento + resíduo corr. constante": lambda: est.multi_factor(X, mG, "cc"),
            "placebo (sentimento embaralhado no tempo)": lambda: est.multi_factor(
                X, np.column_stack([mG[:, 0], G[rng.permutation(window)]]), "diag"),
        }
        perm = G[rng.permutation(window)]
        only = {
            "só fatores: sentimento": G,
            "só fatores: mercado + sentimento": mG,
            "só fatores: placebo (sentimento embaralhado no tempo)": np.column_stack([mG[:, 0], perm]),
        }
        for m in (*METHODS, *FACTOR_ONLY):
            ctx.info = {}
            rank = None
            if m in only:
                Gm = only[m]
                rank = int((Gm.var(axis=0) > 1e-12).sum())
                S = est.multi_factor(X, Gm, "none")
            else:
                S = stat[m](ctx) if m in stat else (sent[m]() if m in sent else news[m].estimate(ctx, F))
            cc, cr = cond(S, rank)
            cols["date"].append(t)
            cols["method"].append(m)
            cols["cond_cov"].append(cc)
            cols["cond_corr"].append(cr)
            cols["p"].append(len(names))
            cols["rank"].append(rank or len(names))
            cols["delta"].append(ctx.info.get("delta", np.nan))
            cols["news_days"].append(news_days if m in news else 0)
    con.close()
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table({"date": pa.array(cols.pop("date"), pa.date32()), **cols}), tmp, compression="zstd")
    tmp.rename(path)
    return f"{first.year}: {len(mk.days) - window} pregões em {time.time() - t_start:.0f}s"


def run(cfg: Config, n: int = 100, window: int = 252, start: str = "2011-01-01", end: str = "2023-12-31",
        min_coverage: float = 0.95, workers: int = 4, force: bool = False) -> Path:
    out = out_dir(cfg, n, window)
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = date.fromisoformat(start), date.fromisoformat(end)
    jobs = []
    for y in range(y0.year, y1.year + 1):
        path = out / f"year={y}.parquet"
        if path.exists() and not force:
            print(f"[stability] {y}: já feito ({path.name})", flush=True)
            continue
        jobs.append((cfg, n, window, min_coverage, max(y0, date(y, 1, 1)), min(y1, date(y, 12, 31)), path))
    # spawn: fork trava com as threads do DuckDB (ver CLAUDE.md)
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn")) as ex:
        for msg in ex.map(_year, *zip(*jobs)) if jobs else []:
            print(f"[stability] {msg}", flush=True)
    return out


# ---------------------------------------------------------------- gráficos

# Codificação composta (cor = família, traço = membro): 16 séries não cabem em
# 8 cores categóricas distintas. Paleta de referência validada (dataviz).
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
STYLE = {
    "amostral": ("#0b0b0b", "-", 1.4),
    "LW identidade": ("#2a78d6", "-", 1.0),
    "LW fator único": ("#2a78d6", "--", 1.0),
    "LW corr. constante": ("#2a78d6", ":", 1.2),
    "LW não linear": ("#2a78d6", "-.", 1.0),
    "RMT": ("#6250d6", "-", 1.0),
    "fator + resíduo diagonal": ("#1baf7a", ":", 1.2),
    "fator + resíduo corr. constante": ("#1baf7a", "-.", 1.0),
    "fator + resíduo setor": ("#1baf7a", "-", 1.0),
    "fator + resíduo notícias": ("#eb6834", "--", 1.0),
    "fator + resíduo setor + notícias": ("#eb6834", "-", 1.0),
    "placebo (setor + notícias embaralhadas)": (MUTED, "--", 1.0),
    "sentimento + resíduo diagonal": ("#e34948", ":", 1.2),
    "mercado + sentimento + resíduo diagonal": ("#e34948", "-", 1.0),
    "mercado + sentimento + resíduo corr. constante": ("#e34948", "-.", 1.0),
    "placebo (sentimento embaralhado no tempo)": (MUTED, ":", 1.2),
}


def load(cfg: Config, n: int = 100, window: int = 252):
    import pandas as pd
    files = sorted(out_dir(cfg, n, window).glob("year=*.parquet"))
    if not files:
        raise ConfigError(f"sem resultados em {out_dir(cfg, n, window)}; rode `stability run` antes")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _beta_zero(ax):
    """Antes de 2013 o β das notícias é 0 (menos de min_train = 24 meses de painel)."""
    import pandas as pd
    ax.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2013-01-01"), color="#f0efec", zorder=0,
               label="β das notícias = 0 (menos de 24 meses de treino)")


def _axes(ax, title: str, ylabel: str, log: bool = True):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def plot(cfg: Config, n: int = 100, window: int = 252, measure: str = "cond_cov", out: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load(cfg, n, window)
    wide = df.pivot(index="date", columns="method", values=measure)
    out = out or out_dir(cfg, n, window)
    what = "Σ̂" if measure == "cond_cov" else "correlação implícita"
    paths = []

    # 1. todas as séries no mesmo gráfico
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _beta_zero(ax)
    for m in METHODS:
        c, ls, lw = STYLE[m]
        ax.plot(wide.index, wide[m], color=c, linestyle=ls, linewidth=lw, label=m)
    _axes(ax, f"Número de condição λmax/λmin de {what}, por pregão (n = {n}, janela de {window} pregões)",
          "λmax/λmin (escala log)")
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    paths.append(out / f"{measure}_todos.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 2. um painel por método, com a matriz crua (amostral) como referência
    others = [m for m in METHODS if m != RAW]
    fig, axs = plt.subplots(4, 4, figsize=(20, 13), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, m in zip(axs.flat, [RAW, *others]):
        ax.set_facecolor("#fcfcfb")
        if m != RAW:
            ax.plot(wide.index, wide[RAW], color=MUTED, linewidth=0.8, label="amostral")
        c, ls, lw = STYLE[m]
        ax.plot(wide.index, wide[m], color=c if c != MUTED else INK, linewidth=1.0, label=m)
        _axes(ax, m, "")
    fig.suptitle(f"λmax/λmin de {what} por método; em cinza, a matriz amostral (crua)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    paths.append(out / f"{measure}_paineis.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 3. antes x depois das notícias: razão contra o alvo só de setor
    base = wide["fator + resíduo setor"]
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _beta_zero(ax)
    for m, c in (("fator + resíduo setor + notícias", "#eb6834"),
                 ("placebo (setor + notícias embaralhadas)", MUTED)):
        ax.plot(wide.index, wide[m] / base, color=c, linewidth=1.0, label=f"{m} / setor")
    ax.axhline(1, color="#c3c2b7", linewidth=1)
    _axes(ax, f"Efeito das notícias na estabilidade: λmax/λmin de {what} relativo ao alvo só de setor",
          "razão (> 1 = menos estável)", log=False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    paths.append(out / f"{measure}_razao_setor.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 4. fatores de sentimento: razão contra o mesmo modelo só com o mercado
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for m, b, c, ls in (("mercado + sentimento + resíduo diagonal", "fator + resíduo diagonal", "#e34948", "-"),
                        ("mercado + sentimento + resíduo corr. constante", "fator + resíduo corr. constante",
                         "#2a78d6", "-"),
                        ("placebo (sentimento embaralhado no tempo)", "fator + resíduo diagonal", MUTED, "-")):
        ax.plot(wide.index, wide[m] / wide[b], color=c, linestyle=ls, linewidth=1.0, label=f"{m} / {b}")
    ax.axhline(1, color="#c3c2b7", linewidth=1)
    _axes(ax, f"Efeito dos fatores de sentimento: λmax/λmin de {what} relativo ao modelo só com o mercado",
          "razão (> 1 = menos estável)", log=False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    paths.append(out / f"{measure}_razao_sentimento.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 5. só a parte comum B'Σ_g B: κ nos autovalores não nulos (outra medida)
    if all(m in wide for m in FACTOR_ONLY):
        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for m, c, ls in zip(FACTOR_ONLY, ("#e34948", "#2a78d6", MUTED), (":", "-", "-")):
            ax.plot(wide.index, wide[m], color=c, linestyle=ls, linewidth=1.0, label=m)
        _axes(ax, f"Só a parte comum B′Σ_f B (sem resíduo): λ₁/λₖ de {what} nos k autovalores não nulos",
              "λ₁/λₖ (escala log)")
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        paths.append(out / f"{measure}_so_fatores.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def summary(cfg: Config, n: int = 100, window: int = 252, measure: str = "cond_cov"):
    """Mediana, p10, p90 e máximo por método e período, e % de dias acima do setor."""
    import pandas as pd
    df = load(cfg, n, window)
    wide = df.pivot(index="date", columns="method", values=measure)
    base = wide["fator + resíduo setor"]
    rows = []
    for m in (*METHODS, *(f for f in FACTOR_ONLY if f in wide)):
        s = wide[m]
        rows.append({"método": m, "mediana": s.median(), "p10": s.quantile(0.1), "p90": s.quantile(0.9),
                     "máximo": s.max(), "mediana/amostral": (s / wide[RAW]).median(),
                     "dias > setor (%)": 100 * (s > base).mean()})
    by_year = wide.groupby(wide.index.year).median()
    return pd.DataFrame(rows).set_index("método"), by_year
