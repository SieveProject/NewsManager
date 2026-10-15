"""Motor do backtest: janela deslizante, rebalanceamento, custos e P&L diário.

Linha do tempo de um rebalanceamento no pregão t:

  1. universo point-in-time em t (news.universe com rebalance_date <= t, ou a
     lista fixa), menos os ativos com menos de `min_coverage` de retornos
     válidos nos `insample_days` pregões ANTES de t;
  2. o modelo recebe esses retornos (só datas < t) e devolve os pesos-alvo;
  3. a troca acontece no fechamento de t-1: giro = Σ|w_alvo - w_atual|, com
     w_atual = pesos derivados pelos preços desde o último rebalanceamento
     (ativos que saíram do universo são vendidos); custo = cost_bps x giro,
     cobrado sobre o patrimônio antes do retorno de t:
        NAV_t = NAV_{t-1} (1 - custo) (1 + r_t)
     O primeiro rebalanceamento compra a carteira inteira (giro 1) e paga custo;
  4. entre rebalanceamentos a carteira é comprada e mantida: cada posição
     rende o próprio retorno e os pesos derivam. Retorno faltante (emenda do
     arquivo, deslistagem) conta como 0: a posição fica parada até a próxima
     venda. O mesmo vale para o benchmark.

Saída em data/marts/backtests/<name>/: config.json, summary.json,
daily.parquet (um pregão por linha), positions.parquet (peso de início do dia,
retorno e contribuição por ativo), trades.parquet e rebalances.parquet (com o
que o modelo pôs em ctx.info).
"""

from __future__ import annotations

import bisect
import json
import shutil
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..config import Config, ConfigError
from ..portfolio.optimize import project_box_simplex
from . import metrics
from .plugin import Context, Env, load_model
from .settings import BacktestConfig

TOL_SUM, TOL_BOUND = 1e-6, 1e-9


def runs_dir(cfg: Config) -> Path:
    return cfg.marts / "backtests"


# ---------------------------------------------------------------- calendário

def schedule(days: list[date], rule: str) -> list[int]:
    """Índices (em `days`) dos pregões de rebalanceamento. O primeiro pregão
    sempre rebalanceia (monta a carteira); depois, o 1º pregão de cada
    semana/mês/trimestre/ano, todo pregão (D) ou a cada k pregões (<k>D)."""
    if not days:
        return []
    if rule == "D":
        return list(range(len(days)))
    if rule.endswith("D"):
        return list(range(0, len(days), int(rule[:-1])))
    key = {"W": lambda d: d.isocalendar()[:2], "M": lambda d: (d.year, d.month),
           "Q": lambda d: (d.year, (d.month - 1) // 3), "A": lambda d: d.year}[rule]
    out = [0]
    for i in range(1, len(days)):
        if key(days[i]) != key(days[i - 1]):
            out.append(i)
    return out


# ---------------------------------------------------------------- dados

class Market:
    """Retornos diários (pregões x tickers) carregados uma vez, NaN onde falta."""

    def __init__(self, con, tickers: list[str], first: date, last: date):
        self.days = [d for (d,) in con.execute(
            "SELECT date FROM news.trading_calendar WHERE date BETWEEN ? AND ? ORDER BY date", [first, last]).fetchall()]
        self.tickers = sorted(set(tickers))
        self.col = {t: j for j, t in enumerate(self.tickers)}
        pos = {d: i for i, d in enumerate(self.days)}
        self.R = np.full((len(self.days), len(self.tickers)), np.nan)
        # Lista literal: com `IN (SELECT unnest(?))` o DuckDB não poda os row
        # groups do Parquet e a consulta levou 15 min em vez de 0,1 s.
        lit = ", ".join("'" + t.replace("'", "''") + "'" for t in self.tickers)
        res = con.execute(f"SELECT ticker, date, ret FROM news.returns WHERE date BETWEEN ? AND ? "
                          f"AND ticker IN ({lit})", [first, last]).fetchnumpy()
        rows = [pos[d] for d in res["date"].astype("datetime64[D]").astype(object)]
        cols = [self.col[t] for t in res["ticker"]]
        self.R[rows, cols] = res["ret"]
        self.missing = [t for t in self.tickers if np.isnan(self.R[:, self.col[t]]).all()]


class Universe:
    def __init__(self, con, bt: BacktestConfig):
        self.n = bt.universe.get("n")
        if self.n is not None:
            rows = con.execute("SELECT rebalance_date, ticker FROM news.universe WHERE n_target = ? "
                               "ORDER BY rebalance_date, rank", [int(self.n)]).fetchall()
            if not rows:
                raise ConfigError(f"sem news.universe n={self.n}; rode `python -m newsmanager universe --n {self.n}`")
            by: dict[date, list[str]] = {}
            for d, t in rows:
                by.setdefault(d, []).append(t)
            self.dates = sorted(by)
            self.by = by
        else:
            self.fixed = list(dict.fromkeys(bt.universe["tickers"]))

    def at(self, t: date) -> list[str]:
        if self.n is None:
            return self.fixed
        k = bisect.bisect_right(self.dates, t) - 1
        if k < 0:
            raise ConfigError(f"news.universe n={self.n} começa em {self.dates[0]}: não há universo em {t}")
        return self.by[self.dates[k]]

    def all_tickers(self, first: date, last: date) -> list[str]:
        if self.n is None:
            return self.fixed
        ds = [d for d in self.dates if d <= last]
        keep = [d for d in ds if d >= first] + [d for d in ds if d < first][-1:]
        return sorted({t for d in keep for t in self.by[d]})


# ---------------------------------------------------------------- carteira

class Book:
    """Pesos correntes de uma carteira (somam 1) e simulação comprada-e-mantida."""

    def __init__(self, cost_rate: float):
        self.cost_rate = cost_rate
        self.w: dict[str, float] = {}

    def rebalance(self, tickers: list[str], w: np.ndarray) -> tuple[float, float, list[tuple]]:
        new = dict(zip(tickers, w))
        names = list(dict.fromkeys([*self.w, *tickers]))
        trades = [(t, self.w.get(t, 0.0), new.get(t, 0.0)) for t in names]
        turnover = float(sum(abs(b - a) for _, a, b in trades))
        self.tickers, self.target = tickers, np.asarray(w, float)
        return turnover, self.cost_rate * turnover, trades

    def hold(self, Rp: np.ndarray, cost: float) -> dict:
        """Rp: dias x p (NaN -> 0). Devolve retornos e pesos de início de cada dia."""
        Rp = np.nan_to_num(Rp)
        V = np.cumprod(1 + Rp, axis=0) * self.target
        tot = V.sum(axis=1)
        prev = np.concatenate([[1.0], tot[:-1]])
        gross = tot / prev - 1
        W0 = np.vstack([self.target[None, :], V[:-1] / tot[:-1, None]])
        net = gross.copy()
        net[0] = (1 - cost) * (1 + gross[0]) - 1
        self.w = {t: float(x) for t, x in zip(self.tickers, V[-1] / tot[-1]) if abs(x) > 1e-12}
        return {"gross": gross, "net": net, "W0": W0, "R": Rp}


def _check_weights(raw, tickers: list[str], lo: float, hi: float, mode: str) -> tuple[np.ndarray, float]:
    """Confere o vetor do plugin; projeta no conjunto viável se preciso.
    Devolve (pesos, distância L1 da projeção; 0 se o vetor já era válido)."""
    p = len(tickers)
    if isinstance(raw, dict):
        unknown = set(raw) - set(tickers)
        if unknown:
            raise ValueError(f"pesos para tickers fora do universo elegível: {sorted(unknown)[:5]}")
        w = np.array([float(raw.get(t, 0.0)) for t in tickers])
    else:
        w = np.asarray(raw, float).ravel()
    if w.shape != (p,):
        raise ValueError(f"o plugin devolveu {w.shape[0]} pesos para {p} ativos")
    if not np.isfinite(w).all():
        raise ValueError("o plugin devolveu pesos NaN/inf")
    ok = abs(w.sum() - 1) <= TOL_SUM and (w >= lo - TOL_BOUND).all() and (w <= hi + TOL_BOUND).all()
    if ok:
        return w, 0.0
    if mode == "error":
        raise ValueError(f"pesos fora dos limites: Σw={w.sum():.6f}, min={w.min():.4f}, max={w.max():.4f} "
                         f"(limites [{lo}, {hi}])")
    v = project_box_simplex(w, lo, hi)
    return v, float(np.abs(v - w).sum())


# ---------------------------------------------------------------- execução

def _git() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "newsmanager"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": head, "dirty_code": dirty}
    except OSError:
        return {}


def run(cfg: Config, bt: BacktestConfig, out: Path | None = None, verbose: bool = True) -> dict:
    t_start = time.time()
    model, prov = load_model(bt.model["plugin"], bt.model["params"])
    con = query.connect(cfg)
    uni = Universe(con, bt)

    cal = [d for (d,) in con.execute("SELECT date FROM news.trading_calendar ORDER BY date").fetchall()]
    i0 = next((i for i, d in enumerate(cal) if d >= bt.start), None)
    i_end = max((i for i, d in enumerate(cal) if d <= bt.end), default=None)
    if i0 is None or i_end is None or i_end < i0:
        raise ConfigError(f"sem pregões entre {bt.start} e {bt.end} (calendário: {cal[0]} a {cal[-1]})")
    if i0 < bt.insample_days:
        raise ConfigError(f"start={bt.start} não deixa {bt.insample_days} pregões in-sample (o calendário começa em {cal[0]})")
    bench_ticker = None if bt.benchmark.lower() in ("none", "equal_weight") else bt.benchmark
    tickers = uni.all_tickers(cal[i0], cal[i_end]) + ([bench_ticker] if bench_ticker else [])
    mk = Market(con, tickers, cal[i0 - bt.insample_days], cal[i_end])
    off = bt.insample_days                       # índice de cal[i0] em mk.days
    days = mk.days[off:]
    reb = [off + k for k in schedule(days, bt.rebalance)]
    if bench_ticker and bench_ticker in mk.missing:
        raise ConfigError(f"benchmark {bench_ticker} sem retornos no período")

    env = Env(cfg=cfg, con=con, bt=bt, rebalance_dates=[mk.days[i] for i in reb], universe_n=uni.n)
    model.setup(env)
    book, bbook = Book(bt.cost_bps / 1e4), Book(bt.cost_bps / 1e4)
    lo, hi = bt.min_weight, bt.max_weight

    T = len(days)
    daily = {k: np.zeros(T) for k in ("ret", "ret_gross", "cost", "turnover", "bench_ret")}
    daily["rebalance"] = np.zeros(T, bool)
    daily["n_positions"] = np.zeros(T, np.int32)
    daily["gross_exposure"] = np.zeros(T)
    daily["max_weight"] = np.zeros(T)
    pos_cols = {k: [] for k in ("date", "ticker", "weight", "ret", "contrib")}
    trade_cols = {k: [] for k in ("date", "ticker", "w_before", "w_after")}
    rb_rows: list[dict] = []
    n_proj = 0
    report_every = max(1, len(reb) // 20)

    for k, i in enumerate(reb):
        i_next = reb[k + 1] if k + 1 < len(reb) else len(mk.days)
        t = mk.days[i]
        U = [x for x in uni.at(t) if x in mk.col]
        cols = np.array([mk.col[x] for x in U], int)
        W = mk.R[i - bt.insample_days:i][:, cols]
        cov = (~np.isnan(W)).mean(axis=0)
        keep = cov >= bt.min_coverage
        names = [x for x, kk in zip(U, keep) if kk]
        cols, W = cols[keep], W[:, keep]
        p = len(names)
        if p == 0:
            raise ConfigError(f"{t}: nenhum ativo com cobertura >= {bt.min_coverage} na janela")
        if p * hi < 1 - TOL_SUM:
            raise ConfigError(f"{t}: {p} ativos x max_weight {hi} < 1; aumente max_weight ou o universo")
        prev = np.array([book.w.get(x, 0.0) for x in names])
        ctx = Context(date=t, tickers=names, returns=W.copy(), dates=mk.days[i - bt.insample_days:i],
                      prev_weights=prev, min_weight=lo, max_weight=hi,
                      benchmark_returns=(mk.R[i - bt.insample_days:i, mk.col[bench_ticker]].copy()
                                         if bench_ticker else None), env=env)
        t0 = time.time()
        try:
            raw = model.weights(ctx)
            w, dist = _check_weights(raw, names, lo, hi, bt.on_violation)
        except ValueError as e:
            raise ConfigError(f"{t}: {e}") from e
        secs = time.time() - t0
        n_proj += dist > 0
        turnover, cost, trades = book.rebalance(names, w)
        for x, a, b in trades:
            trade_cols["date"].append(t)
            trade_cols["ticker"].append(x)
            trade_cols["w_before"].append(a)
            trade_cols["w_after"].append(b)
        Rp = mk.R[i:i_next][:, cols]
        h = book.hold(Rp, cost)
        s = slice(i - off, i_next - off)
        daily["ret"][s], daily["ret_gross"][s] = h["net"], h["gross"]
        daily["cost"][i - off], daily["turnover"][i - off] = cost, turnover
        daily["rebalance"][i - off] = True
        nz = np.abs(h["W0"]) > 1e-10
        daily["n_positions"][s] = nz.sum(axis=1)
        daily["gross_exposure"][s] = np.abs(h["W0"]).sum(axis=1)
        daily["max_weight"][s] = h["W0"].max(axis=1)
        di, dj = np.nonzero(nz)
        pos_cols["date"] += [mk.days[i + a] for a in di]
        pos_cols["ticker"] += [names[b] for b in dj]
        pos_cols["weight"].append(h["W0"][di, dj])
        pos_cols["ret"].append(h["R"][di, dj])
        pos_cols["contrib"].append(h["W0"][di, dj] * h["R"][di, dj])

        if bench_ticker:
            daily["bench_ret"][s] = np.nan_to_num(mk.R[i:i_next, mk.col[bench_ticker]])
        elif bt.benchmark.lower() == "equal_weight":
            _, bcost, _ = bbook.rebalance(names, np.full(p, 1 / p))
            daily["bench_ret"][s] = bbook.hold(Rp, bcost)["net"]

        row = {"date": t, "n_universe": len(U), "n_eligible": p, "n_assets": int((np.abs(w) > 1e-10).sum()),
               "turnover": turnover, "cost": cost, "projected": dist > 0, "projection_l1": dist,
               "eff_n": float(1 / np.sum(w ** 2)), "w_max": float(w.max()), "w_min": float(w.min()),
               "gross": float(np.abs(w).sum()), "n_short": int((w < -1e-10).sum()),
               "period_days": i_next - i, "seconds": secs}
        for key, v in ctx.info.items():
            if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)):
                row[f"info_{key}"] = float(v)
        rb_rows.append(row)
        if verbose and ((k + 1) % report_every == 0 or k + 1 == len(reb)):
            nav = np.prod(1 + daily["ret"][:i_next - off])
            print(f"[backtest] {bt.name}: {t} ({k + 1}/{len(reb)}) p={p} giro={turnover:.2f} "
                  f"NAV={nav:.3f} ({time.time() - t_start:.0f}s)", flush=True)
    con.close()

    has_bench = bt.benchmark.lower() != "none"
    turn = np.array([r["turnover"] for r in rb_rows])
    summ = {
        "name": bt.name, "first": str(days[0]), "last": str(days[-1]), "rebalances": len(reb),
        "projected_rebalances": int(n_proj), "seconds": round(time.time() - t_start, 1),
        "net": metrics.summary(daily["ret"], daily["bench_ret"] if has_bench else None, bt.risk_free,
                               turnover=turn[1:], cost=daily["cost"]),
        "gross": metrics.summary(daily["ret_gross"], None, bt.risk_free),
    }
    if has_bench:
        summ["benchmark"] = metrics.summary(daily["bench_ret"], None, bt.risk_free)

    out = out or runs_dir(cfg) / bt.name
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    nav, bnav = np.cumprod(1 + daily["ret"]), np.cumprod(1 + daily["bench_ret"])
    pq.write_table(pa.table({"date": pa.array(days, pa.date32()), **daily, "nav": nav,
                             "drawdown": metrics.drawdown(daily["ret"]),
                             **({"bench_nav": bnav, "bench_drawdown": metrics.drawdown(daily["bench_ret"])}
                                if has_bench else {})}),
                   tmp / "daily.parquet", compression="zstd")
    pq.write_table(pa.table({"date": pa.array(pos_cols["date"], pa.date32()), "ticker": pos_cols["ticker"],
                             **{c: np.concatenate(pos_cols[c]) for c in ("weight", "ret", "contrib")}}),
                   tmp / "positions.parquet", compression="zstd")
    pq.write_table(pa.table({"date": pa.array(trade_cols["date"], pa.date32()), "ticker": trade_cols["ticker"],
                             "w_before": trade_cols["w_before"], "w_after": trade_cols["w_after"]}),
                   tmp / "trades.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(rb_rows), tmp / "rebalances.parquet", compression="zstd")
    meta = {"config": bt.to_json(), "model": prov, "benchmark_kind": "ticker" if bench_ticker else bt.benchmark,
            "git": _git(), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "missing_tickers": mk.missing}
    (tmp / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (tmp / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    if verbose:
        report(summ)
        print(f"[backtest] {out}")
    return summ


def report(s: dict) -> None:
    n = s["net"]
    pct = lambda x: "—" if x is None else f"{x:7.2%}"       # noqa: E731
    num = lambda x: "—" if x is None else f"{x:7.3f}"       # noqa: E731
    print(f"\n{s['name']}: {s['first']} a {s['last']}, {s['rebalances']} rebalanceamentos"
          f" ({s['projected_rebalances']} projetados)")
    print(f"  CAGR {pct(n['cagr'])}  vol {pct(n['ann_vol'])}  Sharpe {num(n['sharpe'])}  "
          f"Sortino {num(n['sortino'])}  maxDD {pct(n['max_drawdown'])}")
    print(f"  giro/rebal. {n['turnover_per_rebalance']:.2f}  giro/ano {n['turnover_per_year']:.2f}  "
          f"custo/ano {pct(n['cost_per_year'])}  (bruto: CAGR {pct(s['gross']['cagr'])})")
    if "beta" in n:
        print(f"  benchmark: CAGR {pct(n['bench_cagr'])}  vol {pct(n['bench_vol'])}  beta {num(n['beta'])}  "
              f"alfa {pct(n['alpha_ann'])}  TE {pct(n['tracking_error'])}  IR {num(n['information_ratio'])}")
