"""Motor de backtest (newsmanager/backtester) contra uma simulação dia a dia
escrita à parte, num banco DuckDB sintético. Sem dados reais.

    python tests/test_backtester.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newsmanager import config  # noqa: E402
from newsmanager.backtester import Model, engine, metrics, settings  # noqa: E402
from newsmanager.config import ConfigError  # noqa: E402
from newsmanager.portfolio import optimize  # noqa: E402

HERE = str(Path(__file__).resolve())
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


# ------------------------------------------------------------ plugins de teste

class Fixed(Model):
    """Pesos fixos; confere que nada do futuro chega ao modelo."""

    def __init__(self, w):
        self.w = np.array(w, float)
        self.seen = []

    def weights(self, ctx):
        assert max(ctx.dates) < ctx.date, "janela in-sample invadiu o período de carteira"
        assert ctx.returns.shape == (len(ctx.dates), ctx.p)
        ctx.info["calls"] = len(self.seen)
        self.seen.append(ctx.date)
        return dict(zip(ctx.tickers, self.w[: ctx.p]))


def momentum(ctx, lookback=5):
    """Plugin-função: tudo no ativo com maior retorno recente (testa a projeção)."""
    r = np.nansum(ctx.returns[-lookback:], axis=0)
    w = np.zeros(ctx.p)
    w[int(np.argmax(r))] = 1.0
    return w


# ------------------------------------------------------------ dados sintéticos

def make_db(path: Path, days: list[date], R: np.ndarray, universe: dict[date, list[str]]) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA news")
    con.execute("CREATE TABLE news.trading_calendar (date DATE, n_tickers BIGINT, cal_i BIGINT)")
    con.executemany("INSERT INTO news.trading_calendar VALUES (?, ?, ?)",
                    [(d, len(TICKERS), i) for i, d in enumerate(days)])
    con.execute("CREATE TABLE news.returns (ticker VARCHAR, date DATE, ret DOUBLE)")
    con.executemany("INSERT INTO news.returns VALUES (?, ?, ?)",
                    [(t, d, float(R[i, j])) for i, d in enumerate(days) for j, t in enumerate(TICKERS + ["BEN"])
                     if np.isfinite(R[i, j])])
    con.execute("CREATE TABLE news.universe (rebalance_date DATE, ticker VARCHAR, rank BIGINT, n_target INTEGER)")
    con.executemany("INSERT INTO news.universe VALUES (?, ?, ?, 4)",
                    [(d, t, k + 1) for d, ts in universe.items() for k, t in enumerate(ts)])
    con.close()


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def reference(days, R, cols, reb_idx, targets, cost_rate, universe_at):
    """Simulação ingênua, um dia de cada vez, com dicionários."""
    w: dict[str, float] = {}
    out = []
    targets = iter(targets)
    for i in range(reb_idx[0], len(days)):
        c = 0.0
        if i in reb_idx:
            tgt = next(targets)
            names = set(w) | set(tgt)
            c = cost_rate * sum(abs(tgt.get(t, 0) - w.get(t, 0)) for t in names)
            w = dict(tgt)
        r = {t: (0.0 if not np.isfinite(R[i, cols[t]]) else R[i, cols[t]]) for t in w}
        g = sum(w[t] * r[t] for t in w)
        out.append((1 - c) * (1 + g) - 1)
        w = {t: w[t] * (1 + r[t]) / (1 + g) for t in w}
    return np.array(out)


def main() -> int:
    rng = np.random.default_rng(3)
    days = weekdays(date(2019, 1, 1), 160)
    R = rng.normal(0.0005, 0.01, (len(days), len(TICKERS) + 1))
    R[100:, 3] = np.nan                       # DDD é deslistada no dia 100
    R[70, :] = np.nan                         # emenda: dia sem retorno para ninguém
    cols = {t: j for j, t in enumerate(TICKERS + ["BEN"])}
    # universo muda: CCC entra em março; DDD (30% da carteira) é deslistada em maio e sai em junho
    month_starts = [d for k, d in enumerate(days) if k == 0 or d.month != days[k - 1].month]
    uni = {}
    for d in month_starts:
        ts = ["AAA", "BBB", "DDD"] if d < date(2019, 3, 1) else ["AAA", "DDD", "BBB", "CCC"]
        if d >= date(2019, 6, 1):
            ts = ["AAA", "BBB", "CCC"]
        uni[d] = ts

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = dataclasses.replace(config.load(), root=tmp, marts=tmp / "marts", db=tmp / "t.duckdb",
                                  curated=tmp / "curated")
        make_db(cfg.db, days, R, uni)

        base = dict(start="2019-03-01", end="2019-08-30", insample_days=20, rebalance="M",
                    benchmark="BEN", max_weight=1.0, min_weight=0.0, cost_bps=25.0,
                    universe={"n": 4}, min_coverage=0.0)

        # 1. pesos fixos x simulação de referência (giro, custo, deriva, deslistagem, emenda)
        bt = settings.BacktestConfig(name="fixo", **base,
                                     model={"plugin": f"{HERE}:Fixed", "params": {"w": [0.5, 0.3, 0.2, 0.0]}})
        s = engine.run(cfg, bt, verbose=False)
        out = cfg.marts / "backtests" / "fixo"
        import pyarrow.parquet as pq
        daily = pq.read_table(out / "daily.parquet").to_pydict()
        reb = pq.read_table(out / "rebalances.parquet").to_pydict()
        i0 = days.index(min(d for d in days if d >= date(2019, 3, 1)))
        reb_idx = [days.index(d) for d in reb["date"]]
        assert reb_idx[0] == i0 and all(days[i].day <= 3 for i in reb_idx), reb["date"]
        # pesos-alvo: dict de Fixed recortado às p primeiras posições do universo elegível
        tg = []
        for i in reb_idx:
            names = [x for x in uni[max(d for d in uni if d <= days[i])]]
            tg.append(dict(zip(names, [0.5, 0.3, 0.2, 0.0][: len(names)])))
        # Fixed devolve w[:p]; com p = 3 os pesos somam 1 (0.5+0.3+0.2)
        ref = reference(days, R, cols, reb_idx, tg, 25e-4, None)
        got = np.array(daily["ret"])
        assert len(got) == len(ref), (len(got), len(ref))
        assert np.allclose(got, ref, atol=1e-14), np.abs(got - ref).max()
        bench = np.nan_to_num(R[i0:, cols["BEN"]])
        assert np.allclose(daily["bench_ret"], bench)
        assert np.isclose(daily["nav"][-1], np.prod(1 + ref))
        assert reb["turnover"][0] == 1.0 and np.isclose(reb["cost"][0], 25e-4)
        assert reb["info_calls"] == list(range(len(reb_idx)))
        pos = pq.read_table(out / "positions.parquet").to_pandas()
        g = pos.groupby("date")["contrib"].sum().to_numpy()
        assert np.allclose(g, daily["ret_gross"], atol=1e-14)
        assert np.allclose(pos.groupby("date")["weight"].sum(), 1.0)
        tr = pq.read_table(out / "trades.parquet").to_pandas()
        first = tr[tr["date"] == tr["date"].min()]
        assert np.isclose(first["w_after"].sum(), 1) and (first["w_before"] == 0).all()
        assert json.loads((out / "summary.json").read_text())["net"]["final_nav"] == s["net"]["final_nav"]
        print("ok  pesos fixos = simulação de referência (custo, deriva, deslistagem, emenda, benchmark)")

        # 2. plugin-função + teto de peso: momentum põe 100% num ativo -> projeção para o teto
        bt = settings.BacktestConfig(name="mom", **{**base, "max_weight": 0.4, "rebalance": "W"},
                                     model={"plugin": f"{HERE}:momentum", "params": {"lookback": 3}})
        s = engine.run(cfg, bt, verbose=False)
        reb = pq.read_table(cfg.marts / "backtests" / "mom" / "rebalances.parquet").to_pydict()
        assert s["projected_rebalances"] == len(reb["date"]) and max(reb["w_max"]) <= 0.4 + 1e-12
        assert all(abs(x - 1) < 1e-12 for x in reb["gross"])
        assert all(days[days.index(d) - 1].isocalendar()[1] != d.isocalendar()[1] or d == reb["date"][0]
                   for d in reb["date"])
        print(f"ok  plugin-função, rebalanceamento semanal ({len(reb['date'])}x) e projeção no teto de 40%")

        # 3. on_violation = "error" interrompe
        bt = settings.BacktestConfig(name="err", **{**base, "max_weight": 0.4, "on_violation": "error"},
                                     model={"plugin": f"{HERE}:momentum", "params": {}})
        try:
            engine.run(cfg, bt, verbose=False)
            raise AssertionError("deveria falhar")
        except ConfigError as e:
            assert "fora dos limites" in str(e)
        # teto inviável: 3 ativos x 0.3 < 1
        bt = settings.BacktestConfig(name="inv", **{**base, "max_weight": 0.3},
                                     model={"plugin": "newsmanager.backtester.models.basic:EqualWeight"})
        try:
            engine.run(cfg, bt, verbose=False)
            raise AssertionError("deveria falhar")
        except ConfigError as e:
            assert "max_weight" in str(e)
        print("ok  pesos inválidos com on_violation='error' e teto inviável param a run")

        # 4. modelos prontos rodam com venda a descoberto e benchmark equal_weight
        bt = settings.BacktestConfig(name="mv", **{**base, "min_weight": -0.5, "max_weight": 0.8,
                                                   "benchmark": "equal_weight", "rebalance": "10D"},
                                     model={"plugin": "newsmanager.backtester.models.basic:MinVariance",
                                            "params": {"estimator": "LW corr. constante"}})
        s = engine.run(cfg, bt, verbose=False)
        reb = pq.read_table(cfg.marts / "backtests" / "mv" / "rebalances.parquet").to_pydict()
        assert min(reb["w_min"]) >= -0.5 - 1e-9 and max(reb["w_max"]) <= 0.8 + 1e-9
        assert "info_cond" in reb and "beta" in s["net"]
        print("ok  MinVariance com venda a descoberto, a cada 10 pregões, benchmark 1/N")

        # 4b. cada otimizador roda no motor e respeita os limites
        plugins = {"mean_variance": "MeanVariance", "max_sharpe": "MaxSharpe", "hrp": "HRP",
                   "max_diversification": "MaxDiversification", "min_tracking_error": "MinTrackingError",
                   "min_variance": "MinVariance", "inverse_volatility": "InverseVolatility"}
        for name, cls in plugins.items():
            params = {} if name == "inverse_volatility" else {"cov": "LW fator único"}
            bt = settings.BacktestConfig(name=f"o_{name}", **{**base, "max_weight": 0.45},
                                         model={"plugin": f"newsmanager.backtester.models.{name}:{cls}",
                                                "params": params})
            s = engine.run(cfg, bt, verbose=False)
            reb = pq.read_table(cfg.marts / "backtests" / f"o_{name}" / "rebalances.parquet").to_pydict()
            assert max(reb["w_max"]) <= 0.45 + 1e-9 and min(reb["w_min"]) >= -1e-12, name
            assert s["projected_rebalances"] == 0, name
        print(f"ok  {len(plugins)} plugins de modelo rodam no motor e respeitam o teto")

    # 5. calendário de rebalanceamento
    ds = weekdays(date(2020, 1, 1), 300)
    for rule, key in (("W", lambda d: d.isocalendar()[:2]), ("M", lambda d: (d.year, d.month)),
                      ("Q", lambda d: (d.year, (d.month - 1) // 3)), ("A", lambda d: d.year)):
        idx = engine.schedule(ds, rule)
        assert idx[0] == 0 and len(idx) == len({key(d) for d in ds}), rule
    assert engine.schedule(ds, "7D") == list(range(0, 300, 7))
    print("ok  calendários D/W/M/Q/A/<k>D")

    # 6. otimização com limites: sem limite ativo = GMV fechada; com teto, KKT
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 8)) * 0.01
    S = np.cov(A, rowvar=False)
    w0 = optimize.gmv(S)
    w = optimize.min_variance(S, -np.inf, np.inf)
    assert np.allclose(w, w0)
    w = optimize.min_variance(S, -10, 10, tol=1e-13, max_iter=200000)
    assert np.allclose(w, w0, atol=1e-6)
    w = optimize.min_variance(S, 0.0, 0.15, tol=1e-14, max_iter=200000)
    g = 2 * S @ w
    free = (w > 1e-9) & (w < 0.15 - 1e-9)
    lam = g[free].mean()
    assert np.allclose(g[free], lam, rtol=1e-4), "KKT: gradiente igual nos ativos livres"
    assert (g[w <= 1e-9] >= lam - 1e-8).all() and (g[w >= 0.15 - 1e-9] <= lam + 1e-8).all()
    assert abs(w.sum() - 1) < 1e-12 and w.max() <= 0.15 + 1e-12
    print("ok  min_variance: GMV sem limites, KKT com teto de 15%")

    # 6b. otimizadores novos contra soluções fechadas
    mu = A.mean(axis=0) + 0.0003
    w, r = optimize.max_ratio(S, mu, -50, 50)
    t = np.linalg.solve(S, mu)
    t /= t.sum()
    assert np.allclose(w, t, atol=1e-6) and np.isclose(r, mu @ t / np.sqrt(t @ S @ t))
    Si, one, g = np.linalg.inv(S), np.ones(8), 5.0
    eta = (one @ Si @ mu - g) / (one @ Si @ one)
    assert np.allclose(optimize.mean_variance(S, mu, g, -50, 50), Si @ (mu - eta * one) / g, atol=1e-8)
    D = np.diag(np.linspace(0.01, 0.05, 6) ** 2)
    w, _ = optimize.max_ratio(D, np.sqrt(np.diag(D)), 0, 1)
    assert np.allclose(w, (1 / np.sqrt(np.diag(D))) / (1 / np.sqrt(np.diag(D))).sum(), atol=1e-6)
    b = A @ np.full(8, 1 / 8)                       # benchmark = 1/N: TE mínimo é zero em w = 1/N
    c = np.cov(np.column_stack([A, b]), rowvar=False)[:8, 8]
    assert np.allclose(optimize.min_tracking_error(S, c, 0, 1), np.full(8, 1 / 8), atol=1e-6)
    # HRP: Σ diagonal -> variância inversa; soma 1 e positivo em geral
    v = np.linspace(1, 4, 5) * 1e-4
    assert np.allclose(optimize.hrp(np.diag(v)), (1 / v) / (1 / v).sum())
    h = optimize.hrp(S)
    assert np.isclose(h.sum(), 1) and (h > 0).all()
    mb, phi = optimize.bayes_stein_mean(A, S)
    xb = A.mean(axis=0)
    m0 = (one @ Si @ xb) / (one @ Si @ one)
    ph = (8 + 2) / ((8 + 2) + len(A) * (xb - m0) @ Si @ (xb - m0))
    assert 0 < phi < 1 and np.isclose(phi, ph) and np.allclose(mb, (1 - ph) * xb + ph * m0)
    print("ok  tangente, Markowitz, máxima diversificação, TE mínimo, HRP e Bayes-Stein contra fórmulas fechadas")

    # 7. sobrescritas do TOML e medidas
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "x.toml"
        f.write_text('start = "2013-01-01"\nend = "2014-01-01"\n[model]\nplugin = "a:b"\n')
        bt = settings.load(f, ["max_weight=0.05", "model.params.estimator='RMT'", "universe={tickers=['A','B']}"])
        assert bt.name == "x" and bt.max_weight == 0.05 and bt.model["params"]["estimator"] == "RMT"
        assert bt.universe == {"tickers": ["A", "B"]}
    r = np.array([0.1, -0.5, 0.2, 0.0])
    assert np.allclose(metrics.drawdown(r), [0, -0.5, -0.4, -0.4])
    m = metrics.summary(r, r)
    assert np.isclose(m["beta"], 1) and np.isclose(m["tracking_error"], 0)
    print("ok  TOML com --set e medidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
