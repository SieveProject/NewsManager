"""Backtest mensal: estimadores de Σ -> carteiras de variância mínima.

Para cada rebalanceamento t do universo (news.universe) de 2013-01 em diante:

  1. X_in = retornos dos 252 pregões antes de t; X_out = do pregão t até o
     anterior ao próximo rebalanceamento.
  2. Cada estimador produz Σ̂ só com X_in (e, nos alvos por regressão, com
     notícias < t e com a regressão treinada nos meses ANTERIORES a t).
  3. GMV com e sem venda a descoberto; 1/N como referência ingênua.
  4. Carteira comprada e mantida no mês (pesos derivam com os preços);
     retorno faltante no mês (dia inválido, deslistagem) conta como 0.
  5. Giro = Σ|w_novo - w_derivado| no rebalanceamento; custo 10 bps por unidade.

Alvos por regressão ("histórico", "histórico + setor", "notícias"): a
regressão da H1 (h1.py), MQO empilhado nos meses < t com y e x centrados no
mês, prevê ẑ_ij = média(z_hist) + (x_ij - x̄)'b. O alvo é R = tanh(ẑ),
projetado em correlação válida, e Σ̂ = shrinkage ótimo de S para
D^½ R D^½ (estimators.shrink_to_corr). Só os regressores mudam entre os três:
a diferença "notícias" - "histórico + setor" é o valor das notícias.

Placebo: o alvo "setor + notícias" com os atributos de notícias embaralhados entre as
ações do mês (mesma permutação para todos os atributos; regressão treinada nos
dados reais). Se as notícias fossem só regularização genérica, o placebo
empataria com o alvo verdadeiro.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..config import Config
from . import data, estimators as est, features, h1, optimize

MIN_TRAIN = h1.MIN_TRAIN
COST = 0.0010
BLOCK = 10
BOOT = 2000

# Alvos-prior (Becquin e Esmeir 2023): a correlação-alvo NÃO usa a correlação
# amostral, só o nível médio dela (como o alvo de correlação constante) e os
# atributos de setor/notícias: R_ij = tanh(z̄ + β'f_ij), β treinado nos meses
# anteriores sem z_hist. Com β = 0 é exatamente o LW de correlação constante,
# então "LW corr. constante" -> "alvo setor" -> "alvo setor + notícias" é uma
# família aninhada. (Primeira versão incluía z_hist no alvo: δ ≈ 1, Σ̂ ≈ a
# previsão da regressão, mal condicionada -- a GMV sem restrição chegou a 35%
# de volatilidade, placebo incluso.)
TARGETS = {
    "alvo setor": h1.SECTOR,
    "alvo notícias": h1.NEWS,
    "alvo setor + notícias": [*h1.SECTOR, *h1.NEWS],
}
PLACEBO_OF = "alvo setor + notícias"
# Seção 10 do método (fixada antes de rodar): notícias na correlação RESIDUAL ao
# fator de mercado. Σ̂ = σ²_m b b' + shrink(E -> D^½ R_prior D^½).
RES_TARGETS = {
    "fator + resíduo setor": h1.SECTOR,
    "fator + resíduo notícias": h1.NEWS,
    "fator + resíduo setor + notícias": [*h1.SECTOR, *h1.NEWS],
}
RES_PLACEBO_OF = "fator + resíduo setor + notícias"
NEWS_FEATURES = [f for f in h1.NEWS]
ESTIMATORS = {
    "amostral": est.sample_cov,
    "LW identidade": est.lw_identity,
    "LW fator único": est.lw_single_factor,
    "LW corr. constante": est.lw_const_corr,
    "LW não linear": est.lw_nonlinear,
    "RMT": est.rmt_clip,
}
REFERENCES = ("fator + resíduo setor", "LW fator único", "LW corr. constante")


def out_dir(cfg: Config):
    return cfg.marts / "backtest"


class _Trainer:
    """MQO empilhado nos meses < t, y e x centrados em cada mês.

    X'X e X'y de cada mês são calculados uma vez; o ajuste em t é um solve
    sobre a soma acumulada dos meses anteriores (sem varrer o painel de novo).
    """

    def __init__(self, panel: dict, models: dict[str, list[str]]):
        dates = panel["date"]
        order = np.argsort(dates, kind="stable")
        ds = dates[order]
        self.ud, starts = np.unique(ds, return_index=True)
        bounds = list(starts) + [len(ds)]
        self.cum = {}
        for name, regs in models.items():
            k = len(regs)
            XtX, Xty = np.zeros((len(self.ud), k, k)), np.zeros((len(self.ud), k))
            for i in range(len(self.ud)):
                rows = order[bounds[i]:bounds[i + 1]]
                X = np.column_stack([panel[r][rows] for r in regs])
                y = panel["z_fut"][rows]
                X = X - X.mean(axis=0)
                y = y - y.mean()
                XtX[i], Xty[i] = X.T @ X, X.T @ y
            self.cum[name] = (np.cumsum(XtX, axis=0), np.cumsum(Xty, axis=0))

    def coef(self, name: str, before) -> np.ndarray:
        return self.coef_first(name, int(np.searchsorted(self.ud, before)))  # meses estritamente antes

    def coef_first(self, name: str, k: int) -> np.ndarray:
        """Coeficientes treinados nos k primeiros meses do painel."""
        XtX, Xty = self.cum[name]
        A = XtX[k - 1]
        return np.linalg.solve(A + 1e-9 * np.eye(len(A)), Xty[k - 1])


def _target(b: np.ndarray, regs: list[str], z_hist: np.ndarray, F: dict[str, np.ndarray]) -> np.ndarray:
    """Matriz de correlação-alvo p x p prevista pela regressão."""
    p = len(z_hist)
    iu = np.triu_indices(p, 1)
    cols = []
    for r in regs:
        v = z_hist[iu] if r == "z_hist" else F[r][iu]
        cols.append(v)
    X = np.column_stack(cols)
    zh = z_hist[iu]
    ok = np.isfinite(zh)
    level = zh[ok].mean()
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
    zhat = level + (X - X.mean(axis=0)) @ b
    R = np.eye(p)
    R[iu] = np.tanh(zhat)
    R.T[iu] = R[iu]
    return est.nearest_corr(R)


def _features_for_regression(F: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mesmas transformações do painel da H1 (log1p, menos exposição e setor)."""
    return {f: (m if f == "exposure" or f in features.BINARY else np.log1p(m)) for f, m in F.items()}


def _hold(w: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retornos diários da carteira comprada e mantida; pesos derivados no fim."""
    R = np.nan_to_num(X_out)
    V = np.cumprod(1 + R, axis=0) * w              # valor por ativo ao fim de cada dia
    tot = V.sum(axis=1)
    prev = np.concatenate([[w.sum()], tot[:-1]])
    r = tot / prev - 1
    w_end = V[-1] / tot[-1] if len(V) else w
    return r, w_end


def run(cfg: Config, n: int = 100, news_days: int = 365, half_life_days: float = 90.0,
        seed: int = 7) -> dict:
    t0 = time.time()
    panel = h1.build_panel(cfg, n=n, news_days=news_days, half_life_days=half_life_days)
    panel_res = h1.build_panel(cfg, n=n, news_days=news_days, half_life_days=half_life_days, residual=True)
    con = query.connect(cfg)
    dates = [d for (d,) in con.execute(
        "SELECT DISTINCT rebalance_date FROM news.universe WHERE n_target = ? ORDER BY 1", [n]).fetchall()]
    end_of_data = con.execute("SELECT max(date) FROM news.trading_calendar").fetchone()[0]
    rng = np.random.default_rng(seed)
    trainer = _Trainer(panel, TARGETS)
    trainer_res = _Trainer(panel_res, RES_TARGETS)
    res_fixed = {"fator + resíduo diagonal": "identity", "fator + resíduo corr. constante": "cc"}

    names = ["1/N"] + [f"{e} {k}" for e in [*ESTIMATORS, *TARGETS, PLACEBO_OF + " placebo", *res_fixed,
                                            *RES_TARGETS, RES_PLACEBO_OF + " placebo"]
                       for k in ("GMV", "GMV long-only")]
    daily = {k: [] for k in names}
    day_index: list = []
    turnover = {k: [] for k in names}
    deltas = {k: [] for k in [*TARGETS, PLACEBO_OF + " placebo", "fator + resíduo corr. constante",
                              *RES_TARGETS, RES_PLACEBO_OF + " placebo"]}
    prev_end: dict[str, dict[str, float]] = {}
    corr_loss: dict[str, list[float]] = {}
    cond: dict[str, list[float]] = {}

    for k in range(MIN_TRAIN, len(dates)):
        d = dates[k]
        tk = [r[0] for r in con.execute(
            "SELECT ticker FROM news.universe WHERE n_target = ? AND rebalance_date = ? ORDER BY rank", [n, d]).fetchall()]
        p = len(tk)
        w_in = data.trading_window(con, d, before=252)
        nxt = dates[k + 1] if k + 1 < len(dates) else None
        f_end = (con.execute("SELECT max(date) FROM news.trading_calendar WHERE date < ?", [nxt]).fetchone()[0]
                 if nxt else end_of_data)
        X_in = data.returns_matrix(con, tk, *w_in)
        X_out = data.returns_matrix(con, tk, d, f_end)
        day_index += [x for (x,) in con.execute(
            "SELECT date FROM news.trading_calendar WHERE date BETWEEN ? AND ? ORDER BY date", [d, f_end]).fetchall()]

        Ch, _ = data.pairwise_corr(X_in, min_obs=126)
        z_hist = np.arctanh(np.clip(Ch, -0.999, 0.999))
        F = _features_for_regression(features.pair_features(con, cfg, d, tk, news_days, half_life_days))
        perm = rng.permutation(p)
        F_placebo = {f: (m[np.ix_(perm, perm)] if f in NEWS_FEATURES else m) for f, m in F.items()}

        sigmas: dict[str, np.ndarray] = {name: f(X_in) for name, f in ESTIMATORS.items()}
        for name, regs in TARGETS.items():
            b = trainer.coef(name, d)
            sigmas[name], dl = est.shrink_to_corr(X_in, _target(b, regs, z_hist, F), return_delta=True)
            deltas[name].append(dl)
            if name == PLACEBO_OF:
                sigmas[PLACEBO_OF + " placebo"], dl = est.shrink_to_corr(
                    X_in, _target(b, regs, z_hist, F_placebo), return_delta=True)
                deltas[PLACEBO_OF + " placebo"].append(dl)

        # família residual (seção 10)
        m_in = data.market_factor(X_in)
        a_in, b_in = data.market_betas(X_in, m_in)
        Ce, _ = data.pairwise_corr(data.residuals(X_in, b_in, m_in, a_in), min_obs=126)
        z_res = np.arctanh(np.clip(Ce, -0.999, 0.999))
        for name, prior in res_fixed.items():
            sigmas[name], dl = est.factor_residual(X_in, prior, return_delta=True)
            if name in deltas:
                deltas[name].append(dl)
        for name, regs in RES_TARGETS.items():
            b = trainer_res.coef(name, d)
            sigmas[name], dl = est.factor_residual(X_in, _target(b, regs, z_res, F), return_delta=True)
            deltas[name].append(dl)
            if name == RES_PLACEBO_OF:
                sigmas[name + " placebo"], dl = est.factor_residual(
                    X_in, _target(b, regs, z_res, F_placebo), return_delta=True)
                deltas[name + " placebo"].append(dl)

        # perda estatística: erro quadrático médio da correlação estimada contra a
        # realizada no mês (pares com >= 15 dias em comum)
        Cf, _ = data.pairwise_corr(X_out, min_obs=15)
        iu = np.triu_indices(p, 1)
        okf = np.isfinite(Cf[iu])
        for name, S in sigmas.items():
            sd = np.sqrt(np.diag(S))
            C = S / np.outer(sd, sd)
            corr_loss.setdefault(name, []).append(float(np.mean((C[iu][okf] - Cf[iu][okf]) ** 2)))
            # estabilidade (López de Prado): número de condição da correlação
            ev = np.linalg.eigvalsh(C)
            cond.setdefault(name, []).append(float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"))

        weights = {"1/N": np.full(p, 1 / p)}
        for name, S in sigmas.items():
            weights[f"{name} GMV"] = optimize.gmv(S)
            weights[f"{name} GMV long-only"] = optimize.gmv_long_only(S)

        for name, w in weights.items():
            r, w_end = _hold(w, X_out)
            old = prev_end.get(name, {})
            new = dict(zip(tk, w))
            to = sum(abs(new.get(t, 0.0) - old.get(t, 0.0)) for t in set(new) | set(old))
            turnover[name].append(to if old else np.nan)
            r = r.copy()
            if old:
                r[0] -= COST * to
            daily[name].append(r)
            prev_end[name] = dict(zip(tk, w_end))
        if (k - MIN_TRAIN + 1) % 24 == 0:
            print(f"[backtest] {d} ({k - MIN_TRAIN + 1}/{len(dates) - MIN_TRAIN}, {time.time() - t0:.0f}s)")
    con.close()

    R = {name: np.concatenate(v) for name, v in daily.items()}
    res = {"params": {"n": n, "news_days": news_days, "half_life_days": half_life_days, "cost_bps": COST * 1e4,
                      "first": str(dates[MIN_TRAIN]), "last": str(day_index[-1]), "months": len(dates) - MIN_TRAIN},
           "delta_mean": {k: round(float(np.mean(v)), 3) for k, v in deltas.items()},
           "metrics": {name: metrics(r, turnover[name]) for name, r in R.items()},
           "corr_loss": {}, "condition": {}, "tests": {}}
    for name, v in cond.items():
        v = np.array(v)
        res["condition"][name] = {"median": round(float(np.median(v)), 1), "p90": round(float(np.quantile(v, 0.9)), 1),
                                  "max": round(float(v.max()), 1)}
    base_loss = np.array(corr_loss["LW corr. constante"])
    for name, v in corr_loss.items():
        v = np.array(v)
        res["corr_loss"][name] = {"mse": round(float(v.mean()), 5),
                                  "vs_lw_cc": round(float((v - base_loss).mean() / base_loss.mean()), 4),
                                  "t": round(h1._nw_t(base_loss - v), 2) if name != "LW corr. constante" else None}
    for ref in REFERENCES:
        for kind in ("GMV", "GMV long-only"):
            base = f"{ref} {kind}"
            res["tests"][base] = {name: variance_test(R[name], R[base], rng)
                                  for name in R if name.endswith(" " + kind) and name != base}
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    tag = f"n={n}_w={news_days}_h={half_life_days:g}"
    pq.write_table(pa.table({"date": pa.array(day_index), **{k: v for k, v in R.items()}}),
                   d / f"daily_{tag}.parquet", compression="zstd")
    (d / f"results_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    report(res)
    print(f"[backtest] {d / f'results_{tag}.json'} ({time.time() - t0:.0f}s)")
    return res


def metrics(r: np.ndarray, to: list[float]) -> dict:
    vol = r.std(ddof=1) * np.sqrt(252)
    mean = r.mean() * 252
    wealth = np.cumprod(1 + r)
    dd = (wealth / np.maximum.accumulate(wealth) - 1).min()
    return {"vol": round(float(vol), 4), "ret": round(float(mean), 4),
            "sharpe": round(float(mean / vol), 3) if vol > 0 else None,
            "max_dd": round(float(dd), 4), "turnover": round(float(np.nanmean(to)), 3)}


def variance_test(a: np.ndarray, b: np.ndarray, rng, block: int = BLOCK, reps: int = BOOT) -> dict:
    """log(σ²_a/σ²_b) com IC por bootstrap circular em blocos (pares de dias
    reamostrados juntos). Aproxima Ledoit e Wolf (2011), sem studentização."""
    n = len(a)
    stat = np.log(a.var(ddof=1) / b.var(ddof=1))
    nb = int(np.ceil(n / block))
    boots = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        boots[i] = np.log(a[idx].var(ddof=1) / b[idx].var(ddof=1))
    p = float(np.mean(np.abs(boots - stat) >= abs(stat)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"var_ratio": round(float(np.exp(stat)), 4), "ci95": [round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)],
            "p": round(p, 4)}


def report(res: dict) -> None:
    pm = res["params"]
    print(f"\nbacktest n={pm['n']}, {pm['months']} meses ({pm['first']} a {pm['last']}), custo {pm['cost_bps']:g} bps")
    print(f"δ médio dos alvos-prior: {res['delta_mean']}")
    print(f"\n  {'estratégia':42} {'vol':>7} {'ret':>7} {'Sharpe':>7} {'maxDD':>7} {'giro':>6}")
    for kind in ("GMV", "GMV long-only"):
        rows = [(n, m) for n, m in res["metrics"].items()
                if n == "1/N" or (n.endswith(" " + kind) and (kind.endswith("long-only") == n.endswith("long-only")))]
        for name, m in sorted(rows, key=lambda x: x[1]["vol"]):
            print(f"  {name:42} {m['vol']:7.2%} {m['ret']:7.2%} {m['sharpe']:7.3f} {m['max_dd']:7.1%} {m['turnover']:6.2f}")
        print()
    print("número de condição da correlação estimada, λmax/λmin (López de Prado): mediana, p90, máximo")
    for name, c in sorted(res["condition"].items(), key=lambda x: x[1]["median"]):
        print(f"  {name:42} {c['median']:10,.1f} {c['p90']:10,.1f} {c['max']:12,.1f}")
    print()
    print("erro quadrático da correlação estimada contra a realizada no mês (t > 0: melhor que LW corr. constante)")
    for name, c in sorted(res["corr_loss"].items(), key=lambda x: x[1]["mse"]):
        print(f"  {name:42} {c['mse']:.5f}  {c['vs_lw_cc']:+.2%}  t={c['t']}")
    print()
    for base, tests in res["tests"].items():
        print(f"razão de variâncias contra {base} (p bootstrap em blocos):")
        for name, t in sorted(tests.items(), key=lambda x: x[1]["var_ratio"]):
            print(f"  {name:42} {t['var_ratio']:.3f}  IC95 {t['ci95']}  p={t['p']:.3f}")
        print()
