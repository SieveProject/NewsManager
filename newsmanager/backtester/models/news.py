"""Variância mínima com Σ̂ = fator de mercado + resíduo encolhido para um
alvo de correlação previsto por setor e/ou notícias (seção 10 de
docs/metodo_covariancia.md):

    Σ̂ = σ²_m b b' + shrink(E -> D_e^½ R D_e^½),   R_ij = tanh(z̄ + β'f_ij)

f_ij são os atributos de par de newsmanager/portfolio/features.py (só notícias
< t). β vem de um MQO empilhado da correlação residual REALIZADA em meses
passados contra f_ij (painel da H1 residual, universo news.universe n),
usando só os meses cujo período de carteira terminou antes de t. Com menos de
`min_train` meses, β = 0 (alvo = correlação residual constante).

    features = "setor" | "noticias" | "setor+noticias"
    placebo  = true: atributos de notícias embaralhados entre as ações em cada
               data (β treinado nos dados reais), o controle da seção 9
"""

from __future__ import annotations

import bisect
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ...config import ConfigError
from ...portfolio import data, estimators as est, features, h1
from ...portfolio.backtest import _features_for_regression, _target, _Trainer
from ...portfolio.optimize import min_variance
from ..plugin import Context, Env, Model
from .basic import describe

SETS = {"setor": h1.SECTOR, "noticias": h1.NEWS, "setor+noticias": [*h1.SECTOR, *h1.NEWS]}


def _panel(env: Env, n: int, news_days: int, half_life: float) -> dict:
    """Painel residual da H1, em cache enquanto grafo, universo e setores não mudarem."""
    cfg = env.cfg
    deps = [cfg.curated / "universe" / f"n={n}.parquet", cfg.curated / "sectors"]
    deps += sorted((cfg.curated / "graph").glob("v=*/events.parquet"))
    stamp = []
    for d in deps:
        files = sorted(d.rglob("*.parquet")) if d.is_dir() else [d]
        stamp += [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.exists()]
    key = hashlib.sha256(json.dumps([n, news_days, half_life, stamp]).encode()).hexdigest()[:12]
    path = cfg.marts / "backtests" / "_cache" / f"panel_res_n={n}_w={news_days}_h={half_life:g}_{key}.parquet"
    if path.exists():
        t = pq.read_table(path)
        return {c: (np.array(t[c].to_pylist(), dtype=object) if c in ("date", "a", "b") else t[c].to_numpy())
                for c in t.column_names}
    panel = h1.build_panel(cfg, n=n, news_days=news_days, half_life_days=half_life, residual=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: (pa.array(list(v)) if v.dtype == object else v) for k, v in panel.items()}),
                   path, compression="zstd")
    return panel


class NewsFactorResidual(Model):
    def __init__(self, features: str = "setor+noticias", placebo: bool = False, news_days: int = 365,
                 half_life: float = 90.0, min_train: int = h1.MIN_TRAIN, panel_n: int | None = None,
                 seed: int = 7):
        if features not in SETS:
            raise ValueError(f"features {features!r}: use um de {list(SETS)}")
        if placebo and features == "setor":
            raise ValueError("placebo embaralha as notícias: não faz sentido com features='setor'")
        self.regs = SETS[features]
        self.placebo, self.news_days, self.half_life = placebo, news_days, half_life
        self.min_train, self.panel_n = min_train, panel_n
        self.rng = np.random.default_rng(seed)

    def setup(self, env: Env) -> None:
        n = self.panel_n or env.universe_n
        if n is None:
            raise ConfigError("NewsFactorResidual precisa de universe = {n = N} ou de model.params.panel_n")
        self.env = env
        panel = _panel(env, n, self.news_days, self.half_life)
        self.trainer = _Trainer(panel, {"m": self.regs})
        self.months = list(self.trainer.ud)

    def weights(self, ctx: Context) -> np.ndarray:
        X = ctx.returns
        m = data.market_factor(X)
        a, b = data.market_betas(X, m)
        Ce, _ = data.pairwise_corr(data.residuals(X, b, m, a), min_obs=len(X) // 2)
        z = np.arctanh(np.clip(Ce, -0.999, 0.999))
        F = _features_for_regression(features.pair_features(
            self.env.con, self.env.cfg, ctx.date, ctx.tickers, self.news_days, self.half_life))
        if self.placebo:
            perm = self.rng.permutation(ctx.p)
            F = {f: (M[np.ix_(perm, perm)] if f in h1.NEWS else M) for f, M in F.items()}
        # meses do painel cujo período de carteira [m_k, m_{k+1}) já acabou antes de t
        k = max(bisect.bisect_right(self.months, ctx.date) - 1, 0)
        beta = (self.trainer.coef_first("m", k) if k >= self.min_train else np.zeros(len(self.regs)))
        S, delta = est.factor_residual(X, _target(beta, self.regs, z, F), return_delta=True)
        w = min_variance(S, ctx.min_weight, ctx.max_weight)
        describe(ctx, S, w)
        ctx.info.update({"delta": delta, "trained_months": k})
        for r, v in zip(self.regs, beta):
            ctx.info[f"beta_{r}"] = float(v)
        return w
