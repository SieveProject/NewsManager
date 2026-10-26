"""Estimadores de Σ que os plugins de otimização usam (parâmetro `cov`).

Estatísticos (só retornos da janela):
  amostral, LW identidade, LW fator único, LW corr. constante, LW não linear,
  RMT, fator + resíduo diagonal, fator + resíduo corr. constante

Com setor/notícias (seção 10 de docs/metodo_covariancia.md):
  fator + resíduo setor, fator + resíduo notícias, fator + resíduo setor + notícias

Black-Litterman (seção 16): fator + resíduo setor + BL notícias

    Σ̂ = σ²_m b b' + shrink(E -> D_e^½ R D_e^½),   R_ij = tanh(z̄ + β'f_ij)

f_ij são os atributos de par de newsmanager/portfolio/features.py (só notícias
< t). β vem de um MQO empilhado da correlação residual REALIZADA em meses
passados contra f_ij (painel da H1 residual, universo news.universe n), usando
só os meses cujo período de carteira terminou antes de t. Com menos de
`min_train` meses, β = 0 (alvo = correlação residual constante).

`cov_params` (só para os de notícias): placebo (notícias embaralhadas entre as
ações em cada data, o controle da seção 9), news_days, half_life, min_train,
panel_n, seed.
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
from ...portfolio.backtest import _features_for_regression, _target, _Trainer, _zhat
from ..plugin import Context, Env

STATISTICAL = {
    "amostral": est.sample_cov,
    "LW identidade": est.lw_identity,
    "LW fator único": est.lw_single_factor,
    "LW corr. constante": est.lw_const_corr,
    "LW não linear": est.lw_nonlinear,
    "RMT": est.rmt_clip,
    "fator + resíduo diagonal": lambda X: est.factor_residual(X, "identity"),
    "fator + resíduo corr. constante": lambda X: est.factor_residual(X, "cc"),
}
NEWS = {
    "fator + resíduo setor": h1.SECTOR,
    "fator + resíduo notícias": h1.NEWS,
    "fator + resíduo setor + notícias": [*h1.SECTOR, *h1.NEWS],
}
BL = "fator + resíduo setor + BL notícias"
NAMES = (*STATISTICAL, *NEWS, BL)


class Statistical:
    def __init__(self, name: str):
        self.name = name

    def setup(self, env: Env) -> None:
        pass

    def __call__(self, ctx: Context) -> np.ndarray:
        out = STATISTICAL[self.name](ctx.returns)
        return out[0] if isinstance(out, tuple) else out


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


class NewsResidual:
    def __init__(self, name: str, placebo: bool = False, news_days: int = 365, half_life: float = 90.0,
                 min_train: int = h1.MIN_TRAIN, panel_n: int | None = None, seed: int = 7):
        if placebo and name == "fator + resíduo setor":
            raise ValueError("placebo embaralha as notícias: não faz sentido com o alvo só de setor")
        self.name, self.regs = name, NEWS[name]
        self.placebo, self.news_days, self.half_life = placebo, news_days, half_life
        self.min_train, self.panel_n = min_train, panel_n
        self.rng = np.random.default_rng(seed)

    def setup(self, env: Env) -> None:
        n = self.panel_n or env.universe_n
        if n is None:
            raise ConfigError(f"cov {self.name!r} precisa de universe = {{n = N}} ou de cov_params.panel_n")
        self.env = env
        self.trainer = _Trainer(_panel(env, n, self.news_days, self.half_life), {"m": self.regs})
        self.months = list(self.trainer.ud)

    def features(self, ctx: Context, news_days: int | None = None) -> dict[str, np.ndarray]:
        """Atributos de par em ctx.date (notícias dos `news_days` dias anteriores)."""
        return _features_for_regression(features.pair_features(
            self.env.con, self.env.cfg, ctx.date, ctx.tickers, news_days or self.news_days, self.half_life))

    def __call__(self, ctx: Context) -> np.ndarray:
        return self.estimate(ctx, self.features(ctx))

    def estimate(self, ctx: Context, F: dict[str, np.ndarray]) -> np.ndarray:
        X = ctx.returns
        m = data.market_factor(X)
        a, b = data.market_betas(X, m)
        Ce, _ = data.pairwise_corr(data.residuals(X, b, m, a), min_obs=len(X) // 2)
        z = np.arctanh(np.clip(Ce, -0.999, 0.999))
        if self.placebo:
            perm = self.rng.permutation(ctx.p)
            F = {f: (M[np.ix_(perm, perm)] if f in h1.NEWS else M) for f, M in F.items()}
        # meses do painel cujo período de carteira [m_k, m_{k+1}) já acabou antes de t
        k = max(bisect.bisect_right(self.months, ctx.date) - 1, 0)
        beta = (self.trainer.coef_first("m", k) if k >= self.min_train else np.zeros(len(self.regs)))
        S, delta = est.factor_residual(X, _target(beta, self.regs, z, F), return_delta=True)
        ctx.info.update({"delta": delta, "trained_months": k})
        for r, v in zip(self.regs, beta):
            ctx.info[f"beta_{r}"] = float(v)
        return S


class _Omega:
    """Erro das visões por classe de evidência (seção 16), treinado no painel
    residual nos k primeiros meses, com o β desses mesmos meses.

    Classe do par: -1 sem notícia (sem visão); 0 com notícia e cooccur = 0;
    1-4 quartis de cooccur > 0 nos meses de treino.
      ω_c = média(ê² | c) - média(1/(n_fut - 3) | c)
    ê = resíduo do MQO com y e x centrados no mês (o mesmo de _Trainer)."""

    def __init__(self, panel: dict, regs: list[str], months: list):
        pos = {d: k for k, d in enumerate(months)}
        self.month = np.array([pos[d] for d in panel["date"]])
        nm = len(months)
        cnt = np.bincount(self.month, minlength=nm)

        def center(v):
            return v - (np.bincount(self.month, v, nm) / cnt)[self.month]

        self.y = center(panel["z_fut"])
        self.X = np.column_stack([center(panel[r].astype(float)) for r in regs])
        self.news = np.any([panel[f] != 0 for f in h1.NEWS], axis=0)
        self.cooc = panel["cooccur"]
        self.noise = 1 / (panel["n_fut"] - 3)
        self.cache: dict[int, tuple] = {}

    def __call__(self, k: int, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cortes dos quartis, ω_c bruto das classes 0-4, nº de pares de treino por classe)."""
        if k not in self.cache:
            tr = self.month < k
            e2 = (self.y[tr] - self.X[tr] @ beta) ** 2
            cooc = self.cooc[tr]
            cuts = np.quantile(cooc[cooc > 0], [0.25, 0.5, 0.75])
            c = classes(self.news[tr], cooc, cuts)
            n = np.bincount(c[c >= 0], minlength=5)
            om = np.array([e2[c == g].mean() - self.noise[tr][c == g].mean() if n[g] else np.nan
                           for g in range(5)])
            self.cache[k] = (cuts, om, n)
        return self.cache[k]


def classes(news: np.ndarray, cooc: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.where(~news, -1, np.where(cooc > 0, 1 + np.searchsorted(cuts, cooc, side="left"), 0))


class NewsBL(NewsResidual):
    """Σ̂ = σ²_m b b' + BL(Σ̂_e do "fator + resíduo setor", visões das notícias), seção 16.

    `omega`: "classe" (ω por classe de evidência) ou "homogeneo" (média das
    classes ponderada pelos pares de treino, a ablação)."""

    def __init__(self, name: str = BL, placebo: bool = False, omega: str = "classe", **kw):
        if omega not in ("classe", "homogeneo"):
            raise ValueError(f"omega {omega!r}: use 'classe' ou 'homogeneo'")
        NewsResidual.__init__(self, "fator + resíduo setor + notícias", placebo=placebo, **kw)
        self.name, self.omega = name, omega

    def setup(self, env: Env) -> None:
        n = self.panel_n or env.universe_n
        if n is None:
            raise ConfigError(f"cov {self.name!r} precisa de universe = {{n = N}} ou de cov_params.panel_n")
        self.env = env
        panel = _panel(env, n, self.news_days, self.half_life)
        self.trainer = _Trainer(panel, {"m": self.regs, "setor": h1.SECTOR})
        self.months = list(self.trainer.ud)
        self.errors = _Omega(panel, self.regs, self.months)

    def estimate(self, ctx: Context, F: dict[str, np.ndarray]) -> np.ndarray:
        X = ctx.returns
        T = len(X)
        m = data.market_factor(X)
        a, b = data.market_betas(X, m)
        Ce, _ = data.pairwise_corr(data.residuals(X, b, m, a), min_obs=T // 2)
        z = np.arctanh(np.clip(Ce, -0.999, 0.999))
        if self.placebo:
            perm = self.rng.permutation(ctx.p)
            F = {f: (M[np.ix_(perm, perm)] if f in h1.NEWS else M) for f, M in F.items()}
        k = max(bisect.bisect_right(self.months, ctx.date) - 1, 0)
        trained = k >= self.min_train
        b_set = self.trainer.coef_first("setor", k) if trained else np.zeros(len(h1.SECTOR))
        S, delta = est.factor_residual(X, _target(b_set, h1.SECTOR, z, F), return_delta=True)
        common = m.var(ddof=1) * np.outer(b, b)
        info = {"n_views": 0, "cg_iter": 0, "psd_projected": 0.0}
        if trained:
            beta = self.trainer.coef_first("m", k)
            cuts, om, n_tr = self.errors(k, beta)
            om = np.maximum(om, 1 / (T - 3))
            if self.omega == "homogeneo":
                ok = n_tr > 0
                om = np.where(ok, (om[ok] * n_tr[ok]).sum() / n_tr[ok].sum(), np.nan)
            Se0 = S - common
            iu = np.triu_indices(ctx.p, 1)
            news = np.any([F[f][iu] != 0 for f in h1.NEWS], axis=0)
            c = classes(news, F["cooccur"][iu], cuts)
            view = c >= 0
            view[view] = np.isfinite(om[c[view]])
            i, j = iu[0][view], iu[1][view]
            r = np.tanh(_zhat(beta, self.regs, z, F)[view])
            ss = Se0[i, i] * Se0[j, j]
            Se, info = est.black_litterman_cov(Se0, i, j, r * np.sqrt(ss), (1 - r * r) ** 2 * ss * om[c[view]], T)
            S = common + Se
            for g in range(5):
                ctx.info[f"omega_{g}"] = float(om[g])
                ctx.info[f"views_{g}"] = int((c[view] == g).sum())
        ctx.info.update({"delta": delta, "trained_months": k, **info})
        return S


def make(name: str, params: dict | None = None):
    params = params or {}
    if name in STATISTICAL:
        if params:
            raise ValueError(f"cov {name!r} não aceita cov_params {sorted(params)}")
        return Statistical(name)
    if name in NEWS:
        return NewsResidual(name, **params)
    if name == BL:
        return NewsBL(name, **params)
    raise ValueError(f"cov {name!r}: use um de {list(NAMES)}")
