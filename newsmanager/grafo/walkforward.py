"""Avaliação walk-forward da seção 18.5 (G5, G7, G10-G13 e robustez).

    python -m newsmanager.grafo dev        # grades do dev (2013-2018) -> logs/secao18_escolhas.json
    python -m newsmanager.grafo holdout    # 2019-2023, uma vez, com as escolhas congeladas
    python -m newsmanager.grafo placebos   # P1-P6 no M2 escolhido (holdout)
    python -m newsmanager.grafo robustez   # M2 contra B5, uma dimensão por vez

Em cada rebalanceamento t (1º pregão do mês, ou da semana/trimestre na
robustez): U_t = S&P 500 point-in-time com cobertura; X = 252 pregões antes
de t; carteira mantida até o pregão anterior ao próximo t, com pesos
derivando (engine.Book: giro contra os pesos derivados, custo sobre o giro).

O laço começa em 2011-01 para treinar o que é treinado (β do setor no R0, θ
do M6) só com meses cujo período de carteira terminou antes de t; carteiras
e perdas só no período avaliado.

Saída em data/marts/grafo/wf/<tag>/: diario.parquet (data, estimador,
carteira, retorno bruto, giro no dia de troca, σ ex ante), mensal.parquet
(perdas e diagnósticos por mês e estimador), info.json.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..backtester.engine import Book
from ..config import Config
from ..portfolio import data, estimators as est, optimize
from ..portfolio.backtest import _target
from . import dados, estimadores as ge, rede

DEV = ("2013-01-01", "2018-12-31")
HOLDOUT = ("2019-01-01", "2023-12-31")
TRAIN_FROM = "2011-01-01"
MIN_TRAIN = 24
SEED = 7
CAP = 0.05
M1_AGENTS = ("FEDERAL_RESERVE", "CRUDE_OIL", "CHINA", "US_DOLLAR", "GOLD_COMMODITY")
M1_PROXIES = ("-DGS2", "USO", "FXI", "UUP", "GLD")
KIND_LAYERS = {"empresa": ("company",), "governo": ("central_bank", "government"), "commodity": ("commodity",),
               "pais": ("country",), "moeda": ("currency",), "macro": ("macro",), "setor": ("sector", "index")}

# grades do dev (seção 18.5; as marcadas com (*) foram fixadas no commit da implementação)
GRID = {
    "M2": [{"tau": tau, "norm": nm} for tau in (30, 90, 180, 365) for nm in rede.NORMS],
    "B8": [{"C": c} for c in (0.25, 0.5, 1.0)],
    "M4": [{"lam0": l, "gamma": g} for l in (0.02, 0.05, 0.1, 0.2) for g in (0.0, 0.5, 1.0, 2.0)],
    "M5": [{"q": q} for q in (3, 5, 10)],
    "M1": [{"kappa": k, "c": c} for k in (0.0, 0.5, 1.0, 2.0) for c in (0.1, 1.0, 10.0)],         # (*)
    "M2W": [{"alpha": a, "beta": b} for a in (0.25, 0.5, 1.0) for b in (0.5, 1.0, 2.0)],          # (*)
    "M8b": [{"lam": x} for x in (0.25, 0.5, 0.75)],
}
DEFAULTS = {"M2": {"tau": 90, "norm": "idf"}}


@dataclass
class Spec:
    universo: str = "sp500"
    insample: int = 252
    rebalance: str = "M"            # M, W ou Q
    tau: float = 90.0               # meia-vida padrão dos snapshots (M2 pode pedir outra)
    window: int = 365
    agents: str = "todos"           # ou "sem_ticker"
    sem_wrap: bool = False          # exclui artigos com >= 5 tickers distintos
    retornos: str = "simples"       # ou "log" (só na estimação)
    winsor: bool = False            # 1%/99% por ação, na estimação
    cost_bps: float = 10.0
    kernel: str = "gram"            # gram, difusao, higham (M2)

    def label(self) -> str:
        base = Spec()
        diff = {k: v for k, v in asdict(self).items() if getattr(base, k) != v}
        return "padrao" if not diff else "_".join(f"{k}={v}" for k, v in diff.items())


def wf_dir(cfg: Config, tag: str) -> Path:
    return dados.out_dir(cfg) / "wf" / tag


# ------------------------------------------------------------------ mês

@dataclass
class Mes:
    t: object
    i: int
    U: list
    X: np.ndarray                  # retornos para estimar (log/winsor conforme a spec)
    X_raw: np.ndarray
    X_out: np.ndarray
    spy_in: np.ndarray
    sec: dict
    size: np.ndarray
    mom: np.ndarray
    snaps: dict                    # (tau, agents) -> Snapshot
    ev: object
    spec: Spec
    F: np.ndarray | None = None    # fatores do M1 na janela
    cache: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.U)

    def snap(self, tau: float | None = None) -> rede.Snapshot:
        key = (tau or self.spec.tau, self.spec.agents)
        if key not in self.snaps:
            self.snaps[key] = self.ev.snapshot(self.t, self.U, tau=key[0], window_days=self.spec.window,
                                               agents=self.spec.agents, exclude_wrap=self.spec.sem_wrap)
        return self.snaps[key]

    def cg(self, tau=None, norm="idf", kind="sinal") -> np.ndarray:
        key = ("cg", tau or self.spec.tau, norm, kind, self.spec.kernel)
        if key not in self.cache:
            s = self.snap(tau)
            C = s.projection(norm, kind)
            if self.spec.kernel != "gram":
                W = s.direct("raw")
                C = rede.diffusion(W, 1.0) if self.spec.kernel == "difusao" else rede.higham(W, 0.5)
            self.cache[key] = C
        return self.cache[key]


def _resid_corr(X: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, float]:
    """Correlação residual realizada no período (b da janela) e z̄ in-sample."""
    m = data.market_factor(X)
    a, b = data.market_betas(X, m)
    Ce, _ = data.pairwise_corr(data.residuals(X, b, m, a), min_obs=len(X) // 2)
    Cf, _ = data.pairwise_corr(data.residuals(X_out, b, data.market_factor(X_out)), min_obs=max(5, int(0.7 * len(X_out))))
    return Cf, Ce


class Trainer:
    """MQO empilhado (y e x centrados por mês) com acumulação sequencial:
    o mês n entra no treino quando o seu período de carteira acaba."""

    def __init__(self, k: int, nonneg: bool = False):
        self.XtX, self.Xty, self.n = np.zeros((k, k)), np.zeros(k), 0
        self.nonneg = nonneg

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[ok] - X[ok].mean(axis=0), y[ok] - y[ok].mean()
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.n += 1

    def coef(self) -> np.ndarray | None:
        if self.n < MIN_TRAIN:
            return None
        A = self.XtX + 1e-9 * np.eye(len(self.XtX))
        if not self.nonneg:
            return np.linalg.solve(A, self.Xty)
        from scipy.optimize import lsq_linear
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(L, self.Xty)
        return lsq_linear(L.T, z, bounds=(0, np.inf)).x


# ------------------------------------------------------------------ estimadores

def _layers(mes: Mes) -> tuple[list[str], list[np.ndarray]]:
    """Camadas do M6: difusão de cada grupo de classe e Gram por tipo de agente."""
    if "layers" in mes.cache:
        return mes.cache["layers"]
    s = mes.snap()
    names, L = [], []
    for g in rede.GROUP_NAMES:
        W = s.expand(s.Wg[g])
        mx = np.abs(W).max()
        names.append(f"dir_{g}")
        L.append(rede.diffusion(W / mx, 1.0) if mx > 0 else np.eye(mes.p))
    B = s.exposure("idf").tocsc()
    for lab, kinds in KIND_LAYERS.items():
        cols = np.flatnonzero(np.isin(s.agent_kind, kinds))
        names.append(f"exp_{lab}")
        L.append(s.expand(rede.cosine_rows(B[:, cols].tocsr()), diag=1.0) if len(cols) else np.eye(mes.p))
    mes.cache["layers"] = (names, L)
    return names, L


def _m1_prior(mes: Mes, B_ols: np.ndarray, kappa: float) -> np.ndarray:
    s = mes.snap()
    K = len(M1_AGENTS) + 1
    P = np.zeros((K, mes.p))
    P[0] = B_ols[0]
    ix = {mes.ev.names[a]: k for k, a in enumerate(s.agents)}
    for k, ag in enumerate(M1_AGENTS, start=1):
        if ag not in ix:
            continue
        col = np.asarray(s.A[:, ix[ag]].todense()).ravel()
        a = np.where(s.st >= 0, col[np.maximum(s.st, 0)], 0.0)
        mx = np.abs(a).max()
        if mx > 0:
            P[k] = kappa * B_ols[k].std() * a / mx
    return P


def estimate(name: str, prm: dict, mes: Mes, state: dict) -> tuple[np.ndarray, dict]:
    """Σ̂ de um estimador (código da tabela 18.5) no mês."""
    X, p = mes.X, mes.p
    info: dict = {}
    if name == "B1":
        return est.sample_cov(X), info
    if name == "B2":
        S, info["delta"] = est.lw_identity(X, True)
        return S, info
    if name == "B3":
        S, info["delta"] = est.lw_const_corr(X, True)
        return S, info
    if name == "B4":
        S, info["delta"] = est.lw_single_factor(X, True)
        return S, info
    if name == "B5":
        S, info["delta"] = ge.block_target(X, mes.sec["ff12"], True)
        return S, info
    if name == "B6":
        return est.lw_nonlinear(X), info
    if name == "B7":
        return ge.observable_factor(X, mes.sec["ff12"], mes.size, mes.mom), info
    if name in ("B8", "M3", "M3plac"):
        key = ("poet", prm.get("C", 0.5))
        if key not in mes.cache:
            mes.cache[key] = ge.poet(X, C=prm.get("C", 0.5))
        S0, i0 = mes.cache[key]
        if name == "B8":
            return S0, dict(i0)
        n_pairs = max(i0["n_pairs"], 1)
        iu = np.triu_indices(p, 1)
        if name == "M3":
            s = mes.snap()
            score = (s.direct("raw") != 0) * 10.0 + mes.cg()
            top = np.argsort(-score[iu], kind="stable")[:n_pairs]
        else:
            top = np.random.default_rng(SEED + mes.i).choice(len(iu[0]), n_pairs, replace=False)
        sup = np.zeros((p, p), bool)
        sup[iu[0][top], iu[1][top]] = True
        sup |= sup.T
        return ge.poet(X, k=i0["k"], support=sup)
    if name in ("B9", "M4"):
        lam0, gamma = prm["lam0"], (0.0 if name == "B9" else prm["gamma"])
        Lam = np.full((p, p), lam0)
        if gamma:
            s = mes.snap()
            w = s.direct("sgnlog")
            mw = np.abs(w).max()
            wt = mes.cg() - np.eye(p) + (w / mw if mw > 0 else 0)
            wt = wt / max(np.abs(wt).max(), 1e-12)
            Lam = lam0 * np.exp(-gamma * np.abs(wt))
        key = ("glasso_state", mes.i)          # partida a quente entre as configurações do mês
        S, gi = ge.factor_glasso(X, Lam, init=state.get(key))
        state[key] = gi["state"]
        info = {"glasso_iter": gi["iter"]}
        if name == "B9":                      # G11: aresta de notícia x correlação parcial != 0
            iu = np.triu_indices(p, 1)
            e = (mes.snap().direct("raw") != 0)[iu]
            z = gi["support"][iu]
            info.update({"cmh_a": int((e & z).sum()), "cmh_b": int((e & ~z).sum()),
                         "cmh_c": int((~e & z).sum()), "cmh_d": int((~e & ~z).sum())})
        return S, info
    if name == "B10":
        return ge.ewma(X, 0.97), info
    if name == "R0":
        b = state["trainer_r0"].coef()
        m = data.market_factor(X)
        a_, b_ = data.market_betas(X, m)
        Ce, _ = data.pairwise_corr(data.residuals(X, b_, m, a_), min_obs=len(X) // 2)
        z = np.arctanh(np.clip(Ce, -0.999, 0.999))
        F = _sector_pairs(mes)
        R = _target(np.zeros(2) if b is None else b, ["same_ff49", "same_sic2"], z, F)
        S, info["delta"] = est.factor_residual(X, R, True)
        return S, info
    if name in ("M2", "M2res", "M2abs", "M2bin"):
        q = {**DEFAULTS["M2"], **prm}
        kind = {"M2abs": "abs", "M2bin": "bin"}.get(name, "sinal")
        C = mes.cg(q["tau"], q["norm"], kind)
        if name == "M2res":
            S, info["delta"] = est.factor_residual(X, C, True)
        else:
            S, info["delta"] = est.shrink_to_corr(X, C, True)
        return S, info
    if name in ("M2dgmv", "M2dqlike"):
        q = {**DEFAULTS["M2"], **prm}
        C = mes.cg(q["tau"], q["norm"])
        Z = est.demeaned(X)
        Smp = Z.T @ Z / len(Z)
        sd = np.sqrt(np.diag(Smp))
        T = C * np.outer(sd, sd)
        np.fill_diagonal(T, np.diag(Smp))
        hist = state.setdefault("dcv_hist", deque(maxlen=12))
        grid = np.round(np.arange(0, 1.0001, 0.05), 2)
        if len(hist) < 3:
            _, d0 = est.shrink_to_corr(X, C, True)
            delta = d0
        else:
            crit = name[3:]
            delta = _choose_delta(hist, grid, crit)
        info["delta"] = float(delta)
        state.setdefault("dcv_pending", {})[name] = (Smp, T, mes)
        return delta * T + (1 - delta) * Smp, info
    if name == "M2W":
        s = mes.snap()
        W = s.direct("raw")
        mx = np.abs(W).max()
        K = rede.diffusion(W / mx, prm["beta"]) if mx > 0 else np.eye(p)
        G = mes.cg() + prm["alpha"] * K
        S, info["delta"] = est.shrink_to_corr(X, ge.corr_of(G), True)
        return S, info
    if name in ("M1", "B7p"):
        if mes.F is None:
            raise RuntimeError("fatores do M1 indisponíveis")
        F = np.column_stack([data.market_factor(X), mes.F])     # mercado (sem penalidade) + agentes
        _, B_ols = ge.ridge_factor(X, F, lam=0.0)
        if name == "B7p":
            S, _ = ge.ridge_factor(X, F, lam=0.0)
            return S, info
        P = _m1_prior(mes, B_ols, prm["kappa"])
        S, _ = ge.ridge_factor(X, F, prior=P, lam=prm["c"])
        return S, info
    if name == "M5":
        return ge.spectral_factor(X, mes.cg(), prm["q"]), info
    if name in ("PCAk", "PCAkq", "PCAk+q"):
        k = ge.bai_ng_k(est.demeaned(X))
        q = prm["q"]
        if name == "PCAk":
            return ge.pca_factor(X, max(k, 1)), {"k": k}
        if name == "PCAk+q":
            return ge.pca_factor(X, max(k, 1) + q), {"k": k}
        return ge.pca_factor(X, max(k, 1), extra=ge.laplacian_vectors(mes.cg(), q)), {"k": k}
    if name.startswith("M6"):
        c = state["trainer_m6"].coef()
        names, L = _layers(mes)
        drop = prm.get("drop")
        m = data.market_factor(X)
        a_, b_ = data.market_betas(X, m)
        Ce, _ = data.pairwise_corr(data.residuals(X, b_, m, a_), min_obs=len(X) // 2)
        iu = np.triu_indices(p, 1)
        rbar = np.nanmean(Ce[iu])
        R = np.full((p, p), rbar)
        if c is not None:
            for k, (nm, Lk) in enumerate(zip(names, L)):
                if nm != drop and c[k] > 0:
                    R += c[k] * (Lk - Lk[iu].mean())
            info.update({f"theta_{nm}": float(v) for nm, v in zip(names, c)})
        np.fill_diagonal(R, 1.0)
        S, info["delta"] = est.factor_residual(X, est.nearest_corr(R), True)
        return S, info
    if name in ("M8blk", "FF12blk", "CORRblk"):
        lab = {"M8blk": lambda: _communities(mes), "FF12blk": lambda: mes.sec["ff12"],
               "CORRblk": lambda: _corr_clusters(mes)}[name]()
        if name == "M8blk":                   # G13: semelhança entre partições
            com, ff, cc = ge.merge_small(lab), mes.sec["ff12"], ge.merge_small(_corr_clusters(mes))
            info.update({"n_com": int(com.max() + 1), "ari_ff12": ge.ari(com, ff), "nmi_ff12": ge.nmi(com, ff),
                         "ari_corr": ge.ari(com, cc), "nmi_corr": ge.nmi(com, cc),
                         "ari_ff12_corr": ge.ari(ff, cc), "nmi_ff12_corr": ge.nmi(ff, cc)})
        return ge.block_factor(X, ge.merge_small(lab)), info
    raise ValueError(f"estimador {name!r}")


def _sector_pairs(mes: Mes) -> dict:
    out = {}
    for k in ("ff49", "sic2"):
        c = mes.sec[k]
        M = ((c[:, None] == c[None, :]) & (c[:, None] >= 0)).astype(float)
        np.fill_diagonal(M, 0.0)
        out[f"same_{k}"] = M
    return out


def _communities(mes: Mes) -> np.ndarray:
    if "louvain" not in mes.cache:
        mes.cache["louvain"] = ge.louvain(np.maximum(mes.cg() - np.eye(mes.p), 0), seed=SEED)
    return mes.cache["louvain"]


def _corr_clusters(mes: Mes) -> np.ndarray:
    if "corrcl" not in mes.cache:
        C = ge.sample_corr(mes.X)
        iu = np.triu_indices(mes.p, 1)
        mes.cache["corrcl"] = ge.louvain(np.maximum(C - C[iu].mean(), 0) * (1 - np.eye(mes.p)), seed=SEED)
    return mes.cache["corrcl"]


def _gmv(S: np.ndarray) -> np.ndarray:
    try:
        x = np.linalg.solve(S, np.ones(len(S)))
        if not np.isfinite(x).all() or np.linalg.cond(S) > 1e14:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(S) @ np.ones(len(S))
    return x / x.sum()


def _choose_delta(hist, grid, crit: str) -> float:
    """δ da grade que minimiza, nos meses guardados, a variância realizada da
    GMV (crit "gmv") ou a QLIKE média (crit "qlike")."""
    loss = np.zeros(len(grid))
    for Smp, T, X_out in hist:
        R = np.nan_to_num(X_out)
        Sr = R.T @ R / len(R)
        for g, d in enumerate(grid):
            S = d * T + (1 - d) * Smp
            if crit == "gmv":
                w = _gmv(S)
                loss[g] += np.var(R @ w)
            else:
                sgn, ld = np.linalg.slogdet(S)
                loss[g] += (ld + np.trace(np.linalg.solve(S, Sr))) / len(S) if sgn > 0 else np.inf
    return float(grid[int(np.argmin(loss))])


# ------------------------------------------------------------------ carteiras

def weights(port: str, S: np.ndarray, mes: Mes) -> np.ndarray | None:
    if port == "gmv":
        return _gmv(S)
    if port == "gmv_lo":
        return optimize.min_variance(S, 0.0, CAP)
    if port == "hrp":
        return optimize.hrp(S)
    if port == "erc":
        if np.linalg.eigvalsh(S)[0] <= 0:
            return None
        return optimize.project_box_simplex(ge.erc(S), 0.0, 1.0)
    if port == "minte":
        ok = np.isfinite(mes.spy_in)
        Z = est.demeaned(mes.X_raw[ok])
        bb = mes.spy_in[ok] - mes.spy_in[ok].mean()
        return optimize.min_tracking_error(S, Z.T @ bb / (len(bb) - 1), 0.0, CAP)
    if port == "mvo":
        z = (mes.mom - mes.mom.mean()) / max(mes.mom.std(), 1e-12)
        mu = 0.05 * np.sqrt(np.diag(S)) * z
        return optimize.mean_variance(S, mu, 5.0, 0.0, CAP)
    raise ValueError(port)


# ------------------------------------------------------------------ perdas do mês

def month_losses(S: np.ndarray, mes: Mes, edges: np.ndarray, Cf_tot: np.ndarray) -> dict:
    R = np.nan_to_num(mes.X_out)
    h, p = R.shape
    Sr = R.T @ R / h
    sgn, ld = np.linalg.slogdet(S)
    out = {}
    if sgn > 0 and np.linalg.cond(S) < 1e14:
        out["qlike"] = float((ld + np.trace(np.linalg.solve(S, Sr))) / p)
    else:
        out["qlike"] = np.nan
    out["frob_cov"] = float(np.linalg.norm(S - Sr) ** 2 / p ** 2)
    C = ge.corr_of(S)
    iu = np.triu_indices(p, 1)
    ok = np.isfinite(Cf_tot[iu])
    e2 = (C[iu] - Cf_tot[iu]) ** 2
    out["frob_corr"] = float(np.mean(e2[ok]))
    e = edges[iu]
    out["err_aresta"] = float(np.mean(e2[ok & e])) if (ok & e).any() else np.nan
    out["err_sem_aresta"] = float(np.mean(e2[ok & ~e]))
    from scipy.stats import spearmanr
    out["spearman"] = float(spearmanr(C[iu][ok], Cf_tot[iu][ok]).statistic)
    ev = np.linalg.eigvalsh(C)
    out["cond"] = float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf
    return out


# ------------------------------------------------------------------ laço

def _rebalance_idx(days, rule: str, first: str, last: str) -> list[int]:
    out = []
    for i, d in enumerate(days):
        if not (first <= str(d) <= last):
            continue
        prev = days[i - 1] if i else None
        key = {"M": lambda x: (x.year, x.month), "W": lambda x: x.isocalendar()[:2],
               "Q": lambda x: (x.year, (x.month - 1) // 3)}[rule]
        if prev is None or key(d) != key(prev):
            out.append(i)
    return out


class World:
    """Dados carregados uma vez por processo (painel, eventos, séries do M1)."""

    def __init__(self, cfg: Config, need_m1: bool = True):
        self.cfg = cfg
        con = query.connect(cfg)
        self.pn = dados.Panel(cfg, con)
        self.ev = rede.Events(con, mentions_glob=str(cfg.curated / "mentions" / "*.parquet"))
        self.ev.prepare_mentions(con, self.pn.tickers)
        self.news_n100 = {}
        for d, t in con.execute("SELECT rebalance_date, ticker FROM news.universe WHERE n_target = 100").fetchall():
            self.news_n100.setdefault(d, []).append(t)
        con.close()
        self.F = None
        if need_m1:
            cols = [-dados.fred_changes(cfg, "DGS2", self.pn.days)]
            cols += [self.pn.R[:, self.pn.col[x]] for x in M1_PROXIES[1:]]
            self.F = np.column_stack(cols)

    def universe(self, i: int, spec: Spec) -> list[str]:
        if spec.universo == "n100":
            import bisect
            ds = sorted(self.news_n100)
            k = bisect.bisect_right(ds, self.pn.days[i]) - 1
            return self.pn.universe(i, "lista", spec.insample, extra={"tickers": self.news_n100[ds[k]]})
        return self.pn.universe(i, spec.universo, spec.insample)


def run(cfg: Config, tag: str, spec: Spec, ests: list[tuple[str, str, dict]], ports: dict[str, list[str]],
        period: tuple[str, str], world: World | None = None, train: bool = True, losses: bool = True,
        verbose: bool = True) -> Path:
    """ests: (rótulo, código, parâmetros); ports: rótulo -> carteiras."""
    t0 = time.time()
    W = world or World(cfg)
    pn = W.pn
    out = wf_dir(cfg, tag)
    out.mkdir(parents=True, exist_ok=True)
    reb = _rebalance_idx(pn.days, spec.rebalance, TRAIN_FROM if train else period[0], "2099-12-31")
    first_eval = next(k for k, i in enumerate(reb) if str(pn.days[i]) >= period[0])
    need_r0 = any(c == "R0" for _, c, _ in ests)
    need_m6 = any(c.startswith("M6") for _, c, _ in ests)
    state: dict = {}
    if need_r0:
        state["trainer_r0"] = Trainer(2)
    books = {(lab, pt): Book(spec.cost_bps / 1e4) for lab, _, _ in ests for pt in ports[lab]}
    daily = {k: [] for k in ("date", "est", "port", "gross", "turnover", "exante_sd")}
    monthly: list[dict] = []
    layer_names = None
    end = len(pn.days)
    for n, i in enumerate(reb):
        i_next = reb[n + 1] if n + 1 < len(reb) else end
        if str(pn.days[i]) > period[1]:
            break
        evaluate = n >= first_eval
        if not evaluate and not (need_r0 or need_m6):
            continue
        U = W.universe(i, spec)
        X_raw = pn.window(i, U, spec.insample)
        X = X_raw
        if spec.retornos == "log":
            X = np.log1p(X)
        if spec.winsor:
            lo, hi = np.nanpercentile(X, [1, 99], axis=0)
            X = np.clip(X, lo, hi)
        X_out = pn.R[i:i_next][:, [pn.col[x] for x in U]]
        mes = Mes(t=pn.days[i], i=i, U=U, X=X, X_raw=X_raw, X_out=X_out,
                  spy_in=pn.R[i - spec.insample:i, pn.col["SPY"]], sec=pn.sectors.arrays(U),
                  size=pn.size(i, U, spec.insample), mom=pn.momentum(i, U), snaps={}, ev=W.ev, spec=spec,
                  F=None if W.F is None else W.F[i - spec.insample:i])
        if evaluate:
            Cf_tot, _ = data.pairwise_corr(X_out, min_obs=max(5, int(0.7 * len(X_out))))
            edges = mes.snap().direct("raw") != 0
            for lab, code, prm in ests:
                S, info = estimate(code, prm, mes, state)
                row = {"date": str(mes.t), "est": lab, "p": mes.p, **{k: v for k, v in info.items()
                                                                       if isinstance(v, (int, float))}}
                if losses:
                    row.update(month_losses(S, mes, edges, Cf_tot))
                monthly.append(row)
                for pt in ports[lab]:
                    w = _special_weights(pt, S, mes) if pt.startswith("hrp_") or pt == "1n" else weights(pt, S, mes)
                    if w is None:
                        continue
                    row[f"{pt}_effn"] = float(1 / np.sum(w ** 2))
                    row[f"{pt}_wmax"] = float(np.abs(w).max())
                    bk = books[(lab, pt)]
                    to, cost, _ = bk.rebalance(U, w)
                    h = bk.hold(X_out, cost)
                    sd = float(np.sqrt(max(w @ S @ w, 0)))
                    L = len(h["gross"])
                    daily["date"] += [str(d) for d in pn.days[i:i_next]]
                    daily["est"] += [lab] * L
                    daily["port"] += [pt] * L
                    daily["gross"].append(h["gross"])
                    daily["turnover"].append(np.r_[to, np.zeros(L - 1)])
                    daily["exante_sd"].append(np.full(L, sd))
            # δ por validação: guarda o mês com o realizado (o mesmo para as duas variantes)
            pend = state.pop("dcv_pending", {})
            if pend:
                Smp, T, m_ = next(iter(pend.values()))
                state["dcv_hist"].append((Smp, T, m_.X_out))
        # treino sequencial com o mês realizado (entra para os meses seguintes)
        if (need_r0 or need_m6) and spec.rebalance == "M":
            Cf, Ce = _resid_corr(X, X_out)
            iu = np.triu_indices(mes.p, 1)
            if need_r0:
                Fp = _sector_pairs(mes)
                state["trainer_r0"].add(np.column_stack([Fp["same_ff49"][iu], Fp["same_sic2"][iu]]),
                                        np.arctanh(np.clip(Cf[iu], -0.999, 0.999)))
            if need_m6:
                layer_names, L = _layers(mes)
                if "trainer_m6" not in state:
                    state["trainer_m6"] = Trainer(len(L), nonneg=True)
                state["trainer_m6"].add(np.column_stack([Lk[iu] for Lk in L]), Cf[iu])
        if verbose and (n + 1) % 12 == 0:
            print(f"[wf] {tag}: {mes.t} ({n + 1}/{len(reb)}, p={mes.p}, {time.time() - t0:.0f}s)", flush=True)
    tab = {"date": daily["date"], "est": daily["est"], "port": daily["port"],
           **{k: np.concatenate(daily[k]) if daily[k] else np.array([]) for k in ("gross", "turnover", "exante_sd")}}
    pq.write_table(pa.table(tab), out / "diario.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(monthly), out / "mensal.parquet", compression="zstd")
    (out / "info.json").write_text(json.dumps({"spec": asdict(spec), "period": period, "ests": ests, "ports": ports,
                                               "layers": layer_names, "seconds": round(time.time() - t0)},
                                              indent=2, ensure_ascii=False, default=str))
    if verbose:
        print(f"[wf] {out} ({time.time() - t0:.0f}s)")
    return out


def _special_weights(pt: str, S: np.ndarray, mes: Mes) -> np.ndarray:
    """Carteiras que não usam a Σ̂ do estimador do jeito padrão: 1/N e as
    variantes de HRP do M8 (árvore das notícias, variâncias amostrais)."""
    if pt == "1n":
        return np.full(mes.p, 1 / mes.p)
    Ss = est.sample_cov(mes.X)
    if pt == "hrp_a":
        return ge.hrp_dist(Ss, mes.cg())
    if pt.startswith("hrp_b"):
        lam = float(pt.split("=")[1]) if "=" in pt else 0.5
        return ge.hrp_dist(Ss, lam * mes.cg() + (1 - lam) * ge.sample_corr(mes.X))
    if pt == "hrp_c":
        return ge.hrp_communities(Ss, ge.merge_small(_communities(mes)))
    raise ValueError(pt)


# ------------------------------------------------------------------ leitura

def load(cfg: Config, tag: str):
    import pandas as pd
    d = wf_dir(cfg, tag)
    di = pd.read_parquet(d / "diario.parquet")
    me = pd.read_parquet(d / "mensal.parquet")
    return di, me, json.loads((d / "info.json").read_text())


def net_returns(di, cost_bps: float = 10.0):
    """Tabela (pregão x 'est|port') de retornos líquidos: no dia da troca,
    (1 - custo·giro)(1 + bruto) - 1, como engine.Book."""
    import pandas as pd
    r = (1 - cost_bps / 1e4 * di["turnover"]) * (1 + di["gross"]) - 1
    key = di["est"] + "|" + di["port"]
    return pd.DataFrame({"date": pd.to_datetime(di["date"]), "key": key, "r": r}).pivot(
        index="date", columns="key", values="r").sort_index()
