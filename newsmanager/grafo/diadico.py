"""Hipóteses diádicas da seção 18.4: G1, G2, G3, G6, G8 (pares) e G9 (ações).

    python -m newsmanager.grafo diadico-painel [--tau 90] [--norm idf] [--universo sp500]
    python -m newsmanager.grafo diadico [--tag ...]

Painel: em cada 1º pregão t de mês (2011-01 a 2023-12), os pares i < j de
U_t. y = z (atanh) da correlação RESIDUAL ao fator de mercado de U_t nos h
pregões a partir de t (b da janela in-sample, m do próprio período). Um
arquivo por mês em data/marts/grafo/diadico/<tag>/ (pares float32) e outro
com as ações (G9).

Testes (padronizados por data, Fama-MacBeth com Newey-West L = 3 + h/21):
  G1  β de sgnlog(w); QAP com 200 permutações das ações nas matrizes de notícia
  G2  β⁺ - β⁻ com log(1+w⁺) e log(1+w⁻)
  G3  Wald da igualdade dos 6 β de grupo (cadeia, parceria, concorrência,
      legal, controle, impacto), com "outros" como controle
  G6  interação sgnlog(w) x cobertura; β por tercil de tamanho do par
  G8  Δz_fut em Δsgnlog(w)_t contra o reverso Δsgnlog(w)_{t+1} em Δz_t
  G9  beta de baixa futuro (63 pregões) na centralidade de |W|+|G|
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import query
from ..config import Config
from ..portfolio import data
from . import dados, inferencia as inf, rede

HORIZONS = (21, 63, 126)
GROUPS6 = tuple(rede.GROUPS)
CONTROLS = ("z_hist", "same_ff49", "same_sic2", "sizedist", "att_sum", "att_diff")
# variáveis que vêm das notícias (permutadas juntas no QAP)
NEWS_VARS = ("sgnlog", "pos", "neg", *[f"g_{x}" for x in rede.GROUP_NAMES], "g", "att_sum", "att_diff",
             "cooc", "comen")
SUBPERIODS = (("2011", "2014"), ("2015", "2018"), ("2019", "2023"))
QAP_R = 200
SEED = 7


def panel_dir(cfg: Config, tag: str) -> Path:
    return dados.out_dir(cfg) / "diadico" / tag


def default_tag(tau: float, norm: str, universo: str) -> str:
    return f"{universo}_tau={tau:g}_norm={norm}"


def _z(C: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.arctanh(np.clip(C, -0.999, 0.999))


def _centrality(A: np.ndarray, iters: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """(autovetor, PageRank 0,85) de uma matriz de pesos >= 0."""
    p = len(A)
    v = np.full(p, 1 / np.sqrt(p))
    for _ in range(iters):
        nv = A @ v
        n = np.linalg.norm(nv)
        if n == 0:
            break
        nv /= n
        if np.abs(nv - v).max() < 1e-10:
            v = nv
            break
        v = nv
    deg = A.sum(axis=1)
    P = np.divide(A, deg[:, None], out=np.full_like(A, 1 / p), where=deg[:, None] > 0)
    pr = np.full(p, 1 / p)
    for _ in range(iters):
        npr = 0.15 / p + 0.85 * pr @ P
        if np.abs(npr - pr).max() < 1e-12:
            pr = npr
            break
        pr = npr
    return np.abs(v), pr


def _betas(r: np.ndarray, m: np.ndarray, down: bool = False) -> np.ndarray:
    """β de cada coluna de r em m (dias com dado; down: só dias com m < 0)."""
    out = np.full(r.shape[1], np.nan)
    sel = m < 0 if down else np.ones(len(m), bool)
    for j in range(r.shape[1]):
        ok = sel & np.isfinite(r[:, j])
        if ok.sum() >= 10:
            mm, x = m[ok], r[ok, j]
            v = mm.var()
            out[j] = ((x - x.mean()) * (mm - mm.mean())).mean() / v if v > 0 else np.nan
    return out


def _coskew(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.full(r.shape[1], np.nan)
    for j in range(r.shape[1]):
        ok = np.isfinite(r[:, j])
        if ok.sum() >= 10:
            x, mm = r[ok, j] - r[ok, j].mean(), m[ok] - m[ok].mean()
            out[j] = (x * mm ** 2).mean() / (x.std() * (mm ** 2).mean())
    return out


def build(cfg: Config, tau: float = 90.0, window: int = 365, norm: str = "idf", universo: str = "sp500",
          first: str = "2011-01-01", last: str = "2023-12-31", tag: str | None = None) -> Path:
    t0 = time.time()
    tag = tag or default_tag(tau, norm, universo)
    out = panel_dir(cfg, tag)
    out.mkdir(parents=True, exist_ok=True)
    con = query.connect(cfg)
    pn = dados.Panel(cfg, con)
    ev = rede.Events(con, mentions_glob=str(cfg.curated / "mentions" / "*.parquet"))
    ev.prepare_mentions(con, pn.tickers)
    con.close()
    starts = dados.month_starts(pn.days, "2010-01-01", "2023-12-31")
    todo = [k for k, i in enumerate(starts) if first <= str(pn.days[i]) <= last]
    for n, k in enumerate(todo):
        i = starts[k]
        t = pn.days[i]
        f_pairs = out / f"pares_{t}.parquet"
        if f_pairs.exists():
            continue
        U = pn.universe(i)
        p = len(U)
        X_in = pn.window(i, U)
        m_in = data.market_factor(X_in)
        a, b = data.market_betas(X_in, m_in)
        E_in = data.residuals(X_in, b, m_in, a)
        z_hist = _z(data.pairwise_corr(E_in, min_obs=126)[0])
        zf = {}
        for h in HORIZONS:
            if i + h <= len(pn.days):
                Xh = pn.R[i:i + h][:, [pn.col[x] for x in U]]
                Eh = data.residuals(Xh, b, data.market_factor(Xh))
                zf[h] = _z(data.pairwise_corr(Eh, min_obs=int(0.7 * h))[0])
            else:
                zf[h] = np.full((p, p), np.nan)
        # G8: blocos de 21 pregões da janela in-sample (o último é ρ_t)
        blocks = [_z(data.pairwise_corr(E_in[-21 * (q + 1):len(E_in) - 21 * q], min_obs=15)[0]) for q in range(12)]
        z_t, z_tm1 = blocks[0], blocks[1]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                z_bar = np.nanmean(np.stack(blocks), axis=0)

        snap = ev.snapshot(t, U, tau=tau, window_days=window)
        prev = ev.snapshot(pn.days[starts[k - 1]], U, tau=tau, window_days=window) if k > 0 else None
        nxt = ev.snapshot(pn.days[starts[k + 1]], U, tau=tau, window_days=window) if k + 1 < len(starts) else None
        w = snap.direct("sgnlog")
        G = snap.projection(norm)
        mi = np.log1p(snap.mentions())
        cooc = snap.cooccur()
        comen = ev.comention(t, U, window)
        np.fill_diagonal(comen, 0.0)
        sec = pn.sectors.arrays(U)
        size = pn.size(i, U)

        iu = np.triu_indices(p, 1)
        same = lambda c: ((c[:, None] == c[None, :]) & (c[:, None] >= 0))[iu]  # noqa: E731
        cols = {
            "i": iu[0].astype(np.int16), "j": iu[1].astype(np.int16),
            **{f"z_fut{h}": zf[h][iu] for h in HORIZONS},
            "z_hist": z_hist[iu], "z_t": z_t[iu], "z_tm1": z_tm1[iu], "z_bar": z_bar[iu],
            "sgnlog": w[iu], "pos": snap.direct("pos")[iu], "neg": snap.direct("neg")[iu],
            **{f"g_{x}": snap.direct(x)[iu] for x in rede.GROUP_NAMES},
            "g": G[iu], "cooc": np.log1p(cooc[iu]), "comen": np.log1p(comen[iu]),
            "w_prev": prev.direct("sgnlog")[iu] if prev else np.full(len(iu[0]), np.nan),
            "w_next": nxt.direct("sgnlog")[iu] if nxt else np.full(len(iu[0]), np.nan),
            "att_sum": (mi[:, None] + mi[None, :])[iu], "att_diff": np.abs(mi[:, None] - mi[None, :])[iu],
            "same_ff12": same(sec["ff12"]), "same_ff49": same(sec["ff49"]), "same_sic2": same(sec["sic2"]),
            "sizedist": np.abs(size[:, None] - size[None, :])[iu],
            "size_mean": ((size[:, None] + size[None, :]) / 2)[iu],
        }
        pq.write_table(pa.table({c: (v if c in ("i", "j") else np.asarray(v, np.float32)) for c, v in cols.items()}),
                       f_pairs.with_suffix(".tmp"), compression="zstd")
        f_pairs.with_suffix(".tmp").rename(f_pairs)

        # ações (G9)
        Wa = np.abs(snap.direct("raw"))
        Ga = np.abs(G - np.eye(p))
        A = (Wa / Wa.max() if Wa.max() > 0 else Wa) + Ga
        eig, pr = _centrality(A)
        h63 = min(63, len(pn.days) - i)
        Xf = pn.R[i:i + h63][:, [pn.col[x] for x in U]]
        mf = data.market_factor(Xf)
        st = {"ticker": U, "eig": eig, "pagerank": pr, "degree": A.sum(axis=1), "mentions": mi,
              "size": size, "beta_past": b, "ff12": sec["ff12"].astype(float),
              "beta_fut": _betas(Xf, mf), "beta_down_fut": _betas(Xf, mf, down=True), "coskew_fut": _coskew(Xf, mf)}
        pq.write_table(pa.table(st), out / f"acoes_{t}.parquet", compression="zstd")
        if (n + 1) % 12 == 0 or n + 1 == len(todo):
            print(f"[diadico] {t} p={p} ({n + 1}/{len(todo)}, {time.time() - t0:.0f}s)", flush=True)
    (out / "params.json").write_text(json.dumps({"tau": tau, "window": window, "norm": norm, "universo": universo,
                                                 "first": first, "last": last}, indent=2))
    return out


# ------------------------------------------------------------------ testes

def _months(cfg: Config, tag: str) -> list[tuple[str, Path, Path]]:
    d = panel_dir(cfg, tag)
    out = []
    for f in sorted(d.glob("pares_*.parquet")):
        t = f.stem.split("_", 1)[1]
        out.append((t, f, d / f"acoes_{t}.parquet"))
    return out


def _std(M: np.ndarray) -> np.ndarray:
    sd = M.std(axis=0)
    return np.divide(M - M.mean(axis=0), sd, out=np.zeros_like(M), where=sd > 0)


def _ols(y, X):
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)
    return b[1:]


def _load(f: Path, cols: list[str]) -> dict[str, np.ndarray]:
    t = pq.read_table(f, columns=list(dict.fromkeys(["i", "j", *cols])))
    return {c: t[c].to_numpy().astype(float) if c not in ("i", "j") else t[c].to_numpy().astype(np.int64)
            for c in t.column_names}


def _perm_vec(v: np.ndarray, i: np.ndarray, j: np.ndarray, perm: np.ndarray, p: int) -> np.ndarray:
    M = np.zeros((p, p))
    M[i, j] = v
    M[j, i] = v
    return M[perm[i], perm[j]]


def fama_macbeth(cfg: Config, tag: str, y: str, xs: list[str], subset=None, std_y: bool = False,
                 qap: int = 0, extra=None) -> dict:
    """Coeficientes por data (padronizados) e, com qap > 0, as médias sob
    permutações das ações nas variáveis de notícia de xs (as mesmas em todas
    as datas da mesma réplica). subset(d) -> máscara dos pares."""
    months = _months(cfg, tag)
    need = [y, *xs] + (["same_ff12", "z_hist"] if subset else [])
    coefs, dates, perms = [], [], [[] for _ in range(qap)]
    rng = np.random.default_rng(SEED)
    for t, f, _ in months:
        d = _load(f, need + (extra or []))
        ok = np.isfinite(d[y])
        for x in xs:
            ok &= np.isfinite(d[x])
        if subset is not None:
            ok &= subset(d)
        if ok.sum() < 50:
            continue
        yy = d[y][ok]
        yy = _std(yy[:, None])[:, 0] if std_y else yy
        X = _std(np.column_stack([d[x][ok] for x in xs]))
        coefs.append(_ols(yy, X))
        dates.append(t)
        if qap:
            p = int(max(d["i"].max(), d["j"].max())) + 1
            ii, jj = d["i"][ok], d["j"][ok]
            news = [k for k, x in enumerate(xs) if x in NEWS_VARS]
            for r in range(qap):
                perm = rng.permutation(p)
                Xp = X.copy()
                for k in news:
                    Xp[:, k] = _perm_vec(d[xs[k]], d["i"], d["j"], perm, p)[ok]
                Xp[:, news] = _std(Xp[:, news])
                perms[r].append(_ols(yy, Xp))
    C = np.array(coefs)
    return {"dates": dates, "coef": C, "perm": np.array([np.array(x) for x in perms]) if qap else None}


def _lags(h: int) -> int:
    return 3 + h // 21


def _summ(C: np.ndarray, names: list[str], lags: int) -> dict:
    return {n: {"coef": float(C[:, k].mean()), "t": inf.nw_t(C[:, k], lags)} for k, n in enumerate(names)}


def _by_period(dates, x) -> dict:
    out = {}
    for lo, hi in SUBPERIODS:
        m = np.array([lo <= d[:4] <= hi for d in dates])
        out[f"{lo}-{hi}"] = float(np.mean(x[m])) if m.any() else None
    return out


def g1(cfg, tag, h=21, qap=QAP_R) -> dict:
    xs = ["sgnlog", "g", *CONTROLS]
    y = f"z_fut{h}"
    r = fama_macbeth(cfg, tag, y, xs, qap=qap)
    C = r["coef"]
    out = {"h": h, "months": len(r["dates"]), "coef": _summ(C, xs, _lags(h))}
    b = C[:, 0]
    t = inf.nw_t(b, _lags(h))
    from scipy import stats
    out["p_nw"] = float(stats.norm.sf(t))
    if qap:
        null = r["perm"][:, :, 0].mean(axis=1)
        out["p_qap"] = float((1 + np.sum(null >= b.mean())) / (1 + len(null)))
        out["p"] = max(out["p_nw"], out["p_qap"])
    out["by_period"] = _by_period(r["dates"], b)
    for nm, f in (("intra", lambda d: d["same_ff12"] == 1),
                  ("inter", lambda d: (d["same_ff12"] == 0))):
        s = fama_macbeth(cfg, tag, y, xs, subset=f)
        out[f"t_{nm}"] = inf.nw_t(s["coef"][:, 0], _lags(h))
        out[f"coef_{nm}"] = float(s["coef"][:, 0].mean())
    out["sucesso_condicoes"] = {
        "positivo_subperiodos": all(v is not None and v > 0 for v in out["by_period"].values()),
        "t_intra>2": out["t_intra"] > 2, "t_inter>2": out["t_inter"] > 2}
    return out


def g2(cfg, tag, h=21, qap=QAP_R) -> dict:
    from scipy import stats
    xs = ["pos", "neg", "g", *CONTROLS]
    r = fama_macbeth(cfg, tag, f"z_fut{h}", xs, qap=qap)
    C = r["coef"]
    diff = C[:, 0] - C[:, 1]
    t = inf.nw_t(diff, _lags(h))
    out = {"h": h, "coef": _summ(C, xs, _lags(h)), "diff": float(diff.mean()), "t_diff": t,
           "p_nw": float(stats.norm.sf(t)),
           "t_neg": inf.nw_t(C[:, 1], _lags(h)), "p_neg_menor_que_0": float(stats.norm.cdf(inf.nw_t(C[:, 1], _lags(h))))}
    if qap:
        null = (r["perm"][:, :, 0] - r["perm"][:, :, 1]).mean(axis=1)
        out["p_qap"] = float((1 + np.sum(null >= diff.mean())) / (1 + len(null)))
        out["p"] = max(out["p_nw"], out["p_qap"])
    return out


def _wald_equal(C: np.ndarray, lags: int) -> float:
    k = C.shape[1]
    R = np.eye(k)[:-1] - np.eye(k)[1:]
    V = inf.nw_cov(C, lags)
    m = R @ C.mean(axis=0)
    return float(m @ np.linalg.solve(R @ V @ R.T, m))


def g3(cfg, tag, h=21, qap=QAP_R) -> dict:
    from scipy import stats
    gx = [f"g_{x}" for x in GROUPS6]
    xs = [*gx, f"g_{rede.OUTROS}", "g", *CONTROLS]
    r = fama_macbeth(cfg, tag, f"z_fut{h}", xs, qap=qap)
    C = r["coef"]
    W = _wald_equal(C[:, :6], _lags(h))
    out = {"h": h, "coef": _summ(C, xs, _lags(h)), "wald": W, "p_chi2": float(stats.chi2.sf(W, 5))}
    pv = [2 * stats.norm.sf(abs(out["coef"][x]["t"])) for x in gx]
    out["bh"] = dict(zip(gx, inf.bh(np.array(pv)).tolist()))
    if qap:
        null = np.array([_wald_equal(P[:, :6], _lags(h)) for P in r["perm"]])
        out["p_qap"] = float((1 + np.sum(null >= W)) / (1 + len(null)))
        out["p"] = max(out["p_chi2"], out["p_qap"])
    out["expectativas"] = {"cadeia": ">0", "parceria": ">=0 pequena", "legal": "<=0"}
    return out


def g6(cfg, tag, h=21) -> dict:
    xs = ["sgnlog", "inter", "g", *CONTROLS]
    months = _months(cfg, tag)
    coefs, terc = [], {0: [], 1: [], 2: []}
    for t, f, _ in months:
        d = _load(f, [f"z_fut{h}", "sgnlog", "g", *CONTROLS, "size_mean"])
        ok = np.isfinite(d[f"z_fut{h}"]) & np.isfinite(d["size_mean"])
        for x in ["sgnlog", "g", *CONTROLS]:
            ok &= np.isfinite(d[x])
        if ok.sum() < 50:
            continue
        cov = _std(d["att_sum"][ok][:, None])[:, 0]
        ws = _std(d["sgnlog"][ok][:, None])[:, 0]
        X = _std(np.column_stack([d["sgnlog"][ok], ws * cov, d["g"][ok], *[d[c][ok] for c in CONTROLS]]))
        coefs.append(_ols(d[f"z_fut{h}"][ok], X))
        cuts = np.quantile(d["size_mean"][ok], [1 / 3, 2 / 3])
        tt = np.searchsorted(cuts, d["size_mean"][ok])
        for q in range(3):
            m = tt == q
            Xq = _std(np.column_stack([d["sgnlog"][ok][m], d["g"][ok][m], *[d[c][ok][m] for c in CONTROLS]]))
            terc[q].append(_ols(d[f"z_fut{h}"][ok][m], Xq)[0])
    C = np.array(coefs)
    return {"interacao_cobertura": _summ(C, xs, _lags(h))["inter"],
            "beta_por_tercil_tamanho": {q: {"coef": float(np.mean(v)), "t": inf.nw_t(np.array(v), _lags(h))}
                                        for q, v in terc.items()}}


def g8(cfg, tag) -> dict:
    """Direção temporal: Δz_fut = z(ρ_{t,t+21}) - z(ρ_t) em Δw_t = w_t - w_{t-1}
    (+ z(ρ_t), z(ρ_t) - z̄); reverso: Δw_{t+1} = w_{t+1} - w_t em Δz_t = z(ρ_t) - z(ρ_{t-1}) (+ w_t)."""
    from scipy import stats
    months = _months(cfg, tag)
    fw, rv = [], []
    for t, f, _ in months:
        d = _load(f, ["z_fut21", "z_t", "z_tm1", "z_bar", "sgnlog", "w_prev", "w_next", "att_sum", "same_ff49"])
        dz_f = d["z_fut21"] - d["z_t"]
        dw = d["sgnlog"] - d["w_prev"]
        ok = np.isfinite(dz_f) & np.isfinite(dw) & np.isfinite(d["z_t"]) & np.isfinite(d["z_bar"])
        if ok.sum() > 50 and np.std(dw[ok]) > 0:
            X = _std(np.column_stack([dw[ok], d["z_t"][ok], (d["z_t"] - d["z_bar"])[ok], d["att_sum"][ok],
                                      d["same_ff49"][ok]]))
            fw.append(_ols(_std(dz_f[ok][:, None])[:, 0], X)[0])
        dwn = d["w_next"] - d["sgnlog"]
        dz_p = d["z_t"] - d["z_tm1"]
        ok = np.isfinite(dwn) & np.isfinite(dz_p)
        if ok.sum() > 50 and np.std(dwn[ok]) > 0:
            X = _std(np.column_stack([dz_p[ok], d["sgnlog"][ok], d["att_sum"][ok], d["same_ff49"][ok]]))
            rv.append(_ols(_std(dwn[ok][:, None])[:, 0], X)[0])
    fw, rv = np.array(fw), np.array(rv)
    tf, tr = inf.nw_t(fw, 4), inf.nw_t(rv, 4)
    return {"b_frente": float(fw.mean()), "t_frente": tf, "p": float(stats.norm.sf(tf)),
            "b_reverso": float(rv.mean()), "t_reverso": tr, "meses": [len(fw), len(rv)],
            "frente_maior": bool(abs(fw.mean()) > abs(rv.mean()))}


def g9(cfg, tag) -> dict:
    from scipy import stats
    months = _months(cfg, tag)
    res = {k: [] for k in ("beta_down_fut", "beta_fut", "coskew_fut")}
    extra = {k: [] for k in ("pagerank", "degree")}
    for t, _, fa in months:
        d = pq.read_table(fa).to_pydict()
        d = {k: np.array(v, dtype=float) if k != "ticker" else v for k, v in d.items()}
        ff = d["ff12"]
        D = np.column_stack([(ff == s).astype(float) for s in np.unique(ff)[1:]]) if len(np.unique(ff)) > 1 else np.zeros((len(ff), 0))
        for y in res:
            ok = np.isfinite(d[y]) & np.isfinite(d["size"]) & np.isfinite(d["beta_past"])
            if ok.sum() < 30:
                continue
            X = np.column_stack([_std(np.column_stack([d["eig"][ok], d["size"][ok], d["beta_past"][ok]])), D[ok]])
            res[y].append(_ols(d[y][ok], X)[0])
        ok = np.isfinite(d["beta_down_fut"]) & np.isfinite(d["size"]) & np.isfinite(d["beta_past"])
        for c in extra:
            X = np.column_stack([_std(np.column_stack([d[c][ok], d["size"][ok], d["beta_past"][ok]])), D[ok]])
            extra[c].append(_ols(d["beta_down_fut"][ok], X)[0])
    out = {y: {"coef": float(np.mean(v)), "t": inf.nw_t(np.array(v), 6)} for y, v in res.items()}
    out["p"] = float(stats.norm.sf(out["beta_down_fut"]["t"]))
    out["outras_centralidades_beta_baixa"] = {c: {"coef": float(np.mean(v)), "t": inf.nw_t(np.array(v), 6)}
                                              for c, v in extra.items()}
    return out


def run(cfg: Config, tag: str, qap: int = QAP_R) -> dict:
    t0 = time.time()
    res = {"tag": tag}
    for name, fn in (("G1", lambda: g1(cfg, tag, 21, qap)), ("G2", lambda: g2(cfg, tag, 21, qap)),
                     ("G3", lambda: g3(cfg, tag, 21, qap)), ("G6", lambda: g6(cfg, tag)),
                     ("G8", lambda: g8(cfg, tag)), ("G9", lambda: g9(cfg, tag))):
        res[name] = fn()
        print(f"[diadico] {name} ({time.time() - t0:.0f}s): {json.dumps(_brief(res[name]), ensure_ascii=False)}",
              flush=True)
    res["robustez_h"] = {h: {"G1": g1(cfg, tag, h, qap=0)["coef"]["sgnlog"]} for h in (63, 126)}
    out = panel_dir(cfg, tag) / "resultados.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"[diadico] {out} ({time.time() - t0:.0f}s)")
    return res


def _brief(r: dict) -> dict:
    keep = ("p", "p_nw", "p_qap", "p_chi2", "t_diff", "wald", "t_frente", "t_reverso", "sucesso_condicoes")
    return {k: v for k, v in r.items() if k in keep}
