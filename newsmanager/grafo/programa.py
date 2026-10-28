"""Etapas da seção 18 (dev, holdout, placebos, robustez, avaliação).

Ordem (18.10): dev -> escolhas congeladas em logs/secao18_escolhas.json ->
holdout (uma vez) -> placebos -> robustez -> avaliar.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..backtester.engine import Book
from ..config import Config
from ..portfolio import data, estimators as est
from . import dados, estimadores as ge, inferencia as inf, rede
from .walkforward import (DEFAULTS, DEV, GRID, HOLDOUT, SEED, Mes, Spec, World, _gmv, load, net_returns, run,
                          wf_dir)

ROOT = Path(__file__).resolve().parents[2]
ESCOLHAS = ROOT / "logs" / "secao18_escolhas.json"
DEV_LOG = ROOT / "logs" / "secao18_dev.log"
R_PLACEBO = 200


def _lab(code: str, prm: dict) -> str:
    return code + ("[" + ",".join(f"{k}={v}" for k, v in prm.items()) + "]" if prm else "")


def _log(msg: str) -> None:
    DEV_LOG.parent.mkdir(exist_ok=True)
    with DEV_LOG.open("a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


# ------------------------------------------------------------------ dev

def dev(cfg: Config, only: list[str] | None = None) -> dict:
    """Roda as grades do dev (2013-2018) e grava as escolhas (regra da tabela 18.5)."""
    W = World(cfg)
    spec = Spec()
    jobs = {
        "M2": ([(_lab("M2", g), "M2", g) for g in GRID["M2"]], "gmv"),
        "B8": ([(_lab("B8", g), "B8", g) for g in GRID["B8"]], "gmv"),
        "M4": ([(_lab("M4", g), "M4", g) for g in GRID["M4"]], "qlike"),
        "M5": ([(_lab("M5", g), "M5", g) for g in GRID["M5"]], "qlike"),
        "M1": ([(_lab("M1", g), "M1", g) for g in GRID["M1"]], "qlike"),
        "M2W": ([(_lab("M2W", g), "M2W", g) for g in GRID["M2W"]], "gmv"),
        "M8b": ([("M8b", "B1", {})], "hrp"),
    }
    for name, (ests, crit) in jobs.items():
        if only and name not in only:
            continue
        ports = {lab: (["gmv"] if crit != "hrp" else [f"hrp_b={g['lam']}" for g in GRID["M8b"]]) for lab, _, _ in ests}
        _log(f"dev {name}: {len(ests)} configurações, critério {crit}")
        run(cfg, f"dev_{name}", spec, ests, ports, DEV, world=W, train=False, losses=(crit == "qlike"))
    return escolher(cfg)


def _crit_table(cfg: Config, tag: str, crit: str) -> dict[str, float]:
    di, me, _ = load(cfg, tag)
    if crit == "qlike":
        return me.groupby("est")["qlike"].mean().to_dict()
    R = net_returns(di)
    return {k: float(R[k].var()) for k in R.columns}


def escolher(cfg: Config) -> dict:
    """Escolhas do dev. Empate (< 0,1% do melhor) fica com o padrão (M2: τ=90, IDF)."""
    out = {"criado": time.strftime("%Y-%m-%d %H:%M:%S"), "dev": DEV}
    tab = _crit_table(cfg, "dev_M2", "gmv")
    m2 = {k.split("|")[0]: v for k, v in tab.items()}
    best = min(m2, key=m2.get)
    default = _lab("M2", DEFAULTS["M2"])
    pick = default if m2[default] <= m2[best] * 1.001 else best
    out["M2"] = next(g for g in GRID["M2"] if _lab("M2", g) == pick)
    out["M2_dev_var"] = m2
    di, _, _ = load(cfg, "dev_M2")
    R = net_returns(di)
    sr = np.array([R[c].mean() / R[c].std() for c in R.columns])
    out["M2_trials"] = {"n": len(sr), "var_sr": float(sr.var(ddof=1))}
    for name, crit, key in (("B8", "gmv", "C"), ("M5", "qlike", "q"), ("M1", "qlike", None), ("M2W", "gmv", None)):
        t = {k.split("|")[0]: v for k, v in _crit_table(cfg, f"dev_{name}", crit).items()}
        lab = min(t, key=t.get)
        out[name] = next(g for g in GRID[name] if _lab(name, g) == lab)
        out[f"{name}_dev"] = t
    t = {k.split("|")[0]: v for k, v in _crit_table(cfg, "dev_M4", "qlike").items()}
    out["M4"] = next(g for g in GRID["M4"] if _lab("M4", g) == min(t, key=t.get))
    b9 = {k: v for k, v in t.items() if "gamma=0.0" in k}
    out["B9"] = {"lam0": next(g for g in GRID["M4"] if _lab("M4", g) == min(b9, key=b9.get))["lam0"]}
    out["M4_dev"] = t
    t = _crit_table(cfg, "dev_M8b", "gmv")
    lab = min(t, key=t.get)
    out["M8b"] = {"lam": float(lab.split("=")[1])}
    out["M8b_dev"] = t
    ESCOLHAS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _log(f"escolhas do dev: { {k: v for k, v in out.items() if not k.endswith(('_dev', '_dev_var'))} }")
    return out


def escolhas() -> dict:
    if not ESCOLHAS.exists():
        raise RuntimeError(f"{ESCOLHAS} não existe: rode `python -m newsmanager.grafo dev` antes")
    return json.loads(ESCOLHAS.read_text())


# ------------------------------------------------------------------ holdout

CORE_PORTS = ["gmv", "gmv_lo", "hrp", "erc", "minte", "mvo"]


def holdout_ests(ch: dict) -> tuple[list, dict]:
    m2 = ch["M2"]
    q = ch["M5"]["q"]
    ests = [("B0", "B1", {}), ("B1", "B1", {}), ("B2", "B2", {}), ("B3", "B3", {}), ("B4", "B4", {}),
            ("B5", "B5", {}), ("B6", "B6", {}), ("B7", "B7", {}), ("B8", "B8", ch["B8"]),
            ("B9", "B9", {"lam0": ch["B9"]["lam0"]}), ("B10", "B10", {}), ("R0", "R0", {}),
            ("M2", "M2", m2), ("M2abs", "M2abs", m2), ("M2bin", "M2bin", m2), ("M2dgmv", "M2dgmv", m2),
            ("M2dqlike", "M2dqlike", m2), ("M2res", "M2res", m2), ("M2W", "M2W", ch["M2W"]),
            ("M1", "M1", ch["M1"]), ("B7p", "B7p", {}), ("M3", "M3", ch["B8"]), ("M3plac", "M3plac", ch["B8"]),
            ("M4", "M4", ch["M4"]), ("M5", "M5", {"q": q}), ("PCAk", "PCAk", {"q": q}), ("PCAkq", "PCAkq", {"q": q}),
            ("PCAk+q", "PCAk+q", {"q": q}), ("M6", "M6", {}), ("M8blk", "M8blk", {}), ("FF12blk", "FF12blk", {}),
            ("CORRblk", "CORRblk", {}), ("M8", "B1", {})]
    core = {"B1", "B2", "B3", "B4", "B5", "B6", "B8", "R0", "M2", "M3", "M4", "M8blk"}
    ports = {lab: (CORE_PORTS if lab in core else ["gmv"]) for lab, _, _ in ests}
    ports["B0"] = ["1n"]
    ports["B1"] = ["gmv", "gmv_lo", "hrp", "minte", "mvo"]          # B11 = B1|hrp
    ports["M8"] = ["hrp_a", f"hrp_b={ch['M8b']['lam']}", "hrp_c"]
    return ests, ports


def holdout(cfg: Config, force: bool = False) -> Path:
    ch = escolhas()
    out = wf_dir(cfg, "holdout")
    if (out / "diario.parquet").exists() and not force:
        raise RuntimeError(f"{out} já existe: o holdout roda uma vez (use --force só para refazer o mesmo código)")
    ests, ports = holdout_ests(ch)
    _log(f"holdout: {len(ests)} estimadores, escolhas {ESCOLHAS}")
    W = World(cfg)
    p = run(cfg, "holdout", Spec(), ests, ports, HOLDOUT, world=W, train=True)
    # ablação do M6 por camada (só GMV)
    names = json.loads((p / "info.json").read_text())["layers"] or []
    abl = [("M6", "M6", {})] + [(f"M6-{n}", "M6", {"drop": n}) for n in names]
    run(cfg, "holdout_m6", Spec(), abl, {lab: ["gmv"] for lab, _, _ in abl}, HOLDOUT, world=W, train=True,
        losses=False)
    return p


# ------------------------------------------------------------------ placebos (G10, G12)

class FastShrink:
    """estimators.shrink_to_corr com as partes que só dependem de X calculadas
    uma vez (para os 200 placebos do mês)."""

    def __init__(self, X):
        Z = est.demeaned(X)
        self.t, n = Z.shape
        self.sample = Z.T @ Z / self.t
        self.var = np.diag(self.sample).copy()
        self.sd = np.sqrt(self.var)
        phi_mat, theta = est._phi_theta(Z, self.sample, self.var, self.t, n)
        self.phi = phi_mat.sum()
        self.tr = np.trace(phi_mat)
        self.rt = np.outer(1 / self.sd, self.sd) * theta

    def __call__(self, R):
        prior = R * np.outer(self.sd, self.sd)
        np.fill_diagonal(prior, self.var)
        rho = self.tr + (R * self.rt).sum()
        gamma = np.linalg.norm(self.sample - prior, "fro") ** 2
        d = float(np.clip((self.phi - rho) / gamma / self.t, 0, 1)) if gamma > 0 else 1.0
        return d * prior + (1 - d) * self.sample


def placebos(cfg: Config, R: int = R_PLACEBO) -> Path:
    ch = escolhas()
    m2 = {**DEFAULTS["M2"], **ch["M2"]}
    q = ch["M5"]["q"]
    W = World(cfg, need_m1=False)
    pn = W.pn
    spec = Spec()
    from .walkforward import _rebalance_idx
    reb = _rebalance_idx(pn.days, "M", HOLDOUT[0], "2099-12-31")
    rng = np.random.default_rng(SEED)
    kinds = ["P1", "P2", "P3", "P4"]
    books = {(k, r): Book(spec.cost_bps / 1e4) for k in kinds for r in range(R)}
    books.update({(k, 0): Book(spec.cost_bps / 1e4) for k in ("M2", "P5", "P6", "P6b")})
    rets = {key: [] for key in books}
    days: list = []
    g12 = {"real": [], "P2": []}
    t0 = time.time()
    for n, i in enumerate(reb):
        if str(pn.days[i]) > HOLDOUT[1]:
            break
        i_next = reb[n + 1] if n + 1 < len(reb) else len(pn.days)
        t = pn.days[i]
        U = W.universe(i, spec)
        X = pn.window(i, U)
        X_out = pn.R[i:i_next][:, [pn.col[x] for x in U]]
        days += [str(d) for d in pn.days[i:i_next]]
        snap = W.ev.snapshot(t, U, tau=m2["tau"])
        C = snap.projection(m2["norm"])
        fs = FastShrink(X)
        sec = pn.sectors.arrays(U)["ff12"]
        targets = {("M2", 0): C}
        old = W.ev.snapshot(dados.shift_days(t, -730), U, tau=m2["tau"])
        targets[("P5", 0)] = old.projection(m2["norm"])
        targets[("P6", 0)] = rede.count_cosine(W.ev.comention(t, U))
        targets[("P6b", 0)] = rede.count_cosine(snap.cooccur(diag=True))
        # G12: subespaço espectral contra os autovetores da covariância dos 63 pregões seguintes
        Xf = np.nan_to_num(pn.R[i:i + 63][:, [pn.col[x] for x in U]])
        Vf = np.linalg.eigh(Xf.T @ Xf / len(Xf) - np.outer(Xf.mean(0), Xf.mean(0)))[1][:, ::-1][:, :q]
        g12["real"].append(ge.principal_cosines(ge.laplacian_vectors(C, q), Vf).mean())
        p2row = []
        for r in range(R):
            targets[("P1", r)] = snap.projection(B=rede.placebo_p1(snap, m2["norm"], rng))
            C2 = snap.projection(B=rede.placebo_p2(snap, m2["norm"], rng))
            targets[("P2", r)] = C2
            p2row.append(ge.principal_cosines(ge.laplacian_vectors(C2, q), Vf).mean())
            targets[("P3", r)] = rede.placebo_p3(C, sec, rng)
            targets[("P4", r)] = snap.projection(B=rede.placebo_p4(snap, m2["norm"], rng))
        g12["P2"].append(p2row)
        for key, Ct in targets.items():
            w = _gmv(fs(Ct))
            bk = books[key]
            to, cost, _ = bk.rebalance(U, w)
            h = bk.hold(X_out, cost)
            rets[key].append(h["net"])
        print(f"[placebos] {t} p={len(U)} ({n + 1}, {time.time() - t0:.0f}s)", flush=True)
    out = wf_dir(cfg, "placebos")
    out.mkdir(parents=True, exist_ok=True)
    cols = {"date": days}
    for (k, r), v in rets.items():
        cols[f"{k}_{r}"] = np.concatenate(v)
    pq.write_table(pa.table(cols), out / "diario.parquet", compression="zstd")
    (out / "g12.json").write_text(json.dumps({"real": [float(x) for x in g12["real"]],
                                              "P2": [[float(y) for y in x] for x in g12["P2"]], "q": q}))
    print(f"[placebos] {out} ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ robustez

def robustez_specs(ch: dict) -> dict[str, tuple[Spec, dict, list[str]]]:
    m2 = {**DEFAULTS["M2"], **ch["M2"]}
    base = Spec()
    out = {
        "janela=126": (replace(base, insample=126), m2, ["gmv"]),
        "janela=504": (replace(base, insample=504), m2, ["gmv"]),
        "rebal=W": (replace(base, rebalance="W"), m2, ["gmv"]),
        "rebal=Q": (replace(base, rebalance="Q"), m2, ["gmv"]),
        **{f"tau={x}": (base, {**m2, "tau": x}, ["gmv"]) for x in (30, 90, 180, 365) if x != m2["tau"]},
        **{f"norm={x}": (base, {**m2, "norm": x}, ["gmv"]) for x in rede.NORMS if x != m2["norm"]},
        "kernel=difusao": (replace(base, kernel="difusao"), m2, ["gmv"]),
        "kernel=higham": (replace(base, kernel="higham"), m2, ["gmv"]),
        "universo=sp100": (replace(base, universo="sp100"), m2, ["gmv"]),
        "universo=exfin": (replace(base, universo="exfin"), m2, ["gmv"]),
        "universo=n100": (replace(base, universo="n100"), m2, ["gmv"]),
        "retornos=log": (replace(base, retornos="log"), m2, ["gmv"]),
        "winsor": (replace(base, winsor=True), m2, ["gmv"]),
        "agentes=sem_ticker": (replace(base, agents="sem_ticker"), m2, ["gmv"]),
        "sem_wrap": (replace(base, sem_wrap=True), m2, ["gmv"]),
        "long-only": (base, m2, ["gmv_lo"]),
    }
    return out


def robustez(cfg: Config) -> None:
    ch = escolhas()
    W = World(cfg, need_m1=False)
    for name, (spec, m2, ports) in robustez_specs(ch).items():
        tag = f"rob_{name}"
        if (wf_dir(cfg, tag) / "diario.parquet").exists():
            continue
        ests = [("B5", "B5", {}), ("M2", "M2", m2)]
        run(cfg, tag, spec, ests, {"B5": ports, "M2": ports}, HOLDOUT, world=W, train=False, losses=False)


# ------------------------------------------------------------------ avaliação

def _vr(a, b, seed=SEED, **kw):
    return inf.var_ratio(np.asarray(a), np.asarray(b), np.random.default_rng(seed), **kw)


def avaliar(cfg: Config) -> dict:
    from scipy import stats
    ch = escolhas()
    di, me, info = load(cfg, "holdout")
    R = net_returns(di)
    # o ERC fica sem carteira quando Σ̂ não é PD e não entra em nenhum teste daqui:
    # não deixar os meses dele derrubarem os dias das outras carteiras
    R = R.dropna(how="any", axis=0, subset=[c for c in R.columns if not c.endswith("|erc")])
    res: dict = {"escolhas": {k: v for k, v in ch.items() if not k.endswith(("_dev", "_dev_var"))}}

    # G5
    a, b = R["M2|gmv"], R["B5|gmv"]
    g5 = _vr(a, b)
    g5["stationary"] = _vr(a, b, stationary=True)
    ql = me.pivot(index="date", columns="est", values="qlike")
    ql = ql.dropna(axis=1, how="any")
    g5["dm_qlike"] = inf.dm_hln((ql["M2"] - ql["B5"]).to_numpy())
    g5["mcs_qlike"] = inf.mcs(ql.to_numpy(), list(ql.columns))
    gm = R[[c for c in R.columns if c.endswith("|gmv")]]
    g5["mcs_gmv_r2"] = inf.mcs((gm ** 2).to_numpy(), [c.split("|")[0] for c in gm.columns])
    pairs = [("M2", "B5"), ("M2", "B3"), ("M2", "B4"), ("M2res", "R0"), ("M3", "B8"), ("M4", "B9"),
             ("M1", "B7p"), ("M8blk", "FF12blk"), ("M5", "PCAk")]
    g5["pares"] = {f"{x}/{y}": _vr(R[f"{x}|gmv"], R[f"{y}|gmv"]) for x, y in pairs}
    # Romano-Wolf: log da razão de variâncias com bootstrap conjunto (mesmos índices)
    rng = np.random.default_rng(SEED)
    A = np.column_stack([R[f"{x}|gmv"] for x, _ in pairs])
    Bm = np.column_stack([R[f"{y}|gmv"] for _, y in pairs])
    stat = -np.log(A.var(axis=0, ddof=1) / Bm.var(axis=0, ddof=1))
    boots = np.empty((2000, len(pairs)))
    for k in range(2000):
        idx = inf.circular_idx(len(A), 10, rng)
        boots[k] = -np.log(A[idx].var(axis=0, ddof=1) / Bm[idx].var(axis=0, ddof=1))
    se = boots.std(axis=0)
    g5["romano_wolf"] = dict(zip([f"{x}/{y}" for x, y in pairs],
                                 inf.romano_wolf(stat / se, (boots - stat) / se).tolist()))
    graph = [c for c in gm.columns if c.split("|")[0].startswith(("M", "R0")) and not c.startswith("M3plac")]
    g5["spa_contra_B5"] = inf.spa((R["B5|gmv"] ** 2).to_numpy(), (R[graph] ** 2).to_numpy())
    g5["dsr_M2"] = inf.deflated_sharpe(R["M2|gmv"].to_numpy(), ch["M2_trials"]["n"], ch["M2_trials"]["var_sr"])
    res["G5"] = g5

    # G7: diferença mensal (B5 - M2) em regimes
    from datetime import date as _date
    from .. import query
    con = query.connect(cfg)
    pn = dados.Panel(cfg, con, tickers=[])
    con.close()
    sp = pn.R[:, pn.col["SPY"]]
    dates = list(ql.index)
    vol = []
    for d in dates:
        i = pn.pos[_date.fromisoformat(d)]
        vol.append(np.nanstd(sp[i - 21:i]))
    dev_cut = []
    for k, d in enumerate(pn.days):
        if DEV[0] <= str(d) <= DEV[1] and (k == 0 or d.month != pn.days[k - 1].month):
            dev_cut.append(np.nanstd(sp[k - 21:k]))
    cuts = np.quantile(dev_cut, [1 / 3, 2 / 3])
    terc = np.searchsorted(cuts, vol)
    gv = R.groupby(R.index.to_period("M")).var()
    import pandas as pd
    per = [pd.Period(d, "M") for d in dates]
    dq = (ql["B5"] - ql["M2"]).to_numpy()
    dv = np.array([gv.loc[x, "B5|gmv"] - gv.loc[x, "M2|gmv"] for x in per])
    ep = np.array([[d[:7] in ("2018-10", "2018-11", "2018-12"), d[:7] in ("2020-01", "2020-02", "2020-03"),
                    d[:4] == "2022"] for d in dates], float)
    X = np.column_stack([np.ones(len(dq)), terc == 1, terc == 2, ep])
    names = ["const(tercil1)", "tercil2", "tercil3", "2018Q4", "2020Q1", "2022"]
    # episódio ou tercil sem nenhum mês no período (2018Q4 fica fora do holdout) sai da regressão
    keep = [k for k in range(X.shape[1]) if k == 0 or X[:, k].any()]
    X = X[:, keep]
    g7 = {}
    for nm, y in (("qlike", dq), ("var_gmv", dv)):
        bh, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ bh
        V = inf.nw_cov(X * e[:, None], 4) * len(y)
        XtXi = np.linalg.inv(X.T @ X / len(y))
        se_ = np.sqrt(np.diag(XtXi @ V @ XtXi) / len(y))
        g7[nm] = {n: None for n in names}
        g7[nm].update({names[k]: {"coef": float(c), "t": float(c / s) if s > 0 else None}
                       for k, c, s in zip(keep, bh, se_)})
    res["G7"] = g7

    # G10, G12 (placebos)
    pd_ = wf_dir(cfg, "placebos")
    if (pd_ / "diario.parquet").exists():
        P = pq.read_table(pd_ / "diario.parquet").to_pandas()
        v_m2 = P["M2_0"].var()
        g10 = {"var_M2": float(v_m2)}
        for k in ("P1", "P2", "P3", "P4"):
            vs = np.array([P[f"{k}_{r}"].var() for r in range(R_PLACEBO) if f"{k}_{r}" in P])
            g10[k] = {"p_emp": float((1 + np.sum(vs <= v_m2)) / (1 + len(vs))),
                      "pct5": float(np.quantile(vs, 0.05)), "mediana": float(np.median(vs))}
        for k in ("P5", "P6", "P6b"):
            g10[k] = _vr(P["M2_0"], P[f"{k}_0"])
        g10["p"] = max(g10["P2"]["p_emp"], g10["P6"]["p"] if g10["P6"]["var_ratio"] < 1 else 1.0)
        res["G10"] = g10
        g = json.loads((pd_ / "g12.json").read_text())
        real = np.mean(g["real"])
        null = np.array(g["P2"]).mean(axis=0)
        res["G12"] = {"cos_medio": float(real), "p": float((1 + np.sum(null >= real)) / (1 + len(null))),
                      "p95_placebo": float(np.quantile(null, 0.95)), "q": g["q"]}

    # G11
    d_ll = -0.5 * (ql["M4"] - ql["B9"]).to_numpy() if "M4" in ql and "B9" in ql else None
    g11 = {}
    if d_ll is not None:
        dm = inf.dm_hln(-d_ll)
        g11 = {"ganho_loglik": float(d_ll.mean()), "dm": dm, "p": dm["p_b_better"]}
    b9 = me[me["est"] == "B9"]
    if "cmh_a" in b9:
        tabs = [np.array([[r.cmh_a, r.cmh_b], [r.cmh_c, r.cmh_d]], float) for r in b9.itertuples()]
        g11["cmh"] = inf.cmh(tabs)
    res["G11"] = g11

    # G13
    g13 = _vr(R[f"M8|hrp_a"], R["B1|hrp"])
    g13["hrp_b"] = _vr(R[f"M8|hrp_b={ch['M8b']['lam']}"], R["B1|hrp"])
    g13["hrp_c"] = _vr(R["M8|hrp_c"], R["B1|hrp"])
    m8 = me[me["est"] == "M8blk"]
    g13["particoes"] = {c: float(m8[c].mean()) for c in m8.columns if c.startswith(("ari", "nmi", "n_com"))}
    g13["p"] = g13["p"] if g13["var_ratio"] < 1 else 1.0
    res["G13"] = g13

    # tabela descritiva
    res["tabela"] = descritiva(di, me)
    # robustez
    rob = {}
    for f in sorted(dados.out_dir(cfg).joinpath("wf").glob("rob_*")):
        try:
            d2, _, _ = load(cfg, f.name)
        except FileNotFoundError:
            continue
        Rr = net_returns(d2).dropna()
        pt = [c.split("|")[1] for c in Rr.columns if c.startswith("M2|")][0]
        rob[f.name[4:]] = _vr(Rr[f"M2|{pt}"], Rr[f"B5|{pt}"])
    for c in (0.0, 5.0, 20.0):
        Rc = net_returns(di, c).dropna()
        rob[f"custo={c:g}"] = _vr(Rc["M2|gmv"], Rc["B5|gmv"])
    for y in range(2019, 2024):
        m = R.index.year == y
        rob[f"ano={y}"] = _vr(R.loc[m, "M2|gmv"], R.loc[m, "B5|gmv"])
    res["robustez"] = rob

    # família confirmatória (Holm), com os testes diádicos
    dd = dados.out_dir(cfg) / "diadico" / "sp500_tau=90_norm=idf" / "resultados.json"
    fam = {}
    if dd.exists():
        D = json.loads(dd.read_text())
        fam.update({"G1": D["G1"].get("p"), "G2": D["G2"].get("p"), "G3": D["G3"].get("p"),
                    "G8": D["G8"]["p"], "G9": D["G9"]["p"]})
    fam["G5"] = g5["p"] if g5["var_ratio"] < 1 else 1.0
    fam.update({k: res[k]["p"] for k in ("G10", "G11", "G12", "G13") if k in res and "p" in res[k]})
    keys = [k for k, v in fam.items() if v is not None]
    adj = inf.holm(np.array([fam[k] for k in keys]))
    res["holm"] = {k: {"p": fam[k], "p_holm": float(a)} for k, a in zip(keys, adj)}
    res["G5"]["sucesso"] = bool(g5["var_ratio"] < 1 and g5["p"] < 0.01 and res["holm"].get("G5", {}).get("p_holm", 1) < 0.05)
    out = ROOT / "logs" / "secao18_avaliacao.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"[avaliar] {out}")
    return res


def descritiva(di, me) -> dict:
    """Vol., Sharpe, MDD, CVaR, giro, viés de Barra e razão de previsão por estimador|carteira."""
    from ..backtester import metrics as M
    R = net_returns(di)
    sd = di.assign(key=di["est"] + "|" + di["port"]).pivot_table(index="date", columns="key", values="exante_sd")
    sd.index = R.index
    turn = di.assign(key=di["est"] + "|" + di["port"]).groupby("key")["turnover"].sum()
    years = len(R) / 252
    out = {}
    for c in R.columns:
        r = R[c].dropna().to_numpy()
        s = M.summary(r)
        z = (R[c] / sd[c]).dropna()
        out[c] = {"vol": s["ann_vol"], "sharpe": s["sharpe"], "sortino": s["sortino"], "mdd": s["max_drawdown"],
                  "cvar5": s["cvar95_1d"], "giro_ano": float(turn[c] / years), "vies_barra": float(z.std()),
                  "prev_risco": float((sd[c] * np.sqrt(252)).mean() / s["ann_vol"])}
    for c in ("qlike", "frob_cov", "frob_corr", "spearman", "err_aresta", "err_sem_aresta", "cond"):
        if c in me:
            for e, v in me.groupby("est")[c].mean().items():
                out.setdefault(f"{e}|estat", {})[c] = float(v)
    return out
