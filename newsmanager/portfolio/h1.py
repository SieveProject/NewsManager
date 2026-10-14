"""H1: as notícias preveem a correlação futura além da correlação histórica?

Para cada rebalanceamento t do universo (news.universe) e cada par (i, j):

    y = z(ρ_fut)   correlação realizada no período de carteira [t, t seguinte)
    x = z(ρ_hist)  correlação nos 252 pregões antes de t
        + atributos de notícias < t (features.py)

com z = atanh (Fisher). Dois testes, os dois sem olhar o futuro:

  Fama-MacBeth   uma regressão cross-section por data, regressores padronizados
                 na data (coeficiente = efeito de 1 desvio-padrão), média dos
                 coeficientes com t de Newey-West (4 defasagens).
  fora da amostra  em cada t, MQO empilhado nas datas < t (mínimo 24) prevê a
                 cross-section de t (y e x centrados na data: o nível médio da
                 correlação de mercado não é o que se quer prever aqui). R² =
                 1 - SSE/SST somado nas datas de teste; o ganho de cada modelo
                 sobre M0 vem com t de Newey-West da diferença de SSE por data.

Modelos: M0 persistência; M1 + co-menção do FNSPID (Tang et al.); M2 +
co-ocorrência no mesmo artigo extraído; M3 + relações tipadas e exposição; M4
tudo. M3 contra M2 e M4 contra M1/M2 respondem se o CONTEÚDO das relações
importa além de as empresas saírem juntas no texto.
"""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Config, ConfigError
from .. import query
from . import data, features

DIRECT = [f for f in features.FEATURES if f.startswith("direct_")]
SECTOR = ["same_ff49", "same_sic2"]
NEWS = ["comention", "cooccur", *DIRECT, "exposure"]
MODELS = {
    "M0 persistência": ["z_hist"],
    "M1 + co-menção FNSPID": ["z_hist", "comention"],
    "M2 + co-ocorrência LLM": ["z_hist", "cooccur"],
    "M3 + relações e exposição": ["z_hist", *DIRECT, "exposure"],
    "M4 notícias": ["z_hist", *NEWS],
    "S0 persistência + setor": ["z_hist", *SECTOR],
    "S1 + co-menção FNSPID": ["z_hist", *SECTOR, "comention"],
    "S4 + notícias": ["z_hist", *SECTOR, *NEWS],
}
# (modelo, base): o ganho de cada modelo é medido contra a base certa. As
# linhas S* respondem se as notícias acrescentam ALÉM do setor.
COMPARISONS = [
    ("M1 + co-menção FNSPID", "M0 persistência"),
    ("M2 + co-ocorrência LLM", "M0 persistência"),
    ("M3 + relações e exposição", "M0 persistência"),
    ("M4 notícias", "M0 persistência"),
    ("M4 notícias", "M1 + co-menção FNSPID"),
    ("S0 persistência + setor", "M0 persistência"),
    ("S1 + co-menção FNSPID", "S0 persistência + setor"),
    ("S4 + notícias", "S0 persistência + setor"),
    ("S4 + notícias", "S1 + co-menção FNSPID"),
]
MIN_TRAIN = 24
NW_LAGS = 4


def out_dir(cfg: Config):
    return cfg.marts / "h1"


def build_panel(cfg: Config, n: int = 100, news_days: int = 365, half_life_days: float = 90.0,
                insample_days: int = 252, min_fut_obs: int = 15, residual: bool = False) -> dict[str, np.ndarray]:
    """Painel de pares-data. `residual`: correlações dos resíduos ao fator de
    mercado (seção 10 do método); b estimado só na janela in-sample e aplicado
    ao mês seguinte com o m_t do próprio mês."""
    con = query.connect(cfg)
    dates = [d for (d,) in con.execute(
        "SELECT DISTINCT rebalance_date FROM news.universe WHERE n_target = ? ORDER BY 1", [n]).fetchall()]
    if not dates:
        raise ConfigError(f"sem universo n={n}; rode `python -m newsmanager universe --n {n}`")
    end_of_data = con.execute("SELECT max(date) FROM news.trading_calendar").fetchone()[0]
    cols: dict[str, list] = {k: [] for k in ("date", "a", "b", "z_hist", "z_fut", "n_fut", *features.FEATURES)}
    t0 = time.time()
    for k, d in enumerate(dates):
        tk = [r[0] for r in con.execute(
            "SELECT ticker FROM news.universe WHERE n_target = ? AND rebalance_date = ? ORDER BY rank", [n, d]).fetchall()]
        w_in = data.trading_window(con, d, before=insample_days)
        nxt = dates[k + 1] if k + 1 < len(dates) else None
        f_end = (con.execute("SELECT max(date) FROM news.trading_calendar WHERE date < ?", [nxt]).fetchone()[0]
                 if nxt else end_of_data)
        X_in = data.returns_matrix(con, tk, *w_in)
        X_out = data.returns_matrix(con, tk, d, f_end)
        if residual:
            m_in = data.market_factor(X_in)
            a, b = data.market_betas(X_in, m_in)
            X_in = data.residuals(X_in, b, m_in, a)
            X_out = data.residuals(X_out, b, data.market_factor(X_out))
        Ch, _ = data.pairwise_corr(X_in, min_obs=insample_days // 2)
        Cf, Nf = data.pairwise_corr(X_out, min_obs=min_fut_obs)
        F = features.pair_features(con, cfg, d, tk, news_days=news_days, half_life_days=half_life_days)
        i, j = np.triu_indices(len(tk), 1)
        ok = np.isfinite(Ch[i, j]) & np.isfinite(Cf[i, j]) & (np.abs(Ch[i, j]) < 1) & (np.abs(Cf[i, j]) < 1)
        i, j = i[ok], j[ok]
        cols["date"] += [d] * len(i)
        cols["a"] += [tk[x] for x in i]
        cols["b"] += [tk[x] for x in j]
        cols["z_hist"].append(np.arctanh(Ch[i, j]))
        cols["z_fut"].append(np.arctanh(Cf[i, j]))
        cols["n_fut"].append(Nf[i, j])
        for f in features.FEATURES:
            v = F[f][i, j]
            cols[f].append(v if f == "exposure" or f in features.BINARY else np.log1p(v))
    con.close()
    panel = {k: (np.array(v, dtype=object) if k in ("date", "a", "b") else np.concatenate(v))
             for k, v in cols.items()}
    print(f"[h1] painel n={n}{' residual' if residual else ''}: {len(dates)} datas, {len(panel['z_fut']):,} pares-data ({time.time() - t0:.0f}s)")
    return panel


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))


def _nw_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    x = np.asarray(x, float)
    n, e = len(x), x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else float("nan")


def _std(M: np.ndarray) -> np.ndarray:
    sd = M.std(axis=0)
    return np.divide(M - M.mean(axis=0), sd, out=np.zeros_like(M), where=sd > 0)


def fama_macbeth(panel: dict, regs: list[str]) -> dict:
    dates = panel["date"]
    coefs, r2 = [], []
    for d in np.unique(dates):
        m = dates == d
        X = np.column_stack([np.ones(m.sum()), _std(np.column_stack([panel[r][m] for r in regs]))])
        b, r = _ols(X, panel["z_fut"][m])
        coefs.append(b[1:])
        r2.append(r)
    C = np.array(coefs)
    return {"coef": dict(zip(regs, C.mean(axis=0).round(4).tolist())),
            "t": dict(zip(regs, [round(_nw_t(C[:, k]), 2) for k in range(len(regs))])),
            "r2": round(float(np.mean(r2)), 4), "dates": len(r2)}


def oos(panel: dict, models: dict[str, list[str]], comparisons: list[tuple[str, str]],
        min_train: int = MIN_TRAIN) -> dict:
    """R² cross-section fora da amostra (janela expansiva) e ganhos entre modelos."""
    dates = panel["date"]
    ud = np.unique(dates)
    masks = [dates == d for d in ud]
    yc = [panel["z_fut"][m] - panel["z_fut"][m].mean() for m in masks]
    sst = np.array([y @ y for y in yc[min_train:]])
    sse = {}
    for name, regs in models.items():
        Xc = [np.column_stack([panel[r][m] for r in regs]) for m in masks]
        Xc = [x - x.mean(axis=0) for x in Xc]
        XtX, Xty, e2 = np.zeros((len(regs), len(regs))), np.zeros(len(regs)), []
        for k in range(len(ud)):
            if k >= min_train:
                b = np.linalg.solve(XtX + 1e-9 * np.eye(len(regs)), Xty)
                e = yc[k] - Xc[k] @ b
                e2.append(e @ e)
            XtX += Xc[k].T @ Xc[k]
            Xty += Xc[k].T @ yc[k]
        sse[name] = np.array(e2)
    test_dates = ud[min_train:]
    periods = (("2013", "2016"), ("2017", "2019"), ("2020", "2023"))
    out = {"r2_oos": {n: round(1 - s.sum() / sst.sum(), 4) for n, s in sse.items()}, "gains": []}
    for model, base in comparisons:
        diff = sse[base] - sse[model]               # > 0: o modelo erra menos que a base
        per = {}
        for lo, hi in periods:
            m = np.array([lo <= str(d)[:4] <= hi for d in test_dates])
            per[f"{lo}-{hi}"] = round(float(diff[m].sum() / sst[m].sum()), 5)
        out["gains"].append({"model": model, "base": base, "gain": round(float(diff.sum() / sst.sum()), 5),
                             "t": round(_nw_t(diff), 2), "frac_better": round(float(np.mean(diff > 0)), 3),
                             "by_period": per})
    out["_test"] = {"first": str(test_dates[0]), "last": str(test_dates[-1]), "dates": len(test_dates)}
    return out


def run(cfg: Config, n: int = 100, news_days: int = 365, half_life_days: float = 90.0,
        residual: bool = False) -> dict:
    panel = build_panel(cfg, n=n, news_days=news_days, half_life_days=half_life_days, residual=residual)
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    tag = f"n={n}_w={news_days}_h={half_life_days:g}" + ("_res" if residual else "")
    pq.write_table(pa.table({k: (pa.array(list(v)) if v.dtype == object else v) for k, v in panel.items()}),
                   d / f"panel_{tag}.parquet", compression="zstd")
    cov = {f: round(float(np.mean(panel[f] > 0)), 3) for f in features.FEATURES}
    res = {"params": {"n": n, "news_days": news_days, "half_life_days": half_life_days, "residual": residual},
           "pairs": int(len(panel["z_fut"])), "coverage": cov,
           "fama_macbeth": {name: fama_macbeth(panel, regs) for name, regs in MODELS.items()},
           "oos": oos(panel, MODELS, COMPARISONS)}
    (d / f"results_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    report(res)
    print(f"[h1] {d / f'results_{tag}.json'}")
    return res


def report(res: dict) -> None:
    print("\ncobertura (fração dos pares-data com atributo > 0):")
    print("  " + "  ".join(f"{k} {v:.1%}" for k, v in res["coverage"].items()))
    print("\nFama-MacBeth (coeficiente por 1 desvio-padrão; t de Newey-West)")
    for name, r in res["fama_macbeth"].items():
        terms = "  ".join(f"{k}={r['coef'][k]:+.4f} ({r['t'][k]:+.1f})" for k in r["coef"])
        print(f"  {name:28} R²={r['r2']:.4f}  {terms}")
    o = res["oos"]
    print(f"\nfora da amostra, {o['_test']['dates']} datas ({o['_test']['first']} a {o['_test']['last']})")
    for name, r2 in o["r2_oos"].items():
        print(f"  {name:28} R²oos={r2:.4f}")
    print("\nganhos fora da amostra (fração de SST; t de Newey-West da diferença de SSE por data)")
    for g in o["gains"]:
        per = "  ".join(f"{k} {v:+.4f}" for k, v in g["by_period"].items())
        print(f"  {g['model']:26} vs {g['base']:26} {g['gain']:+.4f} (t={g['t']:+.1f}, "
              f"melhor em {g['frac_better']:.0%})  {per}")
