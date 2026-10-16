"""Avaliação da seção 14 de docs/metodo_covariancia.md (fixada antes de rodar).

    python backtests/secao14/avaliar.py

Para cada otimizador, a estatística principal compara Σ̂ "setor + notícias"
com Σ̂ "setor" nos pregões em comum (bootstrap circular em blocos de 10 dias,
2.000 reamostragens, seed 7, p bicaudal):

  mv, maxsharpe   diferença de Sharpe líquido (notícias - setor) > 0
  maxdiv, hrp     razão de variâncias dos retornos líquidos < 1
  minte           razão de variâncias do retorno ativo (r - SPY) < 1

Sucesso do otimizador: melhora com p < 0,01 (Bonferroni: 0,05 / 5) E melhor
que o placebo na mesma estatística. Confirmação: mesma direção em n = 50 e
n = 200 (sem exigir significância).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from newsmanager import config  # noqa: E402
from newsmanager.backtester import analysis as A, metrics as M  # noqa: E402

ALPHA = 0.01
STAT = {"mv": "sharpe", "maxsharpe": "sharpe", "maxdiv": "var", "hrp": "var", "minte": "te"}
NAMES = {"mv": "Markowitz (γ = 5)", "maxsharpe": "máximo Sharpe", "maxdiv": "máxima diversificação",
         "hrp": "HRP", "minte": "tracking error mínimo"}


def stat(kind: str, a, b, bench):
    rng = np.random.default_rng(7)
    if kind == "sharpe":
        r = M.sharpe_diff(a, b, rng)
        return r["diff"], r["p"], r["diff"] > 0
    x, y = (a - bench, b - bench) if kind == "te" else (a, b)
    r = M.variance_ratio(x, y, rng)
    return r["var_ratio"], r["p"], r["var_ratio"] < 1


def compare(cfg, m: str, suffix: str = "") -> dict:
    names = {c: f"s14_{m}_{c}{suffix}" for c in ("setor", "noticias", "placebo")}
    runs = A.load_runs(cfg, list(names.values()))
    idx = A.common_days(runs)
    get = lambda c: runs[names[c]].daily.loc[idx]  # noqa: E731
    bench = get("setor")["bench_ret"].to_numpy()
    base = get("setor")["ret"].to_numpy()
    out = {"vol": {c: runs[names[c]].summary["net"]["ann_vol"] for c in names},
           "sharpe": {c: runs[names[c]].summary["net"]["sharpe"] for c in names},
           "te": {c: runs[names[c]].summary["net"]["tracking_error"] for c in names}}
    for c in ("noticias", "placebo"):
        out[c] = stat(STAT[m], get(c)["ret"].to_numpy(), base, bench)
    return out


def main() -> int:
    cfg = config.load()
    print(f"seção 14: sucesso = melhora com p < {ALPHA} e melhor que o placebo; confirmação em n = 50/200\n")
    for m in STAT:
        r = compare(cfg, m)
        (v, p, good), (vp, pp, _) = r["noticias"], r["placebo"]
        better_than_placebo = v > vp if STAT[m] == "sharpe" else v < vp
        ok = good and p < ALPHA and better_than_placebo
        conf = [compare(cfg, m, f"_n{n}")["noticias"][2] for n in (50, 200)]
        print(f"{NAMES[m]:24} [{STAT[m]}] notícias {v:+.4f} (p={p:.3f})  placebo {vp:+.4f} (p={pp:.3f})  "
              f"-> {'ATENDIDO' if ok else 'não atendido'}; confirmação n50/n200: {conf}")
        print(f"{'':24} vol setor/notícias/placebo {r['vol']['setor']:.2%} / {r['vol']['noticias']:.2%} / "
              f"{r['vol']['placebo']:.2%}; Sharpe {r['sharpe']['setor']:.3f} / {r['sharpe']['noticias']:.3f} / "
              f"{r['sharpe']['placebo']:.3f}; TE {r['te']['setor']:.2%} / {r['te']['noticias']:.2%}")
    print("\nrobustez Markowitz (descritiva):")
    for g in (2, 10):
        r = compare(cfg, "mv", f"_g{g}")
        print(f"  γ = {g:2}: Δ Sharpe notícias {r['noticias'][0]:+.4f} (p={r['noticias'][1]:.3f}), "
              f"placebo {r['placebo'][0]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
