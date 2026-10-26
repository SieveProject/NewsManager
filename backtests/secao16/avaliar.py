"""Avaliação da seção 16 de docs/metodo_covariancia.md (fixada antes de rodar).

    python backtests/secao16/avaliar.py

Principal: GMV sem restrição, n = 100, razão de variâncias dos retornos
diários líquidos "setor + BL notícias" / "setor" nos pregões em comum
(bootstrap circular em blocos de 10 dias, 2.000 reamostragens, seed 7, p
bicaudal). Sucesso: razão < 1 com p < 0,0125 (0,05 / 4 desenhos) E menor que
a razão placebo / setor. Confirmação: razão < 1 em n = 50 e n = 200 (sem
exigir significância). O resto é descritivo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from newsmanager import config  # noqa: E402
from newsmanager.backtester import analysis as A, metrics as M  # noqa: E402

ALPHA = 0.0125
COVS = {"bl": "setor + BL notícias", "placebo": "BL placebo", "homogeneo": "BL Ω homogêneo",
        "noticias": "setor + notícias (δ único)", "lw": "LW fator único"}


def compare(cfg, port: str, covs: list[str], suffix: str = "", active: bool = False) -> dict:
    names = {c: f"s16_{port}_{c}{suffix}" for c in ["setor", *covs]}
    runs = A.load_runs(cfg, list(names.values()))
    idx = A.common_days(runs)
    get = lambda c: runs[names[c]].daily.loc[idx]  # noqa: E731
    bench = get("setor")["bench_ret"].to_numpy() if active else 0.0
    base = get("setor")["ret"].to_numpy() - bench
    out = {c: runs[names[c]].summary["net"] for c in names}
    out["info"] = {c: runs[names[c]].rebalances.filter(like="info_") for c in names}
    for c in covs:
        r = M.variance_ratio(get(c)["ret"].to_numpy() - bench, base, np.random.default_rng(7))
        out[c]["test"] = (r["var_ratio"], r["p"])
    return out


def main() -> int:
    cfg = config.load()
    r = compare(cfg, "gmv", ["bl", "placebo", "homogeneo", "noticias", "lw"])
    (v, p), (vp, _) = r["bl"]["test"], r["placebo"]["test"]
    ok = v < 1 and p < ALPHA and v < vp
    print(f"seção 16, GMV sem restrição, n = 100 (razão contra 'setor'; sucesso: < 1, p < {ALPHA} e < placebo)\n")
    print(f"  {'Σ̂':28} {'vol':>7} {'Sharpe':>7} {'razão':>7} {'p':>6}")
    for c in ["setor", *COVS]:
        t = r[c].get("test")
        print(f"  {COVS.get(c, 'fator + resíduo setor'):28} {r[c]['ann_vol']:7.2%} {r[c]['sharpe']:7.3f} "
              + (f"{t[0]:7.4f} {t[1]:6.3f}" if t else ""))
    print(f"\ncritério 1 (principal): {'ATENDIDO' if ok else 'NÃO atendido'}")
    conf = {}
    for n in (50, 200):
        rn = compare(cfg, "gmv", ["bl", "placebo"], f"_n{n}")
        conf[n] = rn["bl"]["test"][0] < 1
        print(f"  n = {n}: BL {rn['bl']['test'][0]:.4f} (p={rn['bl']['test'][1]:.3f}), "
              f"placebo {rn['placebo']['test'][0]:.4f} (p={rn['placebo']['test'][1]:.3f})")
    print(f"critério 2 (confirmação n = 50/200): {'ATENDIDO' if all(conf.values()) else 'NÃO atendido'} {conf}")

    print("\nrobustez (descritiva):")
    for port, label, active in (("minvar", "variância mínima, teto 10%", False), ("minte", "TE mínimo", True)):
        rr = compare(cfg, port, ["bl", "placebo", "homogeneo"], active=active)
        txt = ", ".join(f"{COVS[c]} {rr[c]['test'][0]:.4f} (p={rr[c]['test'][1]:.3f})"
                        for c in ("bl", "placebo", "homogeneo"))
        print(f"  {label}{' (variância ativa)' if active else ''}: {txt}")
    info = r["info"]["bl"]
    print("\nω_c (escala z) e visões por mês, GMV n = 100, meses com visões:")
    tr = info[info["info_n_views"] > 0]
    for g in range(5):
        print(f"  classe {g}: ω mediano {tr[f'info_omega_{g}'].median():.4f}, visões medianas "
              f"{tr[f'info_views_{g}'].median():.0f}")
    print(f"  CG: iterações medianas {tr['info_cg_iter'].median():.0f}; projeção PSD em "
          f"{int(tr['info_psd_projected'].sum())} de {len(tr)} meses")
    print("\nnúmero de condição mensal (mediana):",
          ", ".join(f"{COVS.get(c, 'setor')} {r['info'][c]['info_cond'].median():.0f}" for c in ["setor", *COVS]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
