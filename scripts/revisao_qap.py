"""Análise posterior (2026-11-10): teste de permutação QAP com mais réplicas.

Completa a inferência dos testes de pares no painel já gravado
(data/marts/grafo/diadico/sp500_tau=90_norm=idf), sem mudar a regressão:

  h1a  coeficiente da exposição comum g na regressão [sgnlog, g, controles]
       (no pré-registro o QAP só foi calculado para a aresta direta);
  h1b  Wald da igualdade dos seis coeficientes de grupo (no pré-registro,
       200 permutações: o menor p possível era 1/201).

Uso:  PYTHONPATH=. .venv/bin/python scripts/revisao_qap.py {h1a|h1b} SEMENTE R SAIDA.json
Rode várias sementes em paralelo e junte com  scripts/revisao_qap.py juntar ARQ... 
"""
import json
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")


def main() -> None:
    if sys.argv[1] == "juntar":
        obs, null, nome = None, [], None
        for f in sys.argv[2:]:
            d = json.load(open(f))
            obs, nome = d["obs"], d["teste"]
            null += d["null"]
        null = np.array(null)
        print(json.dumps({"teste": nome, "obs": obs, "permutacoes": len(null),
                          "p_qap": float((1 + np.sum(null >= obs)) / (1 + len(null))),
                          "null_max": float(null.max()), "null_p99": float(np.quantile(null, 0.99))}, indent=2))
        return
    from newsmanager.config import load
    from newsmanager.grafo import diadico as D
    teste, seed, R, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    cfg = load(None)
    tag = "sp500_tau=90_norm=idf"
    D.SEED = seed
    if teste == "h1a":
        xs = ["sgnlog", "g", *D.CONTROLS]
        r = D.fama_macbeth(cfg, tag, "z_fut21", xs, qap=R)
        obs = float(r["coef"][:, 1].mean())
        null = r["perm"][:, :, 1].mean(axis=1)
    else:
        gx = [f"g_{x}" for x in D.GROUPS6]
        xs = [*gx, f"g_{D.rede.OUTROS}", "g", *D.CONTROLS]
        r = D.fama_macbeth(cfg, tag, "z_fut21", xs, qap=R)
        obs = D._wald_equal(r["coef"][:, :6], D._lags(21))
        null = np.array([D._wald_equal(P[:, :6], D._lags(21)) for P in r["perm"]])
    json.dump({"teste": teste, "seed": seed, "obs": obs, "null": [float(x) for x in null]}, open(out, "w"))


if __name__ == "__main__":
    main()
