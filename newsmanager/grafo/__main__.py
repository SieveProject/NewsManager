"""python -m newsmanager.grafo <etapa> (seção 18 de docs/metodo_covariancia.md)

Ordem (18.10):
  sectors --sp500 (python -m newsmanager)  setores dos membros do S&P 500
  auditoria, amostra, descritivas          etapas 1 e 2
  diadico-painel, diadico                  G1, G2, G3, G6, G8, G9
  dev                                      grades do dev -> logs/secao18_escolhas.json
  holdout                                  2019-2023, uma vez
  placebos, robustez, fomc, g4             G10, G12, robustez, G4
  avaliar                                  G5, G7, G10-G13 e Holm -> logs/secao18_avaliacao.json
"""

from __future__ import annotations

import argparse
import sys

from ..config import ConfigError, load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m newsmanager.grafo")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auditoria")
    sub.add_parser("amostra")
    sm = sub.add_parser("metricas")
    sm.add_argument("planilhas", nargs="+")
    sm.add_argument("--recall", default=None)
    sub.add_parser("anonimizacao")
    sub.add_parser("descritivas")
    sd = sub.add_parser("diadico-painel")
    sd.add_argument("--tau", type=float, default=90.0)
    sd.add_argument("--norm", default="idf")
    sd.add_argument("--universo", default="sp500")
    sd.add_argument("--first", default="2011-01-01")
    sd.add_argument("--last", default="2023-12-31")
    st = sub.add_parser("diadico")
    st.add_argument("--tag", default="sp500_tau=90_norm=idf")
    st.add_argument("--qap", type=int, default=200)
    sv = sub.add_parser("dev")
    sv.add_argument("--only", nargs="*", default=None)
    sub.add_parser("escolher")
    sh = sub.add_parser("holdout")
    sh.add_argument("--force", action="store_true")
    sub.add_parser("placebos")
    sub.add_parser("robustez")
    sub.add_parser("fomc")
    sub.add_parser("g4")
    sub.add_parser("avaliar")
    a = p.parse_args(argv)
    cfg = load(a.config)
    try:
        if a.cmd in ("auditoria", "amostra", "anonimizacao", "descritivas"):
            from . import extracao
            getattr(extracao, a.cmd)(cfg)
        elif a.cmd == "metricas":
            from . import extracao
            extracao.metricas(a.planilhas, a.recall)
        elif a.cmd == "diadico-painel":
            from . import diadico
            diadico.build(cfg, tau=a.tau, norm=a.norm, universo=a.universo, first=a.first, last=a.last)
        elif a.cmd == "diadico":
            from . import diadico
            diadico.run(cfg, a.tag, qap=a.qap)
        elif a.cmd in ("dev", "escolher", "holdout", "placebos", "robustez", "avaliar"):
            from . import programa
            if a.cmd == "dev":
                programa.dev(cfg, a.only)
            elif a.cmd == "holdout":
                programa.holdout(cfg, a.force)
            else:
                getattr(programa, {"escolher": "escolher"}.get(a.cmd, a.cmd))(cfg)
        elif a.cmd == "fomc":
            from . import eventos
            eventos.build_fomc(cfg)
        elif a.cmd == "g4":
            from . import eventos
            eventos.g4(cfg)
    except ConfigError as e:
        print(f"[grafo] erro de configuração: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
