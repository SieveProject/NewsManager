"""python -m newsmanager.backtester {run,compare,notebook,list,stability}"""

from __future__ import annotations

import argparse
import sys

from .. import config
from ..config import EXIT_CONFIG, ConfigError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m newsmanager.backtester")
    p.add_argument("--config", default=None, help="config.toml do NewsManager (caminhos de dados)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sr = sub.add_parser("run", help="roda um ou mais backtests a partir de arquivos TOML")
    sr.add_argument("files", nargs="+")
    sr.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                    help="sobrescreve o TOML (ex.: --set max_weight=0.05 --set model.params.estimator='RMT')")
    sr.add_argument("--name", help="nome da run (só com um arquivo); padrão: name do TOML")

    sc = sub.add_parser("compare", help="tabela de medidas e testes contra uma run-base")
    sc.add_argument("runs", nargs="+")
    sc.add_argument("--base", help="run de referência (padrão: a primeira)")

    sn = sub.add_parser("notebook", help="gera (e executa) um notebook de análise das runs")
    sn.add_argument("runs", nargs="+")
    sn.add_argument("--base", help="run de referência nas comparações (padrão: a primeira)")
    sn.add_argument("--out", help="arquivo .ipynb (padrão: notebooks/<runs>.ipynb)")
    sn.add_argument("--no-execute", action="store_true", help="só escreve, sem executar as células")

    sub.add_parser("list", help="runs já gravadas")

    ss = sub.add_parser("stability", help="número de condição diário de Σ̂ por estimador (stability.py)")
    ss.add_argument("action", choices=["run", "plot", "summary"])
    ss.add_argument("--n", type=int, default=100, help="universo news.universe n")
    ss.add_argument("--window", type=int, default=252, help="pregões de retornos (e de notícias) antes de t")
    ss.add_argument("--start", default="2011-01-01")
    ss.add_argument("--end", default="2023-12-31")
    ss.add_argument("--workers", type=int, default=4, help="anos em paralelo")
    ss.add_argument("--force", action="store_true", help="refaz os anos já gravados")
    ss.add_argument("--measure", default="cond_cov", choices=["cond_cov", "cond_corr"],
                    help="covariância (padrão) ou correlação implícita")

    args = p.parse_args(argv)
    cfg = config.load(args.config)
    try:
        if args.cmd == "run":
            from . import engine, settings
            if args.name and len(args.files) > 1:
                raise ConfigError("--name só com um arquivo")
            for f in args.files:
                bt = settings.load(f, args.set)
                if args.name:
                    bt.name = args.name
                engine.run(cfg, bt)
        elif args.cmd == "compare":
            from . import analysis
            import pandas as pd
            runs = analysis.load_runs(cfg, args.runs)
            base = args.base or next(iter(runs))
            with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
                print(analysis.metrics_table(runs).to_string())
                print(f"\ncontra {base} (pregões em comum; bootstrap circular em blocos de 10):")
                print(analysis.compare(runs, base).to_string())
        elif args.cmd == "notebook":
            from . import notebook
            notebook.write(cfg, args.runs, base=args.base, out=args.out, execute=not args.no_execute)
        elif args.cmd == "stability":
            from . import stability
            if args.action == "run":
                stability.run(cfg, n=args.n, window=args.window, start=args.start, end=args.end,
                              workers=args.workers, force=args.force)
            elif args.action == "plot":
                for f in stability.plot(cfg, n=args.n, window=args.window, measure=args.measure):
                    print(f)
            else:
                import pandas as pd
                tab, by_year = stability.summary(cfg, n=args.n, window=args.window, measure=args.measure)
                with pd.option_context("display.width", 250, "display.max_columns", 30,
                                       "display.float_format", "{:.1f}".format):
                    print(tab.to_string())
                    print()
                    print(by_year.T.to_string())
        elif args.cmd == "list":
            import json
            from .engine import runs_dir
            for d in sorted(runs_dir(cfg).glob("*/summary.json")):
                s = json.loads(d.read_text())
                n = s["net"]
                print(f"{s['name']:40} {s['first']} a {s['last']}  CAGR {n['cagr']:7.2%}  vol {n['ann_vol']:6.2%}  "
                      f"Sharpe {n['sharpe'] or 0:5.2f}")
    except ConfigError as e:
        print(f"\n[backtest] erro de configuração: {e}", file=sys.stderr)
        return EXIT_CONFIG
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
