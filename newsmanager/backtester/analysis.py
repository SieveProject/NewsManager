"""Leitura e comparação de runs (usado pelo notebook e pelo `compare`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from . import metrics
from .engine import runs_dir


@dataclass
class Run:
    name: str
    path: Path
    config: dict
    summary: dict
    daily: object          # pandas.DataFrame indexado por data
    rebalances: object
    _positions: object = None
    _trades: object = None

    @property
    def positions(self):
        """Peso de início do dia, retorno e contribuição por ativo (lido sob demanda)."""
        if self._positions is None:
            import pandas as pd
            self._positions = pd.read_parquet(self.path / "positions.parquet")
            self._positions["date"] = pd.to_datetime(self._positions["date"])
        return self._positions

    @property
    def trades(self):
        if self._trades is None:
            import pandas as pd
            self._trades = pd.read_parquet(self.path / "trades.parquet")
            self._trades["date"] = pd.to_datetime(self._trades["date"])
        return self._trades

    @property
    def has_benchmark(self) -> bool:
        return "bench_nav" in self.daily.columns

    @property
    def benchmark_label(self) -> str:
        return str(self.config["config"]["benchmark"])


def resolve(cfg: Config, name: str) -> Path:
    p = Path(name)
    if (p / "daily.parquet").exists():
        return p
    q = runs_dir(cfg) / name
    if (q / "daily.parquet").exists():
        return q
    raise FileNotFoundError(f"run {name!r} não encontrada (nem {p} nem {q})")


def load(cfg: Config, name: str) -> Run:
    import pandas as pd
    path = resolve(cfg, name)
    daily = pd.read_parquet(path / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    reb = pd.read_parquet(path / "rebalances.parquet")
    reb["date"] = pd.to_datetime(reb["date"])
    return Run(name=path.name, path=path, config=json.loads((path / "config.json").read_text()),
               summary=json.loads((path / "summary.json").read_text()),
               daily=daily.set_index("date"), rebalances=reb.set_index("date"))


def load_runs(cfg: Config, names: list[str]) -> dict[str, Run]:
    return {r.name: r for r in (load(cfg, n) for n in names)}


def common_days(runs: dict[str, Run]):
    idx = None
    for r in runs.values():
        idx = r.daily.index if idx is None else idx.intersection(r.daily.index)
    return idx


def metrics_table(runs: dict[str, Run]):
    """Uma linha por run (medidas líquidas de custo) + o benchmark da primeira."""
    import pandas as pd
    keys = ["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "max_dd_days", "calmar", "skew",
            "excess_kurtosis", "cvar95_1d", "beta", "alpha_ann", "tracking_error", "information_ratio",
            "turnover_per_year", "cost_per_year"]
    rows = {n: {k: r.summary["net"].get(k) for k in keys} for n, r in runs.items()}
    first = next(iter(runs.values()))
    if "benchmark" in first.summary:
        rows[f"benchmark ({first.benchmark_label})"] = {k: first.summary["benchmark"].get(k) for k in keys}
    return pd.DataFrame(rows).T


def compare(runs: dict[str, Run], base: str, block: int = 10, reps: int = 2000, seed: int = 7):
    """Cada run contra `base` nos pregões em comum: razão de variâncias e
    diferença de Sharpe com bootstrap circular em blocos."""
    import pandas as pd
    idx = common_days(runs)
    b = runs[base].daily.loc[idx, "ret"].to_numpy()
    rows = {}
    for n, r in runs.items():
        if n == base:
            continue
        a = r.daily.loc[idx, "ret"].to_numpy()
        rng = np.random.default_rng(seed)
        vr = metrics.variance_ratio(a, b, rng, block, reps)
        sd = metrics.sharpe_diff(a, b, np.random.default_rng(seed), block, reps)
        rows[n] = {"var_ratio": vr["var_ratio"], "var_ci95": vr["ci95"], "var_p": vr["p"],
                   "sharpe_diff": sd["diff"], "sharpe_ci95": sd["ci95"], "sharpe_p": sd["p"],
                   "ann_active_ret": float((a - b).mean() * 252),
                   "active_vol": float((a - b).std(ddof=1) * np.sqrt(252))}
    return pd.DataFrame(rows).T


def sectors(cfg: Config, tickers: list[str], level: str = "ff12_name"):
    """ticker -> setor (news.sectors); 'sem setor' quando falta."""
    from .. import query
    con = query.connect(cfg)
    m = dict(con.execute(f"SELECT ticker, {level} FROM news.sectors WHERE ticker IN (SELECT unnest(?))",
                         [list(tickers)]).fetchall())
    con.close()
    return {t: (m.get(t) or "sem setor") for t in tickers}
