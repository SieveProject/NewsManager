"""Configuração de um backtest (arquivo TOML + sobrescritas da linha de comando)."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from ..config import ConfigError

FREQS = ("D", "W", "M", "Q", "A")


@dataclass
class BacktestConfig:
    name: str
    start: date
    end: date
    insample_days: int = 252
    rebalance: str = "M"                 # D, W, M, Q, A ou "<k>D" (a cada k pregões)
    benchmark: str = "SPY"               # ticker, "equal_weight" ou "none"
    max_weight: float = 1.0              # inf = sem teto
    min_weight: float = 0.0              # < 0 permite venda a descoberto; -inf = sem piso
    cost_bps: float = 10.0               # por unidade de giro (Σ|Δw|), cobrado no pregão do rebalanceamento
    universe: dict = field(default_factory=lambda: {"n": 100})   # {"n": N} (news.universe) ou {"tickers": [...]}
    min_coverage: float = 0.95           # fração mínima de retornos válidos na janela in-sample
    on_violation: str = "project"        # pesos fora dos limites: "project" ou "error"
    risk_free: float = 0.0               # taxa anual constante, só para Sharpe/Sortino
    model: dict = field(default_factory=dict)   # {"plugin": "módulo:Classe" ou "arq.py:Classe", "params": {...}}

    def __post_init__(self):
        self.start, self.end = _date(self.start), _date(self.end)
        if self.end <= self.start:
            raise ConfigError(f"end ({self.end}) <= start ({self.start})")
        if self.insample_days < 2:
            raise ConfigError("insample_days < 2")
        self.rebalance = str(self.rebalance).upper()
        if self.rebalance not in FREQS and not re.fullmatch(r"[1-9]\d*D", self.rebalance):
            raise ConfigError(f"rebalance={self.rebalance!r}: use {', '.join(FREQS)} ou '<k>D'")
        self.max_weight, self.min_weight = float(self.max_weight), float(self.min_weight)
        if self.min_weight >= self.max_weight:
            raise ConfigError("min_weight >= max_weight")
        if self.min_weight > 0:
            raise ConfigError("min_weight > 0 obrigaria a carteira a ter todos os ativos do universo")
        if self.on_violation not in ("project", "error"):
            raise ConfigError("on_violation deve ser 'project' ou 'error'")
        if ("n" in self.universe) == ("tickers" in self.universe):
            raise ConfigError("universe: informe {n = N} ou {tickers = [...]}")
        if not self.model.get("plugin"):
            raise ConfigError("[model] sem 'plugin' (ex.: 'newsmanager.backtester.models.basic:MinVariance')")
        self.model.setdefault("params", {})

    def to_json(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = str(self.start), str(self.end)
        for k in ("max_weight", "min_weight"):
            if d[k] in (float("inf"), float("-inf")):
                d[k] = str(d[k])
        return d


def _date(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))


def _value(text: str):
    """Valor de uma sobrescrita: literal TOML se possível, senão texto."""
    try:
        return tomllib.loads(f"x = {text}")["x"]
    except tomllib.TOMLDecodeError:
        return text


def load(path: str | Path, overrides: list[str] | None = None) -> BacktestConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    raw.setdefault("name", path.stem)
    for o in overrides or []:
        if "=" not in o:
            raise ConfigError(f"--set {o!r}: use chave=valor (ex.: model.params.estimator='RMT')")
        key, val = o.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _value(val.strip())
    for k in ("max_weight", "min_weight"):
        if isinstance(raw.get(k), str):
            raw[k] = float(raw[k])
    known = BacktestConfig.__dataclass_fields__
    extra = set(raw) - set(known)
    if extra:
        raise ConfigError(f"chaves desconhecidas em {path}: {sorted(extra)}")
    return BacktestConfig(**raw)


def dumps(bt: BacktestConfig) -> str:
    return json.dumps(bt.to_json(), indent=2, ensure_ascii=False)
