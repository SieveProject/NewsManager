"""Interface dos modelos-plugin.

Um modelo é uma classe (ou função) que, a cada rebalanceamento, recebe um
`Context` com a janela in-sample e devolve os pesos da carteira:

    from newsmanager.backtester import Model

    class MeuModelo(Model):
        def __init__(self, lookback=60):          # parâmetros vêm de [model.params]
            self.lookback = lookback

        def setup(self, env):                     # opcional: uma vez, antes do 1º rebalanceamento
            ...

        def weights(self, ctx):                   # obrigatório
            X = ctx.returns[-self.lookback:]      # pregões x ativos, NaN onde falta
            ...
            return w                              # vetor alinhado a ctx.tickers, ou {ticker: peso}

Regras que o motor garante e confere:

  - `ctx.returns` só tem pregões ANTERIORES a `ctx.date`. A carteira passa a
    render a partir do retorno de `ctx.date` (decisão no fechamento do pregão
    anterior). Nada do período fora da amostra chega ao modelo.
  - Os pesos somam 1 (carteira totalmente investida) e respeitam
    `ctx.min_weight <= w <= ctx.max_weight`. Com `on_violation = "project"`
    o motor projeta pesos inválidos no conjunto viável (e conta quantas vezes);
    com "error", interrompe.
  - O que o modelo puser em `ctx.info` (números) vira coluna em
    rebalances.parquet: delta do shrinkage, número de condição, etc.
  - `env.con` é uma conexão somente leitura com data/corpus.duckdb (grafo de
    notícias, setores...): use só dados com data < ctx.date.

Uma função `def modelo(ctx, **params) -> pesos` também serve como plugin.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ConfigError


@dataclass
class Env:
    """O que vale para a run inteira (entregue a `Model.setup`)."""
    cfg: Any                       # newsmanager.config.Config
    con: Any                       # duckdb, somente leitura
    bt: Any                        # BacktestConfig
    rebalance_dates: list[date]
    universe_n: int | None         # n do news.universe, se o universo vier de lá


@dataclass
class Context:
    """Uma data de rebalanceamento."""
    date: date                     # 1º pregão da nova carteira
    tickers: list[str]             # ativos elegíveis (universo point-in-time ∩ cobertura)
    returns: np.ndarray            # insample_days x p, retornos simples diários, NaN onde falta
    dates: list[date]              # pregões das linhas de `returns` (todos < date)
    prev_weights: np.ndarray       # pesos atuais (já derivados pelos preços), alinhados a tickers
    min_weight: float
    max_weight: float
    benchmark_returns: np.ndarray | None   # mesma janela; None se benchmark = none/equal_weight
    env: Env
    info: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.tickers)

    def frame(self):
        """`returns` como DataFrame do pandas (datas x tickers)."""
        import pandas as pd
        return pd.DataFrame(self.returns, index=pd.DatetimeIndex(self.dates), columns=self.tickers)


class Model:
    """Base dos plugins. Só `weights` é obrigatório."""

    def setup(self, env: Env) -> None:
        pass

    def weights(self, ctx: Context) -> np.ndarray | dict[str, float]:
        raise NotImplementedError


class _FunctionModel(Model):
    def __init__(self, fn, **params):
        self.fn, self.params = fn, params

    def weights(self, ctx):
        return self.fn(ctx, **self.params)


def _import(spec: str):
    """'pacote.módulo:Nome' ou 'caminho/arquivo.py:Nome'."""
    if ":" not in spec:
        raise ConfigError(f"plugin {spec!r}: use 'módulo:Classe' ou 'arquivo.py:Classe'")
    mod, name = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        path = Path(mod)
        if not path.is_absolute() and not path.exists():   # relativo à raiz do repositório
            path = Path(__file__).resolve().parents[2] / mod
        path = path.resolve()
        if not path.exists():
            raise ConfigError(f"plugin: {path} não existe")
        modname = f"nm_plugin_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
        s = importlib.util.spec_from_file_location(modname, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[modname] = m
        s.loader.exec_module(m)
        source = path
    else:
        m = importlib.import_module(mod)
        source = Path(inspect.getfile(m))
    if not hasattr(m, name):
        raise ConfigError(f"plugin: {mod} não tem {name!r}")
    return getattr(m, name), source


def load_model(spec: str, params: dict) -> tuple[Model, dict]:
    """Instancia o plugin; devolve também a procedência (arquivo e sha256)."""
    obj, source = _import(spec)
    if inspect.isclass(obj):
        model = obj(**params)
        if not callable(getattr(model, "weights", None)):
            raise ConfigError(f"plugin {spec}: a classe não tem weights(ctx)")
    elif callable(obj):
        model = _FunctionModel(obj, **params)
    else:
        raise ConfigError(f"plugin {spec}: não é classe nem função")
    prov = {"plugin": spec, "file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return model, prov
