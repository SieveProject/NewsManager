"""Atalho para plugins/max_sharpe.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.max_sharpe:MaxSharpe` dos TOMLs já registrados."""

from ..plugin import load_class

MaxSharpe = load_class("plugins/max_sharpe.py", "MaxSharpe")
