"""Atalho para plugins/max_diversification.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.max_diversification:MaxDiversification` dos TOMLs já registrados."""

from ..plugin import load_class

MaxDiversification = load_class("plugins/max_diversification.py", "MaxDiversification")
