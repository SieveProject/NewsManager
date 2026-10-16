"""Atalho para plugins/min_variance.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.min_variance:MinVariance` dos TOMLs já registrados."""

from ..plugin import load_class

MinVariance = load_class("plugins/min_variance.py", "MinVariance")
