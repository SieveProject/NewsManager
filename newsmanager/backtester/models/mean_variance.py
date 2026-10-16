"""Atalho para plugins/mean_variance.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.mean_variance:MeanVariance` dos TOMLs já registrados."""

from ..plugin import load_class

MeanVariance = load_class("plugins/mean_variance.py", "MeanVariance")
