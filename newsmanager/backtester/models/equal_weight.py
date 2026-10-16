"""Atalho para plugins/equal_weight.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.equal_weight:EqualWeight` dos TOMLs já registrados."""

from ..plugin import load_class

EqualWeight = load_class("plugins/equal_weight.py", "EqualWeight")
