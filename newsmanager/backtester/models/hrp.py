"""Atalho para plugins/hrp.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.hrp:HRP` dos TOMLs já registrados."""

from ..plugin import load_class

HRP = load_class("plugins/hrp.py", "HRP")
