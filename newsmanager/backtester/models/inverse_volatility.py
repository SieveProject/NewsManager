"""Atalho para plugins/inverse_volatility.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.inverse_volatility:InverseVolatility` dos TOMLs já registrados."""

from ..plugin import load_class

InverseVolatility = load_class("plugins/inverse_volatility.py", "InverseVolatility")
