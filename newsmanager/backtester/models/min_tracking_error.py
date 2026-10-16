"""Atalho para plugins/min_tracking_error.py (onde está a implementação): mantém o caminho
`newsmanager.backtester.models.min_tracking_error:MinTrackingError` dos TOMLs já registrados."""

from ..plugin import load_class

MinTrackingError = load_class("plugins/min_tracking_error.py", "MinTrackingError")
