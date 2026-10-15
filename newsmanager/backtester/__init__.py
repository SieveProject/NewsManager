"""Backtest de carteiras por janela deslizante com modelos-plugin.

    python -m newsmanager.backtester run backtests/<config>.toml [--set chave=valor ...]
    python -m newsmanager.backtester compare <run> <run> ... [--base <run>]
    python -m newsmanager.backtester notebook <run> <run> ... [--out notebooks/x.ipynb]

Ver backtests/README.md.
"""

from .plugin import Context, Env, Model  # noqa: F401
