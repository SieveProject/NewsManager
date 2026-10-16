"""Modelos-plugin prontos, um por módulo: `newsmanager.backtester.models.<módulo>:<Classe>`.

  equal_weight:EqualWeight                 1/N
  inverse_volatility:InverseVolatility     peso ∝ 1/σ
  min_variance:MinVariance                 variância mínima
  mean_variance:MeanVariance               Markowitz com μ de Bayes-Stein
  max_sharpe:MaxSharpe                     carteira tangente com μ de Bayes-Stein
  max_diversification:MaxDiversification   Choueifaty e Coignard (2008)
  hrp:HRP                                  López de Prado (2016)
  min_tracking_error:MinTrackingError      contra o benchmark do backtest

Todos os que usam Σ̂ recebem `cov` (estimador, covariance.py) e, para os
estimadores de notícias, `cov_params`. basic.py e news.py mantêm os nomes
antigos (seção 12).
"""
