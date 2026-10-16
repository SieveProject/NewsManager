"""Infraestrutura dos modelos. As implementações estão em plugins/ na raiz do
repositório, um arquivo por modelo (use `plugins/<arquivo>.py:<Classe>`):

  equal_weight.py:EqualWeight                 1/N
  inverse_volatility.py:InverseVolatility     peso ∝ 1/σ
  min_variance.py:MinVariance                 variância mínima
  mean_variance.py:MeanVariance               Markowitz com μ de Bayes-Stein
  max_sharpe.py:MaxSharpe                     carteira tangente com μ de Bayes-Stein
  max_diversification.py:MaxDiversification   Choueifaty e Coignard (2008)
  hrp.py:HRP                                  López de Prado (2016)
  min_tracking_error.py:MinTrackingError      contra o benchmark do backtest
  risk_parity.py:RiskParity                   paridade de risco (exemplo)

Aqui ficam:
  base.py         CovModel (escolhe a Σ̂ pelo parâmetro `cov`) e describe()
  covariance.py   estimadores de Σ̂, estatísticos e com setor/notícias
  <modelo>.py     atalhos para plugins/<modelo>.py, só para os TOMLs já
                  registrados (seções 12–15)
  basic.py, news.py  nomes antigos da seção 12
"""
