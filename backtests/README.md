# Backtests

Motor em `newsmanager/backtester/`. Cada arquivo `.toml` aqui é uma run.

```bash
python -m newsmanager.backtester run backtests/minvar_setor.toml backtests/minvar_setor_noticias.toml
python -m newsmanager.backtester run backtests/minvar_lw_fator.toml --set max_weight=0.05 --name lw_5pct
python -m newsmanager.backtester compare minvar_setor minvar_setor_noticias minvar_setor_noticias_placebo
python -m newsmanager.backtester notebook minvar_setor minvar_setor_noticias minvar_setor_noticias_placebo
python -m newsmanager.backtester list
```

As runs são gravadas em `data/marts/backtests/<name>/`, e rodar de novo sobrescreve. O notebook vai para
`notebooks/backtest_<runs>.ipynb`, já executado com as figuras. Ele precisa de
`pip install -r requirements-analysis.txt`.

## Configuração

| chave | padrão | significado |
|---|---|---|
| `name` | nome do arquivo | diretório da run |
| `start`, `end` | — | primeiro e último dia da carteira |
| `insample_days` | 252 | pregões ANTES de cada rebalanceamento entregues ao modelo |
| `rebalance` | `"M"` | `D`, `W`, `M`, `Q`, `A` (1º pregão do período) ou `"<k>D"` (a cada k pregões) |
| `benchmark` | `"SPY"` | ticker de `news.returns`, `"equal_weight"` (1/N do mesmo universo, mesmos custos) ou `"none"` |
| `max_weight` | 1.0 | teto por ativo |
| `min_weight` | 0.0 | piso por ativo; negativo permite venda a descoberto |
| `cost_bps` | 10 | custo por unidade de giro Σ\|w_alvo − w_atual\| |
| `universe` | `{ n = 100 }` | `news.universe` (top-n por notícias, point-in-time) ou `{ tickers = [...] }` |
| `min_coverage` | 0.95 | fração mínima de retornos válidos na janela para o ativo ser elegível |
| `on_violation` | `"project"` | pesos fora dos limites: projetar no conjunto viável ou `"error"` |
| `risk_free` | 0.0 | taxa anual constante (só Sharpe/Sortino) |
| `[model] plugin` | — | `"pacote.módulo:Classe"` ou `"arquivo.py:Classe"` (relativo à raiz) |
| `[model] params` | `{}` | argumentos do construtor do plugin |

## Convenções do motor

- **Sem olhar o futuro.** O modelo do pregão t recebe só retornos de t−`insample_days` a t−1. A troca acontece
  no fechamento de t−1 e a carteira rende a partir do retorno de t. O universo vem de `news.universe` com
  `rebalance_date <= t`.
- **Custos.** Na troca, `NAV ← NAV × (1 − cost_bps/1e4 × giro)`. O giro é medido contra os pesos já derivados
  pelos preços, e um ativo que sai do universo é vendido. O primeiro rebalanceamento compra tudo (giro 1) e
  também paga custo.
- **Entre rebalanceamentos a carteira é comprada e mantida**, e os pesos derivam com os preços. Um retorno
  faltante (dia de emenda do arquivo, deslistagem) conta como 0, para a carteira e para o benchmark.
- **Pesos** somam 1 (carteira totalmente investida, sem caixa) e respeitam os limites.

## Saída de uma run

| arquivo | conteúdo |
|---|---|
| `daily.parquet` | um pregão por linha: `ret` (líquido), `ret_gross`, `cost`, `turnover`, `rebalance`, `nav`, `drawdown`, `bench_ret`, `bench_nav`, `n_positions`, `gross_exposure`, `max_weight` |
| `positions.parquet` | por pregão e ativo: peso no início do dia, retorno e contribuição (soma = `ret_gross`) |
| `trades.parquet` | por rebalanceamento e ativo: `w_before` (derivado) e `w_after` (alvo) |
| `rebalances.parquet` | por rebalanceamento: universo, giro, custo, projeção, N efetivo, pesos extremos e as colunas `info_*` do plugin |
| `summary.json` | medidas líquidas, brutas e do benchmark (CAGR, vol., Sharpe, Sortino, drawdown, Calmar, CVaR, beta, alfa, TE, IR, giro, custo...) |
| `config.json` | configuração resolvida, sha256 do arquivo do plugin, commit do git, data |

## Escrevendo um modelo

Veja `plugins/exemplo_risk_parity.py` e a docstring de `newsmanager/backtester/plugin.py`. Mínimo:

```python
from newsmanager.backtester import Model

class MeuModelo(Model):
    def __init__(self, janela=60):
        self.janela = janela

    def weights(self, ctx):
        X = ctx.returns[-self.janela:]          # pregões x ativos, NaN onde falta
        ...
        return w                                # alinhado a ctx.tickers; ou {ticker: peso}
```

`ctx` também traz `tickers`, `dates`, `prev_weights` (pesos atuais, para controlar o giro), `min_weight`,
`max_weight`, `benchmark_returns` e `env` (`env.con` é o DuckDB somente leitura com grafo e setores;
use só dados com data < `ctx.date`). O que o modelo puser em `ctx.info` aparece em `rebalances.parquet`
e nos gráficos de diagnóstico do notebook.

## Modelos prontos

Um plugin por módulo em `newsmanager/backtester/models/`:

| plugin | params |
|---|---|
| `equal_weight:EqualWeight` | — |
| `inverse_volatility:InverseVolatility` | — |
| `min_variance:MinVariance` | `cov`, `cov_params` |
| `mean_variance:MeanVariance` | `cov`, `cov_params`, `gamma` (5) |
| `max_sharpe:MaxSharpe` | `cov`, `cov_params` |
| `max_diversification:MaxDiversification` | `cov`, `cov_params` |
| `hrp:HRP` | `cov`, `cov_params` |
| `min_tracking_error:MinTrackingError` | `cov`, `cov_params` (precisa de `benchmark = <ticker>`) |

`cov` escolhe o estimador de Σ̂ (`models/covariance.py`):
- estatísticos: `amostral`, `LW identidade`, `LW fator único`, `LW corr. constante`, `LW não linear`, `RMT`,
  `fator + resíduo diagonal`, `fator + resíduo corr. constante`;
- com setor e notícias: `fator + resíduo setor`, `fator + resíduo notícias`, `fator + resíduo setor + notícias`.

Nos estimadores com notícias, `cov_params` aceita `placebo`, `news_days`, `half_life`, `min_train`, `panel_n`
e `seed`. Exemplo:

```toml
[model]
plugin = "newsmanager.backtester.models.hrp:HRP"
params = { cov = "fator + resíduo setor + notícias", cov_params = { placebo = true } }
```

`basic:MinVariance` (`estimator`) e `news:NewsFactorResidual` (`features`) são os nomes antigos, mantidos
para os TOMLs das seções 12–13.

O estimador `fator + resíduo ...` é o modelo da seção 10 de `docs/metodo_covariancia.md`. O β da regressão é treinado só
nos meses do painel cujo período de carteira terminou antes de t, o que vale para qualquer frequência de
rebalanceamento. O painel fica em cache em `data/marts/backtests/_cache/` e é recalculado quando grafo,
universo ou setores mudam.
