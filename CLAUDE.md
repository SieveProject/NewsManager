# NewsManager

Infraestrutura para a tese (TCC, finanças) do Gabriel: transforma o corpus de
notícias FNSPID (`nasdaq_exteral_data.csv`, 21,6 GB, Hugging Face) em tuplas
`(agent_a, agent_b, relation_type, direction, strength) + published_at` via LLM
local (Ollama), para cruzar com um painel de preços. Leia `README.md` (desenho e
números medidos) e `deploy/RUNBOOK.md` (sequência numa GPU alugada).

O estado de uma run em andamento, se houver, fica em `CLAUDE.local.md`.

## Como trabalhar neste repositório

- **Nunca baixe o CSV localmente.** O Mac tem ~11 GB livres; tudo lê por HTTP
  range via DuckDB. O corpus é construído na VM.
- **Quem faz commit e push é o usuário.** Edite no Mac, copie para a VM com
  `scp` se precisar testar já, peça o push e, depois dele, confira com hash que o
  arquivo da VM é idêntico ao `origin/main` antes de `git checkout -- . && git pull`
  na VM. Arquivo novo copiado por scp fica não rastreado na VM: apague-o (após
  conferir o hash) antes do pull.
- **Ações que custam dinheiro ou apagam dados exigem confirmação explícita**:
  criar/destruir instância na Vast, iniciar run completa, reiniciar run.
- Responda em português; o usuário escreve em português.
- Comandos longos na VM sempre dentro de `tmux`, com log em `logs/`.
- `pkill -f "padrão"` via ssh mata o próprio shell remoto se o padrão aparecer no
  comando: use `pkill -f "[p]adrão"`.

## Pipeline

```
probe -> plan -> ingest -> curate -> validate -> serve      python -m newsmanager <etapa>
units [--sample-frac F] -> run -> collect -> filter        python -m newsmanager.extract <etapa>
link-prep (Mac) -> link-run (GPU, deploy/run_link.sh) -> link-verify (GPU) -> link-build (Mac)
graph-build (Mac, ~15 s)                                   python -m newsmanager.extract graph-build
prices -> universe --n N -> sectors -> h1 [--residual] -> backtest --n N (Mac)
backtester run backtests/*.toml -> compare -> notebook    python -m newsmanager.backtester <etapa>
```

- `data/raw` (bronze) -> `data/curated/{documents,mentions,summaries}` (silver) ->
  `data/curated/extraction_units` -> `data/extractions/v=<versão>/{runs,relations}` ->
  `data/curated/relations/v=<versão>` -> views em `data/corpus.duckdb` (schema `news`).
- `link` (`newsmanager/extract/link.py`, `prompts/link.txt`): nome do agente ->
  ticker. O modelo leve só ESCOLHE entre candidatos fechados (enum no schema) ou
  NONE; universo = tickers com preço no FNSPID ∪ `symbols`. Saída em
  `data/curated/entities/v=<versão>/` -> `news.entity_map`, `news.relations_linked`.
  O ticker entre parênteses no nome é só evidência ("crude oil (WTI)" ≠ W&T Offshore).
  Depois do LLM: só `company`/`fund` levam ticker; `check_ticker` exige nome
  compatível ('name') ou evidência forte ('evidence'); `link-verify` (sim/não,
  `prompts/link_verify.txt`) derruba colisões com tickers estrangeiros, exceto
  `verify_exempt` (controladoras e ETFs). Use `news.relations_linked` no backtest.
- `graph` (`newsmanager/extract/graph.py`, API em `newsmanager/graph.py`): nós
  únicos por cima do `entity_id` (canônico só se `compatible` com o nome, prefixo
  `US_`, classes de ação, colisão slug×ticker -> `CORN_COMMODITY`, manuais em
  `prompts/graph_aliases.csv`) + eventos datados com `rel_class` (16 classes) e
  `w = ±strength`. `graph.snapshot(con, asof, window_days, half_life_days)` usa
  só `dt < asof` (não há hora em `published_at`). Use `news.graph_*` para o
  grafo, não `entity_a/entity_b` de `relations_linked` (colidem com tickers).
- `prices` (`newsmanager/prices.py`): `Stock_price/full_history.zip` do FNSPID
  (sha256 = oid LFS `03da4f…`, não o etag xet do CDN) -> `news.prices`,
  `news.returns` (só retornos válidos), `news.price_quality`, `news.trading_calendar`.
  O arquivo são coletas emendadas em 4 datas (`SPLICE_DATES`, retorno do dia inválido) e
  3.137 tickers acabam em 2020-04/07 (`truncated`). `universe` (`newsmanager/universe.py`):
  top-n ações por eventos no grafo, point-in-time. Método da tese e resultados em `docs/metodo_covariancia.md`.
- `sectors` (`newsmanager/sectors.py`): CIK pela SEC (ticker atual, senão busca
  de nomes do EDGAR, conferida contra formerNames) -> SIC -> Fama-French 12/49.
  Exige `NM_SEC_USER_AGENT="Nome e-mail"` (o e-mail NÃO vai para o código).
  `h1` (`newsmanager/portfolio/h1.py`): Fama-MacBeth e R² fora da amostra.
  `backtest` (`newsmanager/portfolio/backtest.py`, ~6 min com n=100): estimadores
  (`estimators.py`, portes de Ledoit-Wolf conferidos no teste) -> GMV com e sem
  venda a descoberto, 2013-2023. O desenho residual (seção 10 do método) foi
  pré-especificado no commit 5c01017: não mudar critérios depois de ver resultado.
- `backtester` (`newsmanager/backtester/`, guia em `backtests/README.md`): motor
  genérico de janela deslizante. Um TOML por run (período, in-sample, frequência,
  benchmark, limites de peso, custo, universo, plugin `módulo:Classe` ou
  `arquivo.py:Classe`). O plugin devolve pesos a partir de `ctx` (só dados < t);
  o motor confere/projeta limites, cobra custo sobre o giro contra os pesos
  derivados e grava P&L diário, posições, trocas e `ctx.info` em
  `data/marts/backtests/<name>/`. `notebook` gera e executa um .ipynb de análise.
  Teste: `tests/test_backtester.py` (contra uma simulação dia a dia independente).
  Modelos: um arquivo por modelo em `plugins/` (variância mínima, Markowitz,
  máximo Sharpe, máxima diversificação, HRP, TE mínimo, 1/N, inverso da vol.,
  paridade de risco); `cov` escolhe o estimador de Σ̂
  (`newsmanager/backtester/models/covariance.py`). Os módulos em
  `newsmanager/backtester/models/` com nome de modelo são só atalhos para
  `plugins/`, mantidos para os TOMLs registrados (seções 12–15): não apague.
  `backtest` (acima) é o experimento fixado da seção 10/11; comparações novas
  vão no `backtester`. Filtro de tickers no DuckDB: lista literal, não
  `IN (SELECT unnest(?))` (sem poda do Parquet: 15 min contra 0,1 s).
- `link_version` = hash de template + modelo + KINDS + universo (`ref_hash`) +
  `CANDIDATES_REV`. As listagens da Nasdaq Trader mudam todo dia: a cópia usada
  fica em `data/reference/` (também no Drive).
- `prompt_version` = hash de prompt + schema + `max_chars` + **modelo**. Trocar
  qualquer um abre um diretório novo; resultados nunca se misturam.
- Testes: `tests/test_e2e_subset.py 3 32` (ingestão real, ~100 MB) e
  `tests/test_extract_e2e.py` (extração com Ollama simulado). Rode na VM.

## Decisões já tomadas (não reabrir sem motivo novo)

- **Modelo: `qwen2.5:14b-instruct`.** Comparado ao `deepseek-r1:14b` nos mesmos
  200 artigos: ~2/3 vs ~1/4 de tuplas válidas, 39% mais rápido. O 7B inventava
  relações. Ver README, "Escolha do modelo".
- **lenta.ru excluído** da extração (`EXCLUDED_DOMAINS` em `units.py`): portal
  russo de notícias gerais, 799 mil documentos, zero tickers.
- **Amostra de 25%** (`--sample-frac 0.25`, 401.609 unidades) para caber no
  crédito. É hash determinístico e aninhado: ampliar depois só extrai o novo.
- **Máximo de 8 tuplas por artigo** (prompt + `maxItems` no schema).
- Ruído residual (métricas, grupos genéricos, fontes, ETF→holding…) é tratado
  pelo `filter` (`newsmanager/extract/noise.py`), que **marca** e não apaga:
  use `news.relations_clean` (86,2% das relações) na análise. Indicadores
  macro e setores NÃO são ruído — não amplie as listas sem medir amostras.
- **Link: `qwen2.5:7b-instruct`** (só escolhe num enum fechado). No piloto de
  400 entidades o 3B teve a mesma velocidade e classificou tipos pior.

## Estado atual (2026-10-16)

- Extração `v=a05d0c0dc189`: 572.762 unidades (~34% do corpus; amostra de 25%
  ampliada para 40% e parada por crédito em 2026-10-15), 1.270.582 relações,
  86,1% limpas. Tudo no Drive (`gdrive:NewsManager`). A amostra de 25%
  (401.609 unidades) e os produtos derivados dela estão em `data/archive/*_25pct/`.
- Link `l=c49291db2953` completo para os 304.430 nomes (0 falhas): 773.440
  menções com ticker (33,9%). O verify cobre só os nomes da amostra de 25%: 434
  tickers novos aceitos por evidência (875 menções) ficaram sem verify
  (declarado na seção 13 do método). Runs `l=863c9944d819` e `l=0a3720a0b0ce`
  são do piloto (prompt antigo): não usar no build.
- Grafo: 1.064.801 eventos limpos, 255.384 nós. Resultados da tese até aqui:
  seções 8–13 de `docs/metodo_covariancia.md` (H1 confirmada, H2 rejeitada nos
  três desenhos).
- Nenhuma instância Vast ativa.
- Limitações conhecidas: tickers pequenos usados como nome (AVGO, DHI, RIG, ~250
  menções) e empresas renomeadas (Momo, FTI) ficam sem ticker; IDs canônicos não
  fundem todos os sinônimos ("U.S. Federal Reserve" vs "Federal Reserve"); o
  painel de preços do FNSPID não tem UNH, VZ, MRNA, SPG, CELG, SNY, YHOO, BRK, SNAP.

## Armadilhas já encontradas (o código já trata; não desfaça)

Todas apareceram só no corpus inteiro ou só no Linux/container da Vast:

- **Terceiro formato do CSV** após ~17,7 GB: índice vazio (`,<timestamp>,`). A
  âncora em `shard.py` aceita índice opcional.
- **Leitor CSV paralelo do DuckDB** erra em campos com quebra de linha:
  `parallel=false, strict_mode=true` em `ingest.py`.
- **`fork` trava** o `ProcessPoolExecutor` no Linux (threads do DuckDB): `spawn`.
- **Containers enxergam as CPUs do host** (256 vs cota de 30): `usable_cpus()` lê
  o cgroup; threads do DuckDB limitadas também por memória (1 GB/thread).
  `curate`/`units` usam `NM_HEAVY_MEMORY` (2023 tem 6,3 GB de texto).
- **Ollama no container**: `num_thread` derivado do cgroup (128 threads
  limitavam o decode a ~165 tok/s; 16 dão ~540); contexto fixado com
  `OLLAMA_CONTEXT_LENGTH`; KV cache `q8_0` para 16 slots caberem em 24 GB.
- **`num_ctx` pelo pior caso** (2 caracteres/token no artigo): texto com números
  estourava 4096 em silêncio. Fórmula duplicada em `cli.py` e `deploy/lib.sh`.
- **`keep_alive` deve ser inteiro** (`-1`); a string `"-1"` dá HTTP 400.
- **DeepSeek-R1 ignora `think: false`**: o cliente usa `/api/generate` raw com
  `<think></think>` vazio para modelos `deepseek-r1`.
- **WAL muda durante o upload**: `backup.sh` envia uma cópia instantânea.
- **DuckDB 1.5**: banco não pode se chamar `news.duckdb` (catálogo `news`
  conflita com o schema `news`); é `data/corpus.duckdb`.
- Doc sem URL pode reaparecer em outro ano: `curate` pula `doc_id` já gravado.
- **Disco do Mac**: tabelas intermediárias do DuckDB sobre 21 milhões de linhas
  transbordaram para `.tmp/` e encheram o disco (nem o Claude Code conseguia
  rodar comandos). Consultas grandes no Mac: CTEs + COPY direto para Parquet,
  `memory_limit` e `max_temp_directory_size` (ver `prices.build`).

## Acompanhar uma run

```bash
# na VM: resumo de progresso, falhas, ritmo, ETA, GPU, tmux, backup
.venv/bin/python deploy/status.py
# o mesmo resumo é publicado a cada 10 min em gdrive:NewsManager/status.txt
```

Backup: `deploy/backup.sh` (loop no tmux `backup`) copia `data/extractions`,
`data/curated/relations` e `logs` para `gdrive:NewsManager` com rclone, que nunca
apaga nada no destino.
