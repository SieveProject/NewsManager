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
link-prep (Mac) -> link-run (GPU, deploy/run_link.sh) -> link-build
```

- `data/raw` (bronze) -> `data/curated/{documents,mentions,summaries}` (silver) ->
  `data/curated/extraction_units` -> `data/extractions/v=<versão>/{runs,relations}` ->
  `data/curated/relations/v=<versão>` -> views em `data/corpus.duckdb` (schema `news`).
- `link` (`newsmanager/extract/link.py`, `prompts/link.txt`): nome do agente ->
  ticker. O modelo leve só ESCOLHE entre candidatos fechados (enum no schema) ou
  NONE; universo = tickers com preço no FNSPID ∪ `symbols`. Saída em
  `data/curated/entities/v=<versão>/` -> `news.entity_map`, `news.relations_linked`.
  O ticker entre parênteses no nome é só evidência ("crude oil (WTI)" ≠ W&T Offshore).
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

## Acompanhar uma run

```bash
# na VM: resumo de progresso, falhas, ritmo, ETA, GPU, tmux, backup
.venv/bin/python deploy/status.py
# o mesmo resumo é publicado a cada 10 min em gdrive:NewsManager/status.txt
```

Backup: `deploy/backup.sh` (loop no tmux `backup`) copia `data/extractions`,
`data/curated/relations` e `logs` para `gdrive:NewsManager` com rclone, que nunca
apaga nada no destino.
