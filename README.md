# NewsManager

A DuckDB pipeline for the [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
`nasdaq_exteral_data.csv` news corpus — **21.6 GiB (23,232,979,597 bytes), 12 columns**.

The CSV is never stored locally in one piece. Every stage reads it over HTTP in
ranges and writes compact Parquet instead.

```bash
pip install -r requirements.txt
python -m newsmanager all          # probe → plan → ingest → curate → validate → serve
duckdb data/corpus.duckdb            # then query it
```

---

## Why not just read the CSV

Four properties of this file drive the whole design. All were verified against
the live object, not assumed.

**1. Articles contain raw newlines inside quoted fields.** Splitting the file on
`\n` cuts records in half. Determining quote parity at an arbitrary byte offset
normally means scanning from byte 0 — a 21.6 GB read just to decide where to
start. `shard.py` sidesteps this with a structural anchor (`\n<float>,<ISO
timestamp>,`) whose two leading fields are unquoted, so it cannot match inside
article text. Validated on three 6 MB windows at 2 GB / 8 GB / 15 GB: 2,707
records parsed, **zero ragged rows**.

**2. Type inference is a trap.** `read_csv(..., sample_size=-1)` streams all
21.6 GB purely to pick column types — a 200k-row `LIMIT` query took **198s**
because of it. The schema is pinned in `config.py` and every column is read as
`VARCHAR`; casting happens once, explicitly, in `curate`.

**3. Rows duplicate across tickers — by an amount no sample can pin down.** The
source is one row per *(article, ticker)*, so a story is repeated under every
ticker it mentions. Two honest measurements of how often disagree sharply:

| sample | duplication |
|---|---|
| first 200,000 rows (contiguous) | **22.3%** (156,761 distinct URLs) |
| 19,320 rows across 60 windows spanning the file | **4.3%** |

Both are biased, in opposite directions. The file is sorted by ticker, so the
two copies of a story mentioning NVDA and AAPL sit gigabytes apart — thin
sampled windows almost never capture both and therefore *under*-count, while a
contiguous prefix of heavily co-mentioned A-tickers *over*-counts. The true
corpus-wide figure is only knowable from a full pass, and `nm-extract units`
reports it exactly, once. Don't plan against either number above.

Whatever the rate, the fix is the same: the curated layer splits `documents`
from `mentions`, so per-article statistics stop double-counting multi-ticker
stories and each article is sent to the LLM once.

**4. The corpus is two eras.** Recent rows are nasdaq.com articles with the four
extractive-summary columns populated and `Publisher`/`Author` **100% empty**.
Older rows (toward the end of the file) are Bloomberg-style wire copy with the
summary columns empty. Expect era-dependent nulls; don't read them as corruption.

**5. Past ~17.7 GB there is a third layout — and a Russian news portal.** The
index column goes empty (records start with `,<timestamp>,`), 63% of all rows
carry no ticker and 76% have no article body (headline-only). Bundled in are
799,037 articles from **lenta.ru**, a Russian general-news site (weather,
politics, crime) with zero tickers, covering nearly all of 1999–2009. They stay
in `documents`, but `units` excludes them from LLM extraction
(`EXCLUDED_DOMAINS` in `newsmanager/extract/units.py`): that cut the job from
2,321,501 to **1,607,982** calls.

---

## Sizing — read this before running

| | |
|---|---|
| Source CSV | 21.6 GiB |
| Estimated curated output | ~6.5 GiB |
| Peak temp disk during ingest | `shard_bytes × workers` ≈ 1.5 GiB |
| Free disk actually needed | **~8 GiB** |
| RAM | 16 GiB (10 cores) |

Free space is deliberately *not* pinned here: it drifts as you work, and a
number written once is a number that will be wrong later. `nm probe` measures it
at run time and refuses to start if headroom is short.

The CSV **does not fit** on this machine alongside its own output, which is why
`wget`-then-parse is not an option here regardless of preference. The sharded
ingest holds at most ~1.5 GiB of CSV bytes at any instant.

Measured single-stream throughput is **~7 MB/s**, so a full pass is roughly
**55 minutes**; the 6-way sharded ingest is faster if the CDN cooperates. Run
`python -m newsmanager probe` first — it re-checks the pinned size/etag and warns
if free space is tight.

---

## Stages

```
probe     verify remote size + etag against the pin; report disk headroom
plan      find record-safe shard boundaries → data/manifest.json   (~180 MB of probes)
ingest    fetch shards → data/raw/*.parquet                        (parallel, resumable)
curate    type, dedup, partition → data/curated/{documents,mentions,summaries}
validate  data-quality gate (run against raw and curated)
serve     build data/corpus.duckdb — views over Parquet, not copies
```

Each is a subcommand: `python -m newsmanager <stage>`.

**Resumability is the point of `plan`/`ingest`.** A 21.6 GB single-pass download
that dies at 90% loses everything. Shards are independent, and a shard only
counts as done if its Parquet footer actually reads back — a truncated file from
a killed process is detected and re-fetched rather than silently skipped. Re-run
`ingest` after any failure; it retries only what's missing.

`--mode stream` does a single `read_csv` over the URL with no temp files at all,
if you prefer that tradeoff. It cannot resume.

## Layers

```
data/raw/          bronze — 1:1 with the CSV, zero transformation
data/curated/      silver — typed, deduplicated, partitioned by year
data/marts/        gold   — small aggregates for the analysis loop
data/corpus.duckdb   views over the above (a few hundred KB, not a copy)
```

Raw stays untransformed on purpose: curation bugs get fixed with a `curate`
re-run, never a re-download.

## Data model

```
documents   doc_id, published_at, year, title, url, publisher, author,
            body, body_chars, is_stub, n_symbols          -- one row per article
mentions    doc_id, symbol, published_at, year            -- one row per (article, ticker)
summaries   doc_id, lsa_/luhn_/textrank_/lexrank_summary  -- derived, kept separate
```

`doc_id` is the URL when present, else a hash of the whitespace-normalised body.
URL is the stronger key: syndicated copies under different tickers share one URL,
whereas a pure content hash breaks on trivial whitespace differences.

Dedup runs **one year at a time**, which caps peak memory at the largest single
year instead of the whole corpus. Copies of a story almost always share its
publication date — but on the full corpus 73 URL-less articles were re-dumped
across a year boundary (2015-11 → 2016-01-03). Each year therefore skips any
`doc_id` an earlier year already wrote, keeping `doc_id` unique corpus-wide.

### Querying

```sql
-- per-ticker (the source's original shape, reconstructed)
SELECT symbol, published_at, title FROM news.articles WHERE symbol = 'AAPL';

-- per-article, with no multi-ticker double counting
SELECT year, count(*), avg(body_chars) FROM news.documents GROUP BY 1 ORDER BY 1;

-- join key for a price panel
SELECT * FROM news.daily_symbol_counts WHERE symbol='TSLA' AND dt >= '2020-01-01';

-- coverage per ticker
SELECT * FROM news.coverage ORDER BY n_mentions DESC LIMIT 20;
```

Parquet is columnar, so metadata queries never read the `body` column off disk.
That is what makes `news.coverage` fast over a multi-GB corpus.

## Configuration

Everything lives in `config.toml`. The knobs that matter:

- `shard_bytes` × `workers` — peak temp disk. Lower both if space is tight.
- `drop_summaries` — `true` drops the four summary columns, saving ~33% of the
  text payload. They are derived from `Article` and regenerable.
- `memory_limit` — **per worker**. Total is `memory_limit × workers`; keep it
  under ~60% of RAM.

`[source].expected_bytes` / `expected_etag` pin the upstream object. If HF
re-publishes the file, every stage fails loudly rather than resuming onto stale
shards and producing a half-old, half-new dataset with nothing to indicate it.

---

# Parte 2 — Extração de relações com LLM

Cada artigo vira input de um modelo local (Ollama) que devolve **tuplas de 5
campos acompanhadas da data da notícia**:

```
(agent_a, agent_b, relation_type, direction, strength)  +  published_at
```

> **Para rodar numa VM alugada, siga [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md)** —
> a sequência completa de comandos, com os portões que precisam passar antes de
> cada gasto. O resumo abaixo assume que você já leu aquilo.

## Um comando, do zero às tuplas

```bash
./deploy/run_all.sh          # máquina única
./deploy/run_all.sh 0 4      # worker 0 de 4 VMs
```

Instala o Ollama, baixa o modelo, ingere as notícias, remove duplicatas, extrai
e consolida. Toda etapa é idempotente: **se a VM cair, rode o mesmo comando de
novo** e ele retoma de onde parou.

## Escolha do modelo — leia antes de alugar

**Padrão: `qwen2.5:14b-instruct`** — modelo de instrução, sem raciocínio. Nos
mesmos 200 artigos, com o mesmo prompt, foi comparado ao `deepseek-r1:14b`:

| | deepseek-r1:14b | **qwen2.5:14b-instruct** |
|---|---|---|
| tuplas agente→agente válidas (40 sorteadas) | ~1/4 | **~2/3** |
| artigos preenchidos até o teto de 8 | 30% | **5%** |
| métrica como agente (EBITDA, receita…) | 7,2% | **2,8%** |
| grupo genérico ("customers", "investors") | 10,2% | **4,1%** |
| velocidade numa RTX 4090 | 1,19/s | **1,65/s** |

O R1 (destilado de raciocínio) segue o schema, mas trata o prompt de forma
frouxa: preenche até o teto, extrai organograma ("employs Matt Flake") e
métricas. O Qwen deixa 40% dos artigos vazios — na maioria avisos de
ex-dividendo, listas de cotações e boilerplate, onde o R1 inventava relações.

O modelo entra no `prompt_version`: trocar de modelo abre um diretório novo em
`data/extractions/`, nunca mistura resultados.

**`deepseek-v4-flash` e `deepseek-v4-pro` não servem aqui** — todas as tags
publicadas são `:cloud` e rodam na infraestrutura da Ollama, não na GPU alugada.

### Se usar um DeepSeek-R1: o thinking não desliga

`think: false` é ignorado pelo `deepseek-r1:14b` (Ollama 0.35.1): ele preenche o
campo `thinking` e devolve a resposta vazia ao estourar `num_predict`. Para
modelos `deepseek-r1` o cliente usa o `/api/generate` em modo `raw` com o template
do próprio modelo e um bloco `<think></think>` **vazio** já preenchido — zero
tokens de raciocínio, JSON válido. Outros modelos seguem pelo `/api/chat`.

## O prompt

`prompts/extraction.txt` instrui o modelo a identificar **agentes econômicos**
(empresas, tickers, índices, bancos centrais, governos, moedas, cripto,
commodities, setores, instituições, pessoas) e as **ações** que os conectam,
devolvendo uma tupla por relação. `prompts/schema.json` fixa os 5 campos e é
passado à Ollama como schema de *constrained decoding* — o modelo fica
impossibilitado de emitir JSON inválido.

Os dois arquivos são seus para editar. Ambos são hasheados num `prompt_version`
carimbado em cada linha de saída, então resultados de prompts diferentes nunca
se misturam em silêncio.

Regras que valem a pena conhecer: `direction` é o efeito **sobre agent_b**;
efeito mútuo vira duas tuplas; `strength` é calibrado por faixa (0.8–1.0
explícito e material, 0.1–0.3 especulativo); lista vazia é resposta válida.

## Gravação incremental — nada se perde

Você pediu que a saída fosse salva em tempo de execução, em Parquet. Um detalhe
torna isso menos trivial do que parece: **Parquet grava o footer só no
fechamento**, então um processo morto no meio deixa o arquivo inteiro ilegível —
não apenas a última linha. E gravar um Parquet por prompt criaria milhões de
arquivos de poucos KB.

O desenho usado:

1. Cada resultado vai imediatamente para um **WAL JSONL com fsync**.
2. A cada `--checkpoint-every` (200) registros o buffer vira um **segmento
   Parquet completo**, escrito em `.tmp` e renomeado atomicamente.
3. O WAL só é descartado **depois** do rename.

Consequência prática: **perda máxima numa queda é zero registro**, e nunca
existe Parquet pela metade. Na retomada, um WAL órfão é absorvido em Parquet
antes de qualquer coisa.

Isso também significa que os resultados são consultáveis **durante** a run:

```sql
SELECT count(*) FROM read_parquet('data/extractions/v=<versão>/relations/*.parquet');
```

## Saída

```sql
-- as tuplas, com a data da notícia
SELECT published_at, agent_a, relation_type, direction, strength, agent_b
FROM news.relations ORDER BY published_at DESC;

-- painel diário agente-a-agente (chave de junção com preços)
SELECT * FROM news.agent_edges_daily WHERE agent_a = 'the Fed';

-- relações reabertas por ticker
SELECT * FROM news.relations_by_symbol WHERE symbol = 'NVDA';

-- auditoria de cobertura e falhas
SELECT status, count(*), avg(latency_s) FROM news.extraction_runs GROUP BY 1;
```

`strength` é normalizado para [0,1] na gravação: constrained decoding garante o
*tipo* `number`, não a faixa, e modelos emitem 1.5 ou -0.2 com alguma frequência.

## Filtro de ruído

```bash
python -m newsmanager.extract filter                       # exclui todos os motivos
python -m newsmanager.extract filter --keep comparison,membership
```

Marca cada relação com o motivo de ruído — **marca, não apaga** — em
`relations_flagged.parquet` (coluna `noise`) e cria `news.relations_clean`.
Na run de 401.609 artigos com o Qwen:

| motivo | relações | exemplo |
|---|---|---|
| `generic` | 7,3% | `Allakos → disappoints → investors` |
| `comparison` | 2,1% | `Datadog → outperforms → Computer and Technology sector` |
| `membership` | 1,5% | `Nasdaq 100 → includes → Tesla`, `iShares ETF → affects → PepsiCo` |
| `source` | 1,1% | `Zacks Investment Research → upgrades → Kaman` |
| `self_loop` | 0,8% | `DuPont → expands → DuPont` |
| `metric` | 0,8% | `Comcast → raises → dividend` |
| `orgchart` | 0,2% | `Bank of England → appoints → Mark Carney` |
| **limpas** | **86,2%** | 761.019 relações |

Precisão estimada em amostras de 10–15 por motivo: ~9/10 ou mais em todos.
Num sorteio de 40 do conjunto limpo, ~70% são relações econômicas claras; o
resto é ruído mais difícil de capturar por regra (produto como agente, assunto
interno da empresa, política). **Indicadores macro e setores não são ruído**
(`retail sales → pound`, `Fed → dividend stocks`, `→ banks`): o filtro foi
ajustado para não marcá-los. Sinônimos (`Fed`, `the Fed`, `Federal Reserve`)
não são resolvidos aqui — isso é resolução de entidades, outra etapa.

## O que decide a sua conta de GPU

**O comprimento dos artigos, mais que a quantidade.** Medido em 19.320 artigos:

```
p50=3.518   p75=5.326   p90=6.976   p99=32.285   máx=272.968 caracteres
```

A cauda é brutal — o maior artigo sozinho tem ~68k tokens. `--max-chars` usa
**8.000** por padrão: preserva **85,9% de todos os tokens truncando só 6,3% dos
artigos**.

**`num_ctx`, que a Ollama deixa em 2048.** Um artigo de 8.000 caracteres mais
este prompt dá ~2.600 tokens: o padrão cortaria o fim da maioria dos artigos sem
erro nenhum. A CLI **deriva** o `num_ctx` de `--max-chars` e do tamanho real do
template quando você não passa `--num-ctx`, e **se recusa a iniciar** se o valor
que você passou for pequeno demais. Editar `prompts/extraction.txt` portanto
ajusta o contexto sozinho, em vez de silenciosamente estourá-lo.

**Nunca rodar em CPU por acidente.** `run_all.sh` aborta se `ollama ps` acusar
100% CPU. A run funcionaria — ~100x mais devagar, cobrada por hora.

## Estimar antes de gastar

```bash
python -m newsmanager.extract sweep     # acha o platô de concorrência da placa
python -m newsmanager.extract bench     # mede throughput real
python -m newsmanager.extract project --units-per-s 2.4 --vms 8 --usd-per-gpu-hour 0.40
```

Alugar mais VMs deixa o custo total praticamente igual e derruba o tempo de
parede linearmente — **alugue pelo prazo, não pelo orçamento**.

## Divisão do trabalho entre VMs

Partição determinística por hash: o worker *K* de *N* pega as unidades em que
`md5(unit_id) % N == K`. Sem coordenador, sem banco compartilhado e sem
possibilidade de duas VMs processarem a mesma unidade — o que importa quando as
máquinas são efêmeras. `partition -n <N>` mostra a carga antes de você alugar.

### O corpus vai junto; os resultados voltam

Cada VM grava só a própria partição, em `data/extractions/` local. Os segmentos
são nomeados `w<worker>-<seq>.parquet`, portanto já são únicos entre máquinas e
se unem num diretório só — mas **alguém precisa trazê-los para lá**. Esse é o
`gather.sh`, e ele é o passo que separa uma run paga de uma run perdida:
destruir as VMs sem recolher os segmentos joga fora `(N-1)/N` do que você pagou.

```bash
# 1. numa máquina BARATA (sem GPU): construa o corpus uma vez
python -m newsmanager all && python -m newsmanager.extract units

# 2. distribua o corpus para as VMs alugadas (~6,5 GB, só o curated)
./deploy/gather.sh push vm0 vm1 vm2 vm3

# 3. em cada VM (NM_SKIP_INGEST=1 pula a ingestão, que não usa GPU nenhuma)
NM_SKIP_INGEST=1 ./deploy/run_all.sh <K> 4

# 4. de volta na máquina barata, ANTES de destruir as VMs
./deploy/gather.sh pull vm0 vm1 vm2 vm3
```

O passo 1 fora da GPU não é cosmético: a ingestão leva ~55 min limitada por
rede, e fazê-la na VM alugada significa pagar por uma placa ociosa — numa frota,
por *N* placas ociosas, já que as outras esperam o worker 0 terminar.

O `pull` traz também os `wal-*.jsonl` e os absorve antes de consolidar: os
últimos registros de uma VM que terminou e nunca mais vai reiniciar estão só
neles. É incremental e idempotente — rode durante a run para ir tirando
resultado da máquina antes do fim.

### Falha de configuração não vira loop caro

`run_worker.sh` reinicia um worker que caiu, porque quedas são transitórias. Mas
um erro de configuração — modelo que não foi baixado, `num_ctx` pequeno demais,
corpus que nunca chegou — falha exatamente igual em toda tentativa. As CLIs
saem com **código 2** nesses casos e o `run_worker.sh` aborta na hora, em vez de
gastar 25 minutos de GPU alugada provando cem vezes a mesma coisa.

---

# Parte 3 — Grafo de conhecimento dinâmico

```bash
python -m newsmanager.extract graph-build      # ~15 s no Mac, sem GPU
```

Cada tupla vira um **evento datado** `src --classe(sinal × força)--> dst` entre
**nós únicos**: um por empresa (o ticker), um para o Fed, um para o petróleo. O
grafo "em t" não é materializado: `newsmanager.graph.snapshot` agrega os eventos
de uma janela antes de t, com decaimento exponencial opcional.

```python
from newsmanager import config, query, graph
con = query.connect(config.load())
g = graph.snapshot(con, "2020-03-16", window_days=90, half_life_days=30, tickers_only=True)
for asof, edges in graph.snapshots(con, "2019-01-01", "2020-12-31", freq="1 week"):
    ...                                   # ~40 ms por snapshot
nxg = graph.to_networkx(con, g)           # opcional: pip install networkx
```

**Sem look-ahead.** Todo `published_at` do corpus é meia-noite: não há hora. Uma
notícia do dia d pode ter saído depois do fechamento, então o snapshot em `asof`
usa só `dt < asof` e serve para prever o retorno do próprio dia `asof`.

## Nós: um por agente

O `entity_id` do link não serve direto como nó (medido na run `v=a05d0c0dc189`):

| problema | exemplo | regra |
|---|---|---|
| canônico do LLM junta demais | "mexico central bank", "nigerian banks", "dfs" → Federal Reserve; "moderna" → Merck | sem ticker, o canônico só vale se for `compatible` com o nome (token, prefixo, sigla); senão o nó é o próprio nome |
| junta de menos | `US_FEDERAL_RESERVE` ≠ `FEDERAL_RESERVE`, `US_DOLLAR` ≠ `DOLLAR` | prefixo de país cai quando o nome sem ele existe com o mesmo tipo (não para empresas: "US Steel") |
| classes de ação | GOOG/GOOGL, NWS/NWSA | mesmo nome na listagem → o ticker mais citado com preço; `tickers` guarda todos |
| slug colide com ticker | CORN (milho) = ETF CORN, FTC (agência) = FTC Solar, GDP | sem ticker, o nó leva o tipo: `CORN_COMMODITY` |
| o resto | FOMC → Fed, META → FB, WTI/Brent → petróleo, USD, BTC | manual: `prompts/graph_aliases.csv` |

Resultado: **197.550 nós** (5.654 com ticker e evento). O Fed é um nó com 151
grafias (fed, us federal reserve, fomc, new york fed…) e 21.456 menções.
Há mais nós que `entity_id`s porque os nomes separados do canônico errado
viram nós próprios, quase todos na cauda.

## Arestas: eventos e classes

`direction` é binária, então `w = ±strength`. Os 137 mil `relation_type`
distintos são mapeados por regex (`REL_CLASSES` em `newsmanager/extract/graph.py`)
para 16 classes, e `relation_type` original fica no evento:

```
impact 32,1%  competition 9,2%  ownership 7,4%  ma 7,1%  supply_chain 6,1%
partnership 4,4%  legal_regulatory 4,1%  stance 3,2%  analyst 3,1%  business 2,9%
financing 2,3%  monetary 2,0%  trade_policy 0,9%  governance 0,9%  capital_return 0,3%
other 14,0%
```

Laço criado pela fusão (Snap → "Snapchat shares", ambos `SNAP`) vira ruído
`self_loop`: 26.426 eventos, quase todos "empresa → as próprias ações". Não são
arestas, mas ficam em `news.graph_events_all` como sinal sobre o próprio nó.

## Saída

```
data/curated/graph/v=<versão>/
  nodes.parquet        node_id, label, kind, ticker, tickers, has_price, n_mentions,
                       n_keys, n_events, degree, first_dt, last_dt, graph_version
  node_keys.parquet    chave do link -> node_id, method (ticker|canonical|key|manual)
  events.parquet       event_id, dt, doc_id, src, dst, relation_type, rel_class,
                       sign, strength, w, agent_a, agent_b, noise
  rel_classes.parquet  relation_type -> rel_class
  meta.json            graph_version = hash(GRAPH_REV, link_version, classes, aliases)
```

Views: `news.graph_nodes`, `news.graph_events` (limpos: 741.409, 496.223 pares
distintos), `news.graph_events_all`, `news.graph_edges_weekly`,
`news.graph_edges_monthly`. Cada view de período tem uma linha por
(período, src, dst, classe).

---

# Parte 4 — Preços e universo de ativos

```bash
python -m newsmanager prices                 # baixa o zip (590 MB, ~25 s) e monta o painel (~45 s)
python -m newsmanager universe --n 100       # universo point-in-time de cada mês, 2011-2023
python -m newsmanager universe --asof 2020-08-03 --n 20   # inspecionar uma data
```

A fonte é o `Stock_price/full_history.zip` do próprio FNSPID (7.693 tickers,
1962–2023), conferido pelo sha256 do LFS. O zip é a camada bruta e nunca é
extraído (2,9 GB de CSV): a limpeza lê membro a membro e grava
`data/curated/prices/{daily,quality,calendar}.parquet` (627 MB a partir de 2005).

## O que o arquivo tem de errado (medido)

| problema | tamanho | tratamento |
|---|---|---|
| duas ordens de coluna | 5.705 × 1.988 arquivos | lido pelo cabeçalho |
| **coletas emendadas** em 2015-01-02, 2020-04-02, 2020-07-02 e 2020-07-06 | 22,7 mil retornos | retorno desses dias = inválido (`splice`) |
| lote antigo termina em 2020-07-02 | 3.137 tickers, entre eles BAC, FB, NFLX, MA, CSCO, MCD | `truncated` em `quality` |
| pico de cotação que se reverte | 7,8 mil dias | os dois dias inválidos (`spike`, `spike_revert`) |
| ticker reutilizado (ACB 2006–2016 intercala dois papéis) | — | pego pela regra de pico; o universo exclui quem tem pico na janela |
| preço parado com volume 0 | 437 mil dias, quase só small caps | `stale` |
| pregão faltando na série | 81 mil | retorno de vários dias inválido (`gap`) |

**A emenda é o erro grave.** Cada lote foi ajustado por desdobramentos e
dividendos só até a própria coleta. No dia 2020-07-06 aparecem AAPL −74,9%
(4:1), GOOG −94,9% (20:1), TSLA −92,4% (15:1), e também KO −9,5%, XOM −15,5%
e PFE −17%, pelos dividendos pagos depois de 2020. Dentro de cada lote os
retornos estão certos. As datas de emenda saem do fator de ajuste
f = adj_close/close, que numa série consistente só muda em data ex-dividendo.
Contando os tickers com salto de f acima de 5% no dia: 2020-07-06 (1.918),
2015-01-02 (481: AT&T "sobe" 95%), 2020-04-02 (151: NEE "cai" 76%) e 2020-07-02
(100). O resto do histórico fica em até 60, que são ex-dividendos de fundos. O
build recalcula essa contagem e avisa se surgir outra data. Grandes quedas
reais ficam (AAPL −52% em 2000-09-29).

`news.returns` (`ticker, date, ret`) só tem retornos válidos entre pregões
consecutivos. `news.prices` traz `ret_raw` e `flag` para auditoria.

## Universo

Em cada primeiro pregão do mês t entram as n **ações** (nó `company`, não ETF)
com mais eventos limpos no grafo nos 365 dias antes de t, entre as que têm,
na janela de 252 pregões antes de t:

- ≥ 95% de retornos válidos e nenhum pico;
- preço ≥ US$ 5 e volume financeiro mediano ≥ US$ 5 mi.

Tudo usa só dados < t, **com uma exceção declarada**: quem é truncado pelo fim
da coleta antes do fim do mês de carteira fica de fora. Isso é informação
sobre o arquivo, não sobre retornos. Deslistagens reais não são filtradas: o
backtest tem de tratá-las.

| n | tickers distintos | troca/mês |
|---|---|---|
| 50 | 166 | 4,9% |
| 100 | 334 | 5,0% |
| 200 | 681 | 5,6% |

**Limitação:** depois de 2020-07, 39 das 200 ações mais noticiadas (BAC, FB,
NFLX, MA, CSCO, MCD, SBUX, BLK…) não têm preço no FNSPID e saem do universo.

---

## Testing

```bash
python tests/test_e2e_subset.py 4 48   # pipeline de ingestão, arquivo real, ~200 MB
python tests/test_extract_e2e.py       # pipeline de extração, Ollama simulado
python tests/test_graph.py             # grafo: regras de nó, classes, snapshot sem look-ahead
python tests/test_prices.py            # preços: zip sintético com cada anomalia do FNSPID
```

The ingest test runs every stage for real: range fetch, boundary contiguity,
parquet round-trip, resume, dedup, validation, views.

O teste de extração **não precisa de GPU** — `tests/mock_ollama.py` substitui o
servidor e injeta latência e falhas (`--fail-rate 0.1`). Cobre partição
multi-worker, **queda da VM no meio da escrita com zero perda**, retomada,
retry, consolidação e consulta. Vale rodar nas VMs também, para provar o
encanamento antes de baixar um modelo de verdade.
