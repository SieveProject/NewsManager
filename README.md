# NewsManager

A DuckDB pipeline for the [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
`nasdaq_exteral_data.csv` news corpus — **21.6 GiB (23,232,979,597 bytes), 12 columns**.

The CSV is never stored locally in one piece. Every stage reads it over HTTP in
ranges and writes compact Parquet instead.

```bash
pip install -r requirements.txt
python -m newsmanager all          # probe → plan → ingest → curate → validate → serve
duckdb data/news.duckdb            # then query it
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
serve     build data/news.duckdb — views over Parquet, not copies
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
data/news.duckdb   views over the above (a few hundred KB, not a copy)
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

**`deepseek-v4-flash` e `deepseek-v4-pro` não servem aqui.** São os DeepSeek mais
recentes do catálogo, mas todas as tags publicadas são `:cloud` — rodam na
infraestrutura da Ollama, não na GPU que você alugou. Pagar por GPU para usá-los
é dinheiro jogado fora.

Os DeepSeek com **pesos locais de verdade** são a família `r1`:

| tag | tamanho | onde cabe |
|---|---|---|
| `deepseek-r1:7b`  | 4.7 GB | 12 GB VRAM |
| **`deepseek-r1:14b`** | **9 GB** | **24 GB — padrão do pipeline** |
| `deepseek-r1:32b` | 20 GB | 24 GB aperta; ideal em 48 GB+ |
| `deepseek-r1:70b` | 43 GB | 80 GB |

O padrão é `deepseek-r1:14b` porque, numa placa de 24 GB, 9 GB de pesos deixam
KV cache para ~8 requisições concorrentes — e é do batching que vem o
throughput. O `:32b` cabe, mas sufoca a concorrência e costuma sair **mais
lento** na prática. Meça com `sweep` antes de decidir.

### O thinking vem ligado e precisa ser desligado

R1 é um modelo de raciocínio e a Ollama **habilita o thinking por padrão**. Numa
extração de milhões de artigos a cadeia de raciocínio multiplica os tokens de
saída — que é exatamente o que se paga por hora de GPU — sem melhorar o
preenchimento de um schema fechado. O pipeline envia `think: false` via
`/api/chat` (o `/api/generate` não aceita esse parâmetro) e ainda remove
qualquer bloco `<think>` que escape, para que um resíduo de raciocínio não
derrube o parse do lote inteiro. Use `--think` só se quiser o contrário.

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

## Testing

```bash
python tests/test_e2e_subset.py 4 48   # pipeline de ingestão, arquivo real, ~200 MB
python tests/test_extract_e2e.py       # pipeline de extração, Ollama simulado
```

The ingest test runs every stage for real: range fetch, boundary contiguity,
parquet round-trip, resume, dedup, validation, views.

O teste de extração **não precisa de GPU** — `tests/mock_ollama.py` substitui o
servidor e injeta latência e falhas (`--fail-rate 0.1`). Cobre partição
multi-worker, **queda da VM no meio da escrita com zero perda**, retomada,
retry, consolidação e consulta. Vale rodar nas VMs também, para provar o
encanamento antes de baixar um modelo de verdade.
