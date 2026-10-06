# Runbook — from renting a GPU to having the tuples

Every command, in order, with the gate that has to pass before you spend more.

The rule behind the ordering: **the GPU is the only thing billing you, so nothing
that doesn't need a GPU happens on it.** Building the corpus is network-bound and
takes ~55 minutes. Doing it on the rented box means paying for an idle card; on a
fleet, for N idle cards.

| phase | where | GPU billing | time |
|---|---|---|---|
| 0. corpus | your Mac / cheap VM | no | ~1 h |
| 1. smoke test | 1 rented GPU | yes | ~20 min |
| 2. measure | same VM | yes | ~10 min |
| 3. the run | 1 or N GPUs | yes | from `project` |
| 4. collect | cheap box | no | minutes |

---

## Phase 0 — before you rent anything

On your own machine (needs ~8 GiB free disk and Python 3.11+).

```bash
git clone <your-repo> NewsManager && cd NewsManager
python3 -V                                  # must be 3.11+; 3.10 has no tomllib
python3 -m venv .venv && source .venv/bin/activate   # Homebrew/Ubuntu 24.04 refuse a global pip
pip install -r requirements.txt
```

> **Disk check first.** The corpus build needs ~8 GiB free plus DuckDB spill
> space during `curate`/`units`. If `df -h .` shows less than ~15 GiB, build it on
> the rented VM instead (run `./deploy/run_all.sh` there and skip `gather.sh push`)
> or on a cheap CPU box. That costs ~1 h of idle GPU on one VM, which is cheaper
> than a corpus build that dies at 95% from a full disk.

Prove the plumbing offline — no GPU, no cost:

```bash
python tests/test_e2e_subset.py 3 32        # real remote file, ~100 MB
python tests/test_extract_e2e.py            # full extraction path, mock Ollama
```

Both must print `RESULT: PASS`. Then build the corpus once:

```bash
python -m newsmanager probe                 # re-checks size/etag + disk headroom
python -m newsmanager all                   # ~55 min, streams 21.6 GB, writes ~6.5 GB
python -m newsmanager.extract units         # dedup -> the list of LLM calls
```

**Gate:** note the unit count `units` prints. That is the real size of your job —
every cost estimate from here scales off it.

```bash
python -m newsmanager.extract partition -n 4   # how the load would split over 4 VMs
```

---

## Phase 1 — first rented VM, smoke test

Rent **one** cheap GPU (24 GB is the design point). Do not rent the fleet yet.

### 1.1 On the VM

```bash
git clone <your-repo> NewsManager && cd NewsManager
./deploy/bootstrap.sh 0 1 deepseek-r1:14b
```

Works on full VMs (systemd) and on container rentals like RunPod/Vast (root, no
systemd, no sudo). Python deps go into `.venv/` in the repo; `source
~/worker.env` puts it first on `PATH`, so the `python3` commands below use it.

`bootstrap.sh` checks Python, installs Ollama, configures it for batching
(`OLLAMA_NUM_PARALLEL`, `KEEP_ALIVE=-1`), pulls the model, and **verifies the
model is actually on the GPU**. It writes `~/worker.env`.

**Gate — this is the expensive one.** The script aborts if any part of the model
landed on CPU. If it aborts, do not work around it: a partially-offloaded model
costs the same per hour and delivers a fraction of the throughput. Use
`deepseek-r1:7b`, or lower `NUM_PARALLEL` (each parallel slot costs KV cache).

### 1.2 Ship the corpus from your machine

```bash
# on YOUR machine, not the VM
./deploy/gather.sh push <vm-host>
```

Sends `documents`, `mentions` and `extraction_units` (~6.5 GB). It expects the
repo at `~/NewsManager` on the VM — override with `NM_REMOTE_DIR`.

---

## Phase 2 — measure before committing

On the VM:

```bash
source ~/worker.env
python3 -m newsmanager.extract sweep --n 32      # find this card's concurrency plateau
```

`sweep` only changes *client* concurrency. The Ollama server's
`OLLAMA_NUM_PARALLEL` was fixed at bootstrap, so if the plateau is not 8, re-apply
it and restart the server:

```bash
NUM_PARALLEL=16 ./deploy/bootstrap.sh 0 1 deepseek-r1:14b    # idempotent
python3 -m newsmanager.extract bench --n 40 --concurrency 16
```

`bench` and `sweep` never write results — they don't mark units done and are safe
to repeat. Take the `units/s` that `bench` prints and feed it in:

```bash
python3 -m newsmanager.extract project --units-per-s <measured> --vms 4 \
        --usd-per-gpu-hour <what you are actually paying>
```

**Gate:** this is the first honest cost number you have. Decide the fleet size
here — renting more VMs keeps total cost roughly flat and cuts wall-clock
linearly, so rent for your deadline, not your budget.

### 2.1 Look at the output before buying more of it

```bash
python3 -m newsmanager.extract run --limit 200 --worker-id 0 --workers 1
python3 -m newsmanager.extract collect
duckdb data/corpus.duckdb -c "
  SELECT published_at, agent_a, relation_type, direction, strength, agent_b
  FROM news.relations ORDER BY random() LIMIT 25;"
duckdb data/corpus.duckdb -c "
  SELECT status, count(*), round(avg(latency_s),2) FROM news.extraction_runs GROUP BY 1;"
```

**Gate — read the 25 tuples.** Throughput is worth nothing if the extractions are
junk. Check that agents are real entities, that `direction` matches the article's
sense, and that the failure count is near zero. If the tuples are weak, fix
`prompts/extraction.txt` **now** — the prompt is hashed into `prompt_version`, so
changing it later means re-running everything you already paid for.

---

## Phase 3 — the real run

### Always run it under tmux

`run_worker.sh` survives crashes, but not your SSH session dropping. Without a
multiplexer, a disconnect kills a run you are paying for.

```bash
tmux new -s nm
# inside tmux:
NM_SKIP_INGEST=1 NM_CONCURRENCY=16 ./deploy/run_all.sh 0 1
# detach with Ctrl-b d ; come back with: tmux attach -t nm
```

`NM_SKIP_INGEST=1` is what keeps the GPU off the corpus build. The pushed
`extraction_units` are reused rather than rebuilt, so every VM partitions from an
identical set.

### Fleet of N

```bash
# your machine
./deploy/gather.sh push vm0 vm1 vm2 vm3

# on each VM k of 4, inside tmux
NM_SKIP_INGEST=1 NM_CONCURRENCY=16 ./deploy/run_all.sh <k> 4
```

Worker *k* takes the units where `md5(unit_id) % 4 == k`. No coordinator, no
chance of two VMs doing the same unit. With `N > 1`, `run_all.sh` deliberately
does **not** consolidate — each VM only holds its own quarter.

### Monitoring

```bash
tail -f logs/worker-0.log
nvidia-smi -l 60
ollama ps                                    # PROCESSOR must stay 100% GPU

# results are queryable DURING the run
duckdb -c "SELECT count(*) FROM read_parquet('data/extractions/v=*/runs/*.parquet');"
```

**Abort conditions** — stop and diagnose rather than letting it burn:
- `ollama ps` starts showing any CPU share
- the rate drops far below what `bench` measured
- exit code **2** anywhere: that is a config error, and restarting will not fix it

---

## Phase 4 — collect BEFORE destroying anything

This is the step that decides whether a paid run survives. Each VM holds only its
own partition, and the last up-to-200 records of each worker live only in a WAL
that `collect` cannot read on its own.

```bash
# on your machine, with every VM still alive
NM_MAX_CHARS=8000 ./deploy/gather.sh pull vm0 vm1 vm2 vm3
```

`pull` rsyncs each VM's segments *and* WALs, absorbs the WALs into Parquet, then
consolidates once. It is incremental and idempotent — run it mid-flight too, to
get results off the machines early.

### Verify coverage, then destroy

```bash
duckdb data/corpus.duckdb -c "
  SELECT status, count(*) FROM news.extraction_runs GROUP BY 1;"
duckdb data/corpus.duckdb -c "
  SELECT count(*) AS units_total FROM read_parquet('data/curated/extraction_units/**/*.parquet');"
```

**Gate:** `ok + failed` must equal `units_total`. Only when those match is it safe
to destroy the VMs. If units are missing, a VM still holds them — re-run `pull`
for that host.

Re-run only the failures later by pointing a worker at the same output directory;
resume skips everything already persisted.

### The result

```sql
SELECT published_at, agent_a, relation_type, direction, strength, agent_b
FROM news.relations ORDER BY published_at DESC;

SELECT * FROM news.agent_edges_daily WHERE agent_a = 'the Fed';
SELECT * FROM news.relations_by_symbol WHERE symbol = 'NVDA';
```

---

## Quick reference

| variable | default | what it does |
|---|---|---|
| `NM_MODEL` | `deepseek-r1:14b` | must match what bootstrap pulled |
| `NM_CONCURRENCY` | `8` | match the server's `OLLAMA_NUM_PARALLEL` |
| `NM_MAX_CHARS` | `8000` | **hashed into `prompt_version`** — same value for run and collect |
| `NM_NUM_CTX` | derived | leave unset; derived from `max_chars` + template size |
| `NM_HEAVY_MEMORY` | `9GB` | DuckDB budget for `curate`/`units`; bootstrap sets 40% of RAM. 2023 alone is ~6.3 GB of text |
| `PY` | `.venv/bin/python` | interpreter the scripts use; set up by bootstrap/run_all |
| `NM_SKIP_INGEST` | — | `1` = corpus arrived by push; never ingest here |
| `NM_FORCE_UNITS` | — | `1` = rebuild units instead of reusing the pushed ones |
| `NM_REMOTE_DIR` | `NewsManager` | repo path on the remote host |
| `MAX_RESTARTS` | `100` | crash restarts before giving up |

Variables passed explicitly (`NM_CONCURRENCY=16 ./deploy/run_all.sh 2 4`) take
precedence over `~/worker.env`; the file only fills in what you didn't pass.

Exit codes: `0` done · `1` crashed, restart is appropriate · `2` **misconfigured,
restarting cannot help.**
