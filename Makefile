PY ?= python3

.PHONY: install probe plan ingest curate validate serve all test clean
install:  ; $(PY) -m pip install -r requirements.txt
probe:    ; $(PY) -m newsmanager probe
plan:     ; $(PY) -m newsmanager plan
ingest:   ; $(PY) -m newsmanager ingest
curate:   ; $(PY) -m newsmanager curate
validate: ; $(PY) -m newsmanager validate
serve:    ; $(PY) -m newsmanager serve --marts
all:      ; $(PY) -m newsmanager all --marts
test:     ; $(PY) tests/test_e2e_subset.py 3 32
clean:    ; $(PY) -m newsmanager reset all

# --- extraction ---
.PHONY: units bench sweep extract collect test-extract mock
units:        ; $(PY) -m newsmanager.extract units
bench:        ; $(PY) -m newsmanager.extract bench
sweep:        ; $(PY) -m newsmanager.extract sweep
extract:      ; ./deploy/run_worker.sh
collect:      ; $(PY) -m newsmanager.extract collect
mock:         ; $(PY) tests/mock_ollama.py --port 11500
test-extract: ; $(PY) tests/test_extract_e2e.py

run-all:      ; ./deploy/run_all.sh
