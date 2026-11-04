PY ?= .venv/bin/python

.PHONY: setup data train eval test serve all
setup:
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
data:
	$(PY) scripts/fetch_data.py
train: data
	$(PY) scripts/train.py
eval: train
	$(PY) scripts/evaluate.py
test:
	$(PY) -m pytest -q
serve:
	$(PY) -m uvicorn clinnlp.api:app --host 127.0.0.1 --port 8000
all: setup data train eval test
