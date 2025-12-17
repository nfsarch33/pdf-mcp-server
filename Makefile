UV ?= uv
VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(UV) pip

.PHONY: venv install test clean
.PHONY: smoke prepush

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(UV) venv $(VENV)

install: venv
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

test: install
	. $(VENV)/bin/activate && PYTHONWARNINGS=default pytest

smoke: install
	. $(VENV)/bin/activate && $(PY) scripts/cursor_smoke.py --out-dir /tmp/pdf-handler-prepush-smoke

prepush: test smoke

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__

