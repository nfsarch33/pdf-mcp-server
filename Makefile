UV ?= uv
VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(UV) pip

.PHONY: venv install test clean

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(UV) venv $(VENV)

install: venv
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

test: install
	. $(VENV)/bin/activate && PYTHONWARNINGS=default pytest

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__

