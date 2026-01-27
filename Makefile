UV ?= uv
VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(UV) pip

.PHONY: venv install install-ocr test test-ocr test-quick clean
.PHONY: smoke prepush
.PHONY: lint format-check
.PHONY: check-tesseract

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(UV) venv $(VENV)

install: venv
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

install-ocr: install
	. $(VENV)/bin/activate && $(PIP) install -e ".[ocr]"
	@echo "OCR dependencies installed. Ensure Tesseract is installed:"
	@echo "  macOS: brew install tesseract"
	@echo "  Linux: sudo apt-get install tesseract-ocr"

test: install
	. $(VENV)/bin/activate && PYTHONWARNINGS=default pytest

test-ocr: install-ocr
	. $(VENV)/bin/activate && PYTHONWARNINGS=default pytest tests/test_ocr.py tests/test_phase2_features.py -v

test-quick: install
	. $(VENV)/bin/activate && PYTHONWARNINGS=default pytest -q --tb=short

check-tesseract:
	@which tesseract > /dev/null 2>&1 && tesseract --version | head -1 || echo "Tesseract not installed. Run: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
	@$(PY) -c "import pytesseract; print(f'pytesseract: OK, languages: {pytesseract.get_languages()}')" 2>/dev/null || echo "pytesseract not installed. Run: make install-ocr"

lint: install
	@FILES=$$(git diff --name-only --diff-filter=ACMRTUXB origin/main...HEAD 2>/dev/null | grep -E '\\.py$$' || true); \
	if [ -z "$$FILES" ]; then echo "ruff: no changed python files"; exit 0; fi; \
	. $(VENV)/bin/activate && ruff check $$FILES

format-check: install
	@FILES=$$(git diff --name-only --diff-filter=ACMRTUXB origin/main...HEAD 2>/dev/null | grep -E '\\.py$$' || true); \
	if [ -z "$$FILES" ]; then echo "ruff-format: no changed python files"; exit 0; fi; \
	. $(VENV)/bin/activate && ruff format --check $$FILES

smoke: install
	. $(VENV)/bin/activate && $(PY) scripts/cursor_smoke.py --out-dir /tmp/pdf-handler-prepush-smoke

prepush: lint format-check test smoke

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__

