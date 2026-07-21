.DEFAULT_GOAL := help

.PHONY: help install install-diarize validate modules podcast call transcribe diarize analyze test lint clean

help:
	@echo "VoxYak — modular audio transcription and processing"
	@echo ""
	@echo "  make install          Install the application and development dependencies"
	@echo "  make install-diarize  Install optional speaker diarization dependencies"
	@echo "  make validate         Validate all pipelines in voxyak.yaml"
	@echo "  make modules          List installed pipeline modules"
	@echo "  make podcast          Run the podcast transcription pipeline"
	@echo "  make call             Record, transcribe, and summarize a call"
	@echo "  make transcribe file=<audio> [lang=<code>]"
	@echo "  make diarize transcript=<transcript.json> [audio=<audio>]"
	@echo "  make analyze transcript=<transcript.json>"
	@echo "  make test             Run the test suite"
	@echo "  make lint             Run Ruff"

install:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env"; fi
	uv sync

install-diarize:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env"; fi
	uv sync --extra diarize

validate:
	uv run voxyak pipelines validate

modules:
	uv run voxyak modules

podcast:
	uv run voxyak run podcast

call:
	uv run voxyak run call-summary

transcribe:
	@if [ -z "$(file)" ]; then echo "Usage: make transcribe file=<audio> [lang=<code>]"; exit 1; fi
	uv run voxyak transcribe "$(file)" $(if $(lang),--language "$(lang)",)

diarize:
	@if [ -z "$(transcript)" ]; then echo "Usage: make diarize transcript=<transcript.json> [audio=<audio>]"; exit 1; fi
	uv run voxyak diarize "$(transcript)" $(if $(audio),--audio "$(audio)",)

analyze:
	@if [ -z "$(transcript)" ]; then echo "Usage: make analyze transcript=<transcript.json>"; exit 1; fi
	uv run voxyak analyze "$(transcript)"

test:
	uv run pytest

lint:
	uv run ruff check voxyak tests

clean:
	rm -rf .venv runs .pytest_cache .ruff_cache
	find voxyak tests -type d -name "__pycache__" -exec rm -rf {} +
	find voxyak tests -type f -name "*.pyc" -delete
