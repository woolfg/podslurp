.DEFAULT_GOAL := help

.PHONY: help install run lint clean

help:
	@echo "podslurp — Podcast Search, Download & Transcription CLI"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install      Create/update the virtual env and install all dependencies"
	@echo "  run          Launch the interactive CLI"
	@echo "  transcribe   Transcribe a local audio file (requires file=<path> [lang=<language>])"
	@echo "  lint         Run ruff linter over the source"
	@echo "  clean        Remove generated files (.venv, downloads, transcriptions, caches)"

install:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — fill in your API credentials."; \
	fi
	uv sync

run:
	uv run podslurp

transcribe:
	@if [ -z "$(file)" ]; then \
		echo "Usage: make transcribe file=<path_to_audio> [lang=<language>]"; \
		exit 1; \
	fi
	@if [ -n "$(lang)" ]; then \
		uv run podslurp --transcribe "$(file)" --lang "$(lang)"; \
	else \
		uv run podslurp --transcribe "$(file)"; \
	fi

lint:
	uv run ruff check podslurp/

clean:
	rm -rf .venv downloads transcriptions
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
