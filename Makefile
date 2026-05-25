.DEFAULT_GOAL := help

.PHONY: help install install-diarize run lint clean diarize

help:
	@echo "podslurp — Podcast Search, Download & Transcription CLI"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Create/update the virtual env and install all dependencies"
	@echo "  install-diarize  Also install the speaker diarization extra (pyannote.audio)"
	@echo "  run              Launch the interactive CLI"
	@echo "  transcribe       Transcribe a local audio file (requires file=<path> [lang=<language>])"
	@echo "  diarize          Add speaker labels to an existing transcript (requires json=<path> [audio=<path>])"
	@echo "  lint             Run ruff linter over the source"
	@echo "  clean            Remove generated files (.venv, downloads, transcriptions, caches)"

install:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — fill in your API credentials."; \
	fi
	uv sync

install-diarize:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — fill in your API credentials."; \
	fi
	uv sync --extra diarize

run:
	uv run podslurp

transcribe:
	@if [ -z "$(file)" ]; then \
		echo "Usage: make transcribe file=<path_to_audio> [lang=<language>] [num_speakers=N] [min_speakers=N] [max_speakers=N]"; \
		exit 1; \
	fi
	@PODSLURP_DIARIZE_NUM_SPEAKERS=$(num_speakers) \
	 PODSLURP_DIARIZE_MIN_SPEAKERS=$(min_speakers) \
	 PODSLURP_DIARIZE_MAX_SPEAKERS=$(max_speakers) \
	 $(if $(lang),uv run podslurp --transcribe "$(file)" --lang "$(lang)",uv run podslurp --transcribe "$(file)")

lint:
	uv run ruff check podslurp/

diarize:
	@if [ -z "$(json)" ]; then \
		echo "Usage: make diarize json=<path_to_transcript.json> [audio=<path_to_audio>] [num_speakers=N] [min_speakers=N] [max_speakers=N]"; \
		exit 1; \
	fi
	@PODSLURP_DIARIZE_NUM_SPEAKERS=$(num_speakers) \
	 PODSLURP_DIARIZE_MIN_SPEAKERS=$(min_speakers) \
	 PODSLURP_DIARIZE_MAX_SPEAKERS=$(max_speakers) \
	 $(if $(audio),uv run podslurp --diarize "$(json)" --audio "$(audio)",uv run podslurp --diarize "$(json)")

clean:
	rm -rf .venv downloads transcriptions
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
