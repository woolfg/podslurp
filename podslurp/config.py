"""
config.py — load settings from .env and expose a typed Config dataclass.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_key: str
    api_secret: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    output_dir: Path
    download_dir: Path
    diarize: bool
    pyannote_token: str
    diarize_num_speakers: Optional[int]
    diarize_min_speakers: Optional[int]
    diarize_max_speakers: Optional[int]


def _parse_optional_int(env_var: str) -> Optional[int]:
    val = os.getenv(env_var, "").strip()
    return int(val) if val.isdigit() else None


def load_config() -> Config:
    api_key = os.getenv("PODCASTINDEX_API_KEY", "").strip()
    api_secret = os.getenv("PODCASTINDEX_API_SECRET", "").strip()

    missing = [
        name
        for name, val in (
            ("PODCASTINDEX_API_KEY", api_key),
            ("PODCASTINDEX_API_SECRET", api_secret),
        )
        if not val
    ]
    if missing:
        print(
            f"[error] Missing required environment variable(s): {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your PodcastIndex credentials.",
            file=sys.stderr,
        )
        sys.exit(1)

    diarize = os.getenv("PODSLURP_DIARIZE", "false").strip().lower() in ("1", "true", "yes")
    pyannote_token = os.getenv("PYANNOTE_TOKEN", "").strip()

    if diarize and not pyannote_token:
        print(
            "[error] PODSLURP_DIARIZE is enabled but PYANNOTE_TOKEN is not set.\n"
            "Set PYANNOTE_TOKEN to your HuggingFace access token with access to\n"
            "pyannote/speaker-diarization-3.1 (accept the model's terms of use first).",
            file=sys.stderr,
        )
        sys.exit(1)

    return Config(
        api_key=api_key,
        api_secret=api_secret,
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        output_dir=Path(os.getenv("PODSLURP_OUTPUT_DIR", "./transcriptions")),
        download_dir=Path(os.getenv("PODSLURP_DOWNLOAD_DIR", "./downloads")),
        diarize=diarize,
        pyannote_token=pyannote_token,
        diarize_num_speakers=_parse_optional_int("PODSLURP_DIARIZE_NUM_SPEAKERS"),
        diarize_min_speakers=_parse_optional_int("PODSLURP_DIARIZE_MIN_SPEAKERS"),
        diarize_max_speakers=_parse_optional_int("PODSLURP_DIARIZE_MAX_SPEAKERS"),
    )
