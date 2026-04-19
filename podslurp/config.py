"""
config.py — load settings from .env and expose a typed Config dataclass.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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

    return Config(
        api_key=api_key,
        api_secret=api_secret,
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        output_dir=Path(os.getenv("PODSLURP_OUTPUT_DIR", "./transcriptions")),
        download_dir=Path(os.getenv("PODSLURP_DOWNLOAD_DIR", "./downloads")),
    )
