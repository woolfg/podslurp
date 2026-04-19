"""
downloader.py — stream an audio file from a URL with a tqdm progress bar.
"""
from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

_CHUNK_SIZE = 8192
_TIMEOUT = 30


def download_audio(url: str, dest_dir: Path, filename: str) -> Path:
    """Download *url* into *dest_dir*/*filename*, streaming with a progress bar.

    Returns the full path of the saved file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            open(dest_path, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=True,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

    return dest_path
