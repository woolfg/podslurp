"""Interactive PodcastIndex search and download input."""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field
from tqdm import tqdm

from ...sdk import AudioArtifact, InputModule, ModuleConfig, RunContext


BASE_URL = "https://api.podcastindex.org/api/1.0"


class PodcastIndexConfig(ModuleConfig):
    search_results: int = Field(default=10, ge=1, le=50)
    episode_results: int = Field(default=1000, ge=1, le=1000)
    recent_count: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


def _credentials() -> tuple[str, str]:
    api_key = os.getenv("PODCASTINDEX_API_KEY", "").strip()
    api_secret = os.getenv("PODCASTINDEX_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise ValueError(
            "PODCASTINDEX_API_KEY and PODCASTINDEX_API_SECRET are required "
            "for the podcast-index input."
        )
    return api_key, api_secret


def _headers(api_key: str, api_secret: str) -> dict[str, str]:
    epoch = str(int(time.time()))
    digest = hashlib.sha1((api_key + api_secret + epoch).encode()).hexdigest()
    return {
        "User-Agent": "voxyak/0.2",
        "X-Auth-Key": api_key,
        "X-Auth-Date": epoch,
        "Authorization": digest,
    }


def _request(
    endpoint: str,
    params: dict[str, Any],
    settings: PodcastIndexConfig,
) -> dict[str, Any]:
    api_key, api_secret = _credentials()
    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        headers=_headers(api_key, api_secret),
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("PodcastIndex returned a non-object response.")
    return payload


def _pick(console, prompt: str, maximum: int, *, back: str | None = None) -> int | None:
    while True:
        raw = console.input(prompt).strip()
        if back and raw.lower() == back:
            return None
        if raw.isdigit() and 1 <= int(raw) <= maximum:
            return int(raw) - 1
        suffix = f" or {back!r}" if back else ""
        console.print(f"[red]Enter a number from 1 to {maximum}{suffix}.[/red]")


def _slug(value: str, maximum: int = 80) -> str:
    normalized = re.sub(r"[^\w]+", "_", value.lower(), flags=re.UNICODE)
    return re.sub(r"_+", "_", normalized).strip("_")[:maximum] or "audio"


def _download(
    url: str,
    destination: Path,
    timeout_seconds: int,
) -> None:
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            destination.open("wb") as output,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))


class PodcastIndexInput(InputModule):
    config_model = PodcastIndexConfig

    def preflight(self, config: BaseModel) -> None:
        PodcastIndexConfig.model_validate(config)
        _credentials()

    def run(self, context: RunContext, config: BaseModel) -> AudioArtifact:
        settings = PodcastIndexConfig.model_validate(config)
        console = context.console

        while True:
            query = console.input("[bold cyan]Search for a podcast:[/bold cyan] ").strip()
            if not query:
                continue
            feeds = _request(
                "/search/byterm",
                {"q": query, "max": settings.search_results},
                settings,
            ).get("feeds", [])
            if not feeds:
                console.print("[yellow]No podcasts found.[/yellow]")
                continue
            for index, feed in enumerate(feeds, start=1):
                console.print(
                    f"{index:>2}. [bold]{feed.get('title', '')}[/bold] "
                    f"[dim]{feed.get('author', '')} · {feed.get('language', '')}[/dim]"
                )
            selected = _pick(
                console,
                "Select podcast number (or 's' to search again): ",
                len(feeds),
                back="s",
            )
            if selected is not None:
                feed = feeds[selected]
                break

        episodes = _request(
            "/episodes/byfeedid",
            {"id": feed["id"], "max": settings.episode_results},
            settings,
        ).get("items", [])
        while True:
            keyword = console.input(
                "[bold cyan]Episode keyword filter "
                "(Enter = recent):[/bold cyan] "
            ).strip()
            candidates = (
                [
                    episode
                    for episode in episodes
                    if keyword.lower() in str(episode.get("title", "")).lower()
                ]
                if keyword
                else episodes[: settings.recent_count]
            )
            if not candidates:
                console.print("[yellow]No matching episodes.[/yellow]")
                continue
            for index, episode in enumerate(candidates, start=1):
                date = str(episode.get("datePublishedPretty", ""))[:10]
                console.print(
                    f"{index:>2}. [bold]{episode.get('title', '')}[/bold] [dim]{date}[/dim]"
                )
            selected_episode = _pick(
                console,
                "Select episode number: ",
                len(candidates),
            )
            episode = candidates[selected_episode or 0]
            break

        enclosure_url = str(episode.get("enclosureUrl") or "")
        if not enclosure_url:
            raise ValueError("The selected episode has no audio URL.")

        published = int(episode.get("datePublished") or 0)
        date = datetime.fromtimestamp(published, tz=timezone.utc).strftime("%Y-%m-%d")
        stem = "_".join(
            (
                _slug(str(feed.get("title") or "podcast")),
                date,
                _slug(str(episode.get("title") or "episode")),
            )
        )
        enclosure_type = str(episode.get("enclosureType") or "")
        suffix = ".m4a" if "m4a" in enclosure_type else ".mp3"
        context.work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = context.work_dir / f"{stem}{suffix}"
        console.print(f"[bold]Downloading:[/bold] {audio_path.name}")
        _download(enclosure_url, audio_path, settings.timeout_seconds)
        if audio_path.stat().st_size == 0:
            raise ValueError("Podcast download produced an empty file.")

        raw_language = episode.get("feedLanguage") or feed.get("language")
        language_hint = (
            str(raw_language).split("-")[0].lower() if raw_language else None
        )
        return AudioArtifact(
            path=audio_path.resolve(),
            media_type=enclosure_type or "audio/mpeg",
            tracks={"primary": audio_path.resolve()},
            metadata={
                "input_module": "podcast-index",
                "title": str(episode.get("title") or audio_path.stem),
                "language_hint": language_hint,
                "podcast_title": str(feed.get("title") or ""),
                "episode_url": enclosure_url,
                "feed_url": str(feed.get("url") or ""),
                "published_at": datetime.fromtimestamp(
                    published, tz=timezone.utc
                ).isoformat(),
            },
        )
