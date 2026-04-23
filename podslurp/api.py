"""
api.py — PodcastIndex HTTP client.

Authentication uses SHA-1( api_key + api_secret + epoch ) per the
PodcastIndex API spec: https://podcastindex-org.github.io/docs-api/
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import requests

from .config import Config

_BASE_URL = "https://api.podcastindex.org/api/1.0"
_TIMEOUT = 15  # seconds


def _auth_headers(config: Config) -> dict[str, str]:
    epoch = str(int(time.time()))
    auth_hash = hashlib.sha1(
        (config.api_key + config.api_secret + epoch).encode()
    ).hexdigest()
    return {
        "User-Agent": "podslurp/1.0",
        "X-Auth-Key": config.api_key,
        "X-Auth-Date": epoch,
        "Authorization": auth_hash,
    }


def _get(path: str, params: dict[str, Any], config: Config) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    response = requests.get(
        url,
        params=params,
        headers=_auth_headers(config),
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def search_podcasts(query: str, config: Config, max_results: int = 10) -> list[dict]:
    """Search for podcasts by keyword.

    Returns a list of feed dicts with keys: id, title, author, language,
    description, url, episodeCount.
    """
    data = _get(
        "/search/byterm",
        params={"q": query, "max": max_results},
        config=config,
    )
    return data.get("feeds", [])


def get_episodes(
    feed_id: int,
    config: Config,
    max_results: int = 1000,
) -> list[dict]:
    """Return the most recent episodes for a feed.

    Each dict contains: id, title, description, datePublished,
    datePublishedPretty, enclosureUrl, enclosureType, duration,
    feedLanguage, feedTitle.
    """
    data = _get(
        "/episodes/byfeedid",
        params={"id": feed_id, "max": max_results},
        config=config,
    )
    return data.get("items", [])
