from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

USER_AGENT = "thr3shr-debug/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_EVAL_ROOT = REPO_ROOT / "sample_data" / "debug_evals"


@dataclass(frozen=True)
class SfwPost:
    source_id: str
    post_id: str
    file_url: str
    rating: str
    tags: list[str]


@dataclass(frozen=True)
class SfwSourceInfo:
    id: str
    label: str
    sfw_policy: str
    max_content_tags: int | None = None


class SfwSource(Protocol):
    info: SfwSourceInfo

    def build_query(self, tags: list[str]) -> str: ...

    def fetch_posts(self, tags: list[str], limit: int) -> list[SfwPost]: ...

    def is_sfw_rating(self, rating: str | None) -> bool: ...


def _http_json(url: str, timeout: float = 90.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_bytes(url: str, dest: Path, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def _normalize_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw).strip().replace(" ", "_")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return cleaned


class SafebooruSource:
    info = SfwSourceInfo(
        id="safebooru",
        label="Safebooru",
        sfw_policy="rating:safe",
        max_content_tags=None,
    )
    _API = "https://safebooru.org/index.php"

    def is_sfw_rating(self, rating: str | None) -> bool:
        return str(rating or "").strip().lower() in {"s", "safe", "g", "general"}

    def build_query(self, tags: list[str]) -> str:
        content = _normalize_tags(tags)
        parts = [*content, "rating:safe"]
        return " ".join(parts)

    def fetch_posts(self, tags: list[str], limit: int) -> list[SfwPost]:
        query = self.build_query(tags)
        # Over-fetch slightly so rating filters / missing URLs do not undershoot.
        params = urllib.parse.urlencode(
            {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "json": "1",
                "limit": str(max(limit * 2, limit)),
                "tags": query,
            }
        )
        try:
            payload = _http_json(f"{self._API}?{params}")
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"Safebooru HTTP {err.code}") from err
        if not isinstance(payload, list):
            raise RuntimeError("Safebooru returned unexpected payload")

        posts: list[SfwPost] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            rating = str(row.get("rating") or "")
            if not self.is_sfw_rating(rating):
                continue
            file_url = row.get("sample_url") or row.get("file_url")
            if not file_url:
                continue
            post_id = str(row.get("id") or "")
            if not post_id:
                continue
            known = str(row.get("tags") or "").split()
            posts.append(
                SfwPost(
                    source_id=self.info.id,
                    post_id=post_id,
                    file_url=str(file_url),
                    rating=rating,
                    tags=known,
                )
            )
            if len(posts) >= limit:
                break
        return posts


class DanbooruSource:
    info = SfwSourceInfo(
        id="danbooru",
        label="Danbooru",
        sfw_policy="rating:g",
        # Unauthenticated Danbooru allows ~2 content tags (+ rating often free).
        max_content_tags=2,
    )
    _API = "https://danbooru.donmai.us/posts.json"

    def is_sfw_rating(self, rating: str | None) -> bool:
        return str(rating or "").strip().lower() in {"g", "general"}

    def build_query(self, tags: list[str]) -> str:
        content = _normalize_tags(tags)
        max_tags = self.info.max_content_tags
        if max_tags is not None:
            content = content[:max_tags]
        parts = [*content, "rating:g"]
        return " ".join(parts)

    def fetch_posts(self, tags: list[str], limit: int) -> list[SfwPost]:
        query = self.build_query(tags)
        params = urllib.parse.urlencode(
            {
                "tags": query,
                "limit": str(max(limit * 2, limit)),
            }
        )
        try:
            payload = _http_json(f"{self._API}?{params}")
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"Danbooru HTTP {err.code}") from err
        if not isinstance(payload, list):
            raise RuntimeError("Danbooru returned unexpected payload")

        posts: list[SfwPost] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            rating = str(row.get("rating") or "")
            if not self.is_sfw_rating(rating):
                continue
            if row.get("is_deleted"):
                continue
            file_url = row.get("file_url") or row.get("large_file_url") or row.get("preview_file_url")
            if not file_url:
                continue
            post_id = str(row.get("id") or "")
            if not post_id:
                continue
            known = str(row.get("tag_string") or "").split()
            posts.append(
                SfwPost(
                    source_id=self.info.id,
                    post_id=post_id,
                    file_url=str(file_url),
                    rating=rating,
                    tags=known,
                )
            )
            if len(posts) >= limit:
                break
        return posts


_SOURCES: dict[str, SfwSource] = {
    "safebooru": SafebooruSource(),
    "danbooru": DanbooruSource(),
}


def list_sources() -> list[SfwSourceInfo]:
    return [src.info for src in _SOURCES.values()]


def get_source(source_id: str) -> SfwSource:
    key = str(source_id or "").strip().lower()
    source = _SOURCES.get(key)
    if source is None:
        raise KeyError(f"Unknown SFW source: {source_id}")
    return source


def cache_dir_for(source_id: str) -> Path:
    path = DEBUG_EVAL_ROOT / source_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_post(post: SfwPost) -> Path:
    dest_dir = cache_dir_for(post.source_id)
    ext = Path(urllib.parse.urlparse(post.file_url).path).suffix or ".jpg"
    if len(ext) > 8:
        ext = ".jpg"
    dest = dest_dir / f"{post.post_id}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _download_bytes(post.file_url, dest)
    return dest


def resolve_cached_file(source_id: str, filename: str) -> Path:
    """Return a path under debug_evals or raise ValueError if unsafe/missing."""
    if not source_id or not filename:
        raise ValueError("source and filename are required")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("invalid filename")
    base = cache_dir_for(source_id).resolve()
    candidate = (base / filename).resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("path escapes debug cache")
    if not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate
