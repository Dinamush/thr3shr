"""Remote sample sources for real-life vs anime debug evaluation.

Photos: people/portrait images via LoremFlickr (real Flickr photos) with
optional Wikimedia Commons fallback.
Anime: Safebooru (typical + realism/3d edge-case queries).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .sfw_sources import DEBUG_EVAL_ROOT, USER_AGENT, _download_bytes

logger = logging.getLogger(__name__)

IMAGE_MIME_OK = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}

# People-focused Commons searches (bitmap files only) — used as fallback.
PHOTO_QUERIES = (
    "filetype:bitmap portrait photograph woman",
    "filetype:bitmap portrait photograph man",
    "filetype:bitmap portrait photograph people",
    "filetype:bitmap street photography people",
    "filetype:bitmap candid photograph person",
)

LOREMFlickr_TAGS = (
    "people,portrait",
    "person,face",
    "woman,portrait",
    "man,portrait",
    "crowd,people",
)


@dataclass(frozen=True)
class RealismSample:
    sample_id: str
    label: str  # photo | anime | edge_realistic | edge_3d
    bucket: str  # photo | anime  (ground-truth class for metrics)
    source: str
    file_url: str
    title: str = ""
    query: str = ""


def _http_json(url: str, timeout: float = 90.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_bytes_retry(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    attempts: int = 5,
) -> None:
    """Download with backoff; tolerates Wikimedia/Flickr 429s."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "image/*,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty body")
            dest.write_bytes(data)
            return
        except Exception as err:
            last_err = err
            sleep_s = min(30.0, 1.5 * (2**attempt) + random.random())
            logger.warning(
                "download_retry attempt=%s/%s url=%s err=%s sleep=%.1f",
                attempt + 1,
                attempts,
                url,
                err,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"download failed after {attempts} attempts: {last_err}")


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())[:80]
    return cleaned or "sample"


def fetch_randomuser_people_photos(limit: int) -> list[RealismSample]:
    """Stable remote people portraits (RandomUser CDN stock photos)."""
    samples: list[RealismSample] = []
    # men/women each expose 0..99 portraits on the CDN.
    for i in range(max(0, limit)):
        gender = "women" if i % 2 == 0 else "men"
        idx = (i * 7) % 100
        file_url = f"https://randomuser.me/api/portraits/{gender}/{idx}.jpg"
        samples.append(
            RealismSample(
                sample_id=f"randomuser_{gender}_{idx}",
                label="photo",
                bucket="photo",
                source="randomuser",
                file_url=file_url,
                title=f"{gender}/{idx}",
                query=f"portrait {gender}",
            )
        )
    return samples


def fetch_local_people_photos(limit: int) -> list[RealismSample]:
    """Fallback: local photo folders when remotes fail.

    Set THR3SHR_LOCAL_PHOTO_DIRS to a pathsep/semicolon-separated list of dirs.
    Defaults to ~/Pictures only (no personal album names).
    """
    import os

    raw = os.environ.get("THR3SHR_LOCAL_PHOTO_DIRS", "").strip()
    if raw:
        sep = ";" if ";" in raw else os.pathsep
        roots = [Path(p.strip()) for p in raw.split(sep) if p.strip()]
    else:
        roots = [Path.home() / "Pictures"]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                found.append(path)
            if len(found) >= limit * 3:
                break
        if len(found) >= limit:
            break
    samples: list[RealismSample] = []
    for path in found[:limit]:
        samples.append(
            RealismSample(
                sample_id=_safe_id(f"local_{path.stem}"),
                label="photo",
                bucket="photo",
                source="local_photos",
                file_url=str(path.resolve()),
                title=path.name,
                query=str(path.parent),
            )
        )
    return samples

def fetch_commons_people_photos(limit: int) -> list[RealismSample]:
    """Pull real people photographs from Wikimedia Commons search."""
    if limit < 1:
        return []
    per_query = max(8, (limit + len(PHOTO_QUERIES) - 1) // len(PHOTO_QUERIES))
    samples: list[RealismSample] = []
    seen_urls: set[str] = set()

    for query in PHOTO_QUERIES:
        if len(samples) >= limit:
            break
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(min(50, per_query * 2)),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": "1024",
            }
        )
        url = f"https://commons.wikimedia.org/w/api.php?{params}"
        try:
            payload = _http_json(url)
            time.sleep(0.4)
        except Exception as err:
            logger.warning("commons_search_failed query=%s err=%s", query, err)
            continue
        pages = ((payload or {}).get("query") or {}).get("pages") or {}
        if not isinstance(pages, dict):
            continue
        for page in pages.values():
            if len(samples) >= limit:
                break
            if not isinstance(page, dict):
                continue
            infos = page.get("imageinfo") or []
            if not infos or not isinstance(infos[0], dict):
                continue
            info = infos[0]
            mime = str(info.get("mime") or "").lower()
            if mime not in IMAGE_MIME_OK:
                continue
            file_url = info.get("thumburl") or info.get("url")
            if not file_url or file_url in seen_urls:
                continue
            title = str(page.get("title") or file_url)
            sample_id = _safe_id(f"commons_{page.get('pageid') or title}")
            seen_urls.add(str(file_url))
            samples.append(
                RealismSample(
                    sample_id=sample_id,
                    label="photo",
                    bucket="photo",
                    source="wikimedia_commons",
                    file_url=str(file_url),
                    title=title,
                    query=query,
                )
            )
    return samples[:limit]


def fetch_people_photos(limit: int) -> list[RealismSample]:
    """Prefer RandomUser portraits; pad with Commons then local folders."""
    primary = fetch_randomuser_people_photos(limit)
    if len(primary) >= limit:
        return primary[:limit]
    need = limit - len(primary)
    extras = [
        *fetch_commons_people_photos(need),
        *fetch_local_people_photos(need),
    ]
    seen = {s.sample_id for s in primary}
    for sample in extras:
        if sample.sample_id in seen:
            continue
        primary.append(sample)
        seen.add(sample.sample_id)
        if len(primary) >= limit:
            break
    return primary[:limit]


def fetch_safebooru_anime_bucket(limit: int) -> list[RealismSample]:
    """Pull typical anime + realism/3d edge cases from Safebooru."""
    if limit < 1:
        return []
    typical_n = max(1, limit // 2)
    edge_n = max(1, limit - typical_n)
    per_edge = max(1, (edge_n + 1) // 2)

    samples: list[RealismSample] = []
    seen_ids: set[str] = set()

    def _fetch(tags: str, label: str, want: int) -> None:
        nonlocal samples
        if want < 1:
            return
        params = urllib.parse.urlencode(
            {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "json": "1",
                "limit": str(max(want * 3, want)),
                "tags": f"{tags} rating:safe",
            }
        )
        api = f"https://safebooru.org/index.php?{params}"
        try:
            payload = _http_json(api)
            time.sleep(0.35)
        except urllib.error.HTTPError as err:
            logger.warning("safebooru_failed tags=%s code=%s", tags, err.code)
            return
        except Exception as err:
            logger.warning("safebooru_failed tags=%s err=%s", tags, err)
            return
        if not isinstance(payload, list):
            return
        got = 0
        for row in payload:
            if got >= want or len(samples) >= limit:
                break
            if not isinstance(row, dict):
                continue
            file_url = row.get("sample_url") or row.get("file_url")
            post_id = str(row.get("id") or "")
            if not file_url or not post_id or post_id in seen_ids:
                continue
            seen_ids.add(post_id)
            samples.append(
                RealismSample(
                    sample_id=f"safebooru_{post_id}",
                    label=label,
                    bucket="anime",
                    source="safebooru",
                    file_url=str(file_url),
                    title=post_id,
                    query=tags,
                )
            )
            got += 1

    half = max(1, typical_n // 2)
    _fetch("1girl solo", "anime", half)
    _fetch("2girls", "anime", typical_n - half)
    _fetch("realistic 1girl", "edge_realistic", per_edge)
    _fetch("photorealistic", "edge_realistic", max(0, edge_n - per_edge - per_edge // 2))
    _fetch("3d 1girl", "edge_3d", max(1, edge_n // 3))
    return samples[:limit]


def collect_realism_samples(*, count_per_class: int) -> list[RealismSample]:
    """Balanced photo + anime (incl. edge) sample set."""
    n = max(5, min(int(count_per_class), 40))
    photos = fetch_people_photos(n)
    anime = fetch_safebooru_anime_bucket(n)
    return [*photos, *anime]


def cache_dir_for_realism(source: str) -> Path:
    path = DEBUG_EVAL_ROOT / "realism" / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_realism_sample(sample: RealismSample) -> Path:
    dest_dir = cache_dir_for_realism(sample.source)
    # Local absolute paths: copy into cache so debug preview URLs stay stable.
    if sample.source == "local_photos" or re.match(r"^[A-Za-z]:[\\/]", sample.file_url):
        local = Path(sample.file_url)
        if not local.is_file():
            raise FileNotFoundError(local)
        ext = local.suffix.lower() if local.suffix else ".jpg"
        dest = dest_dir / f"{sample.sample_id}{ext}"
        if not dest.exists() or dest.stat().st_size <= 0:
            dest.write_bytes(local.read_bytes())
        return dest

    ext = Path(urllib.parse.urlparse(sample.file_url).path).suffix or ".jpg"
    if len(ext) > 8 or ext.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    dest = dest_dir / f"{sample.sample_id}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        _download_bytes_retry(sample.file_url, dest)
    except Exception:
        _download_bytes(sample.file_url, dest)
    return dest


def resolve_realism_cached_file(source: str, filename: str) -> Path:
    if not source or not filename:
        raise ValueError("source and filename are required")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("invalid filename")
    base = cache_dir_for_realism(source).resolve()
    candidate = (base / filename).resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("path escapes debug cache")
    if not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate
