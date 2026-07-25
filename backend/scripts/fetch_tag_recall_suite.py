"""Discover/pin suite post IDs and download the tag-recall cache."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tag_recall_eval import (  # noqa: E402
    DEFAULT_SUITE_PATH,
    ensure_suite_cache,
    load_suite,
    rebuild_suite_samples,
    save_suite,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fetch_tag_recall_suite")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE_PATH,
        help="Path to tag_recall_suite.json",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=6,
        help="Target samples per bucket when rebuilding",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Search boards to fill empty sample lists",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="With --rebuild, discard existing pins and re-search",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download cached files even if present",
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    if args.rebuild or args.replace_all:
        suite = rebuild_suite_samples(
            suite,
            samples_per_bucket=args.per_bucket,
            replace_all=bool(args.replace_all),
        )
        save_suite(suite, args.suite)
        logger.info("Wrote pinned suite to %s", args.suite)

    samples, errors = ensure_suite_cache(suite, refresh=args.refresh)
    total_pinned = sum(len(b.get("samples") or []) for b in suite.get("buckets") or [])
    logger.info(
        "Cache ready: %d files, %d pinned samples, %d errors",
        len(samples),
        total_pinned,
        len(errors),
    )
    for err in errors[:20]:
        logger.warning("%s", err)
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
