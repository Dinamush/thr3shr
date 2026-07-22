#!/usr/bin/env python3
"""
Comprehensive multi-model benchmark (benchmark-only; no production wiring).

Models (via dghs-imgutils where available):
  - WD SwinV2 v3, EVA02 Large, ViT Large, ConvNext v3
  - ML-Danbooru
  - Camie (initial + refined, macro_opt mode)
  - PixAI v0.9

Metrics on local SFW Safebooru samples:
  - latency (warm + infer)
  - known-tag recall for site tags 1girl/solo @ threshold
  - taxonomy evidence-tag coverage (vocab)
  - taxonomy evidence hit-rate: how often any evidence tag for each
    destination appears above threshold on images that carry that tag
    in the Safebooru tag string (when present)

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/bench_models_comprehensive.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SAMPLE_DIR = ROOT / "sample_data" / "sfw_safebooru"
META_PATH = SAMPLE_DIR / "manifest.json"
TAXONOMY_PATH = BACKEND / "app" / "data" / "taxonomy.json"
OUT_PATH = ROOT / "sample_data" / "bench_models_report.json"

GENERAL_THRESHOLD = 0.35


def _normalize(tag: str) -> str:
    text = tag.strip().lower()
    parts: list[str] = []
    for ch in text:
        if ch.isalnum():
            parts.append(ch)
        elif ch in {" ", "-", ".", "/", "_"}:
            parts.append("_")
    return "".join(parts).strip("_")


def _as_general_dict(raw: object) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(raw, dict):
        for tag, score in raw.items():
            scores[_normalize(str(tag))] = float(score)
        return scores
    if isinstance(raw, (list, tuple)):
        for part in raw:
            if isinstance(part, dict):
                for tag, score in part.items():
                    key = _normalize(str(tag))
                    scores[key] = max(scores.get(key, 0.0), float(score))
    return scores


def load_taxonomy_evidence() -> dict[str, list[str]]:
    tax = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for bucket in tax["buckets"]:
        out[bucket["folder"]] = [
            _normalize(e["tag"]) for e in bucket.get("evidence", [])
        ]
    return out


def score_with_model(model_id: str, path: Path) -> dict[str, float]:
    if model_id.startswith("wd_"):
        from imgutils.tagging import get_wd14_tags

        mapping = {
            "wd_swinv2_v3": "SwinV2_v3",
            "wd_eva02_large": "EVA02_Large",
            "wd_vit_large": "ViT_Large",
            "wd_convnext_v3": "ConvNext_v3",
        }
        raw = get_wd14_tags(
            str(path),
            model_name=mapping[model_id],
            general_threshold=GENERAL_THRESHOLD,
            no_underline=False,
            drop_overlap=False,
            fmt="general",
        )
        return _as_general_dict(raw)

    if model_id == "ml_danbooru":
        from imgutils.tagging import get_mldanbooru_tags

        raw = get_mldanbooru_tags(
            str(path),
            threshold=GENERAL_THRESHOLD,
            size=448,
            keep_ratio=True,
            drop_overlap=False,
            use_real_name=False,
        )
        return _as_general_dict(raw)

    if model_id.startswith("camie_"):
        from imgutils.tagging import get_camie_tags

        # camie_initial_macro_opt / camie_refined_balanced
        name = None
        mode = None
        for candidate in ("initial", "refined"):
            prefix = f"camie_{candidate}_"
            if model_id.startswith(prefix):
                name = candidate
                mode = model_id[len(prefix) :]
                break
        if name is None or not mode:
            raise ValueError(f"Bad camie model id: {model_id}")
        raw = get_camie_tags(
            str(path),
            model_name=name,
            mode=mode,  # type: ignore[arg-type]
            no_underline=False,
            drop_overlap=False,
            fmt="general",
        )
        return _as_general_dict(raw)

    if model_id == "pixai_v0_9":
        from imgutils.tagging import get_pixai_tags

        raw = get_pixai_tags(
            str(path),
            model_name="v0.9",
            thresholds=GENERAL_THRESHOLD,
            fmt="general",
        )
        return _as_general_dict(raw)

    raise ValueError(f"unknown model_id={model_id}")


def vocab_for_model(model_id: str) -> set[str] | None:
    """Return tag vocabulary when cheaply available; None if unknown."""
    try:
        if model_id.startswith("wd_"):
            from huggingface_hub import hf_hub_download
            import pandas as pd

            repos = {
                "wd_swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
                "wd_eva02_large": "SmilingWolf/wd-eva02-large-tagger-v3",
                "wd_vit_large": "SmilingWolf/wd-vit-large-tagger-v3",
                "wd_convnext_v3": "SmilingWolf/wd-convnext-tagger-v3",
            }
            path = hf_hub_download(repos[model_id], "selected_tags.csv")
            return {_normalize(t) for t in pd.read_csv(path)["name"].tolist()}

        if model_id == "ml_danbooru":
            from huggingface_hub import hf_hub_download
            import pandas as pd

            path = hf_hub_download(
                "deepghs/imgutils-models", "mldanbooru/mldanbooru_tags.csv"
            )
            return {_normalize(t) for t in pd.read_csv(path)["name"].tolist()}

        if model_id.startswith("camie_"):
            from huggingface_hub import hf_hub_download
            import pandas as pd

            name = "refined" if "refined" in model_id else "initial"
            path = hf_hub_download(
                "deepghs/camie_tagger_onnx", f"{name}/selected_tags.csv"
            )
            return {_normalize(t) for t in pd.read_csv(path)["name"].tolist()}

        if model_id == "pixai_v0_9":
            from imgutils.tagging import pixai

            tags_df, _ips = pixai._open_tags("v0.9")
            return {_normalize(str(t)) for t in tags_df["name"].tolist()}
    except Exception as err:
        print(f"  vocab_error {model_id}: {err}", flush=True)
        return None
    return None


def warm_model(model_id: str, path: Path) -> float:
    t0 = time.perf_counter()
    score_with_model(model_id, path)
    return time.perf_counter() - t0


def main() -> int:
    # Match production DLL search so imgutils ORT sessions can use CUDA.
    from app.providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()

    if not META_PATH.exists():
        print(
            f"ERROR: missing {META_PATH}. Run scripts/bench_sfw_sample.py first.",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(META_PATH.read_text(encoding="utf-8"))
    paths = [Path(row["path"]) for row in manifest]
    evidence = load_taxonomy_evidence()
    all_evidence = sorted({t for tags in evidence.values() for t in tags})

    models = [
        "wd_swinv2_v3",
        "wd_eva02_large",
        "wd_vit_large",
        "wd_convnext_v3",
        "ml_danbooru",
        "camie_initial_macro_opt",
        "camie_refined_macro_opt",
        "pixai_v0_9",
    ]

    report: dict[str, object] = {
        "general_threshold": GENERAL_THRESHOLD,
        "sample_count": len(paths),
        "sample_dir": str(SAMPLE_DIR),
        "models": {},
        "notes": [
            "Camie v2 (Camais03/camie-tagger-v2) is not in dghs-imgutils 0.19; "
            "benchmarked ONNX initial/refined mirrors instead.",
            "Published F1 numbers are not directly comparable across vendors "
            "(different splits/thresholds/tag sets).",
            "SFW-only images; taxonomy NSFW folders measured via vocab coverage "
            "and evidence-tag presence on images that carry those tags.",
        ],
    }

    print(f"samples={len(paths)} models={len(models)}", flush=True)

    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        entry: dict[str, object] = {"ok": False}
        try:
            vocab = vocab_for_model(model_id)
            if vocab is not None:
                covered = [t for t in all_evidence if t in vocab]
                missing = [t for t in all_evidence if t not in vocab]
                per_folder = {}
                for folder, tags in evidence.items():
                    hit = [t for t in tags if t in vocab]
                    per_folder[folder] = {
                        "coverage": len(hit) / max(1, len(tags)),
                        "present": hit,
                        "missing": [t for t in tags if t not in vocab],
                    }
                entry["vocab_size"] = len(vocab)
                entry["taxonomy_evidence_coverage"] = len(covered) / max(
                    1, len(all_evidence)
                )
                entry["taxonomy_missing"] = missing
                entry["taxonomy_per_folder"] = per_folder
                print(
                    f"  vocab={len(vocab)} evidence_coverage="
                    f"{entry['taxonomy_evidence_coverage']:.1%}",
                    flush=True,
                )

            warm_s = warm_model(model_id, paths[0])
            print(f"  warm={warm_s:.2f}s", flush=True)

            must_hits = 0
            latencies: list[float] = []
            per_image: list[dict] = []
            # taxonomy evidence recall on images whose site tags include evidence
            tax_denom = {f: 0 for f in evidence}
            tax_numer = {f: 0 for f in evidence}

            for row, path in zip(manifest, paths):
                t0 = time.perf_counter()
                scores = score_with_model(model_id, path)
                dt = time.perf_counter() - t0
                latencies.append(dt)

                must = {_normalize(t) for t in row.get("must_have", [])}
                known = {_normalize(t) for t in row.get("known_general", [])}
                hit = must <= set(scores)
                must_hits += int(hit)
                top = sorted(scores.items(), key=lambda x: -x[1])[:8]

                for folder, ev_tags in evidence.items():
                    # Only evaluate folders whose evidence appears in site tags
                    if not (known & set(ev_tags)):
                        continue
                    tax_denom[folder] += 1
                    if any(scores.get(t, 0.0) >= GENERAL_THRESHOLD for t in ev_tags):
                        tax_numer[folder] += 1

                per_image.append(
                    {
                        "id": row["id"],
                        "must_hit": hit,
                        "latency_s": dt,
                        "top": top,
                        "score_count": len(scores),
                    }
                )
                print(
                    f"  #{row['id']} must_hit={hit} n={len(scores)} "
                    f"{dt*1000:.0f}ms top={[t for t,_ in top[:5]]}",
                    flush=True,
                )

            tax_recall = {
                f: (tax_numer[f] / tax_denom[f] if tax_denom[f] else None)
                for f in evidence
            }
            entry.update(
                {
                    "ok": True,
                    "warm_s": warm_s,
                    "infer_total_s": sum(latencies),
                    "ms_per_image": (sum(latencies) / len(latencies)) * 1000.0,
                    "must_have_recall": must_hits / len(paths),
                    "taxonomy_evidence_recall_on_labeled": tax_recall,
                    "taxonomy_evidence_support": tax_denom,
                    "per_image": per_image,
                }
            )
            print(
                f"  SUMMARY recall={entry['must_have_recall']:.0%} "
                f"ms/img={entry['ms_per_image']:.1f}",
                flush=True,
            )
        except Exception as err:
            entry["ok"] = False
            entry["error"] = f"{type(err).__name__}: {err}"
            entry["traceback"] = traceback.format_exc()
            print(f"  FAILED: {entry['error']}", flush=True)

        report["models"][model_id] = entry

    # Ranking: prioritize must_have recall, then taxonomy coverage, then speed
    ranked = []
    for model_id, entry in report["models"].items():
        if not entry.get("ok"):
            continue
        ranked.append(
            {
                "model": model_id,
                "must_have_recall": entry.get("must_have_recall"),
                "taxonomy_evidence_coverage": entry.get("taxonomy_evidence_coverage"),
                "ms_per_image": entry.get("ms_per_image"),
                "vocab_size": entry.get("vocab_size"),
            }
        )
    ranked.sort(
        key=lambda r: (
            -(r["must_have_recall"] or 0),
            -(r["taxonomy_evidence_coverage"] or 0),
            r["ms_per_image"] or 1e9,
        )
    )
    report["ranking"] = ranked

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}", flush=True)
    print("\n=== RANKING ===", flush=True)
    for i, row in enumerate(ranked, 1):
        print(
            f"{i}. {row['model']}: recall={row['must_have_recall']:.0%} "
            f"tax_cov={row['taxonomy_evidence_coverage']:.0%} "
            f"ms/img={row['ms_per_image']:.1f} vocab={row['vocab_size']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
