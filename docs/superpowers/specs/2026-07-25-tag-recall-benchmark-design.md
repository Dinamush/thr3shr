# Tag Recall Benchmark (ML-Danbooru vs WD SwinV2) — Design

Date: 2026-07-25

## Goal

Compare **ml_danbooru** vs **wd_swinv2_v3** on a **fixed curated image suite**, measuring ability to recover each sample’s **desired tags** via:

- **recall@threshold** — tag score ≥ configured threshold
- **recall@top‑K** — tag appears in the model’s top‑K scores (default K=20)

## Decisions

| Topic | Choice |
|-------|--------|
| Models (v1) | `ml_danbooru`, `wd_swinv2_v3` |
| Samples | Fixed curated suite (pinned Danbooru/Safebooru post IDs) |
| Delivery | CLI + Debug UI |
| Ground truth | Manifest `desired_tags` per sample (optionally verified against post tags at fetch) |
| Threshold default | `0.35` (WD-style general); overrideable |
| Top‑K default | `20` |
| Cache | `sample_data/tag_recall_suite/` (gitignored images); suite JSON committed |
| Out of scope v1 | Taxonomy folder routing, EVA02, live custom pulls |

## Suite format

`backend/app/data/tag_recall_suite.json`:

- `buckets[]`: `{ id, desired_tags[], samples[] }`
- `samples[]`: `{ source, post_id, desired_tags? }`
- Sources: `danbooru` | `safebooru`
- Fetch resolves post → file URL → cache path `sample_data/tag_recall_suite/{source}/{post_id}{ext}`

Buckets (v1): control (`1girl`+`solo`), `monster_girl`, `android`, `robot_girl`, `tentacle_sex`, `fellatio`, `nakadashi`/`cum_in_pussy`, `pokemon_(creature)`, `furry`, `slime_girl`.

## Scoring

Per model, per (sample, desired_tag):

- `@threshold` hit if score ≥ threshold
- `@topK` hit if tag in top‑K by score

Aggregates: micro/macro recall for each metric; per-tag rates; pairwise win/tie/loss (micro recall@threshold primary).

## API

`POST /api/debug/tag-recall-eval`

```json
{
  "models": ["ml_danbooru", "wd_swinv2_v3"],
  "threshold": 0.35,
  "top_k": 20,
  "refresh_cache": false
}
```

Response: per-model summaries + comparison + optional item details (scores for desired tags only to keep payload small).

`GET /api/debug/tag-recall-eval/preview/{source}/{file_name}` — cached thumb.

## CLI

- `scripts/fetch_tag_recall_suite.py` — download/refresh cache from pinned IDs (or expand searches into IDs)
- `scripts/run_tag_recall_eval.py` — run eval → `scripts/out/tag_recall_eval_latest.json`

## UI

Debug page section: threshold / top‑K controls, **Run tag recall benchmark**, side-by-side summary table, per-tag expand.

## False-positive companion

Same suite; probe taxonomy evidence tags for Settings `selected_tags` (min weight default 0.85).

- `POST /api/debug/tag-fp-eval`
- CLI: `scripts/run_tag_fp_eval.py`
- Debug UI: **Tag false-positive benchmark** (micro/macro FPR, top FP tags, folder route FPs; optional second pass at `wd_general_threshold`)
