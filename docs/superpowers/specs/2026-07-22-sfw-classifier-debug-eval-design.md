# SFW Classifier Debug Eval — Design

Date: 2026-07-22

## Goal

Add a Settings **Classifier debug** panel that downloads SFW images from the web for chosen tags, runs the current tagger, and shows both **tag recall** and a **mini classify preview** (no migrate / no run row).

## Decisions

| Topic | Choice |
|-------|--------|
| Metrics | Recall @ confidence threshold **and** destination-folder dry-run preview |
| Sources (v1) | Safebooru (default) + Danbooru (`rating:g` only); pluggable for more later |
| Sample size | User-selectable **5–30**, default **10** |
| Model / destination tags | Current settings `tagger_model`, `selected_tags`, `confidence_threshold` |
| Persistence | Cache files under `sample_data/debug_evals/` (gitignored); no eval history DB |

## UX

Collapsible section in Settings:

1. Source select: `safebooru` | `danbooru`
2. Tags to pull: chip / tag search (AND’d into the board query)
3. Count slider/input: 5–30 (default 10)
4. Read-only notes: active tagger model; destination tags + threshold used for preview
5. **Fetch & evaluate** button (disabled while in flight / no pull tags)
6. Results:
   - Summary: per pull-tag hit rate (score ≥ threshold among downloaded posts that list that tag)
   - Table: thumb, post id/source, known tags (subset), scores for pull tags, would-be `primary_tag` / `needs_review` / reason

## Backend

### API

`GET /api/debug/sfw-sources` → `[{ id, label, sfw_policy }]`

`POST /api/debug/sfw-eval`

Request:

```json
{
  "source": "safebooru",
  "tags": ["1girl", "solo"],
  "count": 10
}
```

Response: `source`, `tags`, `count`, `tagger_model`, `confidence_threshold`, `recall` (per tag), `items` (per image preview fields), `errors` (partial download failures).

`GET /api/debug/sfw-eval/preview/{token}` — short-lived local file preview for thumbs (or reuse path under sample_data with a safe path check).

### Source plugins

Interface (conceptual):

- `id`, `label`
- `build_sfw_query(tags: list[str]) -> str`
- `fetch_posts(query, limit) -> list[PostMeta]`
- `download(post, dest_dir) -> Path`

v1:

- **Safebooru** — existing dapi JSON; force `rating:safe`
- **Danbooru** — public JSON API; force `rating:g`; reject other ratings

Shared HTTP client with User-Agent; timeouts; rate-limit politely.

### Evaluation

For each downloaded image:

1. `extract_scores` with settings `tagger_model` / `wd_general_threshold`
2. Recall: for each requested tag present in post’s known tags, whether model score ≥ `confidence_threshold`
3. Classify preview: `_classify_from_scores` with current `selected_tags` → matched set (same as runs)

No `runs` / `items` DB rows. No migrate.

### Storage

`sample_data/debug_evals/<source>/<post_id>.<ext>` + optional per-request working set. Already covered by `sample_data/` gitignore.

## Security / safety

- Only registered source plugins (no arbitrary URL fetch from client)
- SFW enforced in query **and** post-rating check
- Preview endpoint path-confined to `sample_data/debug_evals/`
- Feature is debug-only; no auth change (local app)

## Out of scope

- Gelbooru / Konachan (add later via plugin)
- Background jobs / progress polling for eval
- Persisted eval history UI
- Changing production classify defaults based on eval

## Testing

- Unit: query builders force SFW; reject unsafe ratings
- API: mocked HTTP + mocked `extract_scores`; recall math; empty tags → 400
- Smoke: UI wires to API (manual or light component check)
