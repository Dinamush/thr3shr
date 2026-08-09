# Drawn taxonomy: `group_sex` + evidence refresh

**Date:** 2026-08-09  
**Status:** Approved for planning  
**Scope:** Drawn/anime `backend/app/data/taxonomy.json` funnel (+ tests). Voyeur tree unchanged. Real-life taxonomy out of scope.

## Problem

Run review showed many “no matching tags” / weak routing cases where:

1. Strong **group** signals (`gangbang`, `group_sex`, `threesome`, `multiple_boys`, …) only reinforced catch-all **`sex`**, with no dedicated destination.
2. **`milf`** evidence was essentially a single tag (`mature_female`), so mature women rarely won.
3. Other favourite folders (`nakadashi`, `fellatio`, `loli`, `shota`, `incest`, `fertilization`, `bestiality`, `Pokemon`, `furry`, `android`, `tentacles`, `monster_girl`, `NTR`, `paizuri`, `footjob`) exist but need denser evidence / cleaner vetoes so they capture comprehensively without stealing Voyeur soft wins.

## Goals

- Add a first-class **`group_sex`** destination that beats vanilla **`sex`** and most other act/milf competition when group evidence is strong.
- Encode an explicit override list that **beats `group_sex`**.
- Expand evidence (and ignore/veto hygiene) for the listed folders so recall improves on real WD score maps.
- Leave **Voyeur** and all `Voyeur/*` buckets unchanged.
- Keep the existing scoring engine (`score_bucket` / `choose_best_destination`); prefer taxonomy JSON + minimal role tweaks over a new precedence system.

## Non-goals

- Changing Voyeur nested taxonomy.
- Real-life adult taxonomy / VLM pipeline.
- UI redesign (aside from `group_sex` appearing in tag search once present in taxonomy).
- Guaranteeing 100% capture of images with no mappable WD tags.

## Precedence (product rules)

When **`group_sex`** would otherwise win:

| Beats `group_sex` (override) | Loses to `group_sex` |
|---|---|
| `loli`, `shota` | `sex` |
| `incest` | `milf` |
| `bestiality` | `fellatio`, `nakadashi`, `paizuri`, `footjob` |
| `NTR` | `fertilization` (same tier as other non-override acts) |
| `Pokemon`, `furry`, `monster_girl`, `android` | |
| `tentacles` | |

Character ladder remains: **`loli` / `shota` still beat acts** via `prefer_character_over_act`.  
**`milf` must not auto-beat `group_sex`** via the character ladder (see Implementation).

## Approach

**Veto + small role tweak** (chosen over full precedence tiers):

1. New **`group_sex`** act bucket.
2. Move group markers off the effective win path for **`sex`** via **`sex.veto`** (and remove or heavily down-rank group tags from `sex.evidence` so they do not inflate sex when group_sex is not selected).
3. **`group_sex.veto`** lists key tags from override folders so those buckets can win when their signals fire.
4. Demote **`milf.role`** from `character` → `theme` so milf no longer yields-steals from acts through the character ladder; milf still competes on score when group_sex is absent/weak.

## Implementation plan (spec level)

### 1. New bucket: `group_sex`

Suggested fields:

- `id` / `folder`: `group_sex`
- `role`: `act`
- `priority`: between specific acts and `sex` (e.g. `13`–`14` band; must be **better than `sex`** on ties — lower number wins today)
- `aliases`: `gangbang`, `Group_sex`, `orgy`, …
- `evidence` (high confidence): `gangbang`, `group_sex`, `orgy`, `threesome`, `mmf_threesome`, `ffm_threesome`, `spitroast`, related explicit group acts
- `gate_tags` + `gated_evidence` (optional): count cues like `multiple_boys`, `3boys`, `4boys`, `6+boys`, `multiple_girls` only when a sex/penis/vaginal/rape gate is present (avoid multi-character SFW false wins)
- `ignore`: bare `1boy`/`2boys`/`hetero`/`penis` without group markers
- `veto`: key tags from override folders, e.g.:
  - loli/shota: `loli`, `shota`, `oppai_loli`, `onee-shota`, …
  - incest: `incest`, `twincest`, `mother_and_son`, …
  - bestiality: `bestiality`, `animal_penis`, `knot`, …
  - NTR: `netorare`, `cheating_(relationship)`
  - Pokemon/furry/monster_girl/android/tentacles: primary evidence tags already used by those buckets (`pokemon_(creature)`, `pokephilia`, `furry`, `monster_girl`, `android`, `tentacle_sex`, …)
- `veto_threshold`: modest (align with existing act veto practice; avoid silencing on noise — prefer ~0.3–0.45 for rare override tags)

### 2. Update `sex`

- Add group markers to **`sex.veto`**: `gangbang`, `group_sex`, `orgy`, `threesome`, `mmf_threesome`, `ffm_threesome`, `spitroast` (and similar).
- Remove `group_sex` (and peers) from **`sex.evidence`** or drop their weights so they no longer help `sex` win when `group_sex` is not in the selected set.
- Keep `sex` as the vanilla / non-group catch-all.

### 3. Update `milf`

- `role`: `theme` (was `character`).
- Expand `evidence` beyond `mature_female` with high-precision mature cues; keep aggressive `ignore` for `loli`, `shota`, `oppai_loli`, age-regression, etc.
- Do **not** treat weak `motherly` / `age_difference` alone as enough to win.

### 4. Evidence refresh (listed folders)

For each of: `nakadashi`, `fellatio`, `loli`, `shota`, `incest`, `fertilization`, `bestiality`, `Pokemon`, `furry`, `android`, `tentacles`, `monster_girl`, `NTR`, `paizuri`, `footjob`:

- Add missing high-precision Danbooru aliases / variants already common in WD outputs.
- Prefer **gated_evidence** for ambiguous species/count/body tags (pattern already used by bestiality / tentacles / android).
- Keep / extend **ignore** lists so soft body tags cannot solo a folder.
- Ensure override folders’ **primary evidence tags** appear on `group_sex.veto` so precedence holds even when scores are close.

**Voyeur / `Voyeur/*`:** no edits.

### 5. Engine / app wiring

- Prefer **JSON-only** changes if veto + milf role suffice.
- If tests show character ladder or selection expansion still mis-orders `milf` vs `group_sex`, add a minimal `choose_best_destination` special case — only after JSON approach fails.
- Update `DOUJIN_FAVOURITE_FOLDERS` to include `group_sex` if Doujin mode should route there.
- Frontend default tag chips: optional follow-up; not required for backend correctness.
- `reload_taxonomy` / process restart after JSON edit in running servers.

### 6. Tests

Extend `backend/tests/test_taxonomy.py` and/or `test_classify_accuracy.py` matrix:

| Scores (sketch) | Expected folder |
|---|---|
| `gangbang` / `group_sex` strong | `group_sex` |
| `sex`/`vaginal` only | `sex` |
| `gangbang` + `loli` | `loli` |
| `gangbang` + `shota` | `shota` |
| `gangbang` + `incest` | `incest` |
| `gangbang` + `netorare` | `NTR` |
| `gangbang` + `bestiality` | `bestiality` |
| `gangbang` + `mature_female` (milf) | `group_sex` |
| `gangbang` + `pokemon_(creature)` / `furry` / `monster_girl` / `android` / `tentacle_sex` | matching override folder |
| `fellatio` strong, no group | `fellatio` |
| Voyeur soft cues only | unchanged Voyeur behavior |

Also assert `resolve_taxonomy_folder("group_sex")` / aliases and that selecting `group_sex` participates in `expand_selected_folders` if grouped (N/A unless `group` set).

### 7. Validation on real data (manual)

After implement: reclassify needs-review from run 125/126 sample or a small inbox slice; confirm group-heavy items land in `group_sex` unless override tags fire; confirm Voyeur regressions absent.

## Risks

- **Veto noise:** a weak `furry`/`android` score could zero `group_sex` incorrectly → tune `veto_threshold` and only veto on high-precision tags.
- **milf role change:** milf may lose more often to other acts when group is absent; mitigate with stronger milf evidence, not by restoring character role.
- **Selected-tag set:** users must add `group_sex` to selected tags (or Doujin favourites) for it to compete.
- **Incomplete WD vocab:** images with only `penis`/`hetero` and no `sex`/`gangbang` may still miss; out of scope beyond evidence expansion.

## Success criteria

- Dedicated `group_sex` folder exists and wins on clear group WD maps when selected.
- Override precedence matches the table above in unit tests.
- milf no longer blocks `group_sex` via character ladder.
- Evidence expansions improve hit-rate on held-out no-match samples without Voyeur regressions.
- Existing taxonomy unit tests remain green; new matrix cases pass.

## Open follow-ups (non-blocking)

- Whether `fertilization` should later join the override list (currently loses to `group_sex` per product choice).
- Optional UI preset chip for `group_sex`.
- Later: explicit precedence tiers if veto lists become unmaintainable.
