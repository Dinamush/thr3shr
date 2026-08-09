# Drawn taxonomy: strengthen strict-safe `SFW` catch-all

**Date:** 2026-08-09  
**Status:** Approved for planning  
**Scope:** Drawn/anime `SFW` fallback (+ ensure `scenery` / `comic` compete). Real-life out of scope.

## Problem

Classify runs often leave plain character art in needs-review (“no matching tags”) because selected destinations are NSFW-heavy. A fallback `SFW` folder already exists, but:

1. It may not be in the user’s selected tags, so it never competes.
2. Evidence / veto hygiene should match a **strict-safe** product rule (no swimsuit/lingerie/fanservice dump).
3. `scenery` and `comic` must remain specialized fallbacks above generic `SFW`.

## Goals

- Route **strict-safe** drawn character art into folder `SFW` when selected.
- Keep `scenery` and `comic` as separate, higher-specificity fallbacks.
- Preserve role ladder: character / act / theme / soft (Voyeur) all beat `SFW`.
- Prefer taxonomy JSON + settings/selection wiring; no engine “force SFW if nothing matched.”

## Non-goals

- Mild fanservice / swimsuit / lingerie as acceptable `SFW` (those must not win `SFW`).
- Real-life tagging domain.
- Merging scenery/comic into `SFW`.
- Guaranteeing every unscored image lands somewhere (garbage / no WD cues may still needs-review).

## Product rules (approved)

| Choice | Decision |
|---|---|
| Safety bar | **Strict safe** (option 1) |
| Folder | Keep existing **`SFW`** |
| Scenery / comic | **Keep separate**; they beat generic SFW |
| Approach | **B — strengthen existing SFW** + ensure selection |

## Current baseline

- `SFW`: `role: fallback`, `priority: 95`, person/portrait evidence, `veto_threshold: 0.35`, broad NSFW/suggestive veto list.
- `scenery` / `comic`: `role: fallback`, priorities 90 / 91.
- Role ladder already makes `fallback` yield to character, act, theme, and soft.
- Tests already cover yield-to-others, comic/scenery vs SFW, and some explicit veto refusals.

## Implementation plan (spec level)

### 1. Taxonomy: `SFW` bucket

In `backend/app/data/taxonomy.json`:

- Keep `folder: "SFW"`, `role: "fallback"`, priority after scenery/comic (~95).
- **Evidence:** retain strong person/portrait cues (`1girl`, `1boy`, multi-person counts, `solo`, `portrait`, `upper_body`, `full_body`, `chibi`). Add only high-precision safe-art tags present in tagger vocab that do not alone pull spicy images (skip noisy tags like bare `looking_at_viewer` if they frequently co-occur with NSFW).
- **Veto:** keep strict blockers; ensure coverage for:
  - nudity / genitals / sex acts / cum / censor bars
  - underwear / lingerie / bikini / swimsuit / panties / leotard
  - focus/suggestive: `ass_focus`, `cameltoe`, `cleavage`, `sexually_suggestive`, erection/bulge, spread_legs
  - specialty NSFW already listed (`bestiality`, `loli`, `shota` as veto noise guards — character folders still win via ladder when selected)
- Do **not** edit Voyeur or act evidence/veto except if a shared soft_veto gap is discovered (prefer SFW-local veto).

### 2. Taxonomy: `scenery` / `comic`

- No structural role changes.
- Only touch if a test proves they fail to beat `SFW` after evidence edits (unlikely).

### 3. Selection wiring

- Add `SFW`, `scenery`, and `comic` to the user’s saved selected tags (settings) so classify runs include them.
- If the frontend has a hard-coded default chip list for drawn classify, add the same three there for new installs; do not remove the user’s NSFW favourites.
- Doujin favourites: **out of scope** unless product later wants SFW for doujin works (comics usually want act routing).

### 4. Engine

- No `choose_best_destination` special case.
- Rely on fallback role + veto + selection.

### 5. Tests

Extend `backend/tests/test_taxonomy.py` (existing `FALLBACKS` block):

| Case | Expected |
|---|---|
| `1girl` + `solo` strong | `SFW` |
| `1girl` + `sex` / `loli` / Voyeur / theme | other folder wins |
| comic / scenery cues | `comic` / `scenery` |
| `1girl` + `nipples` / `penis` / `cameltoe` | `None` (not SFW) |
| `1girl` + `bikini` or `lingerie` or `swimsuit` | `None` (strict) |

### 6. Manual check

After implement: reclassify a small slice of needs-review plain portraits with `SFW` selected; confirm swimsuit/lingerie do not file as SFW; confirm a sex/Voyeur hit still wins when those folders are selected.

## Risks

- **Over-veto:** some fully clothed art with weak `cleavage` noise may miss SFW → tune veto_threshold / drop only the noisiest veto tags if recall tanks (prefer keeping strict).
- **Under-selection:** if `SFW` is not selected, behavior unchanged — wiring step is required.
- **soft_alone_weight:** weak evidence tags must not solo SFW without corroboration (existing engine rule).

## Success criteria

- With `SFW` (+ scenery/comic) selected, plain safe character art routes to `SFW`.
- Strict vetoes keep lingerie/swim/explicit out of `SFW`.
- Acts, themes, Voyeur, scenery, comic still outrank `SFW` in unit tests.
- Existing taxonomy suite stays green.

## Open follow-ups (non-blocking)

- UI preset “Safe art pack” chip group.
- Optional later: rating-model signal if WD alone is insufficient.
