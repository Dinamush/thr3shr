# Drawn Taxonomy `group_sex` + Evidence Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `group_sex` destination with override precedence, demote `milf` off the character ladder, and expand evidence for favourite folders — without touching Voyeur.

**Architecture:** Prefer taxonomy JSON changes only. Use `veto` so override folders silence `group_sex`, and so group markers silence `sex` / non-override acts / `milf`. Keep `score_bucket` / `choose_best_destination` unchanged unless tests prove a JSON-only approach cannot encode precedence.

**Tech Stack:** Python 3, FastAPI backend, `backend/app/data/taxonomy.json`, pytest (`backend/tests/test_taxonomy.py`, `test_doujin_works.py`).

## Global Constraints

- Do **not** edit any Voyeur / `Voyeur/*` bucket in `taxonomy.json`.
- Do **not** change real-life taxonomy / VLM pipeline.
- Prefer JSON + favourites list; touch `taxonomy.py` only if Task 5 proves necessary.
- Every new evidence / veto / gate tag must exist in a tagger vocabulary (`tags.csv` or cached `selected_tags.csv`) — existing test `test_every_evidence_tag_exists_in_a_tagger_vocabulary` must stay green.
- `group_sex` must be selected (or in Doujin favourites) to compete.

---

## File map

| File | Responsibility |
|---|---|
| `backend/app/data/taxonomy.json` | New `group_sex` bucket; update `sex`, `milf`, listed favourites’ evidence/ignore/veto |
| `backend/app/doujin_works.py` | Add `group_sex` to `DOUJIN_FAVOURITE_FOLDERS` |
| `backend/tests/test_taxonomy.py` | Precedence matrix + resolve/alias tests |
| `backend/tests/test_doujin_works.py` | Favourites include `group_sex`; gangbang routes there |
| `backend/app/taxonomy.py` | Touch **only if** JSON veto/role approach fails Task 5 |

---

### Task 1: Failing precedence tests for `group_sex`

**Files:**
- Modify: `backend/tests/test_taxonomy.py` (append near other routing tests, after `test_default_taxonomy_json_loads` helpers / before Voyeur block is fine)
- Test: `backend/tests/test_taxonomy.py`

**Interfaces:**
- Consumes: `choose_best_destination(scores, selected) -> (folder, score, secondary)`, `resolve_taxonomy_folder(name)`
- Produces: failing tests that define the product matrix from the design spec

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_taxonomy.py`:

```python
GROUP_SEX_SELECTED = {
    "group_sex",
    "sex",
    "milf",
    "fellatio",
    "nakadashi",
    "paizuri",
    "footjob",
    "fertilization",
    "loli",
    "shota",
    "incest",
    "bestiality",
    "NTR",
    "Pokemon",
    "furry",
    "monster_girl",
    "android",
    "tentacles",
}


def test_group_sex_bucket_resolves() -> None:
    bucket = resolve_taxonomy_folder("group_sex")
    assert bucket is not None
    assert bucket.folder == "group_sex"
    assert resolve_taxonomy_folder("gangbang").folder == "group_sex"


def test_group_sex_beats_sex_and_vanilla_acts() -> None:
    folder, score, _ = choose_best_destination(
        {"gangbang": 0.92, "sex": 0.9, "vaginal": 0.88}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"
    assert score is not None and score > 0.5

    folder, _, _ = choose_best_destination(
        {"group_sex": 0.9, "fellatio": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"threesome": 0.9, "nakadashi": 0.95, "cum_in_pussy": 0.9},
        GROUP_SEX_SELECTED,
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"mmf_threesome": 0.88, "paizuri": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"orgy": 0.9, "footjob": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"spitroast": 0.9, "fertilization": 0.95, "impregnation": 0.9},
        GROUP_SEX_SELECTED,
    )
    assert folder == "group_sex"


def test_sex_still_wins_without_group_markers() -> None:
    folder, score, _ = choose_best_destination(
        {"sex": 0.9, "vaginal": 0.88, "missionary": 0.85}, GROUP_SEX_SELECTED
    )
    assert folder == "sex"
    assert score is not None and score > 0.5


def test_group_sex_loses_to_overrides() -> None:
    cases = [
        ({"gangbang": 0.95, "loli": 0.9}, "loli"),
        ({"gangbang": 0.95, "shota": 0.9}, "shota"),
        ({"gangbang": 0.95, "incest": 0.9}, "incest"),
        ({"gangbang": 0.95, "netorare": 0.85}, "NTR"),
        ({"gangbang": 0.95, "bestiality": 0.9}, "bestiality"),
        ({"gangbang": 0.95, "pokemon_(creature)": 0.9}, "Pokemon"),
        ({"gangbang": 0.95, "furry": 0.9}, "furry"),
        ({"gangbang": 0.95, "monster_girl": 0.9}, "monster_girl"),
        ({"gangbang": 0.95, "android": 0.9}, "android"),
        ({"gangbang": 0.95, "tentacle_sex": 0.9}, "tentacles"),
    ]
    for scores, expected in cases:
        folder, _, _ = choose_best_destination(scores, GROUP_SEX_SELECTED)
        assert folder == expected, (scores, folder)


def test_group_sex_beats_milf_even_with_strong_mature_female() -> None:
    folder, _, _ = choose_best_destination(
        {"gangbang": 0.85, "mature_female": 0.99}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"


def test_fellatio_still_wins_without_group() -> None:
    folder, score, _ = choose_best_destination(
        {"fellatio": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "fellatio"
    assert score is not None and score > 0.5


def test_voyeur_unchanged_by_group_sex_work() -> None:
    # Soft Voyeur path must still work; selecting group_sex must not break Voyeur group expand.
    folder, score, _ = choose_best_destination({"upskirt": 0.88}, {"Voyeur"})
    assert folder == "Voyeur/upskirt"
    assert score is not None and score > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
python -m pytest tests/test_taxonomy.py::test_group_sex_bucket_resolves tests/test_taxonomy.py::test_group_sex_beats_sex_and_vanilla_acts tests/test_taxonomy.py::test_group_sex_loses_to_overrides tests/test_taxonomy.py::test_group_sex_beats_milf_even_with_strong_mature_female -v
```

Expected: FAIL — `resolve_taxonomy_folder("group_sex")` is `None` and/or assertions fail because the bucket does not exist yet.

- [ ] **Step 3: Commit tests**

```bash
git add backend/tests/test_taxonomy.py
git commit -m "$(cat <<'EOF'
test: add drawn group_sex precedence matrix

Lock product rules for overrides, milf, and non-group sex routing before taxonomy JSON changes.
EOF
)"
```

---

### Task 2: Add `group_sex` bucket + rewire `sex` / non-override vetoes / `milf` role

**Files:**
- Modify: `backend/app/data/taxonomy.json`
  - Insert new bucket **immediately before** the `"sex"` bucket (currently around line 492)
  - Update `"milf"` (around lines 95–108)
  - Update `"sex"` veto/evidence (around lines 492–569)
  - Add group-marker vetoes to non-override act buckets that must lose to `group_sex`

**Interfaces:**
- Consumes: existing `DestinationBucket` fields (`evidence`, `gated_evidence`, `gate_tags`, `veto`, `veto_threshold`, `role`, `priority`)
- Produces: loadable `group_sex` folder; precedence matrix cases from Task 1 start passing

**Critical design note (do not skip):**  
`theme` does **not** yield to `act` in `_ROLE_YIELDS_TO`. Demoting `milf` to `theme` only stops milf from **stealing** when `group_sex` already leads on score. If `mature_female` scores higher than `gangbang`, milf still wins unless **`milf.veto` includes group markers**. Same for `fellatio` / `nakadashi` / `paizuri` / `footjob` / `fertilization`: add group markers to those buckets’ `veto` lists so `group_sex` wins when both fire.

Shared group-marker veto list (reuse verbatim wherever this plan says “GROUP_VETO”):

```json
"gangbang", "group_sex", "orgy", "threesome", "mmf_threesome", "ffm_threesome",
"spitroast", "reverse_spitroast", "double_penetration", "triple_penetration"
```

- [ ] **Step 1: Demote milf + add GROUP_VETO**

In `taxonomy.json`, change the `milf` bucket to:

```json
{
  "id": "milf",
  "folder": "milf",
  "priority": 3,
  "role": "theme",
  "aliases": ["MILF", "mature_woman", "milf"],
  "veto_threshold": 0.35,
  "veto": [
    "gangbang", "group_sex", "orgy", "threesome", "mmf_threesome", "ffm_threesome",
    "spitroast", "reverse_spitroast", "double_penetration", "triple_penetration",
    "loli", "shota", "oppai_loli"
  ],
  "ignore": [
    "mature_male", "old_man", "old_woman", "1girl", "1boy", "motherly",
    "age_difference", "loli", "shota", "oppai_loli"
  ],
  "evidence": [
    { "tag": "mature_female", "weight": 1.0 },
    { "tag": "milf", "weight": 0.95 }
  ]
}
```

Note: keep `milf` evidence minimal here; Task 3 expands it further. If `milf` tag is missing from vocabulary, omit that evidence row (check with `rg "^milf," tags.csv`).

- [ ] **Step 2: Insert `group_sex` bucket before `sex`**

Insert this object as a new element in the `buckets` array, immediately before the `"id": "sex"` object:

```json
{
  "id": "group_sex",
  "folder": "group_sex",
  "priority": 13,
  "role": "act",
  "aliases": ["gangbang", "Group_sex", "orgy", "Group Sex"],
  "veto_threshold": 0.35,
  "veto": [
    "loli", "oppai_loli", "onee-loli", "mesugaki",
    "shota", "onee-shota", "onii-shota", "miniboy",
    "incest", "twincest", "mother_and_son", "father_and_daughter",
    "brother_and_sister", "mother_and_daughter", "father_and_son",
    "bestiality", "animal_penis", "knot",
    "netorare", "cheating_(relationship)",
    "pokemon_(creature)", "pokephilia",
    "furry", "furry_female", "furry_male",
    "monster_girl", "slime_girl",
    "android", "robot_girl", "humanoid_robot", "cyborg",
    "tentacle_sex", "consensual_tentacles", "tentacles_on_male",
    "tentacle_pit", "tentacles_under_clothes"
  ],
  "ignore": [
    "1boy", "2boys", "1girl", "2girls", "hetero", "penis", "erection",
    "nude", "completely_nude", "censored", "blush", "sweat", "multiple_girls"
  ],
  "evidence": [
    { "tag": "gangbang", "weight": 1.0 },
    { "tag": "group_sex", "weight": 1.0 },
    { "tag": "orgy", "weight": 0.95 },
    { "tag": "threesome", "weight": 0.95 },
    { "tag": "mmf_threesome", "weight": 0.95 },
    { "tag": "ffm_threesome", "weight": 0.95 },
    { "tag": "spitroast", "weight": 0.95 },
    { "tag": "reverse_spitroast", "weight": 0.9 },
    { "tag": "double_penetration", "weight": 0.85 },
    { "tag": "triple_penetration", "weight": 0.9 }
  ],
  "gate_tags": [
    "sex", "vaginal", "anal", "rape", "penis", "erection", "ejaculation",
    "cum", "bukkake", "gangbang", "group_sex", "threesome", "orgy"
  ],
  "gated_evidence": [
    { "tag": "multiple_boys", "weight": 0.55 },
    { "tag": "3boys", "weight": 0.6 },
    { "tag": "4boys", "weight": 0.65 },
    { "tag": "5boys", "weight": 0.65 },
    { "tag": "6+boys", "weight": 0.7 }
  ]
}
```

- [ ] **Step 3: Update `sex` — veto group markers; remove group evidence**

In the `sex` bucket:

1. Append these strings to `"veto"` (keep existing veto entries):

```json
"gangbang", "group_sex", "orgy", "threesome", "mmf_threesome", "ffm_threesome",
"spitroast", "reverse_spitroast"
```

(Note: `double_penetration` / `triple_penetration` already help `sex` as evidence for non-group DP; **do not** put them on `sex.veto` — they remain on GROUP_VETO for milf/acts that must lose to group_sex, and on `group_sex.evidence`. If a vanilla DP image has no `gangbang`/`group_sex`/`threesome`, it should still land in `sex`.)

2. **Remove** from `sex.evidence`:

```json
{ "tag": "group_sex", "weight": 0.85 }
```

Leave `shimaidon_(sex)`, positions, etc. intact.

- [ ] **Step 4: Add GROUP_VETO to non-override act buckets**

For each of these buckets — `fertilization`, `nakadashi`, `fellatio`, `paizuri`, `footjob` — add (or merge into) fields:

```json
"veto_threshold": 0.35,
"veto": [
  "gangbang", "group_sex", "orgy", "threesome", "mmf_threesome", "ffm_threesome",
  "spitroast", "reverse_spitroast"
]
```

If a bucket already has `"veto"`, merge the tags (do not wipe existing entries). If it has no `veto_threshold`, add `0.35`.

Do **not** add GROUP_VETO to override folders (`loli`, `shota`, `incest`, `bestiality`, `NTR`, `Pokemon`, `furry`, `monster_girl`, `android`, `tentacles`).

- [ ] **Step 5: Run Task 1 tests**

Run:

```bash
cd backend
python -m pytest tests/test_taxonomy.py::test_group_sex_bucket_resolves tests/test_taxonomy.py::test_group_sex_beats_sex_and_vanilla_acts tests/test_taxonomy.py::test_sex_still_wins_without_group_markers tests/test_taxonomy.py::test_group_sex_loses_to_overrides tests/test_taxonomy.py::test_group_sex_beats_milf_even_with_strong_mature_female tests/test_taxonomy.py::test_fellatio_still_wins_without_group tests/test_taxonomy.py::test_voyeur_unchanged_by_group_sex_work -v
```

Expected: PASS for all of the above.

If an override case fails because the override bucket score is `None` while `group_sex` still scores, check that the override’s primary tag is listed in `group_sex.veto` and that the override bucket itself still scores from that tag.

- [ ] **Step 6: Commit**

```bash
git add backend/app/data/taxonomy.json
git commit -m "$(cat <<'EOF'
feat: add group_sex bucket and veto-based precedence

Route group WD signals away from sex/milf/acts while letting override folders win.
EOF
)"
```

---

### Task 3: Evidence refresh for favourite folders (non-Voyeur)

**Files:**
- Modify: `backend/app/data/taxonomy.json` (listed buckets only)
- Modify: `backend/tests/test_taxonomy.py` (small recall assertions)

**Interfaces:**
- Consumes: same scoring engine
- Produces: denser evidence for listed favourites; vocabulary test still green

- [ ] **Step 1: Write focused failing recall tests**

Append:

```python
def test_milf_accepts_expanded_mature_cues() -> None:
    selected = {"milf", "sex", "group_sex"}
    folder, score, _ = choose_best_destination(
        {"mature_female": 0.7, "huge_breasts": 0.6}, selected
    )
    # huge_breasts alone must not win; mature_female must still route milf
    assert folder == "milf"
    assert score is not None


def test_favourite_act_aliases_still_route() -> None:
    selected = {
        "nakadashi", "fellatio", "paizuri", "footjob", "fertilization",
        "group_sex", "sex",
    }
    assert choose_best_destination({"internal_cumshot": 0.9}, selected)[0] == "nakadashi"
    assert choose_best_destination({"irrumatio": 0.9}, selected)[0] == "fellatio"
    assert choose_best_destination({"perpendicular_paizuri": 0.9}, selected)[0] == "paizuri"
    assert choose_best_destination({"two_footed_footjob": 0.9}, selected)[0] == "footjob"
    assert choose_best_destination({"pregnancy_test": 0.9}, selected)[0] == "fertilization"
```

- [ ] **Step 2: Run to confirm baseline**

Run:

```bash
cd backend
python -m pytest tests/test_taxonomy.py::test_milf_accepts_expanded_mature_cues tests/test_taxonomy.py::test_favourite_act_aliases_still_route -v
```

Expected: `test_favourite_act_aliases_still_route` PASS already; `test_milf_accepts_expanded_mature_cues` PASS if mature_female alone still works (it should). Keep the test as a regression lock while expanding milf.

- [ ] **Step 3: Expand milf evidence (precision-first)**

Update `milf.evidence` to:

```json
"evidence": [
  { "tag": "mature_female", "weight": 1.0 },
  { "tag": "milf", "weight": 0.95 }
],
"gate_tags": ["mature_female", "milf"],
"gated_evidence": [
  { "tag": "huge_breasts", "weight": 0.45 },
  { "tag": "large_breasts", "weight": 0.4 },
  { "tag": "thick_thighs", "weight": 0.4 }
]
```

Rules:
- Do **not** add bare `motherly` / `age_difference` as ungated evidence (already ignored).
- Only add gated tags that exist in `tags.csv`.
- If `milf` is absent from vocabulary, drop that evidence row and keep `mature_female` as the only ungated cue; put gated body tags behind `gate_tags: ["mature_female"]` only.

- [ ] **Step 4: Expand remaining listed folders (additive only)**

Apply these additive changes. Skip any tag not present in vocabulary.

**`nakadashi.evidence` — add if missing:**

```json
{ "tag": "overflow", "weight": 0.5 }
```

Keep `overflow` in ignore if it would solo-win incorrectly; if ignored, **remove from ignore** only when adding as evidence with weight ≤ 0.55 and ensure soft-alone rules still require corroboration OR keep ignore and skip this add. Prefer: leave ignore as-is and instead add:

```json
{ "tag": "cum_in_pussy", "weight": 0.72 }
```

(already present — no-op). Preferred real add if in vocab:

```json
{ "tag": "excessive_cum", "weight": 0.45 }
```

only as gated via `gate_tags: ["internal_cumshot", "cum_in_pussy"]` — if that is too invasive, skip nakadashi changes beyond verifying existing tests.

**`fellatio.evidence` — add if missing and in vocab:**

```json
{ "tag": "oral", "weight": 0.55 }
```

Do **not** add if `oral` stays in `ignore` (it currently is ignored). Leave fellatio as-is unless a high-precision missing tag exists (e.g. `ball_suck` / `penis_in_mouth` if present in `tags.csv`).

**`loli` / `shota`:** add only high-precision missing variants found in vocab that are not already ignored, e.g. if present:

```json
{ "tag": "toddlercon", "weight": 0.7 }
```

for loli — **only if** in `tags.csv`. Otherwise leave character buckets unchanged beyond existing tests.

**`incest.evidence` — add if in vocab and missing:**

```json
{ "tag": "parent_and_child", "weight": 0.7 }
```

Only if removed from `ignore` first; if currently ignored, **do not** promote soft family tags. Prefer adding:

```json
{ "tag": "inseki", "weight": 0.9 }
```

if that tag exists in vocabulary (alias may already resolve folder but not score).

**`fertilization`:** add if missing/in vocab:

```json
{ "tag": "impregnation", "weight": 1.0 }
```

(already present — verify only).

**`bestiality.gated_evidence` — add animals in vocab if missing:** e.g. `fox`, `bear` at weight `0.5`.

**`Pokemon`:** no structural change required if evidence already dense; verify `pokephilia` + `pokemon_(creature)` still win vs `group_sex` (covered in Task 1).

**`furry` / `android` / `tentacles` / `monster_girl` / `NTR` / `paizuri` / `footjob`:** additive high-precision tags only; do not weaken ignore lists that prevent soft solos.

Practical minimum for this task (acceptable if vocab is thin):

1. milf gated expansion (Step 3) — required  
2. At least one real additive evidence tag each for: `bestiality` (gated animal), `incest` (only if safe), `nakadashi` OR `fellatio` — if none safe, document skip in commit message and keep tests green  
3. Ensure override primary tags remain on `group_sex.veto` (already Task 2)

- [ ] **Step 5: Run taxonomy + vocabulary tests**

```bash
cd backend
python -m pytest tests/test_taxonomy.py::test_milf_accepts_expanded_mature_cues tests/test_taxonomy.py::test_favourite_act_aliases_still_route tests/test_taxonomy.py::test_every_evidence_tag_exists_in_a_tagger_vocabulary tests/test_taxonomy.py::test_group_sex_loses_to_overrides tests/test_taxonomy.py::test_group_sex_beats_milf_even_with_strong_mature_female -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/data/taxonomy.json backend/tests/test_taxonomy.py
git commit -m "$(cat <<'EOF'
feat: expand milf and favourite-folder evidence

Improve recall for drawn favourites while keeping group_sex override precedence.
EOF
)"
```

---

### Task 4: Wire `group_sex` into Doujin favourites

**Files:**
- Modify: `backend/app/doujin_works.py` (`DOUJIN_FAVOURITE_FOLDERS`, lines 24–37)
- Modify: `backend/tests/test_doujin_works.py`

**Interfaces:**
- Consumes: `group_sex` folder from taxonomy
- Produces: Doujin mode selects `group_sex` automatically

- [ ] **Step 1: Write failing test**

In `backend/tests/test_doujin_works.py`, add:

```python
def test_doujin_favourites_include_group_sex() -> None:
    assert "group_sex" in DOUJIN_FAVOURITE_FOLDERS
    folder, score, _ = choose_best_destination(
        {"gangbang": 0.9, "sex": 0.85}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "group_sex"
    assert score is not None and score > 0.5
```

- [ ] **Step 2: Run to verify fail**

```bash
cd backend
python -m pytest tests/test_doujin_works.py::test_doujin_favourites_include_group_sex -v
```

Expected: FAIL — `group_sex` not in favourites tuple.

- [ ] **Step 3: Implement**

Update `DOUJIN_FAVOURITE_FOLDERS` in `backend/app/doujin_works.py` to include `"group_sex"` (place after `"milf"` or near act favourites):

```python
DOUJIN_FAVOURITE_FOLDERS: tuple[str, ...] = (
    "loli",
    "shota",
    "milf",
    "group_sex",
    "fertilization",
    "monster_girl",
    "incest",
    "bestiality",
    "Pokemon",
    "NTR",
    "tentacles",
    "furry",
    "android",
)
```

Note: Doujin favourites currently omit `sex` / `fellatio` / etc. That is OK — Task 1 matrix covers full selected sets. This task only ensures Doujin mode can land in `group_sex`.

- [ ] **Step 4: Run Doujin + group_sex tests**

```bash
cd backend
python -m pytest tests/test_doujin_works.py::test_doujin_favourites_include_group_sex tests/test_doujin_works.py::test_milf_and_inseki_taxonomy_routing tests/test_doujin_works.py::test_doujin_favourites_include_bestiality_and_pokemon tests/test_doujin_works.py::test_doujin_favourites_include_ntr_tentacles_furry_android -v
```

Expected: PASS.  
Also confirm milf-only still routes: `{"mature_female": 0.88}` → `milf` when no group markers.

- [ ] **Step 5: Commit**

```bash
git add backend/app/doujin_works.py backend/tests/test_doujin_works.py
git commit -m "$(cat <<'EOF'
feat: include group_sex in Doujin favourite folders

Let Doujin classify/migrate route strong group signals to the new destination.
EOF
)"
```

---

### Task 5: Full regression + optional engine fallback

**Files:**
- Test: `backend/tests/test_taxonomy.py`, `backend/tests/test_doujin_works.py`, `backend/tests/test_classify_accuracy.py` (run only)
- Modify only if needed: `backend/app/taxonomy.py`

**Interfaces:**
- Consumes: Tasks 1–4
- Produces: green suite; engine patch only if JSON cannot express a remaining failure

- [ ] **Step 1: Run full drawn taxonomy regression**

```bash
cd backend
python -m pytest tests/test_taxonomy.py tests/test_doujin_works.py tests/test_classify_accuracy.py -v
```

Expected: PASS.

- [ ] **Step 2: Only if a precedence case still fails after JSON fixes**

Add a minimal special case in `choose_best_destination` **after** candidates are built and **before** return, gated clearly:

```python
# Drawn product rule: group_sex beats milf when both are candidates.
folders = {c[0] for c in candidates}
if "group_sex" in folders and "milf" in folders:
    candidates = [c for c in candidates if c[0] != "milf"]
    candidates.sort(key=lambda item: (-item[1], item[2], _normalize_tag_name(item[0])))
    primary_folder, primary_score, _priority, primary_role = candidates[0]
```

Prefer **not** shipping this if Task 2 `milf.veto` already covers the milf case. Do not add a full precedence-tier system.

- [ ] **Step 3: Final commit only if Step 2 changed code**

```bash
git add backend/app/taxonomy.py backend/tests/test_taxonomy.py
git commit -m "$(cat <<'EOF'
fix: enforce group_sex over milf when both candidates remain

Fallback after taxonomy vetoes; keeps milf from winning on raw score ties.
EOF
)"
```

If Step 2 was unnecessary, skip commit.

- [ ] **Step 4: Manual smoke (operator)**

Restart backend (taxonomy is loaded at process start / `reload_taxonomy`). Reclassify a small needs-review slice from run 125/126:

1. Group-heavy → `group_sex` when selected  
2. `gangbang` + `loli` → `loli`  
3. `gangbang` + `mature_female` → `group_sex`  
4. Voyeur soft item → unchanged Voyeur folder  

No code change required for this step.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| New `group_sex` bucket (evidence, gates, veto, aliases) | Task 2 |
| `sex.veto` group markers; remove `group_sex` from `sex.evidence` | Task 2 |
| Override list beats `group_sex` | Tasks 1–2 |
| Non-override acts / fertilization lose to `group_sex` | Tasks 1–2 (GROUP_VETO on those buckets) |
| `group_sex` beats `milf`; milf role → `theme` | Tasks 1–2 |
| Evidence refresh for listed favourites | Task 3 |
| Voyeur untouched | Tasks 1–2 (explicit test + no edits) |
| Doujin favourites include `group_sex` | Task 4 |
| Engine change only if JSON fails | Task 5 |
| Unit matrix tests | Task 1 |
| Manual validation note | Task 5 Step 4 |

## Placeholder / consistency self-review

- No TBD steps; GROUP_VETO list is copied verbatim in Task 2.
- `priority: 13` for `group_sex` (better than `sex` at 14 on ties).
- `veto_threshold: 0.35` for group/milf/act vetoes (between soft noise and hard hits).
- Doujin favourites intentionally do not add every act folder; full matrix uses `GROUP_SEX_SELECTED`.
