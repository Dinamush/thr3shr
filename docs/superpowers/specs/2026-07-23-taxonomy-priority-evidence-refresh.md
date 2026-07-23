# Taxonomy priority + evidence refresh (2026-07-23)

## Priority (lower wins on score ties)

| P | Bucket | Rationale |
|---|--------|-----------|
| 0 | real_life | Quarantine / style gate destination |
| 1 | **loli** | User-prioritized content |
| 2 | **shota** | User-prioritized content |
| 3 | fertilization | Rare, highly specific |
| 4 | NTR | Specific relationship tag |
| 5 | incest | Specific relationship tag |
| 6 | nakadashi | Common sex act |
| 7 | fellatio | Common sex act |
| 8 | monster_girl | Broad species bucket |
| 9 | furry | Broad style bucket |
| 10 | Pokemon | Creature / franchise |

## Research method

- Cross-checked **tags.csv** post counts and **WD SwinV2 `selected_tags.csv`** vocab.
- Only keep evidence tags that WD can emit (or are soft corroboration with a hard tag).
- Removed `toddler` / `toddlercon` (in tags.csv but **absent from WD** → never registered).
- Replaced `cheating` with WD tag `cheating_(relationship)`.
- Dropped rare/non-WD incest edges (`group_incest`, uncle/aunt/grand*) for cleaner routing.

## Notable evidence adds (WD counts)

- loli: `onee-loli`, `mesugaki`, `lolidom`
- shota: `onee-shota` (~10k), `onii-shota`, keep `miniboy`
- fellatio: `cooperative_fellatio`
- furry: soft `body_fur`, `digitigrade`
- pokemon: `clothed_pokemon`, `riding_pokemon`
- fertilization: soft `uterus`, `cervix` (need hard tag alone)
