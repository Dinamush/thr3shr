# Doujin works classification — Design

**Date:** 2026-07-26  
**Status:** Approved for implementation

## Summary

Add a `doujin_works` run mode that treats each child folder or top-level `.cbz`/`.zip` under the Doujins inbox as one work. Sample pages → WD tag → pool → taxonomy primary + secondary favourite categories. Review/approve/reject like image runs. Migrate **moves** the whole work (preserves folder/archive name) into `categories_root/Doujins/<primary>/<work>/`, then creates Windows directory junctions (folders) or hardlinks (`.cbz`) for secondary tags.

## Favourites

| UI / alias | Destination folder |
|------------|--------------------|
| loli | `loli` |
| shota | `shota` |
| milf | `milf` (new taxonomy bucket) |
| impregnation | `fertilization` |
| monster girl | `monster_girl` |
| inseki | `incest` (alias) |

## Sampling

Cover + evenly spaced pages, default 12 (min 4, max 16). `.cbz` via zip members; folders via sorted image files.

## Pooling

Reuse `pool_presence` with `min_hits=1` or 2 depending on sample count (short works fall back to max).

## Disk

Move only. Junctions/hardlinks for extras — no size inflation.

## UI

Same review table: work name, primary tag, category tag chips (secondaries), approve/reject/migrate.
