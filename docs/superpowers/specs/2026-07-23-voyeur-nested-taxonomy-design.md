# Voyeur nested taxonomy design

Date: 2026-07-23  
Status: approved to implement (user: nested Option A, soft catch-all trumped by character/act)

## Goal

Route sexy-pose / tease content into nested destinations under `Voyeur/`, while hard character and act folders (`loli`, `fellatio`, `NTR`, etc.) always win when they also score.

## Folder layout

```
categories_root/
  Voyeur/
    panties/
    upskirt/
    ass/
    cleavage/
    caught/
  (and catch-all files directly in Voyeur/ when no subfolder wins)
```

Taxonomy `folder` values:

| Bucket id | folder | role |
|-----------|--------|------|
| voyeur_panties | Voyeur/panties | soft |
| voyeur_upskirt | Voyeur/upskirt | soft |
| voyeur_ass | Voyeur/ass | soft |
| voyeur_cleavage | Voyeur/cleavage | soft |
| voyeur_caught | Voyeur/caught | soft |
| voyeur | Voyeur | soft |

Selecting any Voyeur* folder in settings expands the whole Voyeur group for scoring.

## Winner rules

1. Score buckets as today (evidence / gated evidence).
2. If the top candidate is `soft` and any `character` or `act` candidate exists → pick the best character/act instead.
3. Among Voyeur subs, highest score wins; catch-all `Voyeur` only if it scores and no subfolder scored higher (normal score competition within soft).

## Nested paths

`sanitize_folder_name` must not flatten `/`. Add `destination_path(categories_root, folder)` that splits on `/`, sanitizes each segment, and joins.

## Out of scope

- Filename heuristics
- Changing existing act/character evidence
- Auto-creating empty Voyeur dirs outside migrate
