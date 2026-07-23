# Style detector debug compare (real vs anime)

## Goal

Debug-only harness to compare dedicated style detectors against current WD
`real_life` taxonomy routing on the existing realism corpus (RandomUser photos vs
Safebooru anime + photoreal edges). Production classify path stays unchanged.

## Research summary

- WD `realistic` / taxonomy competition is **not** a reliable real-vs-anime gate
  (existing realism eval: NO_GO, ~75% typical photo recall).
- `dghs-imgutils` (already a dependency) exposes `anime_real` / `anime_real_score`
  backed by ONNX models on Hugging Face `deepghs/anime_real_cls`.
- Reported hub metrics: MobileNetV3 v1.4 ~98.8% acc; CAFormer-S36 v1.4 ~99.1% acc.
- No torch/transformers required; fits the existing ORT stack.

## Detectors (debug)

| id | method | notes |
|----|--------|-------|
| `wd_taxonomy` | WD/ML tagger + `real_life` taxonomy | current production-style baseline |
| `imgutils_mobilenet` | `anime_real` `mobilenetv3_v1.4_dist` | fast dedicated ONNX |
| `imgutils_caformer` | `anime_real` `caformer_s36_v1.4` | higher accuracy dedicated ONNX |
| `imgutils_caformer_band` | same + uncertain band | `max(score) < threshold` → uncertain |

Label mapping: imgutils `real` → bucket `photo`; `anime` → `anime`.

## API

`POST /api/debug/style-eval`

```json
{
  "count_per_class": 12,
  "tagger_model": null,
  "detectors": null,
  "uncertain_threshold": 0.85
}
```

Response includes per-detector reports + overall ranking (same GO gates as
realism typical photo-vs-anime: P≥0.95, R≥0.90, anime FP≤0.05). Uncertain
predictions count as not-photo for binary metrics (photo→uncertain = miss).

## UI

Classifier debug page: new “Style detectors” panel to run compare and show
per-detector metrics table + sample disagreements.

## Non-goals

- Wiring winners into production routing
- NSFW-specific corpus (follow-up)
- Installing torch/CLIP for this prototype
