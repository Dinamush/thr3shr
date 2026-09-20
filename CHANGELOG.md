# Changelog

All notable THR3SHR releases are recorded here.

## 1.0.0 — 2026-09-20

First public THR3SHR release (rebrand from Thresh3r).

### Product
- FastAPI + React local workflow: scan → tag → review → migrate
- Multi-model tagging (WD SwinV2 v3, WD EVA02 Large, ML-Danbooru)
- GPU/CPU ONNX Runtime with model unload and cancel-safe classify runs
- GIF/video presence classification with length-scaled frame sampling
- SFW classify mode with parked NSFW destination tags
- Doujin works classify/review/migrate mode
- Isolated real-life adult tagging domain
- Nested Voyeur taxonomy plus `group_sex` and favourite-folder evidence
- Hybrid ML rescue on WD needs-review

### Packaging
- Dual licensing: MIT (code) and CC BY 4.0 (brand/creative docs)
- Model weights stay on Hugging Face; this repo ships code and vocab only
- Version is logged at API startup and returned from `/health` and `/attribution`

## 0.1.0

Initial local classifier UI and FastAPI backend.
