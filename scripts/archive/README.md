# scripts/archive

Retired scripts, kept for provenance (moved here 2026-10-06, audit item 10). Nothing in the production, CI or evaluation pipeline reads them.

| Script | Why retired |
|---|---|
| `09c`–`09f` | One-off historical raw-data rebuild patches, each superseded by the next; only `09_Agmarknet_Weekly_Panel.py` (and `09b_Merge_Onion_2026_Update.py`, the documented onion refresh step) remain in `scripts/`. |
| `34`–`37` | Two-Phase Baseline experiment (Phase 1 long-window model + Sentinel-2 residual model). Rejected: the combined forecast is worse than Phase 1 alone, and Phase 1 alone does not beat M6 under a formal test. See the main README and `Model_Output/MANIFEST.md`. |

These scripts compute the repo root as two directories above their own file, so they will not run from here unchanged. To re-run one, move it back to `scripts/` first.
| `TFT_Model_Test.py` | Abandoned scratch copy of Script 17 from the 2026-09-24 GPU-refactor session (`SMOKE_TEST = True` left on; an older, more elaborate RAPIDS/cuDF variant that `17_TFT_Model.py` superseded the same day). Its `gpu_segment_fill` idea can be lifted into `gpu_utils.py` if wanted. |
