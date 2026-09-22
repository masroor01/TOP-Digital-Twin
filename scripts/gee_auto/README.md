# GEE Automation (`scripts/gee_auto/`)

Python ports of `scripts/gee/gee_0{1-5}_*.js`, for the weekly CI refresh
(`.github/workflows/weekly_refresh.yml`). Replaces the manual "paste into
code.earthengine.google.com, click Run, download from Drive" loop with a
service-account-authenticated script that pulls data directly.

## Why not `Export.table.toDrive` (async export, like the manual JS scripts)?

Tried first — **fails outright for a service account**: `Service accounts
do not have storage quota`, even when the target Drive folder is shared
with the service account's email as Editor. Sharing a folder grants write
*permission*, but the export API call itself still requires the calling
identity to have its own Drive storage quota, which service accounts
categorically don't have.

Switched to a direct synchronous `.getInfo()` fetch instead — no export
task, no Drive, no GCS, no billing dependency at all. Viable because these
per-zone tables are small (a few hundred rows at most). Confirmed exactly
correct by comparing every overlapping row/column against the existing
manually-exported CSVs (bit-for-bit match, see each script's validation
notes below) before trusting any of this near production data.

## Setup (one-time, already done as of 2026-09-22)

- GCP project: `top-digital-twin-data` (billing NOT enabled, not needed —
  Earth Engine API access doesn't require it once the project is
  registered for Earth Engine)
- Service account: `gee-automation@top-digital-twin-data.iam.gserviceaccount.com`
  — roles `roles/earthengine.writer` and `roles/serviceusage.serviceUsageConsumer`
  (the second one is easy to miss; without it every call fails with
  `USER_PROJECT_DENIED` even though Earth Engine access itself works)
- Project registered for Earth Engine at https://code.earthengine.google.com/register
  (one-time human step, already done for `top-digital-twin-data`)
- Key stored as the `GEE_SERVICE_ACCOUNT_KEY` GitHub Actions secret (raw
  JSON key content), written to a temp file by the workflow and pointed
  to via the `GEE_SERVICE_ACCOUNT_KEY` env var (path, not the JSON itself)
  — same pattern as `RCLONE_CONFIG`.

## Scripts

| Script | Source | Notes |
|---|---|---|
| `01_era5_topup.py` | ERA5-Land daily temp | Cheapest, one `getInfo()` per zone |
| `02_chirps_topup.py` | CHIRPS pentad rainfall | One `getInfo()` per zone |
| `03_s2_ndvi_topup.py` | Sentinel-2 NDVI/EVI | **One `getInfo()` PER WINDOW, not per zone** — bundling all ~18 windows into one request consistently hit GEE's "Too many concurrent aggregations" limit even after minutes of backoff (a real compute-budget ceiling for this heavier compositing workload, not a transient blip) |
| `04_modis_ndvi_topup.py` | MODIS 16-day NDVI/EVI (Terra+Aqua) | One `getInfo()` per zone |
| `05_modis_lst_topup.py` | MODIS 8-day LST | One `getInfo()` per zone |

All 5 always re-pull the full `2026-01-01` → today window (not an
incremental append) — mirrors the original JS scripts' own convention,
avoids any append/dedup logic. Each derives its actual usable end date
from the data itself (drops a trailing run of nulls / relies on the
source collection's own real coverage) rather than assuming today's date
is available — CHIRPS and MODIS especially lag real-time by weeks.

Zone coordinates are copied from `scripts/16_Zone_Assignment.py`'s
`ZONES` dict (the current source of truth) — **not** from the original
JS scripts, whose `P1/P2/P3` still show the pre-relocation
Agra/Farrukhabad/Jalandhar coordinates (superseded 2026-07-27, see that
script's own comment). Keep these two lists in sync if the zone
definitions ever change again.

## Validation (2026-09-22)

Every script's overlapping dates/pentads/windows were compared
bit-for-bit against the last manually-exported CSVs before trusting any
of this. ERA5/CHIRPS/MODIS NDVI/MODIS LST: exact match (0.0 max abs
diff) on every numeric column checked. S2: NDVI/valid_px_frac/n_scenes
exact match; EVI differs from the old files because the old files
predate a 2026-09-21 fix (commit `c2e2545`) for an EVI divide-by-zero
corruption bug — the new script correctly implements the current
(fixed) logic, confirmed via `git log` on the JS script's own history
and the old file's mtime (Jul 27, ~2 months before the fix existed).

## Downstream: `14_Satellite_Climate_Features.py`

No changes needed to read this automation's output — same folder/
filename/schema as the manual JS exports (`$TOP_DOWNLOADS_DIR/GEE_2026/
{era5,chirps,s2,modis}/*.csv`).

**Found and fixed while wiring this in**: Script 14 had `END` hardcoded
to `'2026-07-27'` in TWO places (the `END` constant and an independent
copy inside the `_all_weeks` date_range call) — the same silent-
truncation bug class this project has hit repeatedly. Without fixing
this, every week's new GEE data would have been silently discarded back
to July 27 on every run. Now derives `END` from today's date.
