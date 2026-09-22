#!/usr/bin/env python3
"""
Weekly Refresh Orchestrator -- CI (GitHub Actions) variant
=============================================================
Python port of run_weekly_refresh.ps1 for a Linux runner. Same steps, same
safety gates, same file list committed -- ported because the .ps1 hardcodes
Windows paths (C:\\Users\\masro\\...) throughout, which don't exist on a
Linux runner. The local Windows Task Scheduler automation is untouched by
this file; the two are independent, not synced automatically.

What's DIFFERENT from run_weekly_refresh.ps1:
  - Raw trusted CSVs and the scraper live wherever the CALLING WORKFLOW put
    them (TOP_DOWNLOADS_DIR env var, typically a directory the workflow
    populated via `rclone copy` before this script runs), not a fixed local
    Downloads folder.
  - Uses the vendored scraper (scripts/weekly_refresh/vendor/) instead of
    the external Codex-folder script, which a CI runner has no access to.
  - --sleep/--retries are tuned up front (1.5s / 6 retries, vs the
    scraper's own defaults of 0.25s / 4) -- the two real scheduled runs
    that got this far (2026-09-01, 2026-09-08) both hit sustained 429s from
    AGMARKNET during the all-India tomato/onion sweep at the default pace.
    Not yet applied to the local .ps1 path -- a decision for that path
    separately, not silently ported here.
  - Stops after `git commit` (no push) -- same "no auto-push" boundary the
    .ps1 keeps. Whether the calling workflow pushes, opens a PR, or leaves
    it for manual review is a decision made in the workflow YAML, not here.
  - Does not re-upload the refreshed raw CSVs to cloud storage -- that is
    the calling workflow's job (a `rclone copy` step after this script
    exits 0), symmetric with how it fetched them before this script started.

Safety gates (identical intent to the .ps1):
  - A crop's scrape/validate/merge failure only skips THAT crop -- others
    still proceed, and the trusted file for a failed crop is left untouched.
  - If zero crops merge successfully, the job stops before touching the
    pipeline at all (Script 09/23 never run on unchanged data).
  - If Script 09 or Script 23 itself fails (nonzero exit), the job stops
    immediately -- no git commit happens on a broken run.
  - verify.mjs must pass clean or the run stops before committing --
    refuses to ship JS models that don't reproduce Python's predictions.
  - Never pushes. Commits locally (to the runner's checkout) only.

Run: python scripts/weekly_refresh/run_weekly_refresh_ci.py
Env: TOP_DOWNLOADS_DIR must point at a directory containing
     {crop}_all_india_apmcs_2000_2026.csv for tomato/onion/potato.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPER = REPO_ROOT / "scripts" / "weekly_refresh" / "vendor" / "agmarknet_onion_prices.py"
DOWNLOADS = Path(os.environ.get("TOP_DOWNLOADS_DIR", str(REPO_ROOT / "downloads")))
STAGING = REPO_ROOT / "scripts" / "weekly_refresh" / "staging"
LOG_DIR = REPO_ROOT / "logs" / "weekly_refresh"
PYTHON = sys.executable

STAGING.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y-%m-%d_%H%M%S")
LOG_FILE = LOG_DIR / f"refresh_ci_{TS}.log"
_log_fh = open(LOG_FILE, "w", encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


def run_logged(cmd: list[str]) -> int:
    log("RUN: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout.splitlines():
        print(line, flush=True)
        _log_fh.write(line + "\n")
    _log_fh.flush()
    return proc.returncode


def fail_and_exit(reason: str) -> None:
    log(f"=== WEEKLY REFRESH (CI) FAILED: {reason} ===")
    _log_fh.close()
    failed_name = LOG_DIR / f"refresh_ci_{TS}_FAILED.log"
    LOG_FILE.rename(failed_name)
    sys.exit(1)


log("=== Weekly Refresh (CI) Started ===")
year = datetime.now().year
crops = ["Tomato", "Onion", "Potato"]
merged_crops: list[str] = []

# Same restriction as the local .ps1: potato's production panel has only
# ever included West Bengal + Uttarakhand (Script 09's balanced-panel /
# price-clip filters drop everything else regardless) -- see run_weekly_
# refresh.ps1's own comment for the full reasoning, unchanged here.
state_filter = {"Potato": "West Bengal,Uttarakhand"}

for crop in crops:
    crop_lower = crop.lower()
    trusted_file = DOWNLOADS / f"{crop_lower}_all_india_apmcs_2000_2026.csv"
    scrape_file = STAGING / f"{crop_lower}_weekly_scrape.csv"

    scrape_args = [
        PYTHON, str(SCRAPER),
        "--commodity", crop,
        "--start-year", str(year), "--end-year", str(year),
        "--out", str(scrape_file),
        "--log-level", "WARNING",
        # Tuned up from the scraper's own defaults (--sleep 0.25, --retries 4)
        # -- see module docstring above for why.
        "--sleep", "1.5", "--retries", "6",
    ]
    scope_label = "all states"
    if crop in state_filter:
        scrape_args += ["--states", state_filter[crop]]
        scope_label = state_filter[crop]

    log(f"--- {crop}: scraping (year {year}, {scope_label}) ---")
    code = run_logged(scrape_args)
    if code != 0:
        log(f"{crop}: SCRAPE FAILED (exit {code}) -- skipping this crop, trusted file untouched")
        continue

    log(f"--- {crop}: validating ---")
    code = run_logged([PYTHON, str(REPO_ROOT / "scripts" / "weekly_refresh" / "validate_scrape.py"),
                        crop_lower, str(trusted_file), str(scrape_file)])
    if code != 0:
        log(f"{crop}: VALIDATION FAILED -- skipping this crop, trusted file untouched")
        continue

    log(f"--- {crop}: merging ---")
    code = run_logged([PYTHON, str(REPO_ROOT / "scripts" / "weekly_refresh" / "merge_scrape.py"),
                        crop_lower, str(trusted_file), str(scrape_file)])
    if code != 0:
        log(f"{crop}: MERGE FAILED -- check for a .backup_* file next to the trusted CSV, it may need manual restoration")
        continue

    merged_crops.append(crop)
    log(f"{crop}: refresh complete")

if not merged_crops:
    fail_and_exit("no crops merged successfully, stopping before touching the pipeline")
log(f"Crops merged this run: {', '.join(merged_crops)}")

log("--- Script 09: rebuilding weekly panels ---")
env = {**os.environ, "TOP_DOWNLOADS_DIR": str(DOWNLOADS)}
proc = subprocess.run([PYTHON, "scripts/09_Agmarknet_Weekly_Panel.py"], cwd=REPO_ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout.splitlines():
    print(line, flush=True)
    _log_fh.write(line + "\n")
if proc.returncode != 0:
    fail_and_exit(f"Script 09 failed (exit {proc.returncode})")

log("--- Syncing panels into repo (data/agmarknet_weekly/) ---")
sync_files = ["tomato_weekly_panel.csv", "onion_weekly_panel.csv", "potato_weekly_panel.csv", "top_weekly_panel.csv"]
for f in sync_files:
    src = DOWNLOADS / "Agmarknet_Weekly" / f
    if not src.exists():
        fail_and_exit(f"sync source file missing: {src} -- Script 09 may not have written it, refusing to proceed with a stale/incomplete panel copy")
    dst = REPO_ROOT / "data" / "agmarknet_weekly" / f
    dst.write_bytes(src.read_bytes())

log("--- GEE satellite/climate topup (ERA5, CHIRPS, S2 NDVI, MODIS NDVI+LST) ---")
gee_scripts = [
    "scripts/gee_auto/01_era5_topup.py",
    "scripts/gee_auto/02_chirps_topup.py",
    "scripts/gee_auto/03_s2_ndvi_topup.py",
    "scripts/gee_auto/04_modis_ndvi_topup.py",
    "scripts/gee_auto/05_modis_lst_topup.py",
]
gee_ok = True
for gs in gee_scripts:
    code = run_logged([PYTHON, gs])
    if code != 0:
        log(f"{gs}: FAILED (exit {code}) -- skipping Script 14 rebuild this run, "
            f"existing data/satellite_climate/ files left untouched")
        gee_ok = False
        break

if gee_ok:
    log("--- Script 14: rebuilding satellite/climate features ---")
    code = run_logged([PYTHON, "scripts/14_Satellite_Climate_Features.py"])
    if code != 0:
        log(f"Script 14 FAILED (exit {code}) -- existing data/satellite_climate/ "
            f"files left untouched, continuing to Script 23 with whatever "
            f"climate/satellite features were already committed")
else:
    log("GEE topup incomplete this run -- Script 14 not re-run, "
        "continuing with existing committed satellite/climate features")

log("--- Script 23: retraining production models ---")
code = run_logged([PYTHON, "scripts/23_Train_Production_Models.py"])
if code != 0:
    fail_and_exit(f"Script 23 failed (exit {code}) -- production models may be in an inconsistent state, check Model_Output/production_models/ before trusting the dashboard")

log("--- Syncing retrained models into web/ + regenerating JS models ---")
web_model_files = ["feature_columns.json", "feature_ranges.json", "model_uncertainty.json",
                    "macro_climate_staleness.json", "reference_rows.csv", "price_history.csv"]
for f in web_model_files:
    src = REPO_ROOT / "Model_Output" / "production_models" / f
    if not src.exists():
        fail_and_exit(f"web bundle sync source missing: {src} -- Script 23 may not have written it")
    dst = REPO_ROOT / "web" / "data" / "production_models" / f
    dst.write_bytes(src.read_bytes())

# generate_js_models.py runs from web/ in the .ps1; replicate that cwd here.
proc = subprocess.run([PYTHON, "generate_js_models.py"], cwd=REPO_ROOT / "web",
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout.splitlines():
    print(line, flush=True)
    _log_fh.write(line + "\n")
if proc.returncode != 0:
    fail_and_exit(f"generate_js_models.py failed (exit {proc.returncode}) -- web dashboard's bundled JS models NOT updated, still serving the previous training run")

proc = subprocess.run(["node", "backend/src/models/__fixtures__/verify.mjs"], cwd=REPO_ROOT / "web",
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout.splitlines():
    print(line, flush=True)
    _log_fh.write(line + "\n")
if proc.returncode != 0:
    fail_and_exit(f"verify.mjs cross-language parity check FAILED (exit {proc.returncode}) -- newly generated JS models do not reproduce Python predictions, refusing to commit them; check the log above for which model(s) diverged")

log("--- Script 44: final sanity check (informational) ---")
code = run_logged([PYTHON, "scripts/44_Pipeline_Sanity_Check.py"])
if code != 0:
    log("Script 44 reported FAILures after a successful Script 09 + 23 run -- unexpected, needs manual review. NOT aborting the commit since the pipeline itself completed; investigate before trusting the dashboard.")
else:
    log("Script 44: clean.")

log("--- Local git commit (no push) ---")
files_to_add = [
    "data/agmarknet_weekly/tomato_weekly_panel.csv",
    "data/agmarknet_weekly/onion_weekly_panel.csv",
    "data/agmarknet_weekly/potato_weekly_panel.csv",
    "data/satellite_climate/zone_weekly_features.csv",
    "data/satellite_climate/crop_weekly_features.csv",
    "data/satellite_climate/market_zone_features.csv",
    "Model_Output/production_models/feature_ranges.json",
    "Model_Output/production_models/macro_climate_staleness.json",
    "Model_Output/production_models/model_uncertainty.json",
    "Model_Output/production_models/price_history.csv",
    "Model_Output/production_models/reference_rows.csv",
    "Model_Output/production_models/feature_columns.json",
    "Model_Output/production_models/tomato_1w.joblib", "Model_Output/production_models/tomato_4w.joblib",
    "Model_Output/production_models/tomato_13w.joblib", "Model_Output/production_models/tomato_26w.joblib",
    "Model_Output/production_models/onion_1w.joblib", "Model_Output/production_models/onion_4w.joblib",
    "Model_Output/production_models/onion_13w.joblib", "Model_Output/production_models/onion_26w.joblib",
    "Model_Output/production_models/potato_1w.joblib", "Model_Output/production_models/potato_4w.joblib",
    "Model_Output/production_models/potato_13w.joblib", "Model_Output/production_models/potato_26w.joblib",
    "web/data/production_models/feature_columns.json", "web/data/production_models/feature_ranges.json",
    "web/data/production_models/model_uncertainty.json", "web/data/production_models/macro_climate_staleness.json",
    "web/data/production_models/reference_rows.csv", "web/data/production_models/price_history.csv",
    "web/backend/src/models/feature_columns.json",
    "web/backend/src/models/tomato_1w.js", "web/backend/src/models/tomato_4w.js",
    "web/backend/src/models/tomato_13w.js", "web/backend/src/models/tomato_26w.js",
    "web/backend/src/models/onion_1w.js", "web/backend/src/models/onion_4w.js",
    "web/backend/src/models/onion_13w.js", "web/backend/src/models/onion_26w.js",
    "web/backend/src/models/potato_1w.js", "web/backend/src/models/potato_4w.js",
    "web/backend/src/models/potato_13w.js", "web/backend/src/models/potato_26w.js",
]
proc = subprocess.run(["git", "add", *files_to_add], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
_log_fh.write(proc.stdout)
if proc.returncode != 0:
    fail_and_exit(f"git add failed (exit {proc.returncode}) -- see log for git's own error output: {proc.stdout}")

commit_msg = f"Automated weekly data refresh (CI, {TS}) -- crops: {', '.join(merged_crops)}"
proc = subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
_log_fh.write(proc.stdout)
if proc.returncode != 0:
    fail_and_exit(f"git commit failed (exit {proc.returncode}) -- changes are staged but NOT committed: {proc.stdout}")
log(f"Committed locally (not pushed): {commit_msg}")

log("=== Weekly Refresh (CI) Completed OK ===")
_log_fh.close()
sys.exit(0)
