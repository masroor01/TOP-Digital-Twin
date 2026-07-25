# TOP Digital Twin

Price forecasting system for India's Tomato, Onion, Potato (TOP) APMC wholesale
markets. Built for SKUAST-K (HADP-04), targeting *Computers and Electronics in
Agriculture*.

This README is written so you can run, maintain, and extend this project
**without an AI assistant** — every script, every data source, every gotcha
hit during development is documented here. If you're reading this because
Claude Code access lapsed: everything below is enough to keep going on your own.

---

## 1. Environment Setup

- **Python 3.14.x** (this project was built and tested on 3.14.3)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- **Portability**: every script resolves its own project root automatically
  from its file location (`BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`,
  or the pathlib equivalent in Scripts 14/16) — you can move this project to
  any folder or machine and every script will find its own `data/` and
  `Model_Output/` correctly. Scripts 09 and 10 are the one exception: they
  reference external `Downloads/` folders for raw manual-download inputs
  (see §5) — those are inherently machine/session-specific and can't be made
  "portable" in the same sense, since they point to wherever you happened to
  download a raw source file.
- **Windows Bash tool note**: if running these commands from a Git Bash /
  MINGW shell rather than PowerShell, paths and `python` invocation work the
  same way — no special handling needed for this project.

---

## 2. Project Structure

```
TOP_Digital_Twin/
├── scripts/                  All pipeline code, numbered in rough build order
│   └── gee/                  Google Earth Engine JS scripts (run in GEE Code Editor, not locally)
├── data/                     Processed data, organized by layer (see §5)
├── Model_Output/             All results: tables, figures, trained models
│   └── production_models/    The 12 saved LightGBM models + dashboard metadata
├── requirements.txt
├── .gitignore                 Excludes files too large for GitHub (see inline comments)
├── TOP_Digital_Twin.code-workspace   VS Code workspace with a launch config per script
└── .claude/launch.json        Dashboard launch config for the Claude Code browser preview tool
```

GitHub: `https://github.com/masroor01/TOP-Digital-Twin` (private repo).

---

## 3. The Full Pipeline — What Each Script Does

Scripts are numbered in build order, not strict execution order — the
dependency graph below tells you what actually needs to run before what.
Each script prints its own progress and saves its own outputs; run them
from the project root (`cd TOP_Digital_Twin`, then `python scripts/NN_Name.py`).

### Phase A — Raw data → weekly panel

| Script | What it does | Depends on |
|---|---|---|
| `09_Agmarknet_Weekly_Panel.py` | Builds the core weekly price/arrivals panel from raw Agmarknet CSVs (tomato/onion/potato, all-India). Handles ISO-week alignment and gap imputation (see §7 imputation caveat). | Raw Agmarknet CSVs (see §5) |
| `09b_Merge_Onion_2026_Update.py` | One-off/refresh utility: merges the Agmarknet **portal's** separate "Daily Price Report" + "Daily Arrival Report" CSVs into the same row schema as the main onion raw file, matching markets to existing `market_id`s by normalized (state, market) name and assigning new sequential IDs for markets not seen before. Needed because onion's original scraper source doesn't get topped up the way tomato/potato's does — see §5. Run before `09_Agmarknet_Weekly_Panel.py` when refreshing onion. | Onion Daily Price/Arrival Report CSVs (see §5) |
| `10_CMIE_Macro_Parser.py` | Parses CMIE macro Excel exports into `data/cmie_macro/` | Raw CMIE Excel files |
| `10b_Extend_Macro_2026.py` | One-off/refresh utility: extends `data/rbi_dbie/`, `data/ppac_macro/`, and `data/cmie_macro/` CSVs in place with new CMIE Economic Outlook exports (repo/reverse-repo rate, USD/INR, WPI, diesel/LPG, agri credit, agri wages, IIP). Column mappings for each series are validated against known overlapping historical values before trusting them — see the script's own docstring for exact source-file → column notes, including two pre-existing mislabeling quirks found in the already-published data (`agri_wages_rs_day`, `iip_food_proc`) that were kept as-is for continuity rather than silently changed. | New CMIE Excel exports (see §5) |
| `11_Market_Selection_And_DataStructure.py` | Selects/validates the market panel structure | Script 09 output |
| `14_Satellite_Climate_Features.py` | Builds `crop_weekly_features.csv` from raw GEE exports (ERA5, CHIRPS, Sentinel-2, MODIS) | GEE raw exports (see §5), Script 09 |
| `16_Zone_Assignment.py` | Assigns markets to agro-climatic zones | Script 09 |

### Phase B — Layer 5/6 data compilation (infrastructure, policy)

| Script | What it does | Depends on |
|---|---|---|
| `19_Policy_Trade_Events.py` | Builds Layer 6 (export bans, MEP, export duty, market interventions, Operation Greens) from a verified primary-source event log | External event log file (see §5) |
| `20_Labour_Wages_Layer5.py` | Parses state-wise agricultural wage data (Labour Bureau) | External Excel file (see §5) |
| `21_Infrastructure_Layer5.py` | Builds cold storage capacity + road density by state | External CSV/Excel files (see §5) |
| `22_Master_Panel_Join.py` | Joins ALL layers (macro, climate, satellite, wages, infrastructure, policy) onto the base panel. Row-count-asserts after every join to catch silent data corruption. **Not required for most scripts** — Scripts 15/23/25 each rebuild the joined panel themselves inline (see §7 for why). Useful as a standalone sanity check or if you want one consolidated file. | Scripts 09, 10, 14, 19, 20, 21 |

### Phase C — Benchmark + ablation modeling

| Script | What it does | Depends on |
|---|---|---|
| `12_ModuleB_RollingOrigin_MultiHorizon.py` | Core rolling-origin CV framework | Script 09 |
| `13_Benchmark_Models.py` | Naive persistence / ARIMA benchmarks (`table_benchmarks.csv`) | Script 09 |
| `15_Ablation_Study_M0_M4.py` | **The main modeling script.** Trains LightGBM variants M0→M6 (price-only up to the full pipeline), 4-fold rolling-origin CV × 4 horizons × 3 crops = 336 fits. Has a `MARKET_LEVEL_DIAGNOSTIC` flag (see §7) for retraining just 2 variants on the full market panel for higher-power statistical tests. **~10-15 min for the full run.** | Scripts 09, 10, 14, 19, 20, 21 (rebuilds the join itself) |

### Phase D — Statistical validation

| Script | What it does | Depends on |
|---|---|---|
| `18_Diebold_Mariano_Tests.py` | Crop-level DM significance tests on Script 15's ablation results | Script 15 output (`ablation_predictions.csv`) |
| `18b_Market_Level_DM_Check.py` | Higher-power market-level DM tests | Script 15 run with `MARKET_LEVEL_DIAGNOSTIC = True` (`dm_market_level_predictions.csv`) |

### Phase E — Deep learning (secondary model)

| Script | What it does | Depends on |
|---|---|---|
| `17_TFT_Model.py` | Temporal Fusion Transformer, secondary model to LightGBM. Has `SMOKE_TEST`/`TIMING_TEST`/`FAST_MODE` flags (see §7) — the full-capacity run was deliberately deferred until all data layers were ready; it hasn't been run at full scale yet. Requires `pytorch-forecasting`, `torch` (not in `requirements.txt` — install separately if you use this). | Scripts 09, 10, 14, 19, 20, 21 |

### Phase F — Production models, dashboard, interpretability

| Script | What it does | Depends on |
|---|---|---|
| `23_Train_Production_Models.py` | Trains and **saves** 12 final LightGBM models (3 crops × 4 horizons, M6 feature set) — unlike Script 15, these are persisted (`.joblib`) for reuse. Also saves dashboard metadata: feature ranges, reference rows, price history, validated uncertainty (RMSE/MAPE per crop×horizon). **~10 min.** | Scripts 09, 10, 14, 19, 20, 21 |
| `24_Simulation_Dashboard.py` | Interactive Streamlit "what-if" scenario simulator. Run with `streamlit run scripts/24_Simulation_Dashboard.py`, not `python`. See §6. | Script 23 output |
| `25_Horizon_SHAP_Analysis.py` | SHAP feature importance per crop×horizon, grouped by data layer — explains *why* the ablation study shows crop/horizon-dependent results. **~5-10 min.** | Script 23 output (the saved models) |

---

## 4. Regenerating Everything From Scratch

If you're starting over on a fresh machine with all raw data already downloaded:

```bash
# Phase A
python scripts/09_Agmarknet_Weekly_Panel.py
python scripts/10_CMIE_Macro_Parser.py
python scripts/11_Market_Selection_And_DataStructure.py
python scripts/14_Satellite_Climate_Features.py
python scripts/16_Zone_Assignment.py

# Phase B
python scripts/19_Policy_Trade_Events.py
python scripts/20_Labour_Wages_Layer5.py
python scripts/21_Infrastructure_Layer5.py
python scripts/22_Master_Panel_Join.py   # optional consolidated file

# Phase C
python scripts/13_Benchmark_Models.py
python scripts/15_Ablation_Study_M0_M4.py   # ~10-15 min

# Phase D
python scripts/18_Diebold_Mariano_Tests.py
# For 18b, first re-run 15 with MARKET_LEVEL_DIAGNOSTIC = True (edit the script),
# then: python scripts/18b_Market_Level_DM_Check.py
# ...then set MARKET_LEVEL_DIAGNOSTIC back to False before the next full run.

# Phase F
python scripts/23_Train_Production_Models.py   # ~10 min
python scripts/25_Horizon_SHAP_Analysis.py      # ~5-10 min
# Dashboard: streamlit run scripts/24_Simulation_Dashboard.py
```

**If you only added a new week/month of raw data** (routine refresh, not a
full rebuild), you don't need to re-run everything — see §5's per-source
refresh notes, then re-run just Script 09 onward through whichever phases
depend on the layer you updated, finishing with 23 (dashboard) and 15
(ablation study) if you want updated results and forecasts.

---

## 5. Data Sources & Refresh Guide

Most raw sources require **manual download** (no public API) — this has been
your workflow throughout the project, and there's no way around it for most
of these. Here's where everything comes from and how often to refresh it.

### Agmarknet (price + arrivals) — weekly-ish refresh
- Source: [agmarknet.gov.in](https://agmarknet.gov.in) → Price & Arrivals → Download
- Download separately for each crop, place in the same folder as
  `09_Agmarknet_Weekly_Panel.py` expects (see the script's own docstring for
  exact filenames — it looks for `tomato_all_india_apmcs*.csv` etc.)
- **Onion specifically**: its scraper source doesn't get a fresh full-history
  export the way tomato/potato's does (last verified stopping at Dec 2025).
  Instead, download the portal's own "Daily Price Report" + "Daily Arrival
  Report" for onion (same Price & Arrivals page, filter by commodity) and run
  `09b_Merge_Onion_2026_Update.py` first — it merges them into the main raw
  file's schema and writes an updated `onion_all_india_apmcs_2000_2026.csv`.
- Re-run `09_Agmarknet_Weekly_Panel.py` after each download (bump `END_DATE`
  at the top of the script to match your new data's actual cutoff).

### CMIE Macro — monthly refresh
- Source: CMIE Economic Outlook (subscription-based data service) — exports
  download as "Scheme II-NNNNNNNN-X.xlsx" (X = M/A/W/D for monthly/annual/
  weekly/daily frequency), each with a "M"/"C" row tag per date (M = current
  monthly print, C = cumulative/fiscal-YTD — use M for level/index series).
- Place raw Excel exports where `10_CMIE_Macro_Parser.py` expects, re-run it,
  **or** run `10b_Extend_Macro_2026.py` (extends `rbi_dbie`, `ppac_macro`,
  and `cmie_macro` CSVs together from CMIE exports directly in Downloads —
  see the script's docstring for exact expected filenames per series, and
  validate any new series' column mapping against a known historical month
  before trusting it, same as it did).

### RBI DBIE (repo rate, USD/INR, WPI) — monthly refresh
- Source: [DBIE RBI](https://dbie.rbi.org.in), or CMIE Economic Outlook
  exports of the same series (repo/reverse-repo rate, USD/INR, WPI) — see
  `10b_Extend_Macro_2026.py` above, which handles this directly.
- If assembling by hand instead: combine into
  `data/rbi_dbie/rbi_dbie_macro_2017_2025.csv` matching the existing column
  structure — there's no dedicated ingestion script for the RBI-direct path.

### PPAC (diesel/LPG prices) — monthly refresh
- Source: [Petroleum Planning & Analysis Cell](https://ppac.gov.in), or a
  CMIE "Prices of Petroleum Products in Domestic Markets" export — see
  `10b_Extend_Macro_2026.py` above, which handles this directly.
- If assembling by hand instead: combine into
  `data/ppac_macro/ppac_diesel_lpg_2017_2025.csv`.

### Satellite/Climate (ERA5, CHIRPS, Sentinel-2, MODIS) — periodic topup
- Source: **Google Earth Engine** — run the JS scripts in `scripts/gee/`
  (`gee_01` through `gee_05`) in the [GEE Code Editor](https://code.earthengine.google.com),
  not locally. See `scripts/gee/README_GEE_2025.md` for details on the 2025
  topup process — the same pattern applies for future years.
- Export results to Google Drive, download, then run
  `14_Satellite_Climate_Features.py` to process into `crop_weekly_features.csv`.

### Labour Bureau wages — periodic refresh (Layer 5)
- Source: state-wise "Wage Rates in Rural India" series — the file used was
  a **CMIE** Economic Outlook export (`Scheme II-*.xlsx` naming pattern —
  corrected from an earlier "CEIC" mislabel; that naming is CMIE's own
  export convention, see the CMIE Macro note above). Check
  [data.gov.in](https://www.data.gov.in) or your CMIE terminal for
  updated exports.
- Re-run `20_Labour_Wages_Layer5.py` pointing at the new file (update the
  `SRC_FILE` path at the top of the script).

### Cold storage + road density — infrequent refresh (Layer 5, changes slowly)
- Cold storage: sourced from a Rajya Sabha Unstarred Question answer
  (parliamentary record) — check [pib.gov.in](https://pib.gov.in) or
  Rajya Sabha's Q&A archive for updated figures periodically (roughly
  annual is plenty, this data moves slowly).
- Road density: CEIC-sourced MORTH annual series. Also changes slowly —
  annual refresh is enough.
- Re-run `21_Infrastructure_Layer5.py` after updating either source file.

### Policy/trade events — refresh after major policy changes
- The verified event log (`TOP_policy_trade_verified_2017_2026.xlsx`) came
  from a separate scraper project. **Important**: before trusting any new
  policy data source, verify EVERY citation, not just a sample — cross-check
  each PIB press-release ID and DGFT notification number against the actual
  government sites. This has now happened **twice**: one candidate file had
  a 404 URL and a wrong PIB ID next to a real event; a second file
  (`extensive_top_policy_recordss.csv`, offered as a 2026 update) had all 7
  of its new rows fail verification outright — 404s, a domain that doesn't
  even resolve, and 2 real PIB IDs that turned out to be unrelated 2024
  press releases from different ministries. Both looked professionally
  structured. Don't skip this check, ever, even from a source that's been
  reliable before.
- Re-run `19_Policy_Trade_Events.py` after updating the event log.

---

## 6. The Dashboard

**Run locally:**
```bash
python -m streamlit run scripts/24_Simulation_Dashboard.py
```
(Use `python -m streamlit`, not bare `streamlit` — on Windows the `streamlit`
executable often isn't on PATH after `pip install`, `python -m` sidesteps that.)

**Requires** Script 23 to have been run at least once (needs
`Model_Output/production_models/*.joblib` and the metadata JSON/CSV files
alongside them).

**AI policy recommendation (optional)**: the dashboard has a button that
generates a one-paragraph AI policy commentary on whatever scenario you've
built, using the Claude API. It's optional — without a key, the dashboard
still works fully, just with that one section showing an info message
instead of the button.
- Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
  and **set a spending limit there** — the app is public, so any visitor
  who clicks the button triggers one API call (using the cheap Haiku model).
- **Local dev**: copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml` and paste in your key. That file is gitignored —
  never commit a real key.
- **Streamlit Cloud**: App settings → Secrets → paste
  `ANTHROPIC_API_KEY = "sk-ant-..."` in the same TOML format. No redeploy
  needed — secrets take effect on the next app restart/rerun.

**Redeploying to the public URL** (Streamlit Community Cloud):
- The app is already deployed at the live link you've been sharing. Since
  it's connected to GitHub, **any push to `master` triggers an automatic
  rebuild** — you don't need to manually redeploy for code/data changes,
  just `git push` and wait a few minutes.
- If you ever need to redeploy from scratch: go to
  [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
  "Create app" → repo `masroor01/TOP-Digital-Twin`, branch `master`, main
  file `scripts/24_Simulation_Dashboard.py`.
- **If deploy says "repository does not exist"**: the repo is private and
  Streamlit's GitHub App wasn't granted access. Fix at
  [github.com/settings/installations](https://github.com/settings/installations) →
  find "Streamlit" → Configure → grant access to this repo.
- **If the deployed app shows "you do not have access"**: the app's
  visibility is still tied to the private repo. Either make the GitHub repo
  public (Settings → Danger Zone → Change visibility — check first that
  nothing sensitive is in the repo, though as of this writing it's all
  public government data + your own code/models), or use Streamlit's viewer
  invite feature if you're on a paid tier.

---

## 7. Known Gotchas

Things that weren't obvious and cost real debugging time — worth knowing
before you hit them again:

- **Imputation in "latest price"**: the base weekly panel imputes missing
  trading weeks (see the `imputed`/`imputed_method` columns in
  `top_weekly_panel.csv`). **58.6% of markets' most-recent week is imputed,
  not a real trade.** Script 23 accounts for this (`last_observed_price`
  separate from the possibly-imputed "latest" row) — if you write new code
  against this panel, don't assume the last row per market is real data.
- **LightGBM can't extrapolate**: tree-based models only interpolate within
  the range of values seen in training. Widening a dashboard slider beyond
  the historical range doesn't give the model new information — it just
  repeats its most extreme leaf's prediction. The dashboard flags this with
  a warning; keep that pattern if you add more "what-if" inputs.
- **Windows console encoding**: scripts wrap `sys.stdout` in a UTF-8
  `TextIOWrapper` to avoid `UnicodeEncodeError` on em-dashes/special
  characters in PowerShell. If you write new scripts, copy that pattern
  (`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`)
  or avoid non-ASCII characters in print statements.
- **CSV encoding**: always pass `encoding='utf-8'` explicitly to
  `df.to_csv()` — Windows' pandas defaults to the system codepage otherwise,
  which corrupts special characters (found this the hard way with an
  em-dash getting mangled in a saved CSV).
- **Streamlit port conflicts**: if testing the dashboard while it's already
  running elsewhere, use a different `--server.port` rather than fighting
  over 8501 — Streamlit's widgets are React-controlled and don't always
  clean up a stuck connection gracefully.
- **`SMOKE_TEST` / `TIMING_TEST` / `FAST_MODE` flags** (Scripts 15, 17):
  these exist because early full-scale runs took far longer than expected
  (one early TFT attempt projected days). Always sanity-check timing on a
  reduced config before committing to a multi-hour run.
- **Row-count assertions matter**: Script 22's `checked_merge()` pattern
  (assert row count unchanged after every join) caught real bugs during
  development. Reuse this pattern for any new data join — a silently
  non-unique join key will duplicate rows without any obvious symptom.

---

## 8. Where Results Live (`Model_Output/`)

- `ablation_raw_results.csv`, `table_ablation.csv` — Script 15's full M0-M6 results
- `table_diebold_mariano.csv`, `table_dm_market_level_summary.csv` — statistical validation
- `table_shap_by_layer.csv`, `table_shap_top_features.csv` — Script 25's interpretability results
- `fig_*.png` — all paper-ready figures
- `production_models/` — the 12 saved models + dashboard metadata (see §6)

---

## 9. Project Status (as of this writing)

**Done**: all data layers M0-M6, ablation study, statistical validation
(crop- and market-level DM tests), production models, deployed dashboard,
horizon-stratified SHAP analysis.

**Not yet done**: full-capacity TFT run (deliberately deferred), crisis
backtesting case studies, Granger causality / market network analysis,
formal stress-testing module, manuscript draft.
