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

GitHub: `https://github.com/masroor01/TOP-Digital-Twin` (public repo as of
2026-08-10 — see §9 for the visibility history and the separate, deliberately-restricted Streamlit app-sharing setting).

---

## 3. The Full Pipeline — What Each Script Does

Scripts are numbered in build order, not strict execution order — the
dependency graph below tells you what actually needs to run before what.
Each script prints its own progress and saves its own outputs; run them
from the project root (`cd TOP_Digital_Twin`, then `python scripts/NN_Name.py`).

### Phase A — Raw data → weekly panel

| Script | What it does | Depends on |
|---|---|---|
| `09_Agmarknet_Weekly_Panel.py` | Builds the core weekly price/arrivals panel from raw Agmarknet CSVs (tomato/onion/potato, all-India). Handles ISO-week alignment and gap imputation (see §7 imputation caveat). Applies a real-coverage filter (`MIN_REAL_COVERAGE = 0.70`, added 2026-07-27, revised same day from an initial 0.80): markets are dropped unless >=70% of their own real reporting span is real (non-imputed) data — potato additionally keeps its own >=8-of-9-years balanced-panel rule first. Kept market counts: tomato 834 (was 1,725 unfiltered), onion 809 (was 1,580), potato 82 (was 87). Revised down from 80% because the market-level DM test (Script 18b) at 80% found onion's h=1w result resting on only 34/189 significant markets; 70% roughly triples that to a sturdier sample. **Grid-adaptivity fix, 2026-08-01**: each market's coverage grid now starts at its own first real observation rather than a fixed global `START_DATE` — the earlier fixed-grid version silently scored a market as "missing" for every week before it existed in the system, undercounting coverage for markets that onboarded later. Fixing this grew tomato from 517 and onion from 246 to the counts above (potato unaffected — its balanced-panel prefilter already cleared this bar). See `paper_drafts/methods_data_section.txt` Sec 3.3 for the full account. `data/market_coverage_browser.html` is a separate, purely exploratory viewer over ALL markets (unfiltered) — it was never wired into this filter and isn't a decision record. | Raw Agmarknet CSVs (see §5) |
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
| `15b_Tree_Model_Comparison.py` | Compares LightGBM against RandomForest, XGBoost, CatBoost on the same CV framework — validates LightGBM as the production choice. RandomForest is clearly worst; the other three are competitive with no consistent winner. **~2 hours** (RandomForest dominates the runtime). | Same as Script 15 |
| `15c_LSTM_Transformer_Comparison.py` | Compares LightGBM against an LSTM and a small Transformer on the same CV framework. Both deep-learning models underperform every tree model, mostly negative R². **~20-25 min.** | Same as Script 15 |

### Phase D — Statistical validation

| Script | What it does | Depends on |
|---|---|---|
| `18_Diebold_Mariano_Tests.py` | Crop-level DM significance tests on Script 15's ablation results | Script 15 output (`ablation_predictions.csv`) |
| `18b_Market_Level_DM_Check.py` | Higher-power market-level DM tests | Script 15 run with `MARKET_LEVEL_DIAGNOSTIC = True` (`dm_market_level_predictions.csv`) |
| `27_Horizon_Skill_And_MCS.py` | Reshapes Script 15's MASE into a horizon-conditional skill% table (identifies each crop's crossover horizon where the full pipeline first beats naive persistence), then runs a Model Confidence Set (Hansen, Lunde & Nason 2011 — stationary-bootstrap range statistic) jointly testing all 8 variants per crop × horizon with familywise error control, the multi-model alternative to Script 18's pairwise DM tests. **<1 min** (no model fitting, pure statistical test). | Script 15 output (`ablation_predictions.csv`, `table_mase.csv`) |
| `28_Crisis_Backtesting_Case_Studies.py` | Backtests 4 real, verified crisis episodes (tomato's Jun-Sep 2023 spike/crash, onion's Aug 2023-Jan 2024 export-restriction escalation/post-ban crash, potato's Apr 2024 spike and Jan 2025 crash) against Script 15's out-of-sample fold predictions, cross-referenced with Script 19's verified policy-event log. Answers whether "naive wins on average" still holds specifically during the crises policy cares about (it does at h=1w; the data layers pull decisively ahead at h=13w/26w for tomato/onion, but not for potato). **<1 min.** | Script 15 output (`ablation_predictions.csv`), Script 19 output (`export_policy_events.csv`) |
| `29_Granger_Causality_Analysis.py` | Bidirectional Granger causality tests (ADF-checked for stationarity, FDR-corrected within crop) between price and each data layer (arrivals, climate, satellite, macro, policy), plus a light lead-lag network among each crop's top-5-by-volume markets. Complements the ablation study's "does this layer improve accuracy" question with "does this layer have independent predictive content for price at all." **<2 min.** | Script 09 output (weekly panel), Scripts 14/10b/19 outputs |
| `30_Formal_Stress_Testing.py` | Systematises the dashboard's what-if simulator into a fixed, reproducible battery of named stress scenarios, run across every market using the saved M6 production models (Script 23) -- no retraining. Onion's policy scenarios replay the exact verified 2023-24 event values (Script 19); diesel and climate scenarios are calibrated to real historical extremes. Originally surfaced a wrong-signed prediction for onion's export-duty scenario, traced to the same reactive-not-causal relationship Script 29 found; fixed via a monotonic sign constraint on the three export-control features in Script 15/23 (negligible accuracy cost). The constrained model still can't recover a realistic *magnitude* for these scenarios (median response ~0%) -- see Script 31 for why, and for the real-world-calibrated alternative. **<1 min.** | Script 23 output (production models, `reference_rows.csv`), Script 19 output |
| `31_Synthetic_DID_Policy_Effect.py` | Estimates the 2023-24 onion export-restriction episode's causal price effect directly via Synthetic Difference-in-Differences (Arkhangelsky et al. 2021), since Script 30's forecasting model can't recover a reliable counterfactual magnitude (an identification problem, not a training-signal one -- duty, MEP, and the ban only ever moved together once). Part A (cross-crop, tomato/potato as donors) and Part B (within-onion, Nashik export-hub vs non-hub) — see `Model_Output/MANIFEST.md` for the full Part A/B/C breakdown, including 2026-08-01's Part C addendum (event-study trajectories, jackknife CIs, donor-pool robustness, arrivals/quantity effect). **~5 min.** | `data/agmarknet_weekly/top_weekly_panel.csv`, Script 19 output |
| `38_SDID_InSpace_Placebo_Test.py` | Follow-up to Script 31: standard synthetic-control-literature falsification test (Abadie, Diamond & Hainmueller 2010) — treats every individual tomato/potato market as a placebo "treated" unit against the same donor pool, builds a null distribution of placebo ATTs, and judges the real onion postban ATT against it via a rank-based p-value. Result: p=0.131 — the postban effect does not clear significance against this stronger test either. **~5-10 min.** | Script 31's fitted SDID weights |
| `39_SDID_Stacked_MultiEpisode_EventStudy.py` | Extends Script 31's single-episode (2023-24) design to all 3 verified onion export bans in the panel window (Sep 2019, Sep 2020, 2023-24), stacking them into one multi-episode event study — addresses the n=1 identification problem (a single confounded natural experiment can't separate the ban's effect from the coincident weather shock that prompted it) by pooling 3 independent episodes. **~5-10 min.** | Script 19 output, `data/agmarknet_weekly/top_weekly_panel.csv` |

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
| `26_Weekly_To_Daily_Disaggregation.py` | Exploratory: turns the weekly model's 4 forecast points into a smooth daily curve (PCHIP interpolation) with an honest uncertainty band from real historical daily residual std-dev — NOT a new daily-trained model (see §7 for why daily-native models were tried and abandoned). Backs the dashboard's "Show daily price forecast" view. **<1 min.** | Script 23 output |

### Phase G — Exploratory: Two-Phase Residual Architecture (REJECTED, 2026-08-01/04)

Tests whether a 23-year (2003-2026) price/arrivals-only "baseline" model plus
a narrow 2017+ "residual" model (correcting the baseline using the richer
modern-era layers) beats production M6. **Verdict: rejected** — kept in the
repo as a documented negative result, not part of the active pipeline.

| Script | What it does | Depends on |
|---|---|---|
| `32_LongHistory_Panel_Builder.py` | Builds a 2003-2026 price/arrivals panel (exact copy of Script 09's logic, only the start year differs) — validates that Script 09's 2017 floor is a filter choice, not a source limitation, for price/arrivals specifically. | Raw Agmarknet CSVs |
| `33_LongHistory_Validation_Experiment.py` | Controlled single-variable test: does the 23-year window measurably improve accuracy over the 9-year (2017-2026) production window, same market set/features/CV folds/model config? Found a real, modest ~1.2pp average MAPE improvement. | Script 32 output |
| `34_Baseline_Phase_Panel_Join.py` | Builds the full 2003-2026 joined panel (all layers except Sentinel-2, which stays reserved for Phase 2) for the two-phase architecture's Phase 1 model. | Script 32, `10c_RBI_LongHistory_Parser.py` output |
| `35_TwoPhase_Baseline_Model.py` | Trains the long-window Phase 1 baseline model (price/arrivals/climate-non-S2/macro/policy) using 9 annual expanding folds (2017-2025), saves out-of-fold predictions for Script 36. | Script 34 output |
| `36_TwoPhase_Residual_Model.py` | Trains Phase 2 to predict the residual (`actual_log_price - phase1_pred`) from Sentinel-2 vegetation features only. **Result: the combined Phase1+residual forecast is worse than Phase 1 alone and worse than M6 at nearly every crop/horizon cell.** Outputs in `Model_Output/experiments/two_phase/`, not the main results tree. | Script 35 output |
| `37_DM_Test_M6_vs_Phase1Alone.py` | Ad-hoc follow-up: Phase 1 alone (not the combined stack) beat M6 on several raw MAPE cells, notably onion — formally DM-tested to check if that edge is real. **Result: does not survive formal testing (1/12 cells significant at p<0.05).** M6 confirmed as the correct production choice. | Scripts 15, 35 outputs |

### Phase H — Exploratory: Early-Warning Prototype & Market Network (2026-08-08/10)

Two independent extensions built on top of the already-published policy
findings (Sections 8/13 of the comprehensive review report), each iterated
through several rounds of increasingly rigorous validation and reported
honestly including their partial/negative results.

| Script | What it does | Depends on |
|---|---|---|
| `40_Escalation_Signature_Head.py` | Prototype early-warning classifier: a per-crop LightGBM model scoring each week on how closely it resembles the price-escalation pattern that has historically preceded a real policy intervention (backward-looking features only). Validated per-crop with leave-one-episode-out CV plus a placebo-in-time significance test and a data-driven (threshold-based) labeling redesign. 2 of 3 onion episodes (2019, 2023) validate cleanly (p=0.000); the 2020 episode does not (p=0.660) — diagnosed as a real, different-shaped price path, not a labeling artifact. Deliberately not iterated further (data-volume limited, matches the Section 8/Section 10 stopping-rule pattern). | `data/agmarknet_weekly/longhistory/top_weekly_panel_longhistory.csv`, `data/satellite_climate/crop_weekly_features.csv` |
| `41_Escalation_Signature_Nashik_Hub_Test.py` | Follow-up testing whether onion 2020's weak signal was real-but-geographically-diluted (its documented trigger was Nashik-region flooding specifically). Rebuilds the same detector on Nashik-hub-only price. 2020's placebo p-value falls tenfold (0.660→0.064) — a partial, honest result (doesn't cross 0.05, but strongly supports the localization hypothesis). | Script 40's detector, Nashik-hub market list (from Script 31) |
| `42_Market_Leader_Network.py` | Extends Script 29's light top-5-market lead-lag test to the top-20 markets per crop, with a net-leadership-score summary (out-rate − in-rate) designed to surface asymmetry that a small bidirectional-significance test washes out. Finds clear, geographically coherent leader clusters per crop (Karnataka for tomato, Nashik belt for onion, West Bengal for potato) with consumption hubs (Azadpur, Shahdara) as consistent followers. Includes a confound check (leadership isn't just a reporting-coverage artefact — clean for all 3 crops) and a two-period stability check (tomato/potato leaders hold up; onion's single top market shuffles rank, though the Nashik-belt cluster stays dominant throughout). | `data/agmarknet_weekly/top_weekly_panel.csv` |

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
alongside them). The daily price view additionally requires Script 26's
output (`table_dow_pattern.csv`).

**Theme**: `.streamlit/config.toml` sets the project's palette (green
accent, crop-specific colors for tomato/onion/potato) — no setup needed,
Streamlit picks it up automatically.

**Price forecast ticker**: shows the model's prediction at each of its 4
trained horizons, tagged with which season (per the crop's own calendar —
kharif/peak-arrival/lean for tomato, rabi-arrival/kharif/lean for onion,
harvest/storage/lean for potato) that forecast date falls in. The "Show
daily price forecast" toggle (below the ticker) expands into a smoothed
daily curve with a seasonal-shading chart and an honest uncertainty band —
see Script 26 above for what it is and isn't.

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
- **If the deployed app shows "you do not have access"**: this is NOT
  necessarily the GitHub repo's visibility — confirmed 2026-08-10 that
  making the repo public did not by itself fix this message. Streamlit
  Community Cloud has its OWN separate app-level sharing control
  (share.streamlit.io → the app → Settings → Sharing → "Only specific
  people" vs. public/anyone-with-link) that does not automatically follow
  the GitHub repo's visibility. Check that setting directly; if it's
  deliberately set to "Only specific people" (current state as of
  2026-08-10, a deliberate choice, not a bug), add viewers individually
  there rather than via GitHub repo access — GitHub collaborator access
  does not grant Streamlit app viewer access, they're independent
  permission systems.

---

## 7. Known Gotchas

Things that weren't obvious and cost real debugging time — worth knowing
before you hit them again:

- **Imputation in "latest price"**: the base weekly panel imputes missing
  trading weeks (see the `imputed`/`imputed_method` columns in
  `top_weekly_panel.csv`). **76.2% of markets' most-recent week is imputed,
  not a real trade** (as of the 2026-07-27 refresh, post-coverage-filter) —
  this is a recency-lag effect (reporting delay in the newest week), distinct
  from a market's overall coverage quality, so the 70% coverage filter
  doesn't eliminate it. Script 23 accounts for this (`last_observed_price`
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

**Check `Model_Output/MANIFEST.md` before citing or reusing any table/figure** —
it tracks which script produced each output and when it was last regenerated,
so you don't accidentally cite a stale result (this bit us twice on
2026-07-29: `table_benchmarks.csv` and `table_spike_auc.csv` were both
several weeks stale and no one had noticed, since nothing on disk indicated
their age relative to the current pipeline).

- `ablation_raw_results.csv`, `table_ablation.csv`, `table_mase.csv` — Script 15's full M0-M6 results + naive baseline + MASE
- `table_tree_model_comparison*.csv`, `table_lstm_transformer_comparison*.csv` — Scripts 15b/15c's model-family comparisons
- `table_diebold_mariano.csv`, `table_dm_market_level_summary.csv` — pairwise statistical validation (Scripts 18/18b)
- `table_horizon_skill.csv`, `table_horizon_skill_crossover.csv`, `table_mcs.csv`, `table_mcs_membership.csv` — Script 27's horizon-skill and Model Confidence Set results
- `table_crisis_backtests.csv` — Script 28's crisis-episode backtests (tomato/onion/potato)
- `table_granger_layers.csv`, `table_granger_market_network.csv` — Script 29's Granger causality results (data layers and top-5-market network)
- `table_stress_test_results.csv` — Script 30's stress-testing scenario battery results
- `table_sdid_policy_effect.csv`, `table_sdid_hub_policy_effect.csv` — Script 31 Part A/B's Synthetic DID onion export-restriction effect estimates
- `table_sdid_event_study.csv`, `table_sdid_donor_robustness.csv`, `table_sdid_arrivals_effect.csv` — Script 31 Part C's event-study trajectories, jackknife CIs, donor-pool robustness, and arrivals/quantity effect
- `table_sdid_inspace_placebo.csv` — Script 38's in-space placebo falsification test (postban ATT p=0.131)
- `table_sdid_stacked_event_study.csv`, `table_sdid_stacked_summary.csv` — Script 39's stacked 3-episode (2019/2020/2023-24) event study
- `table_dm_m6_vs_phase1alone.csv` — Script 37's DM test (Phase-1-alone vs M6; edge does not survive formal testing)
- `Model_Output/experiments/two_phase/` — Scripts 34-36's two-phase residual architecture outputs, archived as a documented rejected experiment, separate from the production results above
- `table_escalation_signature_*.csv` — Scripts 40/41's early-warning prototype (per-crop LOEO scores, placebo-in-time tests, Nashik-hub localization test)
- `table_market_leader_*.csv` — Script 42's market leader-follower network (rankings, confound check, stability check)
- `table_shap_by_layer.csv`, `table_shap_top_features.csv` — Script 25's interpretability results
- `table_spike_auc.csv`, `table_rolling_origin_metrics.csv` — Script 12's spike-detection classifier
- `table_dow_pattern.csv`, `table_disagg_backtest.csv` — Script 26's daily-disaggregation backtest (feeds the dashboard's daily view)
- `fig_*.png` — all paper-ready figures
- `production_models/` — the 12 saved models + dashboard metadata (see §6)

---

## 9. Project Status (as of 2026-08-10 — see git log for anything after this date)

**Done**: all data layers M0-M6, ablation study, model-family comparisons
(tree ensembles, LSTM/Transformer), statistical validation (crop- and
market-level DM tests, plus a Model Confidence Set jointly testing all
variants), horizon-conditional skill analysis, production models, deployed
dashboard (with a daily-resolution price view), horizon-stratified SHAP
analysis, crisis backtesting, Granger causality, formal stress-testing
(with a monotonic sign fix), Synthetic DID policy-effect estimation
(Parts A/B/C plus two follow-up robustness scripts — in-space placebo,
stacked multi-episode event study), a market-panel coverage-grid bug fix
(2026-08-01, grew the analytical panel from 517/246/82 to 834/809/82
markets, full downstream cascade re-run), a `PANEL_END` staleness bug fix
plus 5 newly-verified 2026 policy events (2026-08-01), a WPI/diesel-LPG
data-vintage refresh (CMIE restates its full history on every pull —
production files had drifted 1.3-13.7% from current vintage; refreshed
2026-08-01), a macro long-history extension that also fixed a pre-existing
2026 gap in `export_veg`/`import_veg`/`crude_oil` (2026-08-03), a
look-ahead leakage fix in `s2_ndvi_anom`'s seasonal climatology with a
full M6 cascade retrain (2026-08-04), a 23-year two-phase residual
architecture — explored fully and **rejected** under a pre-agreed
stopping rule (Scripts 32-37, 2026-08-01/04, see Phase G above), and two
new exploratory investigations folded into the comprehensive review
report as Sections 14-15: an escalation-signature early-warning prototype
(Scripts 40/41, 2026-08-08, deliberately stopped once data-volume-limited)
and a market leader-follower network (Script 42, 2026-08-10, confound-
checked and stability-checked).

**Not yet done / genuinely open**:
- Full-capacity TFT run (deliberately deferred).
- Re-running Scripts 12, 13, 15b, 15c, 17, 18b, 26 on the current panel —
  these predate not just the 2026-08-01 grid fix but also the policy/
  macro-vintage/leakage fixes that landed after it (see
  `Model_Output/MANIFEST.md` for the up-to-date per-file staleness flags).
- Manuscript: 3 of an estimated 6 sections drafted (methods/data, crisis
  backtesting, policy causal-effect investigation).
- A correctness decision on two pre-existing mislabeled macro series kept
  as-is for continuity: `agri_wages_rs_day` is actually an all-occupations
  rural wage series (not agriculture-specific), and `iip_food_proc` is
  actually the overall manufacturing IIP (not the food-processing
  sub-index). Would need a full retrain + re-validation if corrected.
- Confirming the `ANTHROPIC_API_KEY` Streamlit Cloud secret is actually
  live on the deployed dashboard (only ever verified locally with a
  dummy key).
- **Live dashboard viewer access is deliberately restricted, not broken.**
  The GitHub repo itself was switched back to public (2026-08-10,
  `gh repo edit --visibility public`, confirmed via `gh api`), but the app
  still shows "you do not have access" to an anonymous visitor — this is
  Streamlit Community Cloud's own separate app-level sharing setting
  (Settings → Sharing on share.streamlit.io), independent of GitHub repo
  visibility, and the account owner confirmed (2026-08-10) it's
  deliberately kept at "Only specific people" for now rather than public.
  Team members need to be added individually there (not via GitHub repo
  access) to view the deployed app.
- NAFED/APEDA RTI request (sent early in the project for Layer 6 sourcing)
  — likely moot now that a verified policy-event dataset was found instead;
  not formally closed out.
- Onion's leader-market identification (Script 42) is a stable *regional*
  finding (Nashik belt) but not a stable *single-market* one — its
  full-sample #1 market's rank shuffles between the early/late time-period
  split. Not a bug, just an honestly-reported limit of the current data.
