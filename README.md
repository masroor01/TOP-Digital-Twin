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
| `09_Agmarknet_Weekly_Panel.py` | Builds the core weekly price/arrivals panel from raw Agmarknet CSVs (tomato/onion/potato, all-India). Handles ISO-week alignment and gap imputation (see §7 imputation caveat). Applies a real-coverage filter (`MIN_REAL_COVERAGE = 0.70`, added 2026-07-27, revised same day from an initial 0.80): markets are dropped unless >=70% of their own real reporting span is real (non-imputed) data — potato additionally keeps its own >=8-of-9-years balanced-panel rule first. Kept market counts (current, 2026-08-29): tomato 842 (was 1,729 unfiltered), onion 813 (was 1,587), potato 82 (was 87, balanced-panel-prefiltered count — 1,258 raw markets have at least one observation, but only 87 clear the >=8-of-10-years bar). Counts were 834/809/82 as of the 2026-08-01 grid fix (below), 840/814/82 as of 2026-08-21, and are 842/813/82 as of 2026-08-29 (onion's -1 vs. 814 not individually traced — within the normal week-to-week noise of markets sitting right at the 70% threshold, not investigated further since it's a single market) — filter logic itself unchanged. Threshold revised down from 80% to 70% on 2026-07-27 because the market-level DM test (Script 18b) at 80% found onion's h=1w result resting on only 34/189 significant markets; 70% roughly triples that to a sturdier sample. **Grid-adaptivity fix, 2026-08-01**: each market's coverage grid now starts at its own first real observation rather than a fixed global `START_DATE` — the earlier fixed-grid version silently scored a market as "missing" for every week before it existed in the system, undercounting coverage for markets that onboarded later. Fixing this grew tomato from 517 and onion from 246 to (at the time) 834/809 (potato unaffected — its balanced-panel prefilter already cleared this bar). See `Model_Output/MANIFEST.md` for the full account and current counts — `paper_drafts/` is a working folder cleared between manuscript passes, not a stable citation target. `data/market_coverage_browser.html` is a separate, purely exploratory viewer over ALL markets (unfiltered) — it was never wired into this filter and isn't a decision record. | Raw Agmarknet CSVs (see §5) |
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
| `13_Benchmark_Models.py` | Naive persistence / ARIMA benchmarks (`table_benchmarks.csv`). **Fixed 2026-09-02**: added the previously-missing national-level B4_ARIMA h=4/13/26 variants (only h=1 per-market existed before). Re-run surfaced one genuine, diagnosed outlier — tomato fold 3 (2024 test year) shows MAPE>1000%/R²<-450 at h=4 because training data ends mid-spike (the real Jul-2023 tomato escalation) and ARIMA(1,1,1) extrapolates that momentum without mean-reversion; this single fold dominates the crop-level mean, so prefer the median across folds (130% MAPE) over the mean (393%) when citing a single tomato ARIMA figure. Not a bug — see MANIFEST.md for the full diagnosis. | Script 09 |
| `15_Ablation_Study_M0_M4.py` | **The main modeling script.** Trains LightGBM variants M0→M6 (price-only up to the full pipeline), 4-fold rolling-origin CV × 4 horizons × 3 crops = 336 fits. Has a `MARKET_LEVEL_DIAGNOSTIC` flag (see §7) for retraining just 2 variants on the full market panel for higher-power statistical tests. **~10-15 min for the full run.** | Scripts 09, 10, 14, 19, 20, 21 (rebuilds the join itself) |
| `15b_Tree_Model_Comparison.py` | Compares LightGBM against RandomForest, XGBoost, CatBoost on the same CV framework — validates LightGBM as the production choice. RandomForest is clearly worst; the other three are competitive with no consistent winner. **~2 hours** (RandomForest dominates the runtime). | Same as Script 15 |
| `15c_LSTM_Transformer_Comparison.py` | Compares LightGBM against an LSTM and a small Transformer on the same CV framework. Both deep-learning models underperform every tree model, mostly negative R². **~20-25 min.** | Same as Script 15 |

### Phase D — Statistical validation

| Script | What it does | Depends on |
|---|---|---|
| `18_Diebold_Mariano_Tests.py` | Crop-level DM significance tests on Script 15's ablation results. **Fixed 2026-09-02**: was hardcoded to test only M0 vs M4 (stale since Script 15 later expanded to M0-M6) — every headline significance test it ran validated an outdated intermediate variant. Extended to the full M0→M6 chain; M0 vs M6 is now the headline pair. Re-run result: 3/12 crop×horizon cells significant (tomato h=26w, potato h=13w/26w), consistent with the ablation study's own crop/horizon heterogeneity finding. | Script 15 output (`ablation_predictions.csv`) |
| `18b_Market_Level_DM_Check.py` | Higher-power market-level DM tests | Script 15 run with `MARKET_LEVEL_DIAGNOSTIC = True` (`dm_market_level_predictions.csv`) |
| `27_Horizon_Skill_And_MCS.py` | Reshapes Script 15's MASE into a horizon-conditional skill% table (identifies each crop's crossover horizon where the full pipeline first beats naive persistence), then runs a Model Confidence Set (Hansen, Lunde & Nason 2011 — stationary-bootstrap range statistic) jointly testing all 8 variants per crop × horizon with familywise error control, the multi-model alternative to Script 18's pairwise DM tests. **<1 min** (no model fitting, pure statistical test). | Script 15 output (`ablation_predictions.csv`, `table_mase.csv`) |
| `28_Crisis_Backtesting_Case_Studies.py` | Backtests 4 real, verified crisis episodes (tomato's Jun-Sep 2023 spike/crash, onion's Aug 2023-Jan 2024 export-restriction escalation/post-ban crash, potato's Apr 2024 spike and Jan 2025 crash) against Script 15's out-of-sample fold predictions, cross-referenced with Script 19's verified policy-event log. Answers whether "naive wins on average" still holds specifically during the crises policy cares about (it does at h=1w; the data layers pull decisively ahead at h=13w/26w for tomato/onion, but not for potato). **<1 min.** | Script 15 output (`ablation_predictions.csv`), Script 19 output (`export_policy_events.csv`) |
| `29_Granger_Causality_Analysis.py` | Bidirectional Granger causality tests (ADF-checked for stationarity, FDR-corrected within crop) between price and each data layer (arrivals, climate, satellite, macro, policy), plus a light lead-lag network among each crop's top-5-by-volume markets. Complements the ablation study's "does this layer improve accuracy" question with "does this layer have independent predictive content for price at all." **<2 min.** | Script 09 output (weekly panel), Scripts 14/10b/19 outputs |
| `30_Formal_Stress_Testing.py` | Systematises the dashboard's what-if simulator into a fixed, reproducible battery of named stress scenarios, run across every market using the saved M6 production models (Script 23) -- no retraining. Onion's policy scenarios replay the exact verified 2023-24 event values (Script 19); diesel and climate scenarios are calibrated to real historical extremes. Originally surfaced a wrong-signed prediction for onion's export-duty scenario, traced to the same reactive-not-causal relationship Script 29 found; fixed via a monotonic sign constraint on the three export-control features in Script 15/23 (negligible accuracy cost). The constrained model still can't recover a realistic *magnitude* for these scenarios (median response ~0%) -- see Script 31 for why, and for the real-world-calibrated alternative. **<1 min.** | Script 23 output (production models, `reference_rows.csv`), Script 19 output |
| `31_Synthetic_DID_Policy_Effect.py` | Estimates the 2023-24 onion export-restriction episode's causal price effect directly via Synthetic Difference-in-Differences (Arkhangelsky et al. 2021), since Script 30's forecasting model can't recover a reliable counterfactual magnitude (an identification problem, not a training-signal one -- duty, MEP, and the ban only ever moved together once). Part A (cross-crop, tomato/potato as donors) and Part B (within-onion, Nashik export-hub vs non-hub) — see `Model_Output/MANIFEST.md` for the full Part A/B/C breakdown, including 2026-08-01's Part C addendum (event-study trajectories, jackknife CIs, donor-pool robustness, arrivals/quantity effect). **~5 min.** | `data/agmarknet_weekly/top_weekly_panel.csv`, Script 19 output |
| `38_SDID_InSpace_Placebo_Test.py` | Follow-up to Script 31: standard synthetic-control-literature falsification test (Abadie, Diamond & Hainmueller 2010) — treats every individual tomato/potato market as a placebo "treated" unit against the same donor pool, builds a null distribution of placebo ATTs, and judges the real onion postban ATT against it via a rank-based p-value. Result: p=0.131 — the postban effect does not clear significance against this stronger test either. **~5-10 min.** | Script 31's fitted SDID weights |
| `39_SDID_Stacked_MultiEpisode_EventStudy.py` | Extends Script 31's single-episode (2023-24) design to all 3 verified onion export bans in the panel window (Sep 2019, Sep 2020, 2023-24), stacking them into one multi-episode event study — addresses the n=1 identification problem (a single confounded natural experiment can't separate the ban's effect from the coincident weather shock that prompted it) by pooling 3 independent episodes. **~5-10 min.** | Script 19 output, `data/agmarknet_weekly/top_weekly_panel.csv` |
| `38b_SDID_Arrivals_InSpace_Placebo_Test.py` | Arrivals/quantity-outcome analogue of Script 38 (added 2026-08-21) — same in-space placebo design applied to `log1p(arrivals)` instead of price. Result: p=0.680 — fails decisively (placebo noise ~5x larger than for price). **~5-10 min.** | Script 31's fitted SDID weights |
| `39b_SDID_Arrivals_Stacked_MultiEpisode_EventStudy.py` | Arrivals/quantity-outcome analogue of Script 39 (added 2026-08-21) — same 3-episode stacked design applied to arrivals, with per-episode requalification (≥95% arrivals coverage). Onion separates from the placebo band in 10/13 post-ban week-bins, but the effect is wrong-signed and pre-dates the ban — read as corroborating reverse-causality, not an independent ban effect on quantity. **~5-10 min.** | Script 19 output, `data/agmarknet_weekly/top_weekly_panel.csv` |
| `46_Directional_Accuracy_Test.py` | Every other metric in this project (RMSE/MAE/MAPE/R²/MASE) is magnitude-based — this asks whether the model called the right price DIRECTION. Per-market (`market_id`-keyed), looks up the origin price (horizon weeks before the target week) from the raw panel and compares `sign(actual_change)` vs `sign(predicted_change)`, binomial-tested against a 50% null. M0 vs M6 only (that's what `dm_market_level_predictions.csv` has). Cross-validates the ablation story on an independent metric — potato's M0 pulls ahead of M6 at long horizons (h=26w: 78% vs 65%), matching its known "richer features don't help" pattern. B1_Naive scores exactly 0% everywhere by construction (always predicts "no change"). **~1 min.** | `Model_Output/dm_market_level_predictions.csv`, `Model_Output/ablation_predictions.csv`, `data/agmarknet_weekly/top_weekly_panel.csv` |
| `47_Market_Level_Accuracy.py` | The dashboard's "Model Accuracy" KPI is a crop+horizon-level statistic (one shared model per crop/horizon serves every market, so it's the same figure regardless of which market is selected — not a bug, but easy to mistake for one). This computes a hierarchical crop→state→market accuracy from real per-market backtest predictions (M6 variant), with two-level empirical-Bayes shrinkage (market blends toward its own state, state blends toward the crop-wide mean, both weighted by sample size) so every market/state gets a trustworthy figure instead of a hidden or noisy one. **Uses WAPE, not MAPE** (switched 2026-09-02 — see Script 50) — a plain mean-of-percentage-errors let single anomalous rows (e.g. a near-zero recorded price) distort a market's whole figure; WAPE weights each row by its own actual value instead. The React dashboard shows both the market's and its state's own track record alongside the crop-wide average. **~1 min.** | `Model_Output/dm_market_level_predictions.csv`, `Model_Output/production_models/reference_rows.csv` |
| `50_MAPE_vs_WAPE_Comparison.py` | User asked why "Model Accuracy" used MAPE instead of WAPE. Computes both metrics from the SAME already-existing M6 backtest predictions (no retraining) to check empirically before switching anything. Crop-wide: WAPE consistently lower than MAPE across all 12 crop×horizon cells (mean gap 2.6pts, max 5.65pts). Market-level: high correlation overall (0.92), but 5.1% of cells differ by ≥10 accuracy points — worst case a 185-point gap traced to a handful of rows with an anomalous near-zero recorded price. Result: switched to WAPE everywhere the same day (Scripts 23/47, dashboard). **~1 min.** | `Model_Output/dm_market_level_predictions.csv` |
| `48_State_vs_Shared_Model_Comparison.py` | Tests, rather than assumes, whether one model per (crop, state) would beat the current shared/pooled model. Trains a state-restricted M6 model for the top-2 highest-data states per crop (best case for the per-state argument: Uttar Pradesh/Keralam, Maharashtra/Kerala, West Bengal/Uttarakhand), identical 5-fold rolling-origin CV and hyperparameters as the shared model's own backtest, then DM-tests it against the shared model's predictions for the same (market, week) cells. Shared model wins 15/24 comparisons significantly (p<0.05) vs. 5/24 for state-restricted — confirms the bias-variance case for pooling: cutting training data to one state costs more from variance (thinner data feeding a ~60-feature model) than it gains from removing cross-state dilution, even in the best-data states. **~15 min.** | `data/agmarknet_weekly/top_weekly_panel.csv` + the same macro/climate/infra/policy layers as Script 23, `Model_Output/dm_market_level_predictions.csv` |
| `49_Market_Level_Model_Comparison.py` | Direct follow-up to Script 48, one level further: does a model restricted to a single MARKET's own data beat the shared model? Tested on the 3 fullest-history markets per crop (504/504 weeks, no gaps — all 9 are also Script 42's own "market leader" markets), including both Uttarakhand potato markets to test whether Script 48's one state-level win holds market-by-market. Same CV/hyperparameters as the shared model's backtest, DM-tested on identical cells. Shared model wins 9/36 comparisons significantly; market-restricted wins **0/36**. Uttarakhand's state win doesn't replicate per-market — Kashipur APMC shows the same directional pattern but never reaches significance (too little data for DM power alone), Dehradoon APMC is significantly worse restricted at short horizons. Settles the question: full pooling > partial pooling (occasionally) > no pooling (never observed to win). **~2 min.** | `data/agmarknet_weekly/top_weekly_panel.csv` + the same macro/climate/infra/policy layers as Script 23, `Model_Output/dm_market_level_predictions.csv` |

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
| `40_Escalation_Signature_Head.py` | Prototype early-warning classifier: a per-crop LightGBM model scoring each week on how closely it resembles the price-escalation pattern that has historically preceded a real policy intervention (backward-looking features only). Validated per-crop with leave-one-episode-out CV plus a placebo-in-time significance test and a data-driven (threshold-based) labeling redesign. **Three-times-fixed 2026-09-02** (all from independent audits, two review passes): (1) LOEO test set included padding negatives also present in training — fixed; (2) the placebo test's candidate pool was scored mostly in-sample — replaced with genuine K=8-fold time-block out-of-fold scoring; (3) `price_vs_seasonal_norm` — the exact quantity used to DEFINE the escalation label — was also fed to the classifier as a feature, a near-deterministic shortcut; removed, plus the standard +1/+1 permutation-test floor added to the p-value. **Final honest result: mean LOEO AUC tomato 0.860/onion 0.829/potato 0.773 (per-episode discrimination still good) — but ZERO of 8 episodes clear p<0.05 on the placebo-in-time significance test** (closest: onion_2019 p=0.208, potato_2014 p=0.216). This supersedes every earlier number, including this session's own intermediate "potato_2014 significant at p=0.000" figure — update any deck/disclosure document citing prior numbers. | `data/agmarknet_weekly/longhistory/top_weekly_panel_longhistory.csv`, `data/satellite_climate/crop_weekly_features.csv` |
| `41_Escalation_Signature_Nashik_Hub_Test.py` | Follow-up testing whether onion 2020's weak signal was real-but-geographically-diluted (its documented trigger was Nashik-region flooding specifically). Rebuilds the same detector on Nashik-hub-only price. **Three-times-fixed 2026-09-02, same fixes as Script 40.** Final result: none of the 3 episodes reach p<0.05 (onion_2019 p=0.125, onion_2020 p=0.396, onion_2023 p=0.396), but onion_2020's Nashik-hub p is still meaningfully lower than Script 40's corrected national-level p for the same episode — weak, non-significant, but directionally consistent support for localization. Supersedes the original "0.660→0.064" headline and two intermediate figures from earlier in this session. | Script 40's detector, Nashik-hub market list (from Script 31) |
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

**Always run the sanity checker after Script 09 and again after Script 23**:
```bash
python scripts/44_Pipeline_Sanity_Check.py
```
This is not optional busywork — every bug listed in §7 "Known Gotchas" was
originally found by someone noticing a suspicious number by hand, days
after it shipped (a phantom 100%-imputed tail week, a stale `data/` copy,
a forward-fill gap, a NaN-as-truthy slider, a silently-dropped output
column). Script 44 encodes exactly those five failure signatures as
automated checks and exits non-zero if any of them recur — it will not
catch a NEW class of bug it doesn't know about, but it closes the specific
gaps this project has actually hit. **Fixed 2026-09-02** (audit finding,
confirmed and reproduced): the `B1d` market_id-collision check used to
compare only the *count* of distinct market_ids between the panel and
`reference_rows.csv` — a real collision dropping one market could
coincidentally go undetected if the panel's own roster happened to shrink
by exactly one in the same run, since the counts would still match. Now
compares the actual *sets*. Verified the fix closes this exact gap:
deliberately swapped one real market_id for a fake one (same total count,
different set) on a throwaway copy — the old logic would have reported
`[PASS]`; the fixed check correctly reports `[FAIL]`, naming the specific
missing/extra market_id.

---

## 5. Data Sources & Refresh Guide

Most raw sources require **manual download** (no public API) — that's still
true for macro/climate/satellite/policy below. The market layer is the
exception as of 2026-08-29: it's now **fully automated**, see the next
subsection.

### Agmarknet (price + arrivals) — automated weekly, or manual as a fallback
**Automated (current, recommended):** `scripts/weekly_refresh/` scrapes
`api.agmarknet.gov.in` directly — no login, no captcha — validates the
result, merges it, and reruns Scripts 09→23→44, on a Windows Task Scheduler
job (Tuesdays 03:00). See `scripts/weekly_refresh/README.md` for the full
setup and safety model. `END_DATE` in `09_Agmarknet_Weekly_Panel.py`
defaults to today automatically now, so this needs no manual editing.

**Manual fallback**, if the automation isn't set up or you want an ad-hoc
pull outside the schedule:
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
- Re-run `09_Agmarknet_Weekly_Panel.py` after each download — `END_DATE`
  no longer needs manual bumping (see above), it picks up whatever's on disk.

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
- `table_sdid_arrivals_inspace_placebo.csv` — Script 38b's arrivals-outcome placebo test (postban ATT p=0.680, fails decisively)
- `table_sdid_arrivals_stacked_event_study.csv`, `table_sdid_arrivals_stacked_summary.csv` — Script 39b's arrivals-outcome stacked event study (onion separates from placebo band in 10/13 post-ban week-bins, but wrong-signed and pre-existing — corroborates reverse-causality rather than an independent effect)
- `table_directional_accuracy.csv`, `table_directional_accuracy_naive.csv`, `fig_directional_accuracy.png` — Script 46's directional (up/down) accuracy test, per-market, M0 vs M6, binomial-tested against a 50% null
- `table_market_level_accuracy.csv` — Script 47's hierarchical crop→state→market WAPE (raw + shrinkage-blended at each tier, switched from MAPE 2026-09-02) from real backtest predictions, feeding the dashboard's per-market and per-state accuracy figures
- `table_mape_vs_wape_cropwide.csv`, `table_mape_vs_wape_market.csv`, `fig_mape_vs_wape_market_scatter.png` — Script 50's empirical MAPE-vs-WAPE comparison that motivated the switch above
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

## 9. Project Status (as of 2026-08-29 — see git log for anything after this date)

**Done**: all data layers M0-M6, ablation study (extended from 4 to 5
rolling-origin folds on 2026-08-13, 420 LightGBM fits), model-family
comparisons (tree ensembles, LSTM/Transformer), statistical validation
(crop- and market-level DM tests, plus a Model Confidence Set jointly
testing all variants), horizon-conditional skill analysis, production
models, deployed dashboard (with a daily-resolution price view),
horizon-stratified SHAP analysis, crisis backtesting, Granger causality,
formal stress-testing (with a monotonic sign fix), Synthetic DID
policy-effect estimation (Parts A/B/C plus four follow-up robustness
scripts — in-space placebo and stacked multi-episode event study, each
now run for BOTH the price outcome and the arrivals/quantity outcome), a
market-panel coverage-grid bug fix (2026-08-01, grew the analytical panel
from 517/246/82 to 834/809/82 markets, later growing further to
840/814/82 as more 2026 weeks were added; full downstream cascade
re-run), a `PANEL_END` staleness bug fix plus 5 newly-verified 2026
policy events (2026-08-01), a WPI/diesel-LPG data-vintage refresh
(2026-08-01), a macro long-history extension (2026-08-03), a look-ahead
leakage fix in `s2_ndvi_anom`'s seasonal climatology with a full M6
cascade retrain (2026-08-04), a 23-year two-phase residual architecture
explored fully and **rejected** under a pre-agreed stopping rule (Scripts
32-37, 2026-08-01/04, see Phase G above), a project-wide
market-name-vs-`market_id` collision bug found and fixed across eleven
scripts total (2026-08-14 for the first nine, 2026-08-20 for two more
found later — Scripts 38/39), the market-level DM test (18b) re-run
against the corrected data (2026-08-15, no longer deferred), and the
escalation-signature early-warning prototype expanded from a partial
4-episode design to a genuine 8-episode, fully held-out one across all
three crops (Scripts 40/41, 2026-08-19/20), alongside the market
leader-follower network (Script 42, 2026-08-10, confound-checked and
stability-checked). Since then: a fully automated weekly data-refresh
pipeline (`scripts/weekly_refresh/`, 2026-08-29) that scrapes fresh
Agmarknet data unattended (no login/captcha — hits `api.agmarknet.gov.in`
directly), validates it, merges it, and reruns Scripts 09→23→44, scheduled
via Windows Task Scheduler; building and live-testing it found and fixed
two real automation bugs (a reserved-variable collision that silently
launched `python` with no arguments and hung it for ~46 hours; an exit-code-
pollution bug that misreported a genuinely successful scrape as failed) —
see `scripts/weekly_refresh/README.md` and the 2026-08-29 MANIFEST entries
for the full account. The same cycle brought tomato/onion/potato to the
same latest week (2026-08-24) for the first time and dropped potato's
latest-week imputed rate from 80.5% to 4.9%. It also surfaced and fixed a
crash in an uncommitted dashboard UI redesign (`scripts/24_Simulation_
Dashboard.py` — a deprecated Plotly `titlefont` property) found while
reviewing the repo before committing, not something introduced this pass.
Since then: a React/Node rebuild of the dashboard (`web/`, deployed
separately from the Streamlit app) and a directional accuracy test
(Script 46, 2026-08-31) — the project's only metric so far that asks
whether the model calls the right price direction rather than just how
close the magnitude is; cross-validates the ablation study's crop/horizon
heterogeneity finding on an independent axis (see MANIFEST.md).

**Not yet done / genuinely open**:
- Full-capacity TFT run (deliberately deferred). Also still carries the
  market-name/`market_id` collision bug (see below) — a second,
  independent reason not to trust its current numbers.
- Re-running Scripts 12, 13, 15b, 15c, 17, 26 on the current panel — these
  predate not just the 2026-08-01 grid fix but also the policy/
  macro-vintage/leakage fixes that landed after it, and (for all but 15b/
  15c, which were already fixed) the 2026-08-14 collision fix and the
  panel's further growth to 842/813/82 markets (see
  `Model_Output/MANIFEST.md` for the up-to-date per-file staleness
  flags). **18b is no longer on this list** — re-run 2026-08-15.
- Scripts 34/35/36 (the rejected two-phase architecture) still group
  markets by name, not `market_id`, and have never been re-verified
  against the 2026-08-14 collision fix. Low priority since the
  architecture is already a documented rejected result regardless, but
  genuinely open.
- ~~Scripts 40/41's placebo-in-time significance test scored its candidate
  pool mostly in-sample~~ — **DONE 2026-09-02**: replaced with genuine
  K=8-fold time-block out-of-fold scoring, plus a second fix removing a
  label-leakage feature (`price_vs_seasonal_norm` was both the label's
  defining threshold AND a model input) and adding the standard +1/+1
  permutation-test floor (see MANIFEST.md for both scripts). **Action
  item, not yet done**: the fully corrected result is a real, substantial
  downward revision — **zero of 8 episodes now clear p<0.05** (was the
  previously-reported 5 of 8) — any deck, disclosure document, or
  progress-review slide that cited the old escalation-detector placebo
  significance numbers needs updating to match. Not yet swept for/updated
  as of this entry.
- Manuscript: a full section-by-section drafting pass was completed and
  then intentionally cleared from `paper_drafts/` on 2026-08-21 to start
  fresh with a clean audit baseline — `paper_drafts/` is a working folder
  that does not persist between passes, not a place to look for the
  current manuscript state; check chat/session history for where the
  fresh pass currently stands.
- Roughly a week of fixes (2026-08-14 through 2026-08-20: the collision
  fix, the 18b re-run, the escalation-detector expansion) sat uncommitted
  locally before being committed and pushed to `origin/master` on
  2026-08-21 (commit `b8686fa`). No outstanding gap now, but worth
  keeping commit/push a routine part of closing out a work session going
  forward rather than letting another backlog accumulate.
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
