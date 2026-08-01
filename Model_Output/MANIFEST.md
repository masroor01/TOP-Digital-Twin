# Model_Output Manifest

Tracks which script produced each output group, when it was last regenerated,
and whether it reflects the **current** data pipeline. Update the relevant
entry every time you re-run a script that writes here — this file exists
specifically because two stale-output bugs (`table_benchmarks.csv`,
`table_spike_auc.csv`) went unnoticed for weeks in July 2026 before being
caught during the 2026-07-29 results review.

**Current pipeline state (as of 2026-08-01):** market panel filtered to
>=70% real coverage per market (**834 tomato / 809 onion / 82 potato** —
see below), potato zones P1-P3 relocated to Darjeeling/Diamond
Harbour/Dehradun with real climate+satellite data. Any output below dated
before 2026-08-01 was generated on the smaller pre-grid-fix panel (517/246/82
markets) and should not be treated as representing the current pipeline
without re-running its script first.

**2026-08-01 WPI vintage refresh.** Discovered while building an unrelated
long-history data audit (`scripts/10c_RBI_LongHistory_Parser.py`) that CMIE
restates the *entire* history of its item-level WPI series (and PPAC
diesel/LPG) on every pull, not just the newest months — confirmed with the
project owner, a CMIE subscriber. `data/rbi_dbie/rbi_dbie_macro_2017_2025.csv`
and `data/ppac_macro/ppac_diesel_lpg_2017_2025.csv` were built at an earlier
pull and had drifted from CMIE's current vintage across their whole window:
mean |diff| 13.7% for `wpi_tomato`, 9.4% `wpi_onion`, 6.9% `wpi_potato`, 6.6%
`wpi_vegetables_total`, 3.8% `wpi_fruits_vegetables`, 1-3% diesel/LPG (repo
rate/reverse-repo/USD-INR unaffected — discrete/point-in-time, don't get
restated). Refreshed both files to the current vintage, re-ran Script 22
(master join) and Script 23 (retrain). Measured impact: `model_uncertainty.json`
(validated RMSE/MAPE per crop/horizon) came out byte-identical despite tree
counts shifting per model — a real but small effect, not headline-moving.
`reference_rows.csv` (dashboard's live baseline) unchanged, since only
historical training rows were affected, not the latest week. Not treated as
grounds for a full ablation/crisis-backtest/SHAP/stress-test cascade re-run
on this basis alone. If Script 15/25/28/30/31 get re-run for some other
reason, they will pick up the refreshed vintage automatically (no separate
action needed there).

**2026-08-01 grid-adaptivity fix — read this before citing any market
count from before this date.** Script 09's weekly grid was built from one
fixed global `START_DATE` for every market (a plain cartesian product of
markets x weeks), so a market that only began reporting years after
`START_DATE` was scored as "missing" for every week before it existed in
the system — silently dragging its real-coverage ratio down for a reason
that had nothing to do with its actual data quality. Discovered while
investigating whether extending the panel's history back to 2003 was
worthwhile (see `scripts/32_LongHistory_Panel_Builder.py` /
`scripts/33_LongHistory_Validation_Experiment.py`, an exploratory side
study, not part of the main pipeline): at a 23-year window this collapsed
the long-history panel to 162/98/28 qualifying markets; fixing the grid to
start each market at its own first real observation recovered it to
684/746/72. The SAME bug existed at the *production* 9-year (2017-2026)
window too, just with milder consequences — fixing it there and re-running
Script 09 changed the production panel from **517 -> 834 tomato** and
**246 -> 809 onion** markets (potato: 82 -> 82, unchanged, because its
separate >=8-years balanced-panel prefilter already guaranteed every
candidate market existed for nearly the whole window, leaving nothing for
the bug to affect). Median real coverage of the newly-recovered markets is
92-96%, not thin/borderline data — see
`paper_drafts/methods_data_section.txt` Sec 3.3 for the full account.
Given the scale of the change, the entire downstream cascade (ablation,
production models, SHAP, MCS, crisis backtests, Granger causality, stress
test, SDID) was re-run the same day — see individual entries below.
Deferred: Scripts 15b/15c (tree/LSTM model-family comparisons, ~140 min
combined, unlikely to qualitatively change) and 18b (market-level DM,
needs the `MARKET_LEVEL_DIAGNOSTIC` toggle round-trip) — both still
reflect the pre-fix, smaller panel; treat as directionally valid but
numerically stale until re-run.

Status legend: 🟢 current · 🟡 stale, known, re-run pending · ⚫ deprecated/superseded · ⚪ orphaned (no current script produces it)

| Script | Outputs | Last generated | Status | Notes |
|---|---|---|---|---|
| `09_Agmarknet_Weekly_Panel.py` | `data/agmarknet_weekly/*.csv` (not in Model_Output) | 2026-08-01 | 🟢 | Grid-adaptivity fix (see above). 834 tomato / 809 onion / 82 potato markets, 70% real-coverage filter, 2017-2026 window. |
| `11_Market_Selection_And_DataStructure.py` | `filtered_panel_top.csv`, `appendix_market_selection.xlsx`, `fig01-05_*.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Confirms 832/807/82 markets selected at ~100% (small 2-market discrepancy vs Script 09's raw count is a pre-existing, minor date-window edge effect between the two scripts, not a new issue). Median coverage 92.4% tomato / 95.6% onion / 95.6% potato. |
| `12_ModuleB_RollingOrigin_MultiHorizon.py` | `table_rolling_origin_metrics.csv`, `table_spike_auc.csv`, `fig_horizon_r2.png`, `fig_rolling_origin_rmse.png`, `fig_spike_roc.png`, `fig_mape_by_horizon.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix — market set has since grown substantially (esp. onion, 246->809). Own 3-fold structure (test years 2022-2024), independent of Script 15. Re-run pending. |
| `13_Benchmark_Models.py` | `table_benchmarks.csv`, `table_comparison.csv`, `fig_benchmark_comparison.png`, `fig_skill_score.png`, `fig_r2_comparison_heatmap.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. **B1_Naive here is still not the ablation study's naive baseline** — use Script 15's inline B1_Naive for any M0-M6 comparison. Re-run pending. |
| `14_Satellite_Climate_Features.py` | `fig_era5_temperature.png`, `fig_chirps_rainfall_heatmap.png`, `fig_s2_ndvi_anomaly.png`, `fig_satellite_cross_validation.png`, `fig_climate_satellite_correlation.png`, `data/satellite_climate/*.csv` | 2026-07-28 | 🟢 | Crop/zone-level, not market-level — unaffected by the market-panel grid fix. Includes real ERA5/CHIRPS/S2/MODIS for the relocated potato zones. |
| `15_Ablation_Study_M0_M4.py` | `ablation_raw_results.csv`, `ablation_predictions.csv`, `table_ablation.csv`, `table_mase.csv`, `fig_ablation_*.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel (834/809/82 markets). Full M0-M6 + inline B1_Naive. Monotonic constraint on export_banned/export_duty_pct/mep_usd_per_tonne (added 2026-07-31, see note below) retained — negligible accuracy cost. **`table_mase.csv` was a stale one-off**: found 2026-08-01 that this script never actually generated it despite the docstring/this manifest claiming it did (the file on disk was 3+ days old relative to same-day ablation re-runs) — now computed as a real step of this script (Sec 7) instead of a manual artifact. Has two run modes via `MARKET_LEVEL_DIAGNOSTIC`; resting state `False`. |
| `15b_Tree_Model_Comparison.py` | `table_tree_model_comparison*.csv`, `fig_tree_model_comparison.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. ~115 min re-run, deliberately deferred — qualitative conclusion (RandomForest worst, LightGBM/XGBoost/CatBoost competitive) unlikely to change with a larger panel. |
| `15c_LSTM_Transformer_Comparison.py` | `table_lstm_transformer_comparison*.csv`, `fig_lstm_transformer_comparison.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. ~23 min re-run, deliberately deferred alongside 15b. |
| `16_Zone_Assignment.py` | `fig_zone_assignment_*.png`, `data/zone_assignment.csv` | 2026-07-28 | 🟢 | Reflects relocated P1-P3 potato zones; potato's market panel is unchanged by the grid fix (82->82). |
| `17_TFT_Model.py` | `tft_raw_results.csv`, `table_tft_vs_lgbm.csv`, `fig_tft_*.png` | 2026-07-22 | 🟡 | Reduced-scope run only — full-capacity run remains deferred (see README §9), now additionally predates the 2026-08-01 grid fix too. |
| `18_Diebold_Mariano_Tests.py` | `table_diebold_mariano.csv`, `fig_dm_pvalues.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Crop-level, M0 vs M4 headline; conclusions qualitatively unchanged. |
| `18b_Market_Level_DM_Check.py` | `table_dm_market_level_*.csv` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. Needs Script 15 re-run with `MARKET_LEVEL_DIAGNOSTIC = True` first (then flipped back) — deferred alongside 15b/15c. |
| `23_Train_Production_Models.py` | `production_models/` (subfolder) | 2026-08-01 | 🟢 | Retrained on the grid-fixed panel — 1,721 market baselines (up from ~845). Monotonic constraint on the three export-control policy features retained (see Script 15 note); oversampling attempt tried and reverted (see script comments) — magnitude question now handled by Script 31 (SDID) instead of asked of this model. **Retrained again 2026-08-01 (WPI vintage refresh, see Script 22 note)** — tree counts shifted per crop/horizon (e.g. tomato h=1w 56→134) but validated RMSE/MAPE in `model_uncertainty.json` came out byte-identical; `reference_rows.csv` (dashboard's live baseline) unchanged, since only historical training rows were affected, not the latest week. Small, real, non-headline-moving effect — not treated as grounds for a full ablation/crisis-backtest/SHAP/stress-test cascade re-run. |
| `25_Horizon_SHAP_Analysis.py` | `table_shap_by_layer.csv`, `table_shap_top_features.csv`, `fig_shap_*.png` (layer_composition, top_features, beeswarm_onion_4w) | 2026-08-01 | 🟢 | Re-run against the retrained production models. |
| `26_Weekly_To_Daily_Disaggregation.py` | `table_dow_pattern.csv`, `table_disagg_backtest.csv`, `fig_disagg_example.png`, `fig_dow_pattern.png` | 2026-07-30 | 🟡 | Predates the 2026-08-01 grid fix and production-model retrain. Not yet re-run; the day-of-week pattern and noise-band conclusions are unlikely to shift much but the reference series should be refreshed. |
| `27_Horizon_Skill_And_MCS.py` | `table_horizon_skill.csv`, `table_horizon_skill_crossover.csv`, `fig_horizon_skill.png`, `table_mcs.csv`, `table_mcs_membership.csv`, `fig_mcs_membership.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel with the now-real `table_mase.csv`. **M6 crossover horizon improved**: tomato 13w->**4w**, onion 26w->**13w**, potato unchanged at 26w — the larger, correctly-scored panel gives the data layers a statistically confirmed edge earlier than the pre-fix panel showed. B1_Naive remains the sole 90%-MCS survivor at h=1w for all 3 crops. |
| `28_Crisis_Backtesting_Case_Studies.py` | `table_crisis_backtests.csv`, `fig_crisis_tomato_2023.png`, `fig_crisis_onion_2023_24.png`, `fig_crisis_potato_2024_25_spike.png`, `fig_crisis_potato_2024_25_crash.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Conclusion unchanged: naive wins at h=1w even during crises (all 4 episodes); M6 dramatically beats naive at h=13w/26w for the tomato and onion crises; potato's Apr 2024 spike remains the one episode where M6 loses to naive at every horizon. |
| `29_Granger_Causality_Analysis.py` | `table_granger_layers.csv`, `fig_granger_layers.png`, `table_granger_market_network.csv`, `fig_granger_market_network.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Conclusions unchanged: climate Granger-causes price for all 3 crops; arrivals Granger-causes price for tomato/potato but not onion (reverse only); price Granger-causes tomato's `policy_intervention` flag (confirms reactive, not causal, policy timing — this is the mechanism behind Script 30's stress-test finding below). `macro_wpi_own_crop`'s strong F-stats remain very likely mechanical, not causal — do not cite as a novel result. Top-5-by-arrivals market network unchanged (the biggest hub markets were already in the pre-fix panel). |
| `30_Formal_Stress_Testing.py` | `table_stress_test_results.csv`, `fig_stress_test_{tomato,onion,potato}.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel and retrained models. **Sign issue from the earlier (pre-2026-07-31) unconstrained model is fixed** via the monotonic constraint on export_banned/export_duty_pct/mep_usd_per_tonne (Script 15/23) — onion's export-duty/ban scenarios no longer predict a wrong-signed price increase. Current state: median response collapses to ~0.0% for these scenarios (correct sign, but the constrained model can't recover a realistic magnitude either — see Script 31). This is the expected, accepted trade-off: forecasting accuracy cost was negligible, but this model should not be read as sizing a policy counterfactual's magnitude. |
| `31_Synthetic_DID_Policy_Effect.py` | Part A/B: `table_sdid_policy_effect.csv`, `table_sdid_unit_weights.csv`, `fig_sdid_treated_vs_synthetic.png`, `table_sdid_hub_policy_effect.csv`, `fig_sdid_hub_vs_nonhub.png`. Part C: `table_sdid_event_study.csv`, `fig_sdid_event_study.png`, `table_sdid_donor_robustness.csv`, `table_sdid_arrivals_effect.csv`, `fig_sdid_arrivals_treated_vs_synthetic.png` | 2026-08-01 | 🟢 | Estimates the 2023-24 onion export-restriction episode's price effect directly via Synthetic DID (Arkhangelsky et al. 2021), since Script 30's forecasting model can't recover a reliable magnitude (identification problem — duty/MEP/ban only ever moved together once). Part A (cross-crop, tomato/potato as donors): escalation-phase ATT +56.2% is credible (huge gap vs tomato placebo -30.7%), but the post-ban ATT (+14.5%) **fails its own placebo check** — indistinguishable from same-window tomato (+14.5%) and potato (+15.2%) placebo ATTs. Part B (within-onion, Nashik export-hub vs non-hub markets): real 2023 postban ATT +3.7% vs placebo-in-time (fake 2021 dates) ATT -3.9%. **Part C, added 2026-08-01 in response to review feedback that price-only static ATTs "don't fully cover the picture":** (1) *event-study trajectories* (`table_sdid_event_study.csv`) — same fitted weights, no new model fits, decomposes each static ATT into a weekly path; (2) *jackknife 90% CIs* (delete-one over each unit's underlying markets, per Arkhangelsky et al. Sec. 5) — now make the Part A postban placebo failure a formal result, not an eyeball call: onion 90% CI [+12.2%, +16.8%] fully overlaps tomato-placebo's [+12.3%, +16.8%]. The escalation-phase CI [+52.3%, +60.1%] does NOT overlap tomato-placebo's [-31.8%, -29.6%], supporting that estimate. Notably, Part B's hub-design CIs also now separate: real-2023 postban [+0.9%, +6.6%] vs. placebo-in-time [-6.3%, -1.4%] do NOT overlap — nudges Part B from "inconclusive" toward "weak positive signal", though still a small effect on a small market count (8 hub markets); (3) *donor-pool robustness* — the escalation-phase ATT is stable under leave-one-donor-out (top-15 by weight): range [+46.1%, +60.8%] vs. full-sample +56.2%, never close to the tomato placebo; (4) *arrivals/quantity effect* (39 onion / 30 potato / 139 tomato markets with ≥95% arrivals coverage) — tests the ban's actual supply mechanism instead of inferring it from price. Counter-intuitive finding: onion arrivals in the treated markets *fell* relative to synthetic control (escalation -25.8%, postban -14.3%), the opposite of what "keep supply domestic" should produce, while placebo tomato/potato arrivals rose or held flat — reported honestly as an open finding (plausibly farmers withholding stock at crashed prices, not a mechanism failure) rather than resolved. **Overall honest conclusion unchanged: no single design here fully isolates the ban's causal magnitude**, but Part C narrows exactly where the uncertainty lives — real for the escalation phase, genuinely ambiguous for the post-ban relief effect, and an open question on the supply-arrivals mechanism. |
| *(none found)* | `table4_model_metrics.csv`, `fig_feature_importance.png`, `fig_actual_vs_pred_2024.png`, `test_predictions_2024.csv`, `lgbm_{tomato,onion,potato}.txt`, `market_list_by_crop.xlsx`, `fig_shap_{tomato,onion,potato}.png` (July 8 versions, not the July 29 SHAP figures) | 2026-07-08 | ⚪ | No script in the current `scripts/` folder produces these — orphaned from an early prototype, likely predating this repo's current script numbering. Safe to delete once confirmed unneeded; not referenced by README §3. |

## Exploratory, not part of the main numbered pipeline
| Script | Outputs | Last generated | Notes |
|---|---|---|---|
| `32_LongHistory_Panel_Builder.py` | `data/agmarknet_weekly/longhistory/*.csv` | 2026-08-01 | One-off validation experiment: builds a 2003-2026 (23-year) version of the panel to test whether extending training history beyond the 2017 floor helps. Not used by any production script. |
| `33_LongHistory_Validation_Experiment.py` | `table_longhistory_validation.csv`, `table_longhistory_validation_raw.csv` | 2026-08-01 | Controlled comparison (same market set, same folds, same recipe) of a 9-year vs 23-year training window. Finding: real, modest improvement (9/12 crop x horizon cells, ~1.2pp average MAPE reduction) concentrated in onion (all 4 horizons) and tomato (except h=26w, where longer history measurably hurts — plausibly a 2003-2016 regime-break effect: pre-GST, pre-eNAM, pre-2016-demonetization). Potato: mostly neutral to mildly positive. This motivated the Script 09 grid-adaptivity fix (see top of file) but a full 2003-floor production adoption was NOT made — only the grid-adaptivity bug fix was adopted into production, at the existing 2017 floor. |

## Known gaps not yet reflected anywhere above
- Full-capacity TFT run (Script 17) — deferred, see README §9
- Scripts 15b, 15c, 18b — deferred re-runs on the grid-fixed panel (see 2026-08-01 note at top of file)
- Daily-resolution forecasting: tried training genuinely new daily models (LightGBM M6) on 2026-07-29 — abandoned, daily naive persistence won even more decisively than weekly naive, and the daily coverage filter collapsed market counts 3-6x. Script 26 (2026-07-30) instead disaggregates the existing weekly model's forecasts into a smooth daily curve with an honest uncertainty band — not a validated daily forecast, a visualization aid. Wired into the dashboard as a "Daily price view" expander below the ticker. Predates the 2026-08-01 grid fix, not yet re-run.
- A full 2003-floor (23-year) production panel remains a live option for future work — Script 33's validation showed a real, modest benefit, but adopting it means re-running the entire cascade again on an even larger panel; deferred pending a decision on whether the effect size justifies it.
