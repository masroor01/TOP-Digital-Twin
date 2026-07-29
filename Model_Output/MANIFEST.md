# Model_Output Manifest

Tracks which script produced each output group, when it was last regenerated,
and whether it reflects the **current** data pipeline. Update the relevant
entry every time you re-run a script that writes here — this file exists
specifically because two stale-output bugs (`table_benchmarks.csv`,
`table_spike_auc.csv`) went unnoticed for weeks in July 2026 before being
caught during the 2026-07-29 results review. See `README.md` §3 for what
each script actually does; this file only tracks freshness.

**Current pipeline state (as of 2026-07-29):** market panel filtered to
>=70% real coverage per market (517 tomato / 246 onion / 82 potato),
potato zones P1-P3 relocated to Darjeeling/Diamond Harbour/Dehradun with
real climate+satellite data. Any output below dated before 2026-07-27 was
generated on an earlier panel and should not be treated as representing
the current pipeline without re-running its script first.

Status legend: 🟢 current · 🟡 stale, known, re-run pending · ⚫ deprecated/superseded · ⚪ orphaned (no current script produces it)

| Script | Outputs | Last generated | Status | Notes |
|---|---|---|---|---|
| `09_Agmarknet_Weekly_Panel.py` | `data/agmarknet_weekly/*.csv` (not in Model_Output) | 2026-07-28 | 🟢 | 70% coverage filter, full 2017-2026 window |
| `11_Market_Selection_And_DataStructure.py` | `filtered_panel_top.csv`, `appendix_market_selection.xlsx`, `fig01-05_*.png` | 2026-07-15 | 🟡 | Predates the mid-2026 refresh, coverage filter, and zone fix. Re-run if this script's numbers are cited anywhere. |
| `12_ModuleB_RollingOrigin_MultiHorizon.py` | `table_rolling_origin_metrics.csv`, `table_spike_auc.csv`, `fig_horizon_r2.png`, `fig_rolling_origin_rmse.png`, `fig_spike_roc.png`, `fig_mape_by_horizon.png` | 2026-07-29 | 🟢 | Re-run tonight; also fixed a real bug (a string column — `imputed_method` — was crashing LightGBM's feature matrix, missing from the exclude-list since this script predates that column existing). Own 3-fold structure (test years 2022-2024) — independent of Script 15's 4-fold setup, not directly comparable. |
| `13_Benchmark_Models.py` | `table_benchmarks.csv`, `table_comparison.csv`, `fig_benchmark_comparison.png`, `fig_skill_score.png`, `fig_r2_comparison_heatmap.png` | 2026-07-13 | ⚫ | **Deprecated as of 2026-07-29.** Its B1_Naive numbers were being cited as the ablation study's naive baseline despite using a different, mismatched 3-fold structure and stale data — this was silently wrong. Script 15 now computes naive persistence inline on its own folds instead; this script's naive output should no longer be used for that comparison. Its other benchmarks (seasonal, MA4, ARIMA) haven't been re-validated against the current panel. |
| `14_Satellite_Climate_Features.py` | `fig_era5_temperature.png`, `fig_chirps_rainfall_heatmap.png`, `fig_s2_ndvi_anomaly.png`, `fig_satellite_cross_validation.png`, `fig_climate_satellite_correlation.png`, `data/satellite_climate/*.csv` | 2026-07-28 | 🟢 | Includes real ERA5/CHIRPS/S2/MODIS for the relocated potato zones |
| `15_Ablation_Study_M0_M4.py` | `ablation_raw_results.csv`, `ablation_predictions.csv`, `table_ablation.csv`, `table_mase.csv`, `fig_ablation_*.png` | 2026-07-29 | 🟢 | Full M0-M6 + inline B1_Naive (see note on Script 13 above). Has two run modes controlled by `MARKET_LEVEL_DIAGNOSTIC` at the top of the script — check which mode before re-running: `False` = full ablation (this table), `True` = market-level diagnostic for Script 18b (`dm_market_level_*.csv`). Resting state is `True`. |
| `15b_Tree_Model_Comparison.py` | `table_tree_model_comparison*.csv`, `fig_tree_model_comparison.png` | 2026-07-29 | 🟢 | Re-run on current panel (114.8 min, RandomForest 98.8 of those). Same conclusion holds: RandomForest clearly worst, LightGBM/XGBoost/CatBoost competitive with no consistent winner — validates production LightGBM. |
| `15c_LSTM_Transformer_Comparison.py` | `table_lstm_transformer_comparison*.csv`, `fig_lstm_transformer_comparison.png` | 2026-07-29 | 🟢 | Re-run on current panel (23.0 min, much faster than the pre-filter run's 79 min). Same conclusion holds: both models perform poorly, mostly negative R², worse than every tree model. |
| `16_Zone_Assignment.py` | `fig_zone_assignment_*.png`, `data/zone_assignment.csv` | 2026-07-28 | 🟢 | Reflects relocated P1-P3 potato zones, run against the current 82-market potato panel |
| `17_TFT_Model.py` | `tft_raw_results.csv`, `table_tft_vs_lgbm.csv`, `fig_tft_*.png` | 2026-07-22 | 🟡 | Reduced-scope run only (fewer markets, shorter encoder) — full-capacity run is a known open item, not yet done. Also predates the refresh. |
| `18_Diebold_Mariano_Tests.py` | `table_diebold_mariano.csv`, `fig_dm_pvalues.png` | 2026-07-29 | 🟢 | Crop-level, M0 vs M4 headline |
| `18b_Market_Level_DM_Check.py` | `table_dm_market_level_*.csv` | 2026-07-29 | 🟢 | Market-level, M0 vs M6. Depends on Script 15 having been run with `MARKET_LEVEL_DIAGNOSTIC = True` first. |
| `23_Train_Production_Models.py` | `production_models/` (subfolder) | 2026-07-29 | 🟢 | 12 models + dashboard metadata. Own subfolder already — good pattern, mirrored by this reorg. |
| `25_Horizon_SHAP_Analysis.py` | `table_shap_by_layer.csv`, `table_shap_top_features.csv`, `fig_shap_*.png` (layer_composition, top_features, beeswarm_onion_4w) | 2026-07-29 | 🟢 | |
| *(none found)* | `table4_model_metrics.csv`, `fig_feature_importance.png`, `fig_actual_vs_pred_2024.png`, `test_predictions_2024.csv`, `lgbm_{tomato,onion,potato}.txt`, `market_list_by_crop.xlsx`, `fig_shap_{tomato,onion,potato}.png` (July 8 versions, not the July 29 SHAP figures) | 2026-07-08 | ⚪ | No script in the current `scripts/` folder produces these — orphaned from an early prototype, likely predating this repo's current script numbering. Safe to delete once confirmed unneeded; not referenced by README §3. |

## Known gaps not yet reflected anywhere above
- Full-capacity TFT run (Script 17) — deferred, see README §9
- Market-selection appendix (Script 11) — needs a re-run if its figures/tables are used in the paper
