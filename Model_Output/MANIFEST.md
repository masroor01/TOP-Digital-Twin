# Model_Output Manifest

Tracks which script produced each output group, when it was last regenerated,
and whether it reflects the **current** data pipeline. Update the relevant
entry every time you re-run a script that writes here — this file exists
specifically because two stale-output bugs (`table_benchmarks.csv`,
`table_spike_auc.csv`) went unnoticed for weeks in July 2026 before being
caught during the 2026-07-29 results review.

**Current pipeline state (as of 2026-08-21):** market panel filtered to
>=70% real coverage per market (**840 tomato / 814 onion / 82 potato** —
grown further since 2026-08-10 as the panel was extended with more 2026
weeks; potato unchanged), potato zones P1-P3 relocated to
Darjeeling/Diamond Harbour/Dehradun with real climate+satellite data. The
panel/pipeline has moved **five more times since the 2026-08-10 snapshot
this file previously reflected** (a market-name/`market_id` collision fix
across nine scripts plus two more found later, an escalation-detector
expansion from a partial 4-episode design to a genuine 8-episode one, and
two new arrivals-outcome SDID robustness checks — each documented below)
— any output dated before 2026-08-14 should be treated as stale relative
to the current pipeline, not just anything before 2026-08-04.

**2026-08-21 arrivals-outcome SDID robustness checks (new scripts).**
Script 31 Part C.4's arrivals/quantity ATT (onion arrivals fell relative
to synthetic control in every sub-window) had never been tested against a
null distribution, unlike the price-based ATT. Two new scripts close this
gap: `38b_SDID_Arrivals_InSpace_Placebo_Test.py` (in-space permutation,
169 donor markets, postban window) and
`39b_SDID_Arrivals_Stacked_MultiEpisode_EventStudy.py` (stacked design
across all 3 onion export bans, arrivals outcome). Results: the in-space
test **fails decisively** (p=0.680 — the real ATT sits well inside 169
markets' ordinary noise). The stacked test shows onion arrivals
separating from the tomato/potato placebo band in 10 of 13 post-ban
week-bins — but wrong-signed relative to the ban's stated mechanism
(should raise arrivals, not lower them) and already present in the
pre-ban escalation window (-25.8%, more negative than postban's -14.3%).
Read together: the arrivals decline is real (not market-level noise) but
corroborates the existing reverse-causality finding rather than
establishing an independent, ban-caused arrivals effect. See entries
below and the Script 31 entry for the original point estimates.

**2026-08-19/20 escalation-signature detector expansion (Script 40).** A
fact-check pass on the patent/disclosure materials found tomato and
potato's single episodes each had been evaluated in-sample only, while
onion (3 episodes even then) got genuine leave-one-episode-out validation
— an undisclosed confidence gap. Fixed by researching and primary-source-
verifying three further episodes (tomato Jul-2024 PIB PRID=2038421,
tomato Aug-2025 PIB PRID=2154078, potato Jul-2016 DGFT Notification 15),
bringing the design to a genuine 8-episode, fully held-out test across all
three crops (onion 2019/2020/2023, tomato 2023/2024/2025, potato
2014/2016 — the last two potato episodes draw on the separate longhistory
panel, since they predate the main panel's 2017 start). Two hardcoded-by-
crop-name bugs (the placebo `score_type` label and the case-study figure
grid, both keyed to the literal string `'onion'`) were found and fixed in
the same pass — both would otherwise have kept the new episodes mislabelled
`in_sample` even after being added. Honest result: mean held-out AUC
0.801 tomato / 0.831 onion / 0.849 potato; 5 of 8 episodes reach placebo
significance at p<0.05 (both potato, 2 of 3 onion, 1 of 3 tomato) — see
the updated table row below; this supersedes the 2026-08-08 "onion
2019/2023 clean, 2020 not" partial summary previously in this file.

**2026-08-15 market-level DM test re-run (Script 18b) — closes a gap this
file previously listed as deferred.** Script 15's `MARKET_LEVEL_DIAGNOSTIC`
branch was patched to carry `market_id` through to
`dm_market_level_predictions.csv` (it previously carried only the market
name, which is not always unique — see the 2026-08-14 entry below), then
re-run, then Script 18b re-run against the corrected predictions. No
longer 🟡 stale as this file previously said — see the updated table row.

**2026-08-14 market-name-vs-`market_id` collision bug, project-wide.** A
handful of market NAMES repeat across different STATES within the same
crop (Fatehabad APMC in Haryana AND Uttar Pradesh for both tomato and
onion; Balugaon APMC in Odisha AND Assam for tomato; Pratapgarh APMC in
Rajasthan AND Uttar Pradesh for onion) — 4 true `(crop, market)`
collisions out of 800+ markets per crop. Any script that grouped or
pivoted on the market NAME column rather than `market_id` silently merged
these physically distinct markets, corrupting lag/rolling features and
target construction for the affected markets specifically. Found via a
market-count audit, fixed across Scripts 15, 15b, 15c, 23, 25, 29, 31, 33,
42 by switching the join/group key to `market_id` (or, for 31/42's many
pivot call-sites, relabelling colliding names to include state once at
load time). **Two more scripts were found still vulnerable on 2026-08-20
and separately fixed and re-run: 38 (in-space placebo) and 39 (stacked
multi-episode event study)** — both had been missed in the original nine-
script pass. Consistent with every other instance of this bug in the
project, re-running 38/39 with the fix produced results **bit-for-bit
identical** to the pre-fix versions — a confirmed, not assumed, null
effect on those two analyses specifically. **Scripts 34/35/36 (the
rejected two-phase architecture) still group by market name and have not
been re-verified** — low priority since that architecture is already
rejected regardless of this bug's effect, but genuinely open; see the
Exploratory table below.

**2026-08-04 decision: the 23-year two-phase residual architecture is
REJECTED, not "deferred pending a decision."** Script 33's real-but-modest
~1.2pp average MAPE improvement from a longer training window (noted
below) was built out into a full two-phase baseline+residual stack
(Scripts 34-37) and formally tested. Result: the combined Phase1+residual
forecast is worse than both Phase 1 alone and production M6 at nearly
every crop/horizon cell; Phase 1 alone's apparent edge over M6 (notably
onion) does not survive formal DM testing (1/12 cells significant at
p<0.05). M6 remains the sole production/paper architecture — see the
"Exploratory" table below for the full Script 32-37 breakdown.

**2026-08-04 look-ahead leakage fix.** `s2_ndvi_anom`'s seasonal
climatology (Script 14) was a full-sample mean per (zone, ISO-week) — a
2018 row's anomaly could use NDVI observed as late as 2026. Fixed to an
expanding mean over strictly prior ISO-years only (each zone's first S2
year, 2017, now has genuine NaN anomaly instead of a leaked value;
overall `s2_ndvi_anom` coverage drops 35.0%→31.1%, consistent with losing
exactly one year per zone). Full downstream cascade retrained against the
corrected feature: Scripts 15, 23, 25, 27, 28, 30 (NOT 12/13/17/18b/26,
NOT 29/31/38/39 — see their individual entries for exactly which date's
run they reflect). Net accuracy impact small and mixed, as expected for a
smooth seasonal term (e.g. onion h=13w MAPE 40.9%→35.8%); no architectural
or ranking changes.

**2026-08-01 policy-events verification + `PANEL_END` fix.** The
dashboard's "carried forward, ~30w stale" warning on every export-policy
control was traced to `PANEL_END` being frozen at 2025-12-31 while the
price panel had already grown past it — not an inherent external-data lag
like macro/climate, as previously assumed. Extended `PANEL_END` to
2026-12-31, re-ran Scripts 19→22→23. Also verified and added 5 new 2026
policy events to `TOP_policy_trade_verified_2017_2026.xlsx` after
cross-checking a user-supplied AI-drafted policy report against
independent primary sources rather than trusting it at face value (one
claim included a district-level figure that did not match independent
reporting and was deliberately left out rather than included on
convenience). See Script 19's entry below.

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

**2026-08-03 macro long-history extension.** Built to feed the two-phase
architecture's 2003-2026 baseline panel (`scripts/10c_RBI_LongHistory_Parser.py`,
extends repo/reverse-repo/USD-INR/WPI/diesel-LPG back to their real source
floors) and to extend `scripts/10_CMIE_Macro_Parser.py`'s own window from
2017-01-01/2025-12-31 to 2000-01-01/2026-12-31. As a side effect this fixed
a pre-existing gap: `export_veg_usd_mn`, `import_veg_usd_mn`, and
`crude_oil_usd_bbl` were NaN for early-2026 rows in production even though
the already-on-disk source files had that data — just not previously
re-parsed. Validated at 0% drift vs. production across all 6 re-parsed
columns before trusting the extension. Two pre-existing mislabeling
quirks (`agri_wages_rs_day` is all-occupations not agri-specific;
`iip_food_proc` is overall manufacturing not food-processing) were left
untouched, same as always — see README §9.

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
92-96%, not thin/borderline data. (The manuscript's own Data & Study Area
section documents this in full; `paper_drafts/` is a working folder that
gets cleared between manuscript-writing passes, so it is not a stable
citation target from this file — cite the live panel or this manifest
instead.) Given the scale of the change, the entire downstream cascade (ablation,
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
| `09_Agmarknet_Weekly_Panel.py` | `data/agmarknet_weekly/*.csv` (not in Model_Output) | 2026-08-10 (panel extended with more 2026 weeks since; grid/filter logic unchanged since 2026-08-01) | 🟢 | Grid-adaptivity fix (see 2026-08-01 note below). Current market count 840 tomato / 814 onion / 82 potato (grown from 834/809/82 as the panel picked up several 2026 weeks — filter logic and threshold unchanged, just more weeks for borderline markets to clear 70%), 70% real-coverage filter, 2017-2026 window. |
| `11_Market_Selection_And_DataStructure.py` | `filtered_panel_top.csv`, `appendix_market_selection.xlsx`, `fig01-05_*.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Confirms 832/807/82 markets selected at ~100% (small 2-market discrepancy vs Script 09's raw count is a pre-existing, minor date-window edge effect between the two scripts, not a new issue). Median coverage 92.4% tomato / 95.6% onion / 95.6% potato. |
| `12_ModuleB_RollingOrigin_MultiHorizon.py` | `table_rolling_origin_metrics.csv`, `table_spike_auc.csv`, `fig_horizon_r2.png`, `fig_rolling_origin_rmse.png`, `fig_spike_roc.png`, `fig_mape_by_horizon.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix — market set has since grown substantially (esp. onion, 246->809). Own 3-fold structure (test years 2022-2024), independent of Script 15. Re-run pending. |
| `13_Benchmark_Models.py` | `table_benchmarks.csv`, `table_comparison.csv`, `fig_benchmark_comparison.png`, `fig_skill_score.png`, `fig_r2_comparison_heatmap.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. **B1_Naive here is still not the ablation study's naive baseline** — use Script 15's inline B1_Naive for any M0-M6 comparison. Re-run pending. |
| `14_Satellite_Climate_Features.py` | `fig_era5_temperature.png`, `fig_chirps_rainfall_heatmap.png`, `fig_s2_ndvi_anomaly.png`, `fig_satellite_cross_validation.png`, `fig_climate_satellite_correlation.png`, `data/satellite_climate/*.csv` | 2026-08-04 | 🟢 | Crop/zone-level, not market-level — unaffected by the market-panel grid fix. Includes real ERA5/CHIRPS/S2/MODIS for the relocated potato zones. Re-run 2026-08-04 for the `s2_ndvi_anom` look-ahead leakage fix (see note above) — coverage drops 35.0%→31.1% as expected. |
| `15_Ablation_Study_M0_M4.py` | `ablation_raw_results.csv`, `ablation_predictions.csv`, `table_ablation.csv`, `table_mase.csv`, `fig_ablation_*.png` | 2026-08-14 | 🟢 | Re-run on the grid-fixed panel (now 840/809/82 markets — actually 840/814/82, panel grown since). Full M0-M6 + inline B1_Naive. Monotonic constraint on export_banned/export_duty_pct/mep_usd_per_tonne retained — negligible accuracy cost. Cross-validation extended from 4 to 5 rolling-origin folds (2022-2026, last partial) on 2026-08-13, additive to the original four — 420 LightGBM fits total. **`MARKET_LEVEL_DIAGNOSTIC` branch patched 2026-08-14/15** to carry `market_id` (not just market name) through to `dm_market_level_predictions.csv`, feeding the corrected Script 18b re-run. Resting state `MARKET_LEVEL_DIAGNOSTIC = False`. |
| `15b_Tree_Model_Comparison.py` | `table_tree_model_comparison*.csv`, `fig_tree_model_comparison.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. ~115 min re-run, deliberately deferred — qualitative conclusion (RandomForest worst, LightGBM/XGBoost/CatBoost competitive) unlikely to change with a larger panel. |
| `15c_LSTM_Transformer_Comparison.py` | `table_lstm_transformer_comparison*.csv`, `fig_lstm_transformer_comparison.png` | 2026-07-29 | 🟡 | Predates the 2026-08-01 grid fix. ~23 min re-run, deliberately deferred alongside 15b. |
| `16_Zone_Assignment.py` | `fig_zone_assignment_*.png`, `data/zone_assignment.csv` | 2026-07-28 | 🟢 | Reflects relocated P1-P3 potato zones; potato's market panel is unchanged by the grid fix (82->82). |
| `17_TFT_Model.py` | `tft_raw_results.csv`, `table_tft_vs_lgbm.csv`, `fig_tft_*.png` | 2026-07-22 | 🟡 | Reduced-scope run only — full-capacity run remains deferred (see README §9), now additionally predates the 2026-08-01 grid fix too. |
| `18_Diebold_Mariano_Tests.py` | `table_diebold_mariano.csv`, `fig_dm_pvalues.png` | 2026-08-01 | 🟢 | Re-run on the grid-fixed panel. Crop-level, M0 vs M4 headline; conclusions qualitatively unchanged. |
| `18b_Market_Level_DM_Check.py` | `table_dm_market_level_*.csv` | 2026-08-15 | 🟢 | **No longer deferred.** Re-run against the `market_id`-corrected Script 15 diagnostic predictions (see 2026-08-14/15 note above). M0-vs-M6 per-market: at h=1w, significant markets favour M0 for tomato/onion (80.6%/68.9%) but M6 for potato (75.0%) — a reversal in both crops from the last figures this file recorded; at h=13w the pattern flips toward M6 for tomato/onion (96.4%/92.5%) and stays with M0 for potato (100%). Treat any earlier-recorded percentages for this table as superseded. |
| `19_Policy_Trade_Events.py` | `data/policy_trade/policy_weekly_features.csv` (not in Model_Output) | 2026-08-01 | 🟢 | `PANEL_END` extended 2025-12-31→2026-12-31 (see note above) — fixes the dashboard's ~30-week-stale policy-control warning. 5 new 2026 events verified against primary sources and added (CBIC notification, PIB release, 3 press-corroborated buffer-price hikes, sourcing tier kept honest for the latter). |
| `23_Train_Production_Models.py` | `production_models/` (subfolder) | 2026-08-04 | 🟢 | Retrained on the grid-fixed panel — 1,721 market baselines (up from ~845). Monotonic constraint on the three export-control policy features retained; oversampling attempt tried and reverted — magnitude question now handled by Script 31 (SDID) instead of asked of this model. Retrained 2026-08-01 for the WPI vintage refresh (tree counts shifted per crop/horizon but validated RMSE/MAPE came out byte-identical) and **again 2026-08-04 for the `s2_ndvi_anom` leakage fix** — small, real, non-headline-moving effect each time; `reference_rows.csv` (dashboard's live baseline) unchanged both times, since only historical training rows were affected, not the latest week. |
| `25_Horizon_SHAP_Analysis.py` | `table_shap_by_layer.csv`, `table_shap_top_features.csv`, `fig_shap_*.png` (layer_composition, top_features, beeswarm_onion_4w) | 2026-08-04 | 🟢 | Re-run against the retrained production models, most recently for the `s2_ndvi_anom` leakage fix. |
| `26_Weekly_To_Daily_Disaggregation.py` | `table_dow_pattern.csv`, `table_disagg_backtest.csv`, `fig_disagg_example.png`, `fig_dow_pattern.png` | 2026-07-30 | 🟡 | Predates the 2026-08-01 grid fix AND every fix since (policy/PANEL_END, WPI vintage, macro long-history, `s2_ndvi_anom` leakage). Not yet re-run; the day-of-week pattern and noise-band conclusions are unlikely to shift much but the reference series should be refreshed. |
| `27_Horizon_Skill_And_MCS.py` | `table_horizon_skill.csv`, `table_horizon_skill_crossover.csv`, `fig_horizon_skill.png`, `table_mcs.csv`, `table_mcs_membership.csv`, `fig_mcs_membership.png` | 2026-08-04 | 🟢 | Re-run on the grid-fixed panel with the now-real `table_mase.csv`. **M6 crossover horizon improved**: tomato 13w->**4w**, onion 26w->**13w**, potato unchanged at 26w. B1_Naive remains the sole 90%-MCS survivor at h=1w for all 3 crops. Re-run again 2026-08-04 for the `s2_ndvi_anom` leakage fix — no ranking changes. |
| `28_Crisis_Backtesting_Case_Studies.py` | `table_crisis_backtests.csv`, `fig_crisis_tomato_2023.png`, `fig_crisis_onion_2023_24.png`, `fig_crisis_potato_2024_25_spike.png`, `fig_crisis_potato_2024_25_crash.png` | 2026-08-04 | 🟢 | Re-run on the grid-fixed panel, then again 2026-08-04 for the `s2_ndvi_anom` leakage fix. Conclusion unchanged: naive wins at h=1w even during crises (all 4 episodes); M6 dramatically beats naive at h=13w/26w for the tomato and onion crises; potato's Apr 2024 spike remains the one episode where M6 loses to naive at every horizon. |
| `29_Granger_Causality_Analysis.py` | `table_granger_layers.csv`, `fig_granger_layers.png`, `table_granger_market_network.csv`, `fig_granger_market_network.png` | 2026-08-01 | 🟡 | Re-run on the grid-fixed panel; predates the WPI-vintage/macro-long-history/leakage fixes since. Conclusions unlikely to shift (Granger significance is robust to small feature drift) but not re-verified. Its own top-5-by-arrivals market network is now superseded by Script 42's top-20 net-leadership analysis (see below) — Script 42 is the current reference for market lead-lag, not this file's `table_granger_market_network.csv`. |
| `30_Formal_Stress_Testing.py` | `table_stress_test_results.csv`, `fig_stress_test_{tomato,onion,potato}.png` | 2026-08-04 | 🟢 | Re-run on the grid-fixed panel and retrained models, most recently 2026-08-04 for the `s2_ndvi_anom` leakage fix. **Sign issue from the earlier (pre-2026-07-31) unconstrained model is fixed** via the monotonic constraint on export_banned/export_duty_pct/mep_usd_per_tonne (Script 15/23). Current state: median response collapses to ~0.0% for these scenarios (correct sign, but the constrained model can't recover a realistic magnitude either — see Script 31). |
| `31_Synthetic_DID_Policy_Effect.py` | Part A/B: `table_sdid_policy_effect.csv`, `table_sdid_unit_weights.csv`, `fig_sdid_treated_vs_synthetic.png`, `table_sdid_hub_policy_effect.csv`, `fig_sdid_hub_vs_nonhub.png`. Part C: `table_sdid_event_study.csv`, `fig_sdid_event_study.png`, `table_sdid_donor_robustness.csv`, `table_sdid_arrivals_effect.csv`, `fig_sdid_arrivals_treated_vs_synthetic.png` | 2026-08-01 | 🟢 | Estimates the 2023-24 onion export-restriction episode's price effect directly via Synthetic DID (Arkhangelsky et al. 2021), since Script 30's forecasting model can't recover a reliable magnitude (identification problem — duty/MEP/ban only ever moved together once). Part A (cross-crop, tomato/potato as donors): escalation-phase ATT +56.2% is credible (huge gap vs tomato placebo -30.7%), but the post-ban ATT (+14.5%) **fails its own placebo check** — indistinguishable from same-window tomato (+14.5%) and potato (+15.2%) placebo ATTs. Part B (within-onion, Nashik export-hub vs non-hub markets): real 2023 postban ATT +3.7% vs placebo-in-time (fake 2021 dates) ATT -3.9%. **Part C:** (1) event-study trajectories — decomposes each static ATT into a weekly path; (2) jackknife 90% CIs — the Part A postban placebo failure becomes a formal result: onion 90% CI [+12.2%, +16.8%] fully overlaps tomato-placebo's [+12.3%, +16.8%]; the escalation-phase CI [+52.3%, +60.1%] does NOT overlap tomato-placebo's [-31.8%, -29.6%]. Part B's hub-design CIs also separate: real-2023 postban [+0.9%, +6.6%] vs. placebo-in-time [-6.3%, -1.4%] do NOT overlap; (3) donor-pool robustness — escalation-phase ATT stable under leave-one-donor-out: [+46.1%, +60.8%] vs. full-sample +56.2%; (4) arrivals/quantity effect — onion arrivals in treated markets *fell* relative to synthetic control (escalation -25.8%, postban -14.3%), reported as an open finding, not resolved. **Overall honest conclusion: no single design here fully isolates the ban's causal magnitude.** Formally closed as a documented negative/non-identifiable result (2026-08-05). See Scripts 38/39 below for the two follow-up robustness checks this motivated. |
| `38_SDID_InSpace_Placebo_Test.py` | `table_sdid_inspace_placebo.csv`, `fig_sdid_inspace_placebo.png` | 2026-08-20 | 🟢 | Standard synthetic-control falsification test (Abadie, Diamond & Hainmueller 2010): every individual tomato/potato market treated as a placebo unit, one at a time, against the same donor pool. Real onion postban ATT judged against the resulting null distribution: p=0.131 — does not clear significance against this stronger test either, consistent with Part C's jackknife-CI finding above. **Patched 2026-08-20 for the market_id collision fix** (missed in the original nine-script pass) and re-run — result bit-for-bit identical to the pre-fix version (n=214 placebo markets, same mean/std/p-value). |
| `39_SDID_Stacked_MultiEpisode_EventStudy.py` | `table_sdid_stacked_event_study.csv`, `table_sdid_stacked_summary.csv` | 2026-08-20 | 🟢 | Extends the single-episode (2023-24) SDID design to all 3 verified onion export bans in the panel window (Sep 2019, Sep 2020, 2023-24), addressing the underlying n=1 identification problem by pooling 3 independent episodes into one event study. **Patched 2026-08-20 for the market_id collision fix** and re-run — result bit-for-bit identical to the pre-fix version. |
| `38b_SDID_Arrivals_InSpace_Placebo_Test.py` | `table_sdid_arrivals_inspace_placebo.csv`, `fig_sdid_arrivals_inspace_placebo.png` | 2026-08-21 | 🟢 | New. Arrivals-outcome analogue of Script 38, same in-space design applied to log1p(arrivals). Real ATT recomputation matched Script 31 Part C.4's postban row exactly (-0.1548 log-pts, -14.3%) before trusting the placebo result. 169 donor markets (fewer than 38's 214 — arrivals reporting is patchier than price). Placebo std 0.492 log-pts (~5x noisier than price's 0.096). Result: **fails decisively**, p=0.680 (115/169 placebo markets at least as extreme). |
| `39b_SDID_Arrivals_Stacked_MultiEpisode_EventStudy.py` | `table_sdid_arrivals_stacked_event_study.csv`, `table_sdid_arrivals_stacked_summary.csv`, `fig_sdid_arrivals_stacked_event_study.png` | 2026-08-21 | 🟢 | New. Arrivals-outcome analogue of Script 39, per-episode arrivals requalification (95% coverage, gaps <=4wk interpolated, mirroring Script 31 Part C.4). All 3 episodes produced usable fits (onion: 51/45/40 arrivals-qualifying markets for 2019/2020/2023). Onion's stacked mean sits outside the tomato/potato placebo band in 10/13 post-ban week-bins — but wrong-signed (arrivals fell, not rose, under an export ban meant to retain domestic supply) and already negative pre-ban. Read as corroborating the reverse-causality finding, not an independent effect — see 2026-08-21 note above. |
| *(none found)* | `table4_model_metrics.csv`, `fig_feature_importance.png`, `fig_actual_vs_pred_2024.png`, `test_predictions_2024.csv`, `lgbm_{tomato,onion,potato}.txt`, `market_list_by_crop.xlsx`, `fig_shap_{tomato,onion,potato}.png` (July 8 versions, not the July 29 SHAP figures) | 2026-07-08 | ⚪ | No script in the current `scripts/` folder produces these — orphaned from an early prototype, likely predating this repo's current script numbering. Safe to delete once confirmed unneeded; not referenced by README §3. |

## Exploratory, not part of the main numbered pipeline

### Two-phase residual architecture (Scripts 32-37) — REJECTED 2026-08-04
| Script | Outputs | Last generated | Notes |
|---|---|---|---|
| `32_LongHistory_Panel_Builder.py` | `data/agmarknet_weekly/longhistory/*.csv` | 2026-08-01 | One-off validation experiment: builds a 2003-2026 (23-year) version of the panel to test whether extending training history beyond the 2017 floor helps. Reused by Scripts 40/41 (escalation-signature prototype, see below) as a convenient long-history price source — that reuse is unrelated to the two-phase verdict below. |
| `33_LongHistory_Validation_Experiment.py` | `table_longhistory_validation.csv`, `table_longhistory_validation_raw.csv` | 2026-08-01 | Controlled comparison (same market set, same folds, same recipe) of a 9-year vs 23-year training window. Finding: real, modest improvement (9/12 crop x horizon cells, ~1.2pp average MAPE reduction) concentrated in onion (all 4 horizons) and tomato (except h=26w, where longer history measurably hurts — plausibly a 2003-2016 regime-break effect: pre-GST, pre-eNAM, pre-2016-demonetization). This motivated building out Scripts 34-37 to test the idea properly, rather than adopting it on this result alone — see the formal verdict below. |
| `34_Baseline_Phase_Panel_Join.py` | `data/baseline_phase_panel.csv` | 2026-08-04 | Full 2003-2026 joined panel for the two-phase Phase 1 model (all layers except Sentinel-2). |
| `35_TwoPhase_Baseline_Model.py` | `Model_Output/experiments/two_phase/table_baseline_phase_metrics.csv` (the underlying per-row OOF predictions file is gitignored — regenerate via script, too large for GitHub) | 2026-08-03 | Phase 1 (baseline) model, 9 annual expanding folds. Its raw MAPE beat M6 on several cells (notably onion) — see Script 37's formal test of whether that's real. **Not yet re-verified for the market_id collision bug** (groups by market name; see 2026-08-14 note above) — low priority since the architecture is already rejected regardless, but genuinely open. |
| `36_TwoPhase_Residual_Model.py` | `Model_Output/experiments/two_phase/table_twophase_combined_metrics.csv` (the underlying per-row predictions file is gitignored) | 2026-08-04 | **REJECTED.** Phase 2 (Sentinel-2-only residual on top of Phase 1) — combined forecast is worse than Phase 1 alone AND worse than M6 at nearly every crop/horizon cell, in both its full-feature and narrowed forms. Kept in the repo as a documented negative result, not part of the active pipeline. Same unverified collision-bug caveat as Script 35. |
| `37_DM_Test_M6_vs_Phase1Alone.py` | `table_dm_m6_vs_phase1alone.csv` | 2026-08-04 | Formal DM test of Phase-1-alone's apparent raw-MAPE edge over M6. **Result: does not survive formal testing (1/12 cells significant at p<0.05).** This is the decision record — M6 confirmed as the correct production choice, not merely "not yet adopted." |

### Early-warning prototype & market network (Scripts 40-42) — folded into the comprehensive review report as Sections 14-15, 2026-08-08/10
| Script | Outputs | Last generated | Notes |
|---|---|---|---|
| `40_Escalation_Signature_Head.py` | `table_escalation_signature_loeo_percrop.csv`, `table_escalation_signature_scores_percrop.csv`, `table_escalation_signature_placebo_test.csv`, `fig_escalation_signature_percrop_heldout.png`, `fig_escalation_signature_placebo_test.png` | 2026-08-20 | **v5, expanded to a genuine 8-episode design** (onion 2019/2020/2023, tomato 2023/2024/2025, potato 2014/2016 — see 2026-08-19/20 note above; supersedes the earlier 4-episode partial design this row previously described). Every episode now has genuine held-out validation, including tomato/potato which were previously in-sample only. Mean held-out AUC: tomato 0.801, onion 0.831, potato 0.849. 5 of 8 episodes reach placebo significance at p<0.05 (both potato, onion 2019/2023, tomato 2023); onion 2020, tomato 2024, and tomato 2025 do not. |
| `41_Escalation_Signature_Nashik_Hub_Test.py` | `table_escalation_signature_nashik_loeo.csv`, `table_escalation_signature_nashik_placebo.csv`, `fig_escalation_signature_nashik_vs_national.png` | 2026-08-08 | Tests whether onion 2020's weak national-level signal was real-but-geographically-diluted (its trigger was Nashik-region flooding specifically). Nashik-hub-only price: 2020's placebo p falls tenfold (0.660→0.064) — a partial, honest result, doesn't cross 0.05 but strongly supports the localization hypothesis. |
| `42_Market_Leader_Network.py` | `table_market_leader_network.csv`, `table_market_leader_ranking.csv`, `table_market_leader_confound_check.csv`, `table_market_leader_stability.csv`, `fig_market_leader_ranking.png`, `fig_market_leader_stability.png` | 2026-08-10 | Extends Script 29's top-5-market lead-lag test to the top-20 markets per crop with a net-leadership-score summary. Clear, geographically coherent leader clusters per crop (Karnataka for tomato, Nashik belt for onion, West Bengal for potato); consumption hubs (Azadpur, Shahdara) are consistent followers across both tomato AND onion independently. Confound check clean for all 3 crops (leadership isn't a reporting-coverage artefact: ρ=0.06-0.37, p>0.10 throughout). Stability check: tomato/potato leaders hold rank across an early/late time-period split; onion's single top-ranked market shuffles rank (Chandvad #2→#5.5 of 16), though the Nashik-belt cluster as a whole stays dominant in both periods — reported as the honest boundary of the finding, not smoothed over. |

## Known gaps not yet reflected anywhere above
- Full-capacity TFT run (Script 17) — deferred, see README §9. Also still carries the market-name collision bug (groups by `market`, not `market_id`) — a second, independent reason not to trust its numbers even setting the reduced-scope issue aside.
- Scripts 12, 13, 15b, 15c, 17, 26 — deferred re-runs, still genuinely open. (18b is **no longer** on this list — re-run 2026-08-15, see its table row above.) These predate not just the 2026-08-01 grid fix but every fix since (policy/PANEL_END, WPI vintage, macro long-history, `s2_ndvi_anom` leakage, the 2026-08-14 market_id collision fix, and the panel's growth from 834/809/82 to 840/814/82 markets).
- Scripts 34/35/36 (rejected two-phase architecture) — never verified against the market_id collision fix. Low priority given the architecture's already-rejected status, but genuinely open; see their table rows above.
- Daily-resolution forecasting: tried training genuinely new daily models (LightGBM M6) on 2026-07-29 — abandoned, daily naive persistence won even more decisively than weekly naive, and the daily coverage filter collapsed market counts 3-6x. Script 26 (2026-07-30) instead disaggregates the existing weekly model's forecasts into a smooth daily curve with an honest uncertainty band — not a validated daily forecast, a visualization aid. Wired into the dashboard as a "Daily price view" expander below the ticker.
- Arrivals-outcome SDID (Scripts 38b/39b, new 2026-08-21): both robustness checks now done — see the 2026-08-21 note and their table rows above. No longer an open item, listed here only so this section's history stays complete.
- Manuscript: previous manuscript-drafting pass cleared from `paper_drafts/` on 2026-08-21 to start a fresh, section-by-section pass — see README §9 and current chat/session history for status, since this working folder does not persist between passes.
- Two pre-existing mislabeled macro series (`agri_wages_rs_day`, `iip_food_proc`) kept as-is for continuity — see README §9, no commit has addressed this.
- **This session's work (2026-08-14 through 2026-08-21, commit `b8686fa`) was committed and pushed to `origin/master` on 2026-08-21** — prior sessions had left roughly a week of fixes uncommitted locally; confirm this stays current practice going forward rather than accumulating another backlog.
