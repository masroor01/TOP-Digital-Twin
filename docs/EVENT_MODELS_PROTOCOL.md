# Event models (Arm D2): frozen protocol

Frozen 2026-10-07, before the full training run and before any scoring (Script 72). Defaults approved by Dr. Masroor: honesty rule (report "not shown" if the layers do not help), capacity-based alarms as primary, event-specific layer features, several hours of compute, price spikes and crashes only. One potato smoke test (h = 13) was run to check the code; its AUCs were printed but **no design choice was changed because of them**.

## 1. Why
Script 70 showed regression forecasts rank spike/crash risk well (AUC 0.80-0.90) but the six-layer model adds almost nothing over the price-only model for the decision; regression targets (price level) shrink toward the mean, and fixed thresholds break when the yearly event base rate moves (onion 15-54 %). Here the target is the event itself.

## 2. Events and models (Script 71)
- Event at origin week *t*, horizon *h*: spike = price(*t*+*h*)/price(*t*) − 1 >= +30 %; crash <= −25 %. Primary h = 13; secondary h = 4 and 26.
- Two LightGBM binary classifiers per crop x horizon x side x fold, differing **only** in features:
  - **M0E (price-only):** Script 15's M0 features + price-derived event features (momentum 4/13/26 wk, deviation from the 52-wk mean, 13-wk z-score, national-median momentum, share of markets already rising, market vs national level).
  - **LAY (layers):** M0E + Script 15's M6 features (arrivals, macro-logistics, climate, satellite, infrastructure, policy) + drought + ONI + layer-derived anomalies (arrivals shortfall vs 52-wk mean, 4-wk arrivals change, rainfall and heat anomalies vs 104-wk mean, 13-wk diesel change).
- Same rolling-origin folds as Script 15 (train and validation rows chosen by target date; test = following calendar year 2022-2026). Early stopping on validation log-loss; isotonic calibration on the validation rows; 5 seeds (42-46), probabilities averaged. Hyper-parameters fixed (num_leaves 63, min_child_samples 100, lr 0.05, 600 trees max). No threshold or parameter is chosen on a test year.

## 3. Scoring (Script 72)
- **Primary alarm rule (capacity):** each week and crop, alarm the top **20 %** of markets by calibrated probability. Sensitivity: 10 % and 30 %. Secondary: probability rule (alarm if calibrated p >= 0.2 = C/L). Robustness: persistence-2 (alarm only if the market was also in the top group the previous week).
- **Rivals under the same capacity rule:** momentum (13-wk past rise), seasonal naive, the Script 70 regression forecasts M0 and M6 (ranked by forecast return), plus never/always alarm.
- **Metrics:** relative economic value V at C/L = 0.2 (0.1, 0.3), POD, FAR, CSI, AUC, Brier score. Same market-week sample for all rules.
- **Inference:** 13-week block bootstrap over calendar weeks (2,000 resamples), per-fold results.

## 4. Evidence standard (fixed)
For each crop x side at the primary setting (h = 13, capacity 20 %, C/L = 0.2), **"layers help the decision" is passed** only if all hold: (1) V(LAY) − V(M0E) has a 90 % block-bootstrap interval above zero; (2) LAY beats M0E in at least 4 of the 5 test years; (3) V(LAY) exceeds the best of momentum and seasonal naive with an interval above zero. There are 6 crop x side tests (about 0.3 false passes expected at the 5 % one-sided level), so:
- a **general claim** ("the layers improve early-warning decisions") needs at least 3 of 6 passes;
- a **crop claim** needs both sides of that crop to pass;
- otherwise the result is reported as **not shown**, and the layers are described as helping long-horizon error and scenario inputs only.
Everything else (other horizons, capacity levels, probability rule, persistence rule) is sensitivity, not claim.

## 5. Limits stated up front
Few independent episodes; market-weeks within a week are correlated (hence block bootstrap); the cost-loss ratio is illustrative; calibration depends on the validation year's base rate (hence capacity rules as primary); results are out-of-sample scoring of decisions, not evidence that acting would have changed the market.

## 6. Results (run 2026-10-07; Scripts 71 and 72; tables `table_event_models_*.csv`)
**Pre-registered verdict: layers help the decision: not shown (2 of 6 crop x side tests pass; a general claim needs 3).**
Primary setting (h = 13, capacity 20 %, C/L 0.2), V(LAY) vs V(M0E): onion crash -0.005 vs -0.021 (diff +0.015, CI above 0, 5/5 folds), potato crash 0.051 vs 0.022 (+0.029, CI above 0, 5/5); onion spike -0.779 vs -0.766, potato spike 0.088 vs 0.093, tomato crash -0.415 vs -0.400, tomato spike -0.549 vs -0.527 (not passed; tomato LAY significantly worse). The two passes have V near zero (no better than "always alarm"), so they are not decision-useful at this capacity.
**What clearly improved: the event models with calibrated probabilities (exploratory secondary rule, alarm if calibrated p >= 0.2 = C/L).** V(LAY_prob), V(M0E_prob): onion spike 0.04 / -0.54 (Script 70's M6 regression rule: -0.54), onion crash 0.52 / 0.42, tomato spike 0.22 / 0.32, tomato crash 0.06 / 0.21, potato spike 0.48 / 0.45, potato crash 0.30 / 0.45. Against the regression forecasts under capacity alarms, the probability-rule event model is higher in all six cells (interval above zero in 5 of 6). **Layers vs price-only under this rule: no significant difference in any cell** (onion positive but wide intervals; tomato and potato crash significantly worse with layers). AUC: LAY 0.68-0.88, M0E 0.66-0.86; layers raise AUC for onion/potato crash (+0.04-0.07) and lower it for tomato.
Honest reading: robustness of detection improved mainly because the target is now the event and probabilities are calibrated (a price-only event model also gains); the data layers do not reliably improve early-warning decisions in this test. Onion under the probability rule is the one place worth a confirmatory test on new data (rule frozen in advance), not a claim.
