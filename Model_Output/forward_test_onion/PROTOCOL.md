# Onion forward confirmatory test: protocol (frozen 2026-10-08, before any outcome exists)

## Hypothesis
For **onion** at **h = 13 weeks**, the layered event model (LAY) has higher early-warning decision value than the price-only event model (M0E), for **both** price spikes (+30 %) and price crashes (−25 %). Background: Script 72 (retrospective, 2022-2026) found no reliable layer benefit overall; onion under the calibrated-probability rule was the one promising, exploratory, wide-interval signal. This test is the confirmation attempt on weeks whose outcomes were unknown when the models were frozen.

## What is frozen
- Models: Script 71's specification (LightGBM binary classifiers; M0E and LAY feature sets; spike and crash; horizons 4, 13, 26; five seeds 42-46; isotonic calibration on the validation rows). Trained **once** by `scripts/73_Forward_Test_Onion.py freeze` on all onion rows whose target was known at the freeze date (training: targets up to 26 weeks before the last price week; validation and calibration: the last 26 weeks of known targets). Files are SHA-256 hashed in `freeze_manifest.json` (tracked); the model files stay local. The script refuses to retrain.
- Alarm rule: raise an alarm if the calibrated probability is **≥ 0.20** (the cost-loss ratio). Cost of acting = 20 % of the loss avoided.
- Forward origins: every origin week **≥ 2026-07-13** whose outcome is not yet observable when scored (origin + h later than the latest price week). Probabilities for every onion market are appended to `forecast_log.csv` after each weekly refresh (append-only, timestamped, with the code commit and the latest data week; committed to git).

## The single pre-specified analysis
- Run once, when **26 origin weeks** have matured at h = 13 (first origin 2026-07-13, 26th origin 2027-01-04, evaluable from **2027-04-05**). Only the first 26 origin weeks are used. Outcome: realised price at origin + 13 weeks from the weekly panel at evaluation time; event = rise ≥ 30 % (spike) or fall ≥ 25 % (crash).
- Statistic: relative economic value V (cost-loss, C/L = 0.2) of LAY minus V of M0E, per side; **90 % circular block bootstrap over origin weeks (blocks of 4 weeks, 5,000 resamples)**.
- **Confirmed** only if the interval is above zero for **both** spike and crash. Otherwise **not confirmed**, and the claim that the data layers improve early-warning decisions is dropped (the layers' demonstrated value remains long-horizon error and scenario inputs).
- `status` prints counts only; no interim result is used for any claim. h = 4 and 26, capacity alarms and other C/L are descriptive.

## Limits
26 weeks of origins and ~800 clustered markets give modest power: failing to confirm is not proof of no effect, and a confirmation applies to onion only. The 2026 onion spike regime may not repeat. Prices can be revised in the panel after logging; outcomes use the panel at evaluation time. Any model improvement is a new version with its own new test, never a change to these models.
