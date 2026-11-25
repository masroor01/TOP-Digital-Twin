# Protocol E: can the model's response to macro, climate and infrastructure levers be enlarged without losing accuracy? (frozen 2026-11-23, before any estimate)

Planned script: `scripts/93_Lever_Specification_Test.py`, run once against this text; every deviation is reported as a deviation.

## Why
Script 92 (`table_lever_diagnostic.csv`, read-only) moved each lever from 10 % to 90 % of its observed range, everything else at baseline, in all 12 production models:
- Infrastructure levers move the forecast by about 2 to 3.5 % (cold-storage capacity 3.5 %, agricultural wages 2.5 to 3.3 %).
- Some macro levers move it by 1 to 3 % (WPI potato and onion, LPG, import and export values), but diesel (0.75 %), USD/INR (0.5 %), the repo rate (0.35 %) and rainfall and heat variables (below 0.5 %) move it very little.
- The five policy levers move it by exactly 0: the models never split on them (no reliable policy signal, by design).
- Outside the observed range every lever is flat (a tree model cannot extrapolate); the sliders already stop at the observed range.

So most small responses are what the model learned. National macro series change monthly (about a hundred distinct values over the history), so trees use them mostly as a stand-in for time. The request is to retrain so that more levers register. **Movement is not a goal in itself**: a model that is made to react to a lever it cannot predict with is worse and misleading. The goal here is accuracy; lever response is reported as a secondary outcome, and a specification is adopted only if it passes both.

## Specifications (fixed; no others will be run)
- **L0 (control)**: the current production model M6, same code, panel, hyperparameters and seeds.
- **L1 (deviation features)**: the 18 macro levels are replaced by their deviation from the mean of the previous 52 weeks, computed from information available at the origin week only. Climate and infrastructure features are unchanged. In the dashboard a lever sets a level; the app converts it to a deviation by subtracting the trailing mean stored in the baseline row.
- **L2 (L1 plus two monotone constraints)**: as L1, with price non-decreasing in the diesel and crude-oil deviations only, because the direction (higher fuel cost raises or leaves unchanged wholesale prices) is stated here in advance. No other constraint.

## Evaluation (identical to the headline evaluation)
Five expanding folds (test years 2022 to 2026), the same panel, crops, horizons (1, 4, 13, 26 weeks) and skill = (1 minus MASE) x 100 against the no-change forecast. Five seeds per specification; the cell value is the mean over seeds. The control gives the seed noise.

## Primary criteria (all must hold for adoption; an improvement is not required)
1. In each of the 12 crop x horizon cells, the candidate’s mean skill is not below L0’s by more than 2 points (tomato, onion) or by more than the spread of L0 across its seeds (potato).
2. In no cell is the candidate worse than L0 in all 5 folds (fold-level binomial test, p = 0.031).
3. In no cell is the candidate significantly worse than L0 on the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction (p < 0.05).

## Secondary criteria (reported; adoption needs at least 3 of 18 macro levers to qualify)
A macro lever qualifies as made more responsive when its mean in-range effect (Script 92 method, averaged over cells) is at least 1 % and at least 1.5 times L0’s, its sign is the same in at least 4 of 5 seeds, and (diesel, crude) the sign is non-negative.

## Decision
Adopt L1 or L2 only if all primary and the secondary criterion hold; if both qualify, take L1 (simpler). Otherwise keep M6 and report that the small responses are what the data support. **Adoption is a separate step that needs approval**, because retraining changes every live number, the guide and the deck.

## Not allowed
Changing the specifications, folds, seeds or thresholds, or adding candidates, after seeing a result; tuning hyperparameters to enlarge a response; describing any response as causal. Policy levers stay at zero effect and keep their historical-analogue display.

## Limits
The macro series are national and monthly, so any response is an association with the time series, not a market mechanism. Potato seed noise reaches 8 points. Levers remain flat outside the observed range. The test cannot show that a lever has no real-world effect.

## Outputs
`Model_Output/lever_responsiveness/`: `table_spec_skill.csv`, `table_spec_lever_response.csv`, `table_spec_decision.csv`, and a results section appended to this file.

---
## Results (run once on 2026-11-25 against the text above; Script 93, 5 seeds x 2 pairs; tables in this folder)

**Decision: neither L1 nor L2 is adopted. The current model M6 stays.** Both fail primary criterion 1 in one cell: potato at 4 weeks, where the candidate’s mean skill is 6.5 points (L1) and 6.2 points (L2) below M6 against a tolerance of 2.9 points, and the loss is negative in all 5 seeds. Eleven of 12 cells pass criterion 1 for each candidate; criterion 2 (never worse in all 5 folds) and criterion 3 (no significant Diebold-Mariano loss) pass in all 12 cells.

| | L1 | L2 |
|---|---|---|
| Cells passing criterion 1 / 2 / 3 | 11 / 12 / 12 | 11 / 12 / 12 |
| Macro levers qualifying (secondary) | 3 of 18 | 3 of 18 |
| Adopt | no | no |

**Lever response (secondary).** The hypothesis that deviation features make more levers register is not supported for the levers that were the reason for the test. Diesel (4 city) moves the forecast 1.7 % under M6, 1.2 % under L1 and 0.6 % under L2; USD/INR 1.8 %, 2.5 % and 2.6 %; crude oil 3.5 %, 3.9 % and 1.9 %. Under L1 the diesel and crude responses have the opposite sign to the one stated in advance (lower forecasts for higher fuel cost) in all 5 seeds; the L2 constraint corrects the sign but shrinks the size. The three qualifying levers (WPI vegetables total, LPG Delhi, reverse repo; diesel Delhi and the same three under L2) are not the fuel or exchange-rate levers, and the reverse-repo gain (0.29 % to 1.6 %) is five-fold, which is more likely a feature artefact than a mechanism.

**Accuracy (unplanned observation, not part of the decision).** Both candidates had higher mean skill than M6 in 9 of 12 cells and higher in all 5 seeds in 5 (L1) and 6 (L2) cells, for example onion at 26 weeks (+11 and +14 points) and potato at 13 weeks (+11 and +12 points). Only one improvement is significant on the DM test (onion 1 week, p = 0.005 and 0.0002). These gains were not what the protocol set out to test and were seen after the fact; they are a hypothesis for a new protocol with its own frozen criteria, not a result to act on.

**Deviations from the protocol (reported, none changes the decision).**
- The 52-week trailing mean needs at least 26 weeks of history; the first 26 weeks of 2017 therefore get a deviation of 0.
- Seeds 42 to 46.
- The potato tolerance used the larger of 2 points and the spread of M6 across seeds; the protocol said the spread only. For potato at 13 and 26 weeks the spread is under 2 points, so the protocol’s tolerance would be stricter, but both cells improve by more than 2 points, so no cell changes verdict.
- The first launch crashed after run 1 (a stdout bug in my wrapper, not the data); runs were then executed one per process. Saved results are unaffected.

**What this means.** The small lever responses are what the data support. The retraining that was meant to make the levers register does not do so for fuel and exchange rate, and one cell (potato 4 weeks) gets clearly worse. Scenario outputs stay labelled model sensitivities, and the dashboard keeps its "no model response" notice.
