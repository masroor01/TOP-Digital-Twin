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
