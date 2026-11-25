# Protocol F: do 52-week macro deviation features improve forecast accuracy? (frozen 2026-11-25, before any estimate)

Planned scripts: `scripts/94_Deviation_Features_Screen.py` (Stage 1, built on Script 93's harness) and `scripts/95_Deviation_Features_Forward_Log.py` (Stage 2, to be written and committed before the first forward origin week used below). Each stage is run once against this text; every deviation is reported as a deviation.

## Why, and why this is not a confirmation
Protocol E (results in `../lever_responsiveness/PROTOCOL_E_lever_specification.md`) set out to make levers respond and failed that aim. It also showed, after the fact, that replacing the 18 macro levels by their deviation from the previous 52 weeks (L1) gave a higher mean skill than the current model M6 in 9 of 12 cells (for example onion at 26 weeks +11 points, potato at 13 weeks +11 points, tomato at all four horizons +1.4 to +3.5 points), but lower skill for potato at 4 weeks (-6.5 points) and was significant on the Diebold-Mariano test in one cell only. Those seeds, folds and results are exposed. This protocol therefore treats the E result as a **hypothesis**, not as evidence, and asks for two things the E result cannot give: a fresh-seed screen on the same folds (Stage 1), and a confirmation on weeks that did not exist when this protocol was frozen (Stage 2).

## Hypothesis and control
**H1**: macro deviation features (F1) give higher out-of-sample skill than the current macro levels (M6) in a crop x horizon cell.
- **Control M6**: the current production model, same code, panel, hyperparameters, folds and validation-window rule.
- **F1**: exactly L1 of Protocol E: each of the 18 macro levels is replaced by (level minus mean of the previous 52 weeks, at least 26 weeks required, otherwise 0), computed from information available at the origin week only. Nothing else changes. No window other than 52 weeks, no constraint, no other variant is a candidate.
- **F0 (interpretive comparator only, no decision role)**: M6 with the 18 macro levels removed. Its purpose is to say what a gain means: if F0 gains as much as F1, the gain comes from removing the levels, not from the deviation information, and the conclusion will be worded that way.

## Stage 1: screen on the retrospective folds
- Same five expanding folds (test years 2022 to 2026), same panel snapshot as in Protocol E, skill = (1 minus MASE) x 100 against the no-change forecast per the headline definition.
- **Ten new seeds: 101 to 110** (not the Protocol E seeds). Pairs (M6, F1) and (M6, F0) per seed.
- A cell (crop x horizon) **passes Stage 1** if all hold:
  1. mean skill gain of F1 over M6 across the ten seeds is at least +2.0 points;
  2. F1 beats M6 in at least 8 of the 10 seeds;
  3. on the seed-averaged predictions F1 is not worse than M6 in all five folds;
  4. F1 is not significantly worse than M6 on the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction (p < 0.05, crop-level weekly series, Newey-West lag h-1).
- All twelve cells are reported, passing or not. **H1 is "supported at screen level" if at least 6 of 12 cells pass.** If fewer than 6 pass, H1 is rejected and Stage 2 is not run.
- No cell is excluded in advance. Potato at 4 weeks, which lost points in Protocol E, is judged like every other cell.

## Stage 2: forward confirmation
- Freeze the F1 models for every cell that passed Stage 1: trained through origin week 2026-11-02 (the latest origin week in the panel at the freeze), with the Stage 1 hyperparameters, and store the frozen files.
- From origin week 2026-11-09 onward, log F1 and the production M6 forecast side by side every week (Script 95) for all passing cells. M6 forecasts come from the production forecast log; F1 forecasts from the frozen files with deviation inputs computed from the live panel.
- A cell is **scored when 13 origin weeks have realised outcomes** (all markets with an observed target price). Expected maturity if weekly refreshes are not missed: about early February 2027 at 1 week, early March at 4 weeks, early May at 13 weeks, early August 2027 at 26 weeks.
- A cell **confirms** if F1's mean absolute error over the 13 origin weeks is no higher than M6's and F1 is not significantly worse than M6 on the DM test at p < 0.10 (a non-inferiority guard; 13 weeks cannot show superiority). Missed refresh weeks are not filled; the 13 weeks must be consecutive origin weeks with outcomes.

## Adoption
A cell is switched to F1 in production only if it passes Stage 1 and confirms in Stage 2, and only after separate approval (retraining changes the live forecasts, the guide and the deck). Cells adopt independently. If a Stage 1 cell fails Stage 2 it stays on M6.

## Not allowed
Changing F1, F0, the 52-week window, the minimum of 26 weeks, the folds, seeds, thresholds or the number of required passing cells after seeing any result; adding candidates; using the Protocol E seeds or results as evidence for adoption; describing any gain as causal or as a lever response. Lever responses are reported with the Script 92 method for completeness and carry no weight in any decision.

## Limits
- Stage 1 reuses the five test years that Protocol E and earlier work have already examined, so it can screen but not confirm; the forward stage is the confirmation.
- The forward stage covers one season (November to February at 1 and 4 weeks; later months at 13 and 26 weeks) and 13 origin weeks, so it can detect a clear loss but not a small gain.
- The macro series are national and monthly. A gain from deviation features is an association with the time series, not a market mechanism.
- Potato is noisy (seed spread up to about 5 points at one week); the 10 seeds reduce this but do not remove it.
- One panel snapshot is used; a data refresh between Stage 1 and Stage 2 changes later origin weeks only.

## Outputs
`Model_Output/deviation_features/`: `table_screen_skill.csv`, `table_screen_decision.csv` (Stage 1); `forward_log.csv`, `table_forward_score.csv` (Stage 2); a results section appended to this file after each stage.
