# Protocol: can a supply shock be given a causal price effect, using weather as the instrument? Feasibility test

Frozen 2026-10-30, before any estimate was computed. Script: `scripts/87_Weather_Causal_Feasibility.py` (run once against this text; any deviation is reported as a deviation).

## Why this test
The twin's scenarios are sensitivities of a forecasting model, not causal estimates (Section 3). A causal claim needs, for each lever, a design in which the variation in the lever is unrelated to the prices it is meant to explain. Export policy has no such design in this study (the price effect is not identified; Scripts 66-68). Weather is different: a weekly weather anomaly, measured against the normal for that week of the year at that place, is plausibly unrelated to prices except through supply. This test asks only whether the data can support that route. It is a feasibility test, not the final estimate.

## Target quantity
For each crop c, the effect on the national wholesale price of a change in the quantity arriving in the markets: the coefficient beta in
log(price_t) = alpha + beta * log(arrivals_t) + controls_t + e_t,
with log(arrivals_t) instrumented by weather anomalies in the production zones. beta < 0 is expected (more supply, lower price). The scenario it would serve: "what if arrivals were 20 % lower".

## Data and sample
- Weeks: 2017-01-02 to 2026-10-26 (513 weeks), the weekly panels in `data/agmarknet_weekly/{crop}_weekly_panel.csv`.
- Markets: a **fixed set** per crop, the markets with an observed (not imputed) price in at least 80 % of the 513 weeks, so that the entry of new markets in 2024-25 does not move the series.
- Price: log of the median across the fixed set of the weekly price (observed weeks only). Arrivals: log of the sum of weekly arrivals (tonnes) over the fixed set (observed weeks; weeks with fewer than 80 % of the set reporting are dropped).
- Weather: ERA5-Land maximum temperature and CHIRPS rainfall per production zone (`data/satellite_climate/zone_weekly_features.csv`, rainfall now on one scale). An anomaly is the standardised deviation of a week's value from the mean and standard deviation of the same ISO week over 2000-2016 (years before the panel), averaged over the zones of the crop.

## Instruments (fixed in advance, six per crop)
Rainfall anomaly and maximum-temperature anomaly, each averaged over three lag blocks before the arrival week t: weeks t-8 to t-1, t-17 to t-9, and t-26 to t-18 (supply responds to weather with a lag of the crop cycle). No other instrument is tried.

## Controls
Three harmonic pairs of the week of the year, a linear time trend, and an indicator for the lockdown weeks 2020-03-23 to 2020-06-30. For onion, a robustness run adds the weekly export-ban, duty and minimum-export-price indicators.

## Estimation
All by direct calculation with numpy (no IV package is installed). Standard errors are heteroskedasticity- and autocorrelation-consistent (Newey-West, Bartlett kernel, 13 lags).
1. **First stage**: log arrivals on the six instruments and the controls; report the HAC Wald F for the six instruments.
2. **Placebo first stage**: the same with the lead blocks (weeks t+1 to t+8, t+9 to t+17, t+18 to t+26): future weather must not predict current arrivals. Report the joint HAC p value.
3. **Two-stage least squares** for beta, with the HAC standard error, and the **Anderson-Rubin 90 % confidence set** for beta over a grid from -6 to +2 in steps of 0.01 (valid whatever the strength of the instruments).
4. **Over-identification**: two-step GMM Hansen J with the HAC weight (5 degrees of freedom).
5. **Reduced form**: log price on the six instruments and the controls; joint HAC p value.
6. **Zone-level check (descriptive)**: for every production zone with at least 5 markets of the fixed set within 150 km, the same first stage on that zone's own weather anomalies and the arrivals of those markets; report how many zones have F >= 10 and the sign of the 2SLS estimate.

## Decision rule (fixed in advance)
A crop **passes** the feasibility test only if all four hold:
(i) first-stage HAC F >= 10;
(ii) placebo (lead) joint p >= 0.10;
(iii) Hansen J p >= 0.05;
(iv) the Anderson-Rubin 90 % confidence set for beta lies entirely below zero (more supply, lower price).
If at least one crop passes, a causal supply-and-weather module is worth building for that crop, with the assumptions stated. If none passes, we stop: scenarios stay labelled as model sensitivities and the paper reports the negative result. The zone-level check and the onion robustness run inform the write-up and never rescue a failed crop. A pass would support a causal claim only about the supply channel, not about policy, macro-economic variables or any other lever.

## Limits stated up front
- The exclusion restriction (weather affects the price only through the quantity arriving) cannot be tested, only probed by the over-identification test and the placebo. Storage (onion keeps for months), trade and imports, weather effects on quality and demand, and weather outside the sampled zones all work against it.
- The production zones are key regions, not all of supply, so the first stage may be weak; a weak first stage is itself the answer to this test.
- Aggregate weekly series are autocorrelated and 513 weeks is a short sample; the Anderson-Rubin set is reported because it stays valid when the instruments are weak, and will be wide if they are.
- A single run; no specification is tuned after seeing a result.

## Addendum, 2026-10-30 (before any estimate on the real data)
Simulated data only were used for the changes below; no real-data estimate had been computed when they were made (the real-data run is a separate command, `run`, executed after this addendum is committed).
1. **The HAC-based tests were found to be badly oversized for this design.** In simulations with moving-average block instruments and autocorrelated errors, a nominal 10 % Newey-West (13 lags) joint test rejected a true null in 23 to 37 % of samples (0.5 % nominal: 3 to 13 %). Confidence sets built on it would be too narrow and the "set below zero" rule too easy to pass.
2. **Replacement.** Inference that decides the rule is now by **circular-shift permutation** (999 shifts of at least 52 weeks of the instrument rows relative to the outcome series, which keeps the autocorrelation of both): the Anderson-Rubin set, the first-stage test, the placebo test and the reduced-form test. The Anderson-Rubin set is the set of beta with permutation p >= 0.08; that level was chosen by simulation so that the nominal "90 %" set covers the true beta in 88 to 92 % of simulated samples (level 0.10 gave 86 to 88 %, level 0.06 gave 93 to 96 %). The permutation first-stage test has size 7 % at the 5 % level in the same simulations.
3. **Decision rule as amended** (replaces the four conditions above): a crop passes only if (i) the first-stage HAC F is at least 10 **and** the permutation p value of the first stage is below 0.05; (ii) the permutation p value of the placebo (lead) test is at least 0.10; (iii) the Anderson-Rubin set is non-empty (this replaces the Hansen test, whose HAC version is oversized; an empty set is the over-identification failure of the Anderson-Rubin test); (iv) the Anderson-Rubin set lies entirely below zero and does not touch the lower end of the grid (-6).
4. The HAC standard error of the 2SLS estimate, the HAC Hansen J and the HAC p values are still reported, as descriptive columns only, with the oversizing caveat. The finite-sample scaling T/(T-k) is applied to every HAC covariance.
5. Self-test: `scripts/87_Weather_Causal_Feasibility.py selftest` (100 simulated samples at each autocorrelation level 0, 0.6 and 0.85; 2SLS biased towards OLS when the noise is very persistent, which is why the decision does not use it).

## Results (run once, 2026-10-30, after the addendum was committed; Script 87; tables in this folder)
**No crop passes; the pre-specified decision is to stop.** Weather anomalies in the production zones, lagged 1 to 26 weeks, do not predict the quantity arriving at the markets, so there is no first stage to build a causal supply effect on.

| Crop | Fixed markets | Weeks | First-stage F (HAC) | First-stage permutation p | Placebo (lead) p | Reduced-form permutation p | Anderson-Rubin set for beta | Passes |
|---|---|---|---|---|---|---|---|---|
| Tomato | 400 | 495 | 0.4 | 0.90 | 0.69 | 0.99 | whole grid (-6 to +2) | no |
| Onion | 191 | 488 | 0.6 | 0.78 | 0.27 | 0.17 | whole grid | no |
| Potato | 79 | 488 | 1.1 | 0.64 | 0.73 | 0.91 | whole grid | no |

- The first stage fails by a wide margin (F about 0.4 to 1.1 against the threshold of 10), and the Anderson-Rubin set is the whole grid: the data are consistent with any price response to supply, so they say nothing about it.
- Zone by zone (descriptive): first-stage F 0.4 to 5.4 across 8 zones; the best is onion at Pimpalgaon (F 5.4, permutation p 0.05) and tomato at Kolar (F 4.5, p 0.12); neither reaches F = 10. Tomato at Nashik was skipped (singular design).
- The HAC test gave a reduced-form p of 0.002 for onion; the permutation p is 0.17. The difference is the oversizing found in the simulations, and it shows why the HAC p values were not used for the decision.
- The onion run with the export-policy indicators added as controls gives the same picture (F 0.6).
- Placebo (lead) tests pass for all crops (future weather does not predict current arrivals), as they should when there is no relation.

**What this does and does not show.** Within this design (six pre-specified lagged weather blocks in 17 key production zones, aggregate weekly arrivals at a fixed set of wholesale markets) the weather in these zones does not move measured arrivals. It does not show that supply does not affect price, nor that another design (district-level production and area, satellite yield proxies, region-specific supply chains) would fail. Possible reasons, none tested here: the zones cover only part of national supply; wholesale-market arrivals are a noisy measure of supply (private trade, storage, imports, reporting); and the effect of weather on yield may be non-linear. No specification was changed after seeing the result.

**Consequence.** Scenarios stay labelled as sensitivities of the fitted model, not causal estimates. The negative result is reported in the paper (Sections 3.3 and 6.7) as the test of the only lever for which an identification route existed.
