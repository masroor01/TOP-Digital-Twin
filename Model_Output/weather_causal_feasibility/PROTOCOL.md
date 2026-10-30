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
