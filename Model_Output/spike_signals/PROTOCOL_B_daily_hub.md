# Protocol B: do daily prices at the onion hub give an early warning of a national spike? (frozen 2026-11-02, before any estimate)

Script: `scripts/89_Daily_Hub_Early_Detection.py`, run once against this text; every deviation is reported as a deviation.

## Why
Spike onset is not predictable from national weekly price, arrivals, weather, vegetation or diesel (Script 77), and momentum carries no signal. Two things have not been used: **the daily frequency** (the dashboard and the models are weekly) and **the geography** (the market network analysis found that the Nashik-belt markets lead and the consumption markets, such as Delhi, follow, but the forecasting model uses none of it). A descriptive look at weekly data showed the hub moving first in 6 of 11 episodes by a median of one week, which is too coarse to say anything. This test asks whether daily prices at the hub give a usable warning before the rest of the country moves.

## Data
Daily market records for onion: the Agmarknet history file (`onion_all_india_apmcs_2000_2025.csv`) for 2016-11-01 to 2025-11-06 and the portal's Daily Price Arrival Report (the file placed on Drive and used by the weekly refresh) from 2025-11-07 to 2026-10-28. One price per market-day: the arrival-weighted mean of the modal prices over varieties. Records outside 50 to 12,000 Rs per quintal or with no arrivals are dropped (the bounds of the weekly panel). Comparability of the two sources is checked on their overlap and reported.

## Groups (fixed in advance)
- **Hub**: markets whose district is Nashik (the Nashik belt, found to lead in Script 42). Daily hub price = median over the hub markets reporting that day, required to be at least 3 markets.
- **Follower**: all markets outside the Nashik district. Daily follower price = median over markets reporting, at least 30. A **Delhi** series (markets in Delhi) is reported separately.
- **National weekly reference** (what the weekly twin sees): the weekly national median of the weekly panel, observed weeks with at least 100 reporting markets, as in Script 77.

## Signal rule (one primary threshold, not tuned)
For a group, the 7-day trailing median of the daily group price is compared with the median of the daily group price over days t-35 to t-14. The **signal is on** when it is at least **15 %** higher. The same rule is applied to the weekly national series at weekly resolution (weekly median at least 15 % above the median of weeks t-5 to t-3; the signal date is the end of that week). A 25 % threshold is reported descriptively only and is never used to pass or rescue the test.

## Episodes
The unusual onion spike episodes of Script 77 (`table_onion_spike_episodes.csv`: onsets 2017-06-05 to 2026-05-11, 12 episodes), with the same definition (8-week forward return of at least +30 %, exceeding the seasonal norm by at least 20 points, calm start). The onset week t0 is taken from that table without change.

## Quantities
1. **Lead** per episode: the date of the first signal of the follower group in [t0-56 days, t0+28 days] minus the date of the first signal of the hub in the same window (positive: hub first). Lead against the weekly national reference is reported in the same way, since that is what the dashboard sees. Episodes in which a group has no signal in the window have no lead and count as "not earlier".
2. **Hub-only alert**: a day on which the hub signal is on, the follower signal is off and the weekly national price rose by less than 10 % over the previous 8 weeks (a calm start in the sense of the episode definition). Hub-only days less than 14 days apart form one alert, dated by its first day.
3. **Recall**: the share of episodes with at least one hub-only alert in the 56 days before t0. **Precision**: the share of alerts followed by an episode onset within 56 days of the alert date. **Lift and p value**: recall observed against recall when the 12 onset dates are shuffled among eligible Mondays (not within 8 weeks of another episode; 5,000 shuffles); p is the share of shuffles with recall at least as high as observed.

## Decision rule (fixed in advance)
The hub shows a usable early warning only if **all** hold: (i) the median lead of the hub over the follower group, over the episodes in which both fire, is at least **7 days**, and the hub fires first in at least **6** episodes; (ii) recall is at least **6 of 12**; (iii) precision is at least **0.30**; (iv) the permutation p value of the recall is below **0.05**. If all hold, a second protocol (B2) will test whether hub signals improve one- to four-week forecasts for follower markets. If not, we stop: daily hub data give no usable lead, and the finding is reported as a negative result.

## Limits stated up front
Twelve episodes, so every statistic is imprecise; the groups are defined a priori but the hub is chosen from earlier analysis of the same history; daily records are noisy (reporting gaps, few arrivals on some days) and the two data sources differ; a retrospective alert is not a forecast issued in real time; and a pass would support a warning for onion only. A negative result with this power does not prove that no signal exists.
