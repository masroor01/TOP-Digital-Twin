# Protocol D: does news volume about onion give an early warning of a national spike? (frozen 2026-11-02, before any news data is pulled)

## Why
Tracks A and B showed that onset is not predictable from prices, weather, vegetation, diesel or hub prices. Policy announcements, shortage reports and crop-damage stories appear in news before or while prices move. This asks whether news volume adds a lead. It is the last candidate in the spike plan; Track C found no usable supply series.

## Step 0: access feasibility (before any test)
Source: GDELT DOC 2.0 API (free, no login), mode `timelinevol` (share of monitored coverage matching a query, daily). Known limits to check, not assume: a request rate of one per five seconds (a first attempt on 2026-11-02 was refused for rate), and a possible history limit of about three months. Step 0 passes only if a request for a 2019 window returns a daily series. If the API cannot return 2017-2026, the fallback is the GDELT Global Knowledge Graph files or BigQuery; BigQuery needs the user's approval for cost and account. If neither is workable, Track D closes as "not feasible", with no test.

## Series (fixed in advance)
- Query A (supply and price): `onion` AND (`price` OR `prices` OR `shortage` OR `supply`), source country India, English.
- Query B (policy): `onion` AND (`export` OR `ban` OR `duty` OR `MEP` OR `stock limit`), source country India, English.
- Daily share of coverage, 7-day trailing mean. Episodes and calm-start rules as in Protocol B (the 12 episodes in `table_onion_spike_episodes.csv`); no queries are added or tuned after seeing results.

## Signal rule
Signal on when the 7-day mean of a query exceeds the median of days t-90 to t-30 by at least 2 median absolute deviations. A 3-deviation threshold is descriptive only.

## Quantities and decision rule
As Protocol B: lead against the weekly national price signal, recall of news-only alerts (signal on while the weekly national price rose less than 10 percent over the previous 8 weeks) in the 56 days before onset, precision against onset within 56 days, and a permutation p value (5,000 shuffles of the 12 onset dates among eligible Mondays). News shows a usable warning only if all hold: median lead at least 7 days with the news first in at least 6 episodes; recall at least 6 of 12; precision at least 0.30; permutation p below 0.05. If it passes, a second protocol tests forecast improvement. If not, stop.

## Limits stated up front
Twelve episodes. News volume reacts to price rises as well as preceding them, so a lead is hard to read causally even if found. GDELT coverage of Indian regional press is partial and changed over 2017-2026. Query terms were chosen by me before any data, from general knowledge of how onion crises are reported. A negative result does not prove no news signal exists.

## Results (run once, 2026-11-06, after the protocol was committed; Scripts 90 and 91; tables in this folder)
**Not passed; the pre-specified decision is to stop.** News volume about onion gives no usable early warning of a national onion spike.

| Criterion (primary rule, k = 2) | Supply and price query | Policy query | Required |
|---|---|---|---|
| (i) median lead over the weekly price signal; news first | 6 days; first in 4 of 5 episodes where both fire | 6 days; first in 5 of 6 | at least 7 days; first in at least 6 episodes |
| (ii) recall of news-only alerts before onset | 5 of 12 | 5 of 12 | at least 6 of 12 |
| (iii) precision (alerts followed by an onset within 56 days) | 0.23 (26 alerts) | 0.25 (24 alerts) | at least 0.30 |
| (iv) permutation p value of the recall | 0.41 | 0.20 | below 0.05 |

- Both queries fail every criterion. The 3-deviation threshold (descriptive only) also fails.
- **The policy series is degenerate under the frozen rule.** Half of its days have zero volume (51 %), so the median absolute deviation is zero on 56 % of days and the rule fires on 58 % of days. Its alerts therefore carry little information, and its lead and recall should not be read as evidence of a signal. The supply and price series fires on 20 % of days (zero deviation on 0.3 %).
- Leads of about 6 days against the weekly price signal are partly an artefact of dating the weekly signal at the end of its week, as in Protocol B.

### Deviations and data notes (all reported, none changes the decision)
1. The query text was changed only in syntax: the API refuses parentheses around a single term, so `(onion)` became `onion`. The logic is unchanged.
2. The API rate-limited heavily. The pull used 365-day requests (after 120-day requests proved too slow); the earlier 120-day chunks are kept in `raw/` and not used.
3. Both series have the same 41 missing days (GDELT gaps: 2 in 2017, 1 in 2020, 1 in 2023, 17 in 2025, 20 in 2026). They stay missing; rolling windows use the days that exist.
4. Two queries were tested, so the p values are not adjusted; with a Bonferroni level of 0.025 the conclusion is the same.
5. Eligible dates for the permutation start at 2017-04-03 (90 days of history are needed).

**What this does and does not show.** News volume from these two queries, in Indian English-language sources as indexed by GDELT, does not warn of onset beyond chance. It does not show that no news signal exists: GDELT coverage of regional-language and local reports is partial, the two queries were fixed in advance, and there are 12 episodes. This closes the four-track spike investigation: no data source tried gives a usable warning of onset.
