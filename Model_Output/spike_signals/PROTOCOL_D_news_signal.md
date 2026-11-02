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
