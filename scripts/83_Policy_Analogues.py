"""
Script 83 -- Historical analogues for the dashboard's onion export-policy controls (2026-10-28).

The forecasting models do not respond to the export ban / minimum export price / export duty inputs (they never split on
them; Script 15 and 23 comments record why oversampling and forcing were tried and reverted: one escalating episode in which
the three moved together, so the effect is not identified, and Scripts 66-68 found the price effect not identified either).
So the dashboard does not invent a policy effect. Instead, when a policy control is switched on it shows what onion prices
actually did around the real, dated restrictive actions in the verified policy log -- descriptive only, no counterfactual.

For each restrictive action (ban, minimum export price, duty) the script takes every onion market with an observed (not
imputed) weekly price in the event week and in at least 80 % of the weeks from 13 before to 26 after, indexes each market's
price to its own event-week price (= 100) and reports the median and inter-quartile range ACROSS MARKETS by week. The
inter-quartile range is the spread between markets, not a confidence interval. Policy is set reactively to already-high
prices, so the run-up before the event is part of the picture and is reported separately.

Output : web/data/policy_analogues.json (served by the backend), Model_Output/table_policy_analogues.csv
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(BASE, 'data', 'agmarknet_weekly', 'onion_weekly_panel.csv')
EVENTS = os.path.join(BASE, 'data', 'policy_trade', 'export_policy_events.csv')
OUT_JSON = os.path.join(BASE, 'web', 'data', 'policy_analogues.json')
OUT_CSV = os.path.join(BASE, 'Model_Output', 'table_policy_analogues.csv')

PRE, POST, MIN_SHARE = 13, 26, 0.8
KINDS = {'export_ban': 'ban', 'minimum_export_price': 'mep', 'export_duty': 'duty'}

panel = pd.read_csv(PANEL, usecols=['market_id', 'state', 'market', 'week_start', 'modal_price_weighted', 'imputed'],
                    parse_dates=['week_start'])
panel = panel[(panel['imputed'] == 0) & panel['modal_price_weighted'].notna() & (panel['modal_price_weighted'] > 0)]
panel['mkey'] = panel['state'].astype(str) + '|' + panel['market'].astype(str)
wide = panel.pivot_table(index='week_start', columns='mkey', values='modal_price_weighted', aggfunc='mean').sort_index()

ev = pd.read_csv(EVENTS, parse_dates=['effective_date'])
ev = ev[(ev['crop'] == 'onion') & (ev['policy_direction'] == 'restrictive') & ev['policy_type'].isin(KINDS)]
# one row per distinct restrictive action; the 2017 retrospective MEP and the 2019 MEP that precedes the 2019 ban by 16 days
# are left out (the 2019 episode is represented by the ban itself, the 2017 record is a retrospective statement of a standing MEP)
ev = ev[ev['effective_date'] >= '2019-09-20'].sort_values('effective_date')

rows, episodes = [], []
for _, r in ev.iterrows():
    d0 = r['effective_date']
    wk0 = wide.index[wide.index.get_indexer([d0], method='pad')[0]]          # market week containing the event
    i0 = wide.index.get_loc(wk0)
    if i0 - PRE < 0 or i0 + POST >= len(wide):
        continue
    win = wide.iloc[i0 - PRE: i0 + POST + 1]
    ok = win.iloc[PRE].notna() & (win.notna().mean() >= MIN_SHARE)
    mk = win.columns[ok.values]
    idx = win[mk].ffill().bfill().div(win[mk].iloc[PRE]) * 100
    if len(mk) < 30:
        continue
    weeks = list(range(-PRE, POST + 1))
    med, q25, q75 = idx.median(axis=1).values, idx.quantile(.25, axis=1).values, idx.quantile(.75, axis=1).values
    ep = {
        'id': f"{KINDS[r['policy_type']]}_{d0.date()}",
        'kind': KINDS[r['policy_type']],
        'date': str(d0.date()),
        'label': r['description'],
        'verification': r['verification_status'],
        'nMarkets': int(len(mk)),
        'weeks': weeks,
        'median': [round(float(v), 1) for v in med],
        'q25': [round(float(v), 1) for v in q25],
        'q75': [round(float(v), 1) for v in q75],
        'runUpPct': round(float((100 / med[0] - 1) * 100), 1),     # median % rise from 13 weeks before the event to the event week
        'afterPct': {str(h): round(float(med[PRE + h] - 100), 1) for h in (4, 13, 26)},
    }
    episodes.append(ep)
    for w, m_, a_, b_ in zip(weeks, med, q25, q75):
        rows.append({'episode': ep['id'], 'kind': ep['kind'], 'event_date': ep['date'], 'week_rel': w, 'n_markets': ep['nMarkets'],
                     'median_index': round(float(m_), 2), 'q25_index': round(float(a_), 2), 'q75_index': round(float(b_), 2)})
    print(f"{ep['id']:<22} n={ep['nMarkets']:>3}  run-up {ep['runUpPct']:+6.1f}  after 4/13/26 wk: "
          f"{ep['afterPct']['4']:+6.1f} / {ep['afterPct']['13']:+6.1f} / {ep['afterPct']['26']:+6.1f}  ({ep['verification']})")

out = {
    'crop': 'onion',
    'note': ('Observed onion prices around the real dated restrictive export actions in the verified policy log. Each market is indexed '
             'to its own price in the event week (= 100); lines are the median across markets, bands the inter-quartile range across '
             'markets (spread between markets, not a confidence interval). Descriptive only: policy is set in response to already-high '
             'prices, the duty, minimum export price and ban were imposed in one escalating sequence in 2023, and the price effect of '
             'onion export policy is not identified in this project (SDID and placebo checks, Scripts 66-68). It is not a forecast of '
             'what the policy would do to the price now.'),
    'episodes': episodes,
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, lineterminator='\n')
print('wrote', OUT_JSON, 'and', OUT_CSV)
