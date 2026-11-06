# -*- coding: utf-8 -*-
"""
Script 90 -- pull daily GDELT DOC 2.0 news-volume series for Protocol D (Model_Output/spike_signals/PROTOCOL_D_news_signal.md).
Data acquisition only: no estimate, no test. Queries are the two fixed in the protocol. Requests are spaced (the API refuses faster than one per 5 s)
and retried with a longer pause on a refusal. Raw chunks are kept so the pull is auditable.
Output: data/news_gdelt/onion_news_daily.csv (date, query, value) and raw/ chunks.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data', 'news_gdelt')
RAW = os.path.join(OUT, 'raw_365d')   # year-long chunks; the earlier 120-day chunks stay in raw/ (not used)
QUERIES = {
    'A_supply_price': 'onion (price OR prices OR shortage OR supply) sourcecountry:india sourcelang:english',
    'B_policy': 'onion (export OR ban OR duty OR MEP OR "stock limit") sourcecountry:india sourcelang:english',
}
START, END, STEP = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-11-01'), pd.Timedelta(days=365)
GAP, RETRY_GAP, MAX_TRY = 7, 300, 12
START_DELAY = int(os.environ.get('GDELT_START_DELAY', '0'))   # seconds to wait before the first request (lets a rate-limit block clear)


def fetch(q, a, b):
    params = {'query': q, 'mode': 'timelinevol', 'format': 'json', 'startdatetime': a.strftime('%Y%m%d%H%M%S'), 'enddatetime': b.strftime('%Y%m%d%H%M%S')}
    url = 'https://api.gdeltproject.org/api/v2/doc/doc?' + urllib.parse.urlencode(params)
    for k in range(MAX_TRY):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                txt = r.read().decode('utf-8', 'replace')
            if txt.lstrip().startswith('{'):
                return json.loads(txt)
            print('  refused/odd reply (try %d): %s' % (k + 1, txt[:80].replace('\n', ' ')))
        except Exception as e:
            print('  error (try %d): %s' % (k + 1, e))
        time.sleep(RETRY_GAP)
    return None


def main():
    os.makedirs(RAW, exist_ok=True)
    if START_DELAY:
        print('waiting %d s before the first request' % START_DELAY, flush=True); time.sleep(START_DELAY)
    rows, missing = [], []
    for name, q in QUERIES.items():
        a = START
        while a < END:
            b = min(a + STEP, END)
            f = os.path.join(RAW, '%s_%s.json' % (name, a.strftime('%Y%m%d')))
            if os.path.exists(f):
                js = json.load(open(f, encoding='utf-8'))
            else:
                js = fetch(q, a, b)
                time.sleep(GAP)
                if js is None:                      # fallback: the same span in four short requests, kept only if all four succeed
                    print('  year request failed; trying four quarter-sized pieces', flush=True)
                    cuts = [a + (b - a) * i / 4 for i in range(5)]
                    parts = []
                    for lo, hi in zip(cuts[:-1], cuts[1:]):
                        p = fetch(q, lo.floor('D'), hi.floor('D'))
                        time.sleep(GAP)
                        if p is None:
                            parts = None; break
                        parts.append(p)
                    if parts:
                        data = [d for p in parts for s in p.get('timeline', []) for d in s['data']]
                        js = {'timeline': [{'series': 'Volume Intensity', 'data': data}], 'note': 'merged from 4 pieces'}
                if js is not None:
                    json.dump(js, open(f, 'w', encoding='utf-8'))
            if js is None:
                missing.append((name, a.date())); print('MISSING', name, a.date())
            else:
                for s in js.get('timeline', []):
                    for d in s['data']:
                        rows.append((pd.to_datetime(d['date'], format='%Y%m%dT%H%M%SZ'), name, d['value']))
                print(name, a.date(), 'ok', len(js.get('timeline', [{}])[0].get('data', [])) if js.get('timeline') else 0)
            a = b
    df = pd.DataFrame(rows, columns=['date', 'query', 'value']).drop_duplicates(['date', 'query']).sort_values(['query', 'date'])
    df.to_csv(os.path.join(OUT, 'onion_news_daily.csv'), index=False)
    print('rows', len(df), 'missing chunks', missing)
    print(df.groupby('query')['date'].agg(['min', 'max', 'count']))


if __name__ == '__main__':
    main()
