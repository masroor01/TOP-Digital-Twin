# -*- coding: utf-8 -*-
"""
Script 51 — Drought Layer: VEDAS/SAC Trigger-1 Result (district-level)
==============================================================
Acquires "Trigger-1 Result" (param_id 50039) from ISRO/SAC's VEDAS Krishi-DSS
drought-monitoring portal (https://vedas.sac.gov.in), the same national
implementation behind the Maharashtra.xlsx export the user downloaded from
the portal's own UI. Trigger-1 is the official Drought Manual-2016/2020
"Stage-1" indicator (rainfall-deficiency + dry-spell rule, IMD-sourced,
2-week rolling window) that gates whether a district enters the formal
drought-assessment process at all — a clean, government-computed Yes/No
signal, district-by-district.

SCOPE DECISION (validated, not to be silently expanded):
  Only Trigger-1 (param 50039) is acquired here. "Drought Categories"
  (param 10025) was investigated and found to require replicating a
  multi-level cascade of at least 4 further virtual params (Drought
  Intensity Category -> Agriculture/Remote-Sensing/Soil-Moisture severity,
  one of which -- NDVI VCI, param 10017 -- is itself virtual and was not
  traced to completion). That is a materially bigger reverse-engineering
  task than this one clean endpoint and was deliberately deferred -- see
  session notes. Do not fold Drought-Categories logic into this script;
  give it its own script if it's ever taken up.

HOW THIS WAS VALIDATED:
  Endpoint and args format confirmed by direct curl against the live site,
  then the actual date grid and response shape were empirically probed
  (curl sweeps across dates/years) rather than assumed -- both turned out
  to differ from the initial working guess:
    GET https://vedas.sac.gov.in/geoentity-services/api/geoentity-sources/
        19/otf-params/50039/values?prefix={STATE_CODE}D&args=rf;{DATE};2W;{DATE};IMD
    - geoentity-source 19 = district-level aggregation.
    - args = "<rf-arg>;<rf-date>;<dryspell-arg>;<dryspell-date>;<source>".
      rf-arg is always "rf" (RF Deviation, params.js default rf_parameter),
      dryspell-arg is always "2W" (params.js default dryspell selection,
      param 50050), source is "IMD" for that default dryspell param.
      DATE is the SAME calendar date in both slots (format YYYYMMDD), per
      the traced generate_trigger_1_result() logic.
    - No authentication required.
    - Response shape is a dict keyed by an internal district ID, NOT by
      district name directly:
        {"C91S12D182": {"name": "ANAND",
                         "param_values": {"50039": {"<unix_ts>": "Yes"}},
                         "parent_name": "GUJARAT"}, ...}
      The district name is payload[id]["name"]; the value is the single
      entry in payload[id]["param_values"]["50039"].
    - VALID DATES ARE FORTNIGHTLY ONLY (the 1st and 16th of each month),
      and ONLY within mid-June to mid-October (the Kharif/SDRF drought-
      assessment window: 16-Jun, 01-Jul, 16-Jul, 01-Aug, 16-Aug, 01-Sep,
      16-Sep, 01-Oct, 16-Oct -- 9 fortnights/season). Any other date
      (including the 22nd, the last day of a month, or any date outside
      that window) returns an empty `{}`, not an error.
    - DATA STARTS IN THE 2022 SEASON. Every date probed in 2017-2021
      (matching the panel's own start) returned empty; 2022-06-16 was the
      first date with data, and every fortnight from 2022 through the
      live 2026 season (checked through 2026-09-01) returns data. This is
      a REAL coverage gap against the panel's 2017-2025 window, not a
      script bug -- see the probe report and Section "Known limitation"
      below.

Districts vs. panel markets: this script fetches at DISTRICT granularity
and does NOT join to panel markets/zones -- output is a standalone,
district-indexed drought-trigger table. Joining onto the market panel
(nearest-district-per-market, same pattern as Script 16's zone assignment)
is a separate, later step once this table's coverage is confirmed useful.

KNOWN LIMITATION (accept before running --mode full): Trigger-1 only covers
the 2022-2026 Kharif seasons, roughly the last 40% of the panel's 2017-2025
history. It cannot backfill 2017-2021. Whether that's still useful (e.g. as
a recent-seasons feature, or for the crisis-backtesting/event-study scripts
whose episodes fall inside 2022+) is a judgment call for whoever reviews the
probe report -- this script does not make that call for you.

Output (per mode):
  --mode probe : prints a coverage report only, writes nothing.
  --mode full  : data/drought_vedas/trigger1_district_weekly.csv
                 columns: state, state_code, district_raw, district_norm,
                          date, trigger1, fetch_status
                 data/drought_vedas/raw_cache/{state_code}_{date}.json
                   (raw API responses, cached so a re-run doesn't re-fetch)

Run:
  python scripts/51_Drought_Trigger1_VEDAS_Layer.py --mode probe
  python scripts/51_Drought_Trigger1_VEDAS_Layer.py --mode full --start-year 2022 --end-year 2026
"""

import os
import json
import time
import argparse
import datetime as dt

import requests
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR = os.path.join(BASE, 'data', 'drought_vedas')
CACHE_DIR = os.path.join(OUT_DIR, 'raw_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

API_URL = ('https://vedas.sac.gov.in/geoentity-services/api/geoentity-sources/'
           '19/otf-params/50039/values')
PARAM_ID = '50039'
RF_ARG = 'rf'
DRYSPELL_ARG = '2W'
SOURCE = 'IMD'
REQUEST_TIMEOUT = 20
REQUEST_DELAY_S = 1.0  # politeness delay between requests to a government server

# Panel state name -> VEDAS state code, from js/custom/states.js (fetched and
# cached 2026-09-08). Only states that actually appear in the panel for
# onion/potato/tomato are mapped; anything else is intentionally omitted
# rather than guessed.
STATE_CODE_MAP = {
    'Andhra Pradesh':     'C91S40',
    'Assam':              'C91S28',
    'Bihar':               'C91S2',
    'Chandigarh':          'C91S3',
    'Chattisgarh':         'C91S4',   # panel spelling; VEDAS lbl = "Chhattisgarh"
    'Goa':                'C91S11',
    'Gujarat':            'C91S12',
    'Haryana':            'C91S13',
    'Himachal Pradesh':   'C91S26',
    'Jammu and Kashmir':  'C91S14',   # panel spelling; VEDAS lbl = "Jammu & Kashmir"
    'Karnataka':          'C91S16',
    'Kerala':             'C91S17',
    'Keralam':            'C91S17',   # panel spelling variant
    'Madhya Pradesh':     'C91S19',
    'Maharashtra':        'C91S20',
    'Manipur':            'C91S29',
    'NCT of Delhi':        'C91S6',   # VEDAS lbl = "Delhi"
    'Odisha':             'C91S35',
    'Punjab':             'C91S21',
    'Rajasthan':          'C91S22',
    'Tamil Nadu':         'C91S39',
    'Telangana':          'C91S23',
    'Tripura':            'C91S34',
    'Uttar Pradesh':      'C91S25',
    'Uttarakhand':        'C91S24',
    'West Bengal':        'C91S36',
}

# Valid query dates, confirmed empirically (curl sweep, 2026-09-08): only
# the 1st and 16th of each month, only mid-June through mid-October (the
# SDRF/Drought-Manual Kharif assessment window), only 2022 season onward.
# NOT a weekly grid -- do not "densify" this without re-probing the site.
FORTNIGHT_MONTH_DAYS = [(6, 16), (7, 1), (7, 16), (8, 1), (8, 16),
                         (9, 1), (9, 16), (10, 1), (10, 16)]
EARLIEST_SEASON_YEAR = 2022  # confirmed: every 2017-2021 date probed was empty


def kharif_fortnight_dates(start_year, end_year):
    """The 9 valid Trigger-1 query dates per season, across the given
    inclusive year range, clipped to the confirmed-available range."""
    start_year = max(start_year, EARLIEST_SEASON_YEAR)
    dates = []
    for year in range(start_year, end_year + 1):
        for month, day in FORTNIGHT_MONTH_DAYS:
            dates.append(dt.date(year, month, day))
    return dates


def fetch_trigger1(state_code, date_obj, session):
    """Single API call for one state, one date. Returns (dict|None, status_str)."""
    date_str = date_obj.strftime('%Y%m%d')
    args = f'{RF_ARG};{date_str};{DRYSPELL_ARG};{date_str};{SOURCE}'
    params = {'prefix': f'{state_code}D', 'params': PARAM_ID, 'args': args}
    try:
        resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return None, f'request_error: {e}'
    if resp.status_code != 200:
        return None, f'http_{resp.status_code}'
    try:
        payload = resp.json()
    except ValueError:
        return None, 'non_json_response'
    if payload == {}:
        return None, 'empty_response'  # normal off-cycle result, not an error
    if not payload or not isinstance(payload, dict):
        return None, 'unexpected_response_shape'
    return payload, 'ok'


def parse_trigger1_payload(payload):
    """{internal_id: {name, param_values:{'50039':{ts: 'Yes'/'No'}}, parent_name}}
    -> [(district_name, value), ...]. Skips any entry missing the expected
    shape rather than raising, and reports how many were skipped."""
    out, skipped = [], 0
    for entry in payload.values():
        try:
            name = entry['name']
            ts_map = entry['param_values'][PARAM_ID]
            value = next(iter(ts_map.values()))
        except (KeyError, StopIteration, TypeError):
            skipped += 1
            continue
        out.append((name, value))
    return out, skipped


def cache_path(state_code, date_obj):
    return os.path.join(CACHE_DIR, f'{state_code}_{date_obj.strftime("%Y%m%d")}.json')


def fetch_cached(state_code, date_obj, session):
    """Fetch through a per-(state,date) JSON cache so re-runs don't re-hit the API."""
    path = cache_path(state_code, date_obj)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        return cached['payload'], cached['status']
    payload, status = fetch_trigger1(state_code, date_obj, session)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'payload': payload, 'status': status}, f, ensure_ascii=False)
    time.sleep(REQUEST_DELAY_S)
    return payload, status


def normalize_district(name):
    return str(name).strip().upper()


def run_probe():
    """Small, cheap coverage check before ever committing to a full pull:
    - does the endpoint return data at all for a spread of states/years?
    - do returned district names look like they'll match the panel's
      district field, or will a name-normalization/crosswalk step be needed?
    Writes nothing; prints a report."""
    panel = pd.read_csv(PANEL_FILE, usecols=['state', 'district']).drop_duplicates()
    probe_states = ['Maharashtra', 'West Bengal', 'Karnataka', 'Gujarat', 'Rajasthan']
    probe_years = [2022, 2024, 2026]
    session = requests.Session()

    print('=' * 65)
    print('SCRIPT 51 PROBE -- Trigger-1 (VEDAS/SAC) coverage check')
    print('=' * 65)
    for state in probe_states:
        code = STATE_CODE_MAP[state]
        panel_districts = set(normalize_district(d) for d in
                               panel.loc[panel['state'] == state, 'district'].unique())
        print(f'\n[{state}] code={code}  panel districts={len(panel_districts)}')
        for year in probe_years:
            date_obj = dt.date(year, 8, 1)  # always a valid fortnight date
            payload, status = fetch_cached(code, date_obj, session)
            if status != 'ok':
                print(f'  {date_obj}  status={status}')
                continue
            parsed, skipped = parse_trigger1_payload(payload)
            returned = set(normalize_district(n) for n, _ in parsed)
            overlap = returned & panel_districts
            sample = dict(parsed[:3])
            print(f'  {date_obj}  status=ok  districts_returned={len(returned)}  '
                  f'panel_overlap={len(overlap)}/{len(panel_districts)}  '
                  f'unparsed_entries={skipped}  sample={sample}')
            if overlap < returned:
                unmatched = sorted(returned - panel_districts)[:5]
                print(f'    unmatched VEDAS names (sample): {unmatched}')
    print('\n' + '=' * 65)
    print('Probe complete. Inspect panel_overlap above before running --mode full:')
    print('  - overlap near panel-district-count -> district names match cleanly,')
    print('    proceed to full pull with confidence.')
    print('  - overlap low/zero but districts_returned > 0 -> VEDAS district')
    print('    names differ from panel district names (spelling/granularity);')
    print('    a name crosswalk will be needed before any downstream join.')
    print('  - status != ok across the board -> endpoint/args assumption needs')
    print('    re-checking before spending any full-pull budget.')


def run_full(start_year, end_year):
    session = requests.Session()
    dates = kharif_fortnight_dates(start_year, end_year)
    print('=' * 65)
    print(f'SCRIPT 51 FULL PULL -- Trigger-1, {max(start_year, EARLIEST_SEASON_YEAR)}-'
          f'{end_year} Kharif fortnights, {len(STATE_CODE_MAP)} states x '
          f'{len(dates)} fortnight-dates')
    print('=' * 65)

    rows = []
    n_calls = len(STATE_CODE_MAP) * len(dates)
    i = 0
    for state, code in STATE_CODE_MAP.items():
        for date_obj in dates:
            i += 1
            payload, status = fetch_cached(code, date_obj, session)
            if i % 50 == 0:
                print(f'  [{i}/{n_calls}] {state} {date_obj} status={status}')
            if status != 'ok':
                rows.append({'state': state, 'state_code': code, 'district_raw': None,
                              'district_norm': None, 'date': date_obj,
                              'trigger1': None, 'fetch_status': status})
                continue
            parsed, skipped = parse_trigger1_payload(payload)
            if skipped:
                print(f'    WARNING: {state} {date_obj} -- {skipped} entries had an '
                      f'unexpected shape and were skipped')
            for district_raw, value in parsed:
                rows.append({'state': state, 'state_code': code,
                              'district_raw': district_raw,
                              'district_norm': normalize_district(district_raw),
                              'date': date_obj, 'trigger1': value,
                              'fetch_status': 'ok'})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'trigger1_district_weekly.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')

    n_ok = (out['fetch_status'] == 'ok').sum()
    print(f'\nSaved: {out_path}  ({len(out):,} rows, {n_ok:,} ok, '
          f'{len(out) - n_ok:,} failed fetches)')
    if n_ok < len(out):
        failed_by_state = (out[out['fetch_status'] != 'ok']
                            .groupby(['state', 'fetch_status']).size())
        print('\nFailed-fetch breakdown:')
        print(failed_by_state.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    parser.add_argument('--start-year', type=int, default=2022)
    parser.add_argument('--end-year', type=int, default=2026)
    args = parser.parse_args()

    if args.mode == 'probe':
        run_probe()
    else:
        run_full(args.start_year, args.end_year)
