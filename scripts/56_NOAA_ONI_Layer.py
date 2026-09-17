# -*- coding: utf-8 -*-
"""
Script 56 — Climate Layer: NOAA Oceanic Nino Index (ONI)
==============================================================
Acquires the Oceanic Nino Index (ONI) from NOAA's Climate Prediction Center
(https://www.cpc.ncep.noaa.gov) -- a single, national/global-level ENSO
index, 3-month rolling seasons, 1950 to present.

WHY THIS LAYER IS SIMPLER THAN SCRIPTS 51/53 (VEDAS/IDM DROUGHT):
  ONI is a single global time series with no spatial/district dimension at
  all -- no crosswalk, no nearest-grid-point join, no geocode dependency.
  Every panel district/market gets the SAME ONI value for a given week (it's
  a national-level teleconnection index, not a local one). This script only
  acquires and parses it into a clean, join-ready CSV -- it does NOT join
  it onto the weekly panel (that would be a future script, analogous to
  Script 54's role for the drought layers, once/if this layer is actually
  wired into M7-extended features).

HOW THIS WAS VALIDATED (2026-09-14 session, re-confirmed live 2026-09-17):
  - GET https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt returns a
    plain-text, whitespace-separated table: "SEAS YR TOTAL ANOM" header,
    then one row per 3-month rolling season (DJF, JFM, FMA, ... NDJ) per
    year, 1950 to the current season. TOTAL = mean SST (deg C) in the
    Nino 3.4 region; ANOM = the ONI value itself (SST anomaly vs. the
    1991-2020 base period).
  - Official NOAA CPC domain (not the third-party ggweather.com the
    original review doc cited, which is not authoritative and was not
    used here).
  - No auth, no CAPTCHA, no rate limit encountered.
  - This is NOAA's modern standard ENSO index (supersedes the older SOI
    in most current ag-climate literature) -- satisfies the project's
    "SOI/ONI" data-layer ask from the original review docs.

Output:
  data/noaa_oni/oni_seasonal.csv
    columns: season (3-letter code, e.g. "DJF"), season_start_month (1-12,
    the first calendar month of the 3-month window -- DJF's "year" in the
    source file is the YEAR OF THE J/F MONTHS, so DJF 1998's season_start
    is December 1997; handled explicitly below, not assumed), season_end_month,
    year (the source file's own YR column, kept as-is for traceability),
    sst_total, oni_anom

Run:
  python scripts/56_NOAA_ONI_Layer.py --mode probe
  python scripts/56_NOAA_ONI_Layer.py --mode full
"""

import os
import argparse

import requests
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'data', 'noaa_oni')
os.makedirs(OUT_DIR, exist_ok=True)

ONI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
REQUEST_TIMEOUT = 30

# 3-month rolling season codes, in calendar order, with each season's
# (start_month, end_month). DJF spans Dec(prev year) -> Jan -> Feb.
SEASON_MONTHS = {
    'DJF': (12, 2), 'JFM': (1, 3), 'FMA': (2, 4), 'MAM': (3, 5),
    'AMJ': (4, 6), 'MJJ': (5, 7), 'JJA': (6, 8), 'JAS': (7, 9),
    'ASO': (8, 10), 'SON': (9, 11), 'OND': (10, 12), 'NDJ': (11, 1),
}


def fetch_raw(session):
    resp = session.get(ONI_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def parse_oni(text):
    lines = text.strip().splitlines()
    header = lines[0].split()
    assert header == ['SEAS', 'YR', 'TOTAL', 'ANOM'], (
        f'unexpected header, source format may have changed: {header}')
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        season, yr, total, anom = parts
        assert season in SEASON_MONTHS, f'unrecognized season code: {season}'
        start_month, end_month = SEASON_MONTHS[season]
        rows.append({
            'season': season,
            'year': int(yr),
            'season_start_month': start_month,
            'season_end_month': end_month,
            'sst_total': float(total),
            'oni_anom': float(anom),
        })
    df = pd.DataFrame(rows)
    # DJF's YR column is the year of the J/F months (per NOAA's own
    # convention, confirmed against known events -- e.g. "DJF 1998" in the
    # source shows the tail of the strong 1997-98 El Nino, ANOM=2.22,
    # consistent with Jan/Feb 1998 being the peak). So DJF's actual season
    # START is December of (year - 1) -- captured explicitly below rather
    # than left implicit, since getting this backwards would misalign any
    # future join to the weekly panel by a full year for DJF rows only.
    df['season_start_year'] = df.apply(
        lambda r: r['year'] - 1 if r['season'] == 'DJF' else r['year'], axis=1)
    return df


def run_probe():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 56 PROBE -- NOAA ONI coverage check')
    print('=' * 65)

    print(f'\n[1] Fetching {ONI_URL} ...')
    text = fetch_raw(session)
    print(f'  {len(text.splitlines())} lines returned')

    print('\n[2] Parsing ...')
    df = parse_oni(text)
    print(f'  {len(df)} seasonal rows parsed')
    print(f'  Range: {df["season"].iloc[0]} {df["year"].iloc[0]} -> '
          f'{df["season"].iloc[-1]} {df["year"].iloc[-1]}')

    print('\n[3] Sanity checks against known ENSO events ...')
    checks = [
        ('DJF', 1998, 2.0, 2.5, 'strong 1997-98 El Nino peak'),
        ('DJF', 2016, 2.0, 2.7, 'strong 2015-16 El Nino peak'),
        ('OND', 2010, -1.8, -1.2, 'strong 2010-11 La Nina'),
    ]
    all_ok = True
    for season, year, lo, hi, label in checks:
        row = df[(df['season'] == season) & (df['year'] == year)]
        if row.empty:
            print(f'  MISSING: {season} {year} ({label}) not in parsed data')
            all_ok = False
            continue
        val = row['oni_anom'].iloc[0]
        ok = lo <= val <= hi
        all_ok &= ok
        print(f'  {"OK  " if ok else "FAIL"} {season} {year} ({label}): '
              f'ANOM={val:+.2f}, expected [{lo:+.2f}, {hi:+.2f}]')

    print('\n[4] Recent 6 seasons:')
    print(df.tail(6)[['season', 'year', 'sst_total', 'oni_anom']].to_string(index=False))

    print('\n' + '=' * 65)
    if all_ok:
        print('Probe complete -- known-event sanity checks passed. Safe to run --mode full.')
    else:
        print('Probe complete -- one or more sanity checks FAILED. Do not trust this '
              'source/parse until investigated.')


def run_full():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 56 FULL PULL -- NOAA ONI, 1950 to present')
    print('=' * 65)

    print(f'\n[1] Fetching {ONI_URL} ...')
    text = fetch_raw(session)

    print('\n[2] Parsing ...')
    df = parse_oni(text)

    out_path = os.path.join(OUT_DIR, 'oni_seasonal.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f'\nSaved: {out_path}  ({len(df)} seasonal rows, '
          f'{df["season"].iloc[0]} {df["year"].iloc[0]} -> '
          f'{df["season"].iloc[-1]} {df["year"].iloc[-1]})')
    print('\nNOTE: not yet joined onto the weekly panel -- this is acquisition only. '
          'A future script would map each panel week_start to its 3-month season '
          '(using season_start_year/season_start_month above) and optionally add '
          'lagged features (oni_lag_3m, oni_lag_4m, per the original review docs).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    args = parser.parse_args()

    if args.mode == 'probe':
        run_probe()
    else:
        run_full()
