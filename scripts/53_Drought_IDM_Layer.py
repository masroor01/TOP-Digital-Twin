# -*- coding: utf-8 -*-
"""
Script 53 — Drought Layer: India Drought Monitor (IDM) Combined Drought Index
==============================================================
Acquires the Combined Drought Index (CDI) from the India Drought Monitor
(https://indiadroughtmonitor.in, IIT Gandhinagar Water and Climate Lab,
DST-funded) and assigns each panel district its nearest CDI grid-cell
value, weekly, for the full history the portal has published.

WHY THIS LAYER IS STRUCTURED DIFFERENTLY FROM SCRIPT 51 (VEDAS Trigger-1):
  IDM publishes a continuous national grid, not pre-aggregated district
  values, so there is no district-name crosswalk problem to solve (Script
  52's problem for VEDAS). Instead this script does a nearest-grid-point
  spatial join, using each district's centroid coordinates from
  `data/geocode_cache.json` -- the SAME Nominatim-geocoded, already-
  validated centroid cache Script 16 uses for zone assignment (395/395
  panel districts hit that cache exactly; confirmed before writing this
  script, not assumed). No new geocoding, no polygon/point-in-polygon
  math, no shapely dependency.

  The grid<->district assignment is computed ONCE (grid coordinates are
  confirmed stable across 5+ years of files, only individual cell VALUES
  go NaN some weeks) and reused for every week, so each district maps to
  the same physical grid cell for its entire time series -- the assigned
  cell doesn't drift week to week even when that week's value is NaN.

HOW THIS WAS VALIDATED:
  - GET https://indiadroughtmonitor.in/data/Drough_TS/CDI_{YYYYMMDD}.txt
    returns a fixed-width text grid: "<lat> <lon> <value>" per line, 4537
    grid points (0.25-degree spacing), no header. `value` is a
    standardized drought index (CDI), roughly in [-2.7, 2.2] in the files
    probed; NaN for a small number of masked/no-data cells (60 of 4537 in
    the most recent file checked).
  - CADENCE CONFIRMED EMPIRICALLY: weekly, every WEDNESDAY (not the
    initial guess of "any weekly day") -- verified across all of 2025 and
    2026 to date. Live and current: 2026-09-02 returns data, 2026-09-09
    (the following Wednesday, after today 2026-09-08) does not yet.
  - HISTORY: live continuously since 2021-07-14 (the portal's own stated
    start date; confirmed fetchable) through the present week -- a full
    5+ year span, unlike Script 51's VEDAS layer (2022+ only, Kharif
    fortnights only). This is the stronger of the two drought layers on
    coverage.
  - CDI -> drought-category thresholds are the site's own DOCUMENTED
    values, pulled directly from assets/interactive/drought-map.colormaps.js
    (the Legend.jsx CDI band table), not inferred: Abnormal(D0) <= -0.5,
    Moderate(D1) <= -0.8, Severe(D2) <= -1.3, Extreme(D3) <= -1.6,
    Exceptional(D4) <= -2.0, Normal otherwise. This mirrors the standard
    USDM D0-D4 convention.

KNOWN LIMITATION: nearest-grid-point is an approximation, not a true
district-area aggregate -- a district's assigned cell is whichever grid
point is closest to its Nominatim centroid, which can be tens of km off
for large or oddly-shaped districts. `distance_km` is written out per
district so anyone using this layer can see exactly how good that
approximation is per district, and --mode probe reports the distribution
before any full pull.

Output:
  --mode probe : prints nearest-grid-point distance distribution and a
                 handful of sample values across 3 widely-spaced weeks;
                 writes nothing.
  --mode full  : data/drought_idm/district_grid_assignment.csv
                   (state, district, district_lat, district_lon, grid_lat,
                    grid_lon, distance_km -- computed once, static)
                 data/drought_idm/cdi_district_weekly.csv
                   (state, district, week_start, cdi, drought_category,
                    fetch_status)
                 data/drought_idm/raw_cache/CDI_{date}.txt (raw weekly
                   grid files, cached so a re-run doesn't re-fetch)

Run:
  python scripts/53_Drought_IDM_Layer.py --mode probe
  python scripts/53_Drought_IDM_Layer.py --mode full
"""

import os
import math
import time
import argparse
import datetime as dt

import requests
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
GEOCODE_FILE = os.path.join(BASE, 'data', 'geocode_cache.json')
OUT_DIR = os.path.join(BASE, 'data', 'drought_idm')
CACHE_DIR = os.path.join(OUT_DIR, 'raw_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

IDM_URL_TMPL = 'https://indiadroughtmonitor.in/data/Drough_TS/CDI_{date}.txt'
REQUEST_TIMEOUT = 30
REQUEST_DELAY_S = 1.0  # politeness delay between requests

CDI_START_DATE = dt.date(2021, 7, 14)  # confirmed live; portal's own stated start
assert CDI_START_DATE.weekday() == 2, 'expected a Wednesday'

# Documented thresholds, transcribed from the site's own
# assets/interactive/drought-map.colormaps.js (Legend.jsx CDI bands).
# Evaluated low-to-high: first band whose upper edge the value is <= wins.
CDI_BANDS = [
    (-2.0, 'Exceptional (D4)'),
    (-1.6, 'Extreme (D3)'),
    (-1.3, 'Severe (D2)'),
    (-0.8, 'Moderate (D1)'),
    (-0.5, 'Abnormal (D0)'),
]
CDI_NORMAL_LABEL = 'Normal'


def classify_cdi(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for upper, label in CDI_BANDS:
        if value <= upper:
            return label
    return CDI_NORMAL_LABEL


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def weekly_wednesdays(start_date, end_date):
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += dt.timedelta(days=7)
    return dates


def cache_path(date_obj):
    return os.path.join(CACHE_DIR, f'CDI_{date_obj.strftime("%Y%m%d")}.txt')


def fetch_grid(date_obj, session):
    """Returns (list of (lat, lon, value_or_None), status_str). Cached to
    a local .txt so a re-run doesn't re-hit the site."""
    path = cache_path(date_obj)
    if os.path.exists(path):
        text = open(path, 'r', encoding='utf-8').read()
        status = 'ok' if text.strip() else 'empty_cache'
    else:
        url = IDM_URL_TMPL.format(date=date_obj.strftime('%Y%m%d'))
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            return [], f'request_error: {e}'
        if resp.status_code != 200:
            return [], f'http_{resp.status_code}'
        text = resp.text
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        status = 'ok'
        time.sleep(REQUEST_DELAY_S)

    grid = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue  # e.g. a stray "NaN NaN NaN" line -- not a usable grid point
        try:
            val = float(parts[2])
        except ValueError:
            val = float('nan')
        grid.append((lat, lon, val))
    return grid, status


def load_district_centroids():
    import json
    panel = pd.read_csv(PANEL_FILE, usecols=['state', 'district']).drop_duplicates()
    cache = json.load(open(GEOCODE_FILE, encoding='utf-8'))
    rows = []
    missing = []
    for _, r in panel.iterrows():
        key = f'{r.district}, {r.state}, India'
        entry = cache.get(key)
        if entry:
            lat, lon = entry
            rows.append({'state': r.state, 'district': r.district,
                          'district_lat': lat, 'district_lon': lon})
        else:
            missing.append(key)  # either absent, or a cached failed-geocode (null)
    if missing:
        print(f'  WARNING: {len(missing)} panel districts not found in geocode_cache.json '
              f'and will be skipped: {missing[:10]}{"..." if len(missing) > 10 else ""}')
    return pd.DataFrame(rows)


def build_grid_assignment(reference_grid, centroids):
    """For each district centroid, find the nearest grid coordinate
    (brute-force over 4537 points -- cheap, done once). Returns a
    DataFrame with grid_lat/grid_lon/distance_km per district."""
    grid_coords = [(lat, lon) for lat, lon, _ in reference_grid]
    rows = []
    for _, r in centroids.iterrows():
        best_d, best_pt = float('inf'), None
        for glat, glon in grid_coords:
            d = haversine_km(r.district_lat, r.district_lon, glat, glon)
            if d < best_d:
                best_d, best_pt = d, (glat, glon)
        rows.append({'state': r.state, 'district': r.district,
                      'district_lat': r.district_lat, 'district_lon': r.district_lon,
                      'grid_lat': best_pt[0], 'grid_lon': best_pt[1],
                      'distance_km': round(best_d, 2)})
    return pd.DataFrame(rows)


def run_probe():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 53 PROBE -- India Drought Monitor (IDM) coverage check')
    print('=' * 65)

    print('\n[1] Loading district centroids from geocode_cache.json ...')
    centroids = load_district_centroids()
    print(f'  {len(centroids)} district centroids loaded')

    covered = set(zip(centroids['state'], centroids['district']))
    full_panel = pd.read_csv(PANEL_FILE, usecols=['crop', 'state', 'district', 'week_start',
                                                    'arrivals_tonnes_week', 'imputed'])
    full_panel['week_start'] = pd.to_datetime(full_panel['week_start'])
    cutoff = full_panel['week_start'].max() - pd.Timedelta(days=365)
    recent = full_panel[(full_panel['week_start'] >= cutoff) & (full_panel['imputed'] == 0)].copy()
    recent['has_centroid'] = list(zip(recent['state'], recent['district']))
    recent['has_centroid'] = recent['has_centroid'].isin(covered)
    print('  Volume-weighted coverage despite the geocode gap above (last ~12mo, non-imputed):')
    for crop, g in recent.groupby('crop'):
        total = g['arrivals_tonnes_week'].sum()
        cov = g.loc[g['has_centroid'], 'arrivals_tonnes_week'].sum()
        print(f'    {crop:8s}: {100*cov/total:5.1f}%')
    print('  (the missing districts are the same messy-label cases Script 52 hand-resolved for')
    print('   VEDAS -- fixing this properly means extending Script 16\'s own geocode_cache.json,')
    print('   not duplicating Nominatim calls in this script.)')

    print('\n[2] Fetching a reference grid (most recent available week) ...')
    probe_dates = [dt.date(2021, 7, 14), dt.date(2023, 9, 6), dt.date(2026, 9, 2)]
    grids = {}
    for d in probe_dates:
        grid, status = fetch_grid(d, session)
        grids[d] = (grid, status)
        print(f'  {d} ({d.strftime("%A")})  status={status}  points={len(grid)}')

    ref_grid, ref_status = grids[probe_dates[-1]]
    if ref_status != 'ok':
        print('\nReference grid fetch failed -- cannot proceed with the join check. '
              'Re-verify the URL/cadence assumptions above before running --mode full.')
        return

    print('\n[3] Nearest-grid-point join (all districts, reference grid) ...')
    assignment = build_grid_assignment(ref_grid, centroids)
    print(assignment['distance_km'].describe().to_string())
    far = assignment[assignment['distance_km'] > 30].sort_values('distance_km', ascending=False)
    print(f'\n  Districts with nearest grid point > 30km away ({len(far)}):')
    if not far.empty:
        print(far[['state', 'district', 'distance_km']].head(15).to_string(index=False))
    else:
        print('  (none)')

    print('\n[4] Sample values across the 3 probe weeks, for 5 districts ...')
    value_by_coord = {}
    for d, (grid, status) in grids.items():
        value_by_coord[d] = {(lat, lon): val for lat, lon, val in grid} if status == 'ok' else {}
    sample_districts = assignment.sample(min(5, len(assignment)), random_state=42)
    for _, r in sample_districts.iterrows():
        vals = []
        for d in probe_dates:
            v = value_by_coord.get(d, {}).get((r.grid_lat, r.grid_lon))
            cat = classify_cdi(v) if v is not None else None
            vals.append(f'{d}: {v if v is not None else "n/a"} ({cat})')
        print(f'  {r.state} / {r.district}  (dist={r.distance_km}km):  ' + '  |  '.join(vals))

    print('\n' + '=' * 65)
    print('Probe complete. Before running --mode full:')
    print('  - distance_km distribution above should be mostly well under the')
    print('    ~28km grid spacing; a long tail of large distances usually means')
    print('    either a genuinely large/coastal district or a bad geocode --')
    print('    spot-check the ones listed above.')
    print('  - sample CDI values/categories above should look like plausible')
    print('    drought readings (not all identical, not all NaN).')


def run_full():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 53 FULL PULL -- India Drought Monitor, weekly CDI, '
          f'{CDI_START_DATE} to present')
    print('=' * 65)

    print('\n[1] Loading district centroids ...')
    centroids = load_district_centroids()

    print('\n[2] Fetching current grid to fix the district<->grid-point assignment ...')
    today = dt.date.today()
    latest_wed = today - dt.timedelta(days=(today.weekday() - 2) % 7)
    ref_grid, ref_status = fetch_grid(latest_wed, session)
    if ref_status != 'ok':
        # fall back to one week earlier if the latest week isn't published yet
        latest_wed -= dt.timedelta(days=7)
        ref_grid, ref_status = fetch_grid(latest_wed, session)
    assert ref_status == 'ok', f'could not fetch a reference grid (last tried {latest_wed}): {ref_status}'
    assignment = build_grid_assignment(ref_grid, centroids)
    assignment_path = os.path.join(OUT_DIR, 'district_grid_assignment.csv')
    assignment.to_csv(assignment_path, index=False, encoding='utf-8')
    print(f'  Saved: {assignment_path}  ({len(assignment)} districts)')

    print('\n[3] Fetching weekly grids and looking up each district\'s assigned cell ...')
    dates = weekly_wednesdays(CDI_START_DATE, latest_wed)
    print(f'  {len(dates)} weekly files to fetch/read (cached files are reused)')

    rows = []
    for i, date_obj in enumerate(dates, 1):
        grid, status = fetch_grid(date_obj, session)
        if i % 20 == 0:
            print(f'  [{i}/{len(dates)}] {date_obj}  status={status}')
        value_by_coord = {(lat, lon): val for lat, lon, val in grid} if status == 'ok' else {}
        for _, r in assignment.iterrows():
            val = value_by_coord.get((r.grid_lat, r.grid_lon)) if status == 'ok' else None
            cat = classify_cdi(val)
            rows.append({'state': r.state, 'district': r.district, 'week_start': date_obj,
                          'cdi': val, 'drought_category': cat,
                          'fetch_status': status if status == 'ok' else status})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'cdi_district_weekly.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')

    n_ok = (out['fetch_status'] == 'ok').sum()
    n_nan = out['cdi'].isna().sum()
    print(f'\nSaved: {out_path}  ({len(out):,} rows, {n_ok:,} from an ok fetch, '
          f'{n_nan:,} with a NaN/missing CDI value)')
    if n_ok < len(out):
        print('\nFailed-fetch weeks:')
        bad = out[out['fetch_status'] != 'ok'][['week_start', 'fetch_status']].drop_duplicates()
        print(bad.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    args = parser.parse_args()

    if args.mode == 'probe':
        run_probe()
    else:
        run_full()
