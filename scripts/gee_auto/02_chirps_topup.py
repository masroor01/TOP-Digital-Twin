"""
GEE Automation -- CHIRPS pentad rainfall topup (Python port of
scripts/gee/gee_02_CHIRPS_2026.js), for the weekly CI refresh.

Same service-account + direct .getInfo() pattern as 01_era5_topup.py --
see that script's docstring for the full rationale (no Drive/export
needed, service accounts have no Drive storage quota).

Writes CSVs into $TOP_DOWNLOADS_DIR/GEE_2026/chirps/{zone_id}_CHIRPS_2026.csv
-- same folder/filename/schema as the manual JS script, zero changes
needed to Script 14.

Zone list copied from scripts/16_Zone_Assignment.py's ZONES dict (the
CURRENT source of truth) -- NOT from the stale gee_02_CHIRPS_2026.js,
whose P1/P2/P3 still show the pre-relocation Agra/Farrukhabad/Jalandhar
zones (superseded 2026-07-27). See 01_era5_topup.py for the same note.

NOTE: CHIRPS PENTAD has a real processing lag (often several weeks behind
real-time, worse than ERA5-Land) -- the actual end date is derived from
the data itself (last pentad with a non-null value), never assumed.

Deliberately NOT touched here: the known chirps_rain_mm CHIRPS-pentad-vs-
ISO-week misalignment bug (flagged by a peer session, ~38% of weeks
double-counting rainfall) lives in Script 14/09's downstream weekly-panel
join, not in this raw per-pentad pull -- this script intentionally
reproduces the EXISTING raw pentad format unchanged, so that bug (still a
separate open task) is unaffected either way by this port.

Env vars: same as 01_era5_topup.py (GEE_SERVICE_ACCOUNT_KEY, TOP_DOWNLOADS_DIR)
"""
import os
from datetime import date, timedelta
from pathlib import Path

import ee
import pandas as pd

SERVICE_ACCOUNT = 'gee-automation@top-digital-twin-data.iam.gserviceaccount.com'
GCP_PROJECT = 'top-digital-twin-data'
KEY_PATH = os.environ.get(
    'GEE_SERVICE_ACCOUNT_KEY',
    r'C:\Users\masro\AppData\Local\Temp\claude\C--Users-masro-Documents-Claude-Code\5347e58a-b41d-4333-8d13-80bc3c9a2503\scratchpad\gee-automation-key.json',
)
DOWNLOADS = Path(os.environ.get('TOP_DOWNLOADS_DIR', str(Path.home() / 'Downloads')))
OUT_DIR = DOWNLOADS / 'GEE_2026' / 'chirps'

BUFFER_M = 30000
SCALE = 5566  # CHIRPS native resolution 0.05deg ~= 5.5 km
START = '2026-01-01'
EXCESS_THRESH_MM_PENTAD = 50  # >50 mm/pentad = >10 mm/day = excess rain

ZONES = [
    {'id': 'T1_Kolar',          'crop': 'Tomato', 'market': 'Kolar APMC',                          'state': 'Karnataka',       'lon': 78.1320,  'lat': 13.1390},
    {'id': 'T2_Madanapalle',    'crop': 'Tomato', 'market': 'Madanapalle APMC',                    'state': 'Andhra Pradesh',  'lon': 78.5025,  'lat': 13.5504},
    {'id': 'T3_Nashik_Tomato',  'crop': 'Tomato', 'market': 'Nashik APMC',                         'state': 'Maharashtra',     'lon': 73.7898,  'lat': 19.9975},
    {'id': 'T4_Solan',          'crop': 'Tomato', 'market': 'Solan APMC',                          'state': 'Himachal Pradesh','lon': 77.1167,  'lat': 30.9045},
    {'id': 'T5_Navsari',        'crop': 'Tomato', 'market': 'Navsari APMC',                        'state': 'Gujarat',         'lon': 72.9031,  'lat': 20.9476},
    {'id': 'O1_Lasalgaon',      'crop': 'Onion',  'market': 'Lasalgaon APMC',                      'state': 'Maharashtra',     'lon': 74.0088,  'lat': 20.4061},
    {'id': 'O2_Pimpalgaon',     'crop': 'Onion',  'market': 'Pimpalgaon APMC',                     'state': 'Maharashtra',     'lon': 74.0167,  'lat': 20.0833},
    {'id': 'O3_Mahuva',         'crop': 'Onion',  'market': 'Mahuva APMC',                         'state': 'Gujarat',         'lon': 71.7744,  'lat': 21.0888},
    {'id': 'O6_Hubli',          'crop': 'Onion',  'market': 'Hubli APMC',                          'state': 'Karnataka',       'lon': 75.1239,  'lat': 15.3647},
    {'id': 'O7_Solapur',        'crop': 'Onion',  'market': 'Solapur APMC',                        'state': 'Maharashtra',     'lon': 75.9064,  'lat': 17.6854},
    {'id': 'O8_Manmad',         'crop': 'Onion',  'market': 'Manmad APMC',                         'state': 'Maharashtra',     'lon': 74.4367,  'lat': 20.2500},
    {'id': 'O9_Kurnool',        'crop': 'Onion',  'market': 'Kurnool APMC',                        'state': 'Andhra Pradesh',  'lon': 78.0373,  'lat': 15.8281},
    {'id': 'O10_Gondal',        'crop': 'Onion',  'market': 'Gondal APMC',                         'state': 'Gujarat',         'lon': 70.7980,  'lat': 21.9608},
    {'id': 'P1_Darjeeling',     'crop': 'Potato', 'market': 'Darjeeling APMC',                     'state': 'West Bengal',     'lon': 88.263176,'lat': 27.037755},
    {'id': 'P2_DiamondHarbour', 'crop': 'Potato', 'market': 'Diamond Harbour(South 24-pgs) APMC',  'state': 'West Bengal',     'lon': 88.189488,'lat': 22.192689},
    {'id': 'P3_Dehradun',       'crop': 'Potato', 'market': 'Dehradoon APMC',                      'state': 'Uttarakhand',     'lon': 78.043681,'lat': 30.325565},
    {'id': 'P4_Bardhaman',      'crop': 'Potato', 'market': 'Bardhaman APMC',                      'state': 'West Bengal',     'lon': 87.8550,  'lat': 23.2330},
]


def init_ee():
    creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(creds, project=GCP_PROJECT)


def fetch_zone(z, end_exclusive):
    geom = ee.Geometry.Point([z['lon'], z['lat']]).buffer(BUFFER_M)
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
              .filterDate(START, end_exclusive)
              .select('precipitation'))

    def to_feature(img):
        precip = img.select('precipitation')
        daily = precip.divide(5).rename('daily_mm')
        excess = precip.gt(EXCESS_THRESH_MM_PENTAD).rename('excess')
        stacked = daily.addBands(precip.rename('pentad_mm')).addBands(excess)

        mean_stats = stacked.select(['daily_mm', 'pentad_mm']).reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), None, True).combine(ee.Reducer.max(), None, True),
            geometry=geom, scale=SCALE, maxPixels=1e6,
        )
        frac_excess = stacked.select('excess').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE, maxPixels=1e6,
        )
        dt = img.date()
        return ee.Feature(None, {
            'date': dt.format('YYYY-MM-dd'),
            'year': dt.get('year'),
            'month': dt.get('month'),
            'doy_pentad_start': dt.getRelative('day', 'year').add(1),
            'rain_mean_mm': mean_stats.get('daily_mm_mean'),
            'rain_sum_mm': mean_stats.get('daily_mm_sum'),
            'rain_max_mm': mean_stats.get('daily_mm_max'),
            'frac_excess_rain': frac_excess.get('excess'),
        })

    fc = ee.FeatureCollection(chirps.map(to_feature))
    info = fc.getInfo()
    rows = []
    for f in info['features']:
        p = f['properties']
        rows.append({
            'zone_id': z['id'], 'crop': z['crop'], 'market': z['market'], 'state': z['state'],
            'date': p['date'], 'year': p['year'], 'month': p['month'],
            'doy_pentad_start': p['doy_pentad_start'],
            'rain_mean_mm': p.get('rain_mean_mm'), 'rain_sum_mm': p.get('rain_sum_mm'),
            'rain_max_mm': p.get('rain_max_mm'), 'frac_excess_rain': p.get('frac_excess_rain'),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('date').reset_index(drop=True)
    valid = df['rain_mean_mm'].notna()
    if not valid.all():
        last_valid_idx = valid[valid].index.max() if valid.any() else -1
        df = df.iloc[:last_valid_idx + 1]
    return df


def main():
    end_exclusive = (date.today() + timedelta(days=1)).isoformat()
    print(f'CHIRPS topup: requesting {START} to {end_exclusive} (exclusive), deriving real end from data.')
    init_ee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_dates = []
    for z in ZONES:
        df = fetch_zone(z, end_exclusive)
        out_path = OUT_DIR / f"{z['id']}_CHIRPS_2026.csv"
        df.to_csv(out_path, index=False)
        zmax = df['date'].max() if len(df) else None
        max_dates.append(zmax)
        print(f"  {z['id']:20s} {len(df):>4d} pentads, through {zmax}  -> {out_path.name}")

    if len(set(max_dates)) > 1:
        print(f'WARNING: zones disagree on latest available pentad: {sorted(set(max_dates))}')
    print(f'CHIRPS topup complete. Wrote {len(ZONES)} files to {OUT_DIR}')


if __name__ == '__main__':
    main()
