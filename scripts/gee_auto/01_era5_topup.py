"""
GEE Automation -- ERA5-Land daily temperature topup (Python port of
scripts/gee/gee_01_ERA5_topup_2026.js), for the weekly CI refresh.

Authenticates as a dedicated Earth Engine service account (no OAuth/Drive
export needed -- fetches each zone's FeatureCollection directly via
.getInfo(), since service accounts have no Drive storage quota and these
per-zone tables are small enough that a synchronous fetch is simpler and
faster than an async Export.table.toDrive + polling + rclone pull).

Writes CSVs into $TOP_DOWNLOADS_DIR/GEE_2026/era5/{zone_id}_ERA5topup_2026.csv
-- same folder, filename, and column schema as the manual JS script, so
Script 14 (14_Satellite_Climate_Features.py) needs zero changes.

Always re-pulls the full 2026-01-01 -> latest-available window (not an
incremental append) -- mirrors the JS script's own convention exactly, and
avoids any append/dedup logic. ERA5-Land has a real processing lag, so the
actual end date is DERIVED from the data itself (first date with a null
Tmax_C is dropped), never assumed -- same discipline as this project's
other date-handling code (see feedback_working_style.md).

Env vars:
  GEE_SERVICE_ACCOUNT_KEY  -- path to the service account JSON key
  TOP_DOWNLOADS_DIR        -- output root (defaults to ~/Downloads, matching
                              Script 14's own default)
"""
import os
import sys
import json
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
OUT_DIR = DOWNLOADS / 'GEE_2026' / 'era5'

BUFFER_M = 30000  # 30 km production zone radius, matches the original JS scripts
SCALE = 11132     # ERA5-Land native resolution ~0.1deg ~= 11 km
START = '2026-01-01'

# Authoritative zone list -- copied from scripts/16_Zone_Assignment.py's
# ZONES dict (the CURRENT source of truth), NOT from the stale
# scripts/gee/gee_01_ERA5_topup_2026.js, whose P1/P2/P3 still show the
# pre-relocation Agra/Farrukhabad/Jalandhar zones (superseded 2026-07-27).
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
    era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .filterDate(START, end_exclusive)
            .select(['temperature_2m_max', 'temperature_2m_min', 'temperature_2m']))

    def to_feature(img):
        tmax = img.select('temperature_2m_max').subtract(273.15).rename('Tmax_C')
        tmin = img.select('temperature_2m_min').subtract(273.15).rename('Tmin_C')
        tmean = img.select('temperature_2m').subtract(273.15).rename('Tmean_C')
        f30 = tmax.gt(30).rename('flag_above30')
        f35 = tmax.gt(35).rename('flag_above35')
        f38 = tmax.gt(38).rename('flag_above38')
        stacked = tmax.addBands([tmin, tmean, f30, f35, f38])
        stats = stacked.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE, maxPixels=1e6)
        dt = img.date()
        return ee.Feature(None, {
            'date': dt.format('YYYY-MM-dd'),
            'year': dt.get('year'),
            'month': dt.get('month'),
            'doy': dt.getRelative('day', 'year').add(1),
            'Tmax_C': stats.get('Tmax_C'),
            'Tmin_C': stats.get('Tmin_C'),
            'Tmean_C': stats.get('Tmean_C'),
            'flag_above30': stats.get('flag_above30'),
            'flag_above35': stats.get('flag_above35'),
            'flag_above38': stats.get('flag_above38'),
        })

    fc = ee.FeatureCollection(era5.map(to_feature))
    info = fc.getInfo()
    rows = []
    for f in info['features']:
        p = f['properties']
        rows.append({
            'zone_id': z['id'], 'crop': z['crop'], 'market': z['market'], 'state': z['state'],
            'date': p['date'], 'year': p['year'], 'month': p['month'], 'doy': p['doy'],
            'Tmax_C': p.get('Tmax_C'), 'Tmin_C': p.get('Tmin_C'), 'Tmean_C': p.get('Tmean_C'),
            'flag_above30': p.get('flag_above30'), 'flag_above35': p.get('flag_above35'),
            'flag_above38': p.get('flag_above38'),
        })
    df = pd.DataFrame(rows)
    # ERA5-Land has real processing lag -- derive the actual last usable
    # date from the data itself (drop the trailing run of nulls), never
    # assume "today" is available.
    df = df.sort_values('date').reset_index(drop=True)
    valid = df['Tmax_C'].notna()
    if not valid.all():
        last_valid_idx = valid[valid].index.max() if valid.any() else -1
        df = df.iloc[:last_valid_idx + 1]
    return df


def main():
    end_exclusive = (date.today() + timedelta(days=1)).isoformat()
    print(f'ERA5 topup: requesting {START} to {end_exclusive} (exclusive), deriving real end from data.')
    init_ee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_dates = []
    for z in ZONES:
        df = fetch_zone(z, end_exclusive)
        out_path = OUT_DIR / f"{z['id']}_ERA5topup_2026.csv"
        df.to_csv(out_path, index=False)
        zmax = df['date'].max() if len(df) else None
        max_dates.append(zmax)
        print(f"  {z['id']:20s} {len(df):>4d} rows, through {zmax}  -> {out_path.name}")

    if len(set(max_dates)) > 1:
        print(f'WARNING: zones disagree on latest available date: {sorted(set(max_dates))}')
    print(f'ERA5 topup complete. Wrote {len(ZONES)} files to {OUT_DIR}')


if __name__ == '__main__':
    main()
