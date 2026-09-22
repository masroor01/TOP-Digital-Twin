"""
GEE Automation -- MODIS 16-day NDVI/EVI (Terra+Aqua merged) topup (Python
port of scripts/gee/gee_04_MODIS_NDVI_2026.js), for the weekly CI refresh.

Same service-account + direct .getInfo() pattern as 01/02 -- see
01_era5_topup.py's docstring for the full rationale. Structurally like
ERA5/CHIRPS (one reduceRegion per already-discrete collection image), not
like S2's expensive custom-window compositing, so one getInfo() per zone
is expected to be cheap enough (retry/backoff included defensively anyway).

Writes CSVs into $TOP_DOWNLOADS_DIR/GEE_2026/modis/{zone_id}_MODIS_NDVI_2026.csv
-- same folder/filename/schema as the manual JS script.

Zone list copied from scripts/16_Zone_Assignment.py's ZONES dict (current
source of truth) -- see 01_era5_topup.py's note on the stale P1/P2/P3 in
the original JS scripts.

Env vars: same as 01_era5_topup.py (GEE_SERVICE_ACCOUNT_KEY, TOP_DOWNLOADS_DIR)
"""
import os
import time
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
OUT_DIR = DOWNLOADS / 'GEE_2026' / 'modis'

BUFFER_M = 30000
SCALE = 250  # MOD13Q1 / MYD13Q1 native resolution 250 m
START = '2026-01-01'

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


def get_info_with_retry(ee_object, max_retries=5, base_delay=15):
    for attempt in range(max_retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f'    (getInfo failed: {e} -- retrying in {delay}s, attempt {attempt + 1}/{max_retries})')
            time.sleep(delay)


def mask_quality(img):
    qa = img.select('SummaryQA')
    good = qa.lte(1)  # 0=Good, 1=Marginal
    return img.updateMask(good)


def scale_indices(img):
    return (img
            .addBands(img.select('NDVI').multiply(0.0001).rename('NDVI_scaled'))
            .addBands(img.select('EVI').multiply(0.0001).rename('EVI_scaled'))
            .copyProperties(img, ['system:time_start']))


def fetch_zone(z, modis_ndvi):
    geom = ee.Geometry.Point([z['lon'], z['lat']]).buffer(BUFFER_M)

    def to_feature(img):
        stats = img.select(['NDVI_scaled', 'EVI_scaled']).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE, maxPixels=1e9,
        )
        n_valid = img.select('NDVI_scaled').mask().reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=SCALE, maxPixels=1e9,
        )
        dt = ee.Date(img.get('system:time_start'))
        return ee.Feature(None, {
            'date': dt.format('YYYY-MM-dd'), 'year': dt.get('year'), 'month': dt.get('month'),
            'doy': dt.getRelative('day', 'year').add(1),
            'NDVI': stats.get('NDVI_scaled'), 'EVI': stats.get('EVI_scaled'),
            'n_valid_px': n_valid.get('NDVI_scaled'),
        })

    fc = ee.FeatureCollection(modis_ndvi.map(to_feature)).filter(ee.Filter.notNull(['NDVI']))
    info = get_info_with_retry(fc)
    rows = []
    for f in info['features']:
        p = f['properties']
        rows.append({
            'zone_id': z['id'], 'crop': z['crop'], 'market': z['market'], 'state': z['state'],
            'date': p['date'], 'year': p['year'], 'month': p['month'], 'doy': p['doy'],
            'NDVI': p.get('NDVI'), 'EVI': p.get('EVI'), 'n_valid_px': p.get('n_valid_px'),
        })
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def main():
    end_exclusive = (date.today() + timedelta(days=1)).isoformat()
    print(f'MODIS NDVI topup: requesting {START} to {end_exclusive} (exclusive).')
    init_ee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terra = (ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(START, end_exclusive)
             .select(['NDVI', 'EVI', 'SummaryQA']).map(mask_quality).map(scale_indices))
    aqua = (ee.ImageCollection('MODIS/061/MYD13Q1').filterDate(START, end_exclusive)
            .select(['NDVI', 'EVI', 'SummaryQA']).map(mask_quality).map(scale_indices))
    modis_ndvi = terra.merge(aqua).sort('system:time_start')

    for i, z in enumerate(ZONES):
        df = fetch_zone(z, modis_ndvi)
        out_path = OUT_DIR / f"{z['id']}_MODIS_NDVI_2026.csv"
        df.to_csv(out_path, index=False)
        zmax = df['date'].max() if len(df) else None
        print(f"  {z['id']:20s} {len(df):>3d} composites, through {zmax}  -> {out_path.name}")
        if i < len(ZONES) - 1:
            time.sleep(2)

    print(f'MODIS NDVI topup complete. Wrote {len(ZONES)} files to {OUT_DIR}')


if __name__ == '__main__':
    main()
