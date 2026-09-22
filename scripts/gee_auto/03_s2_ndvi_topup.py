"""
GEE Automation -- Sentinel-2 NDVI/EVI bimonthly composites topup (Python
port of scripts/gee/gee_03_S2_NDVI_2026.js), for the weekly CI refresh.

Same service-account + direct .getInfo() pattern as 01/02 -- see
01_era5_topup.py's docstring for the full rationale.

Deliberately reproduces the ORIGINAL script's logic exactly, including one
existing quirk NOT to "fix" here: MIN_VALID_FRAC is declared but never
actually applied as a filter (only notNull(NDVI) filters rows) -- this is
a port, not a behavior change; any bug in the original stays a separate,
explicit decision for later, not something silently altered mid-port.

Writes CSVs into $TOP_DOWNLOADS_DIR/GEE_2026/s2/{zone_id}_S2_NDVI_2026.csv
-- same folder/filename/schema as the manual JS script.

Zone list copied from scripts/16_Zone_Assignment.py's ZONES dict (current
source of truth) -- see 01_era5_topup.py's note on the stale P1/P2/P3 in
the original JS scripts.

Env vars: same as 01_era5_topup.py (GEE_SERVICE_ACCOUNT_KEY, TOP_DOWNLOADS_DIR)
"""
import os
import time
import calendar
from datetime import date, timedelta
from pathlib import Path

import ee
import pandas as pd


def get_info_with_retry(ee_object, max_retries=5, base_delay=15):
    """S2 compositing is heavy enough to occasionally hit GEE's
    'Too many concurrent aggregations' transient rate limit -- retry with
    exponential backoff rather than fail the whole run over a blip."""
    for attempt in range(max_retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f'    (getInfo failed: {e} -- retrying in {delay}s, attempt {attempt + 1}/{max_retries})')
            time.sleep(delay)

SERVICE_ACCOUNT = 'gee-automation@top-digital-twin-data.iam.gserviceaccount.com'
GCP_PROJECT = 'top-digital-twin-data'
KEY_PATH = os.environ.get(
    'GEE_SERVICE_ACCOUNT_KEY',
    r'C:\Users\masro\AppData\Local\Temp\claude\C--Users-masro-Documents-Claude-Code\5347e58a-b41d-4333-8d13-80bc3c9a2503\scratchpad\gee-automation-key.json',
)
DOWNLOADS = Path(os.environ.get('TOP_DOWNLOADS_DIR', str(Path.home() / 'Downloads')))
OUT_DIR = DOWNLOADS / 'GEE_2026' / 's2'

BUFFER_M = 30000
SCALE = 10  # S2 native resolution 10 m (NDVI bands B4, B8)
YEAR = 2026
DATA_START = '2026-01-01'

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


def mask_s2_clouds(img):
    scl = img.select('SCL')
    clear = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)))
    return img.updateMask(clear).set('clear_mask', clear)


def add_indices(img):
    b4 = img.select('B4')
    b8 = img.select('B8')
    b2 = img.select('B2')
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')
    # Same divide-by-zero guard as the JS script's fix for the s2_evi
    # blowup bug (values up to ~1e9 downstream before this guard existed).
    evi_denom = b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
    evi = (b8.subtract(b4).divide(evi_denom).multiply(2.5)
           .updateMask(evi_denom.abs().gte(0.01))
           .clamp(-1, 1).rename('EVI'))
    return img.addBands([ndvi, evi])


def build_windows(data_end_exclusive):
    """15-day windows (1-15, 16-end) for every month, filtered to those
    that start before the real data cutoff -- matches the JS script's own
    'later windows just come back empty' behavior, just skipped up front
    here instead of computed-then-discarded (functionally identical
    output, cheaper)."""
    days_in_month = [calendar.monthrange(YEAR, m)[1] for m in range(1, 13)]
    windows = []
    for m in range(1, 13):
        d_end = days_in_month[m - 1]
        windows.append({
            'start': date(YEAR, m, 1), 'end': date(YEAR, m, 15),
            'date_start': f'{YEAR}-{m:02d}-01', 'date_end': f'{YEAR}-{m:02d}-15',
        })
        windows.append({
            'start': date(YEAR, m, 16), 'end': date(YEAR, m, d_end),
            'date_start': f'{YEAR}-{m:02d}-16', 'date_end': f'{YEAR}-{m:02d}-{d_end}',
        })
    cutoff = date.fromisoformat(data_end_exclusive)
    return [w for w in windows if w['start'] < cutoff]


def fetch_zone(z, s2, windows):
    """One getInfo() call PER WINDOW, not one bundling all windows --
    bundling all ~18 windows' median-composite-then-reduceRegion into a
    single server-side computation graph consistently hit GEE's 'Too many
    concurrent aggregations' limit even after minutes of backoff (a real
    compute-budget ceiling, not a transient blip). Splitting per-window
    keeps each request small enough to stay within GEE's interactive
    compute budget, at the cost of more requests."""
    geom = ee.Geometry.Point([z['lon'], z['lat']]).buffer(BUFFER_M)
    rows = []
    for w in windows:
        sub = s2.filterDate(w['start'].isoformat(), (w['end'] + timedelta(days=1)).isoformat()).filterBounds(geom)
        n_scenes_img = sub.size()
        placeholder = ee.Image.constant([0, 0]).rename(['NDVI', 'EVI']).selfMask()
        comp = ee.Image(ee.Algorithms.If(n_scenes_img.gt(0), sub.median(), placeholder))

        mean_stats = comp.select(['NDVI', 'EVI']).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE, maxPixels=1e13,
        )
        valid_frac = comp.select('NDVI').mask().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE, maxPixels=1e13,
        )
        result = ee.Dictionary({
            'n_scenes': n_scenes_img,
            'NDVI': mean_stats.get('NDVI'),
            'EVI': mean_stats.get('EVI'),
            'valid_px_frac': valid_frac.get('NDVI'),
        })
        p = get_info_with_retry(result)
        time.sleep(1)  # spread load across the many more requests this now makes
        if p.get('NDVI') is None:
            continue  # matches the JS script's notNull(['NDVI']) filter
        rows.append({
            'zone_id': z['id'], 'crop': z['crop'], 'market': z['market'], 'state': z['state'],
            'date_start': w['date_start'], 'date_end': w['date_end'], 'year': YEAR,
            'doy_start': date.fromisoformat(w['date_start']).timetuple().tm_yday,
            'n_scenes': p['n_scenes'], 'NDVI': p.get('NDVI'), 'EVI': p.get('EVI'),
            'valid_px_frac': p.get('valid_px_frac'),
        })
    return pd.DataFrame(rows).sort_values('date_start').reset_index(drop=True)


def main():
    end_exclusive = (date.today() + timedelta(days=1)).isoformat()
    print(f'S2 NDVI topup: requesting {DATA_START} to {end_exclusive} (exclusive).')
    init_ee()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(DATA_START, end_exclusive)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 95))
          .map(mask_s2_clouds)
          .map(add_indices))
    windows = build_windows(end_exclusive)
    print(f'  {len(windows)} candidate composite windows through {end_exclusive}')

    for i, z in enumerate(ZONES):
        df = fetch_zone(z, s2, windows)
        out_path = OUT_DIR / f"{z['id']}_S2_NDVI_2026.csv"
        df.to_csv(out_path, index=False)
        zmax = df['date_end'].max() if len(df) else None
        print(f"  {z['id']:20s} {len(df):>3d} composites, through {zmax}  -> {out_path.name}")
        if i < len(ZONES) - 1:
            time.sleep(5)  # spread load, reduce concurrent-aggregation rate-limit hits

    print(f'S2 NDVI topup complete. Wrote {len(ZONES)} files to {OUT_DIR}')


if __name__ == '__main__':
    main()
