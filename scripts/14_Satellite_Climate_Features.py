# -*- coding: utf-8 -*-
"""
Script 14 — Satellite & Climate Feature Engineering
====================================================
Extracts and processes 4 data sources into weekly district-level
climate and vegetation features for Layer 2 (Satellite) and
Layer 3 (Climate) of the TOP Digital Twin.

Inputs (zip files):
  ERA5-...zip           daily temperature per zone (2000-2024)
  CHIRPS-...zip         pentad rainfall per zone (1981-2024)
  S2-...zip             scene-level NDVI/EVI per zone (2017-2024)
  MODIS-...zip          16-day NDVI+EVI and LST per zone (2000-2024)

Outputs:
  data/satellite_climate/raw/era5/            extracted ERA5 CSVs
  data/satellite_climate/raw/chirps/          extracted CHIRPS CSVs
  data/satellite_climate/raw/s2/              extracted S2 CSVs
  data/satellite_climate/raw/modis/           extracted MODIS CSVs
  data/satellite_climate/zone_weekly_features.csv
      zone_id x week_start: all climate + satellite features
  data/satellite_climate/crop_weekly_features.csv
      crop x week_start: crop-averaged features (joins to market panel)
"""

import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOWNLOADS = Path(r'C:\Users\masro\Downloads')
PROJ      = Path(r'C:\Users\masro\Documents\TOP_Digital_Twin')
OUT_DIR   = PROJ / 'data' / 'satellite_climate'
RAW_DIR   = OUT_DIR / 'raw'
FIG_DIR   = PROJ / 'Model_Output'

ZIP_ERA5       = DOWNLOADS / 'ERA5-20260708T112951Z-3-001.zip'
ZIP_ERA5_TOPUP = DOWNLOADS / 'ERA5_topup-20260708T112808Z-3-001.zip'
ZIP_CHIRPS     = DOWNLOADS / 'CHIRPS-20260708T112910Z-3-001.zip'
ZIP_S2         = DOWNLOADS / 'S2-20260708T112817Z-3-001.zip'
ZIP_MODIS      = DOWNLOADS / 'MODIS-20260708T112651Z-3-001.zip'

START = pd.Timestamp('2017-01-01')
END   = pd.Timestamp('2024-12-31')

# Full ISO-week index 2017-2024 (Mondays)
_all_weeks = pd.date_range('2016-12-26', '2024-12-31', freq='W-MON')
WEEK_INDEX = _all_weeks[(_all_weeks >= START) & (_all_weeks <= END)]

CROPS = ['tomato', 'onion', 'potato']

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def to_week_start(series: pd.Series) -> pd.Series:
    """Map any date to the Monday of its ISO week."""
    dt = pd.to_datetime(series)
    return (dt - pd.to_timedelta(dt.dt.dayofweek, unit='D')).dt.normalize()


def extract_zip(zip_path: Path, subdir: str) -> int:
    dest = RAW_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            fname = Path(entry).name
            if not fname.endswith('.csv'):
                continue
            dest_file = dest / fname
            if not dest_file.exists():
                with zf.open(entry) as src, open(dest_file, 'wb') as dst:
                    dst.write(src.read())
            n += 1
    return n


def reindex_and_ffill(df: pd.DataFrame,
                      value_cols: list,
                      meta_cols: list,
                      limit: int = 4) -> pd.DataFrame:
    """
    For each zone, expand to the full weekly grid and forward-fill
    sparse satellite observations (limit = max weeks to fill).
    """
    frames = []
    for zone_id, grp in df.groupby('zone_id'):
        grp = grp.set_index('week_start').sort_index()

        # Fill meta columns forward first (they're constant per zone)
        for c in meta_cols:
            if c in grp.columns:
                grp[c] = grp[c].ffill().bfill()

        # Reindex to full weekly grid
        grp = grp.reindex(WEEK_INDEX)
        grp['zone_id'] = zone_id

        # Forward fill value columns (cloud gaps, compositing gaps)
        for c in value_cols:
            if c in grp.columns:
                grp[c] = grp[c].ffill(limit=limit)

        # Restore meta from original (not affected by reindex noise)
        src_meta = df[df['zone_id'] == zone_id][meta_cols].iloc[0]
        for c in meta_cols:
            grp[c] = src_meta[c]

        grp.index.name = 'week_start'
        frames.append(grp.reset_index())

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract zips
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 1: Extracting zip files')
print('='*65)

for label, zpath, subdir in [
    ('ERA5',       ZIP_ERA5,       'era5'),
    ('ERA5_topup', ZIP_ERA5_TOPUP, 'era5_topup'),
    ('CHIRPS',     ZIP_CHIRPS,     'chirps'),
    ('S2',         ZIP_S2,         's2'),
    ('MODIS',      ZIP_MODIS,      'modis'),
]:
    n = extract_zip(zpath, subdir)
    print(f'  {label:<8}  {n:>3} files  →  {RAW_DIR / subdir}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: ERA5 — daily temperature → weekly aggregates
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 2: ERA5 — daily → weekly')
print('='*65)

era5_frames = []
# Load ERA5 main (2000-2020-07-09) and ERA5_topup (2020-07-10 to 2024-12-31)
for era5_subdir in ['era5', 'era5_topup']:
    for fpath in sorted((RAW_DIR / era5_subdir).glob('*.csv')):
        df = pd.read_csv(fpath, parse_dates=['date'], low_memory=False)
        df = df[(df['date'] >= START) & (df['date'] <= END)].copy()
        if df.empty:
            continue
        era5_frames.append(df)

era5_raw = pd.concat(era5_frames, ignore_index=True)
# Remove any overlap between main and topup (keep one row per zone × date)
era5_raw = era5_raw.drop_duplicates(subset=['zone_id', 'date']).copy()

era5_raw['week_start'] = to_week_start(era5_raw['date'])
era5_raw['dtr'] = era5_raw['Tmax_C'] - era5_raw['Tmin_C']

era5 = era5_raw.groupby(['zone_id', 'crop', 'week_start']).agg(
    era5_tmax    = ('Tmax_C',       'mean'),
    era5_tmin    = ('Tmin_C',       'mean'),
    era5_tmean   = ('Tmean_C',      'mean'),
    era5_dtr     = ('dtr',          'mean'),
    era5_heat_35 = ('flag_above35', 'sum'),   # sum of daily fractional flags (0-7)
    era5_heat_38 = ('flag_above38', 'sum'),
    era5_n_days  = ('date',         'count'),
).reset_index()
era5['crop'] = era5['crop'].str.lower()

print(f'  Zones processed : {era5["zone_id"].nunique()}')
print(f'  Date range      : {era5["week_start"].min().date()} → {era5["week_start"].max().date()}')
print(f'  Rows            : {len(era5):,}')
print(f'  ERA5 Tmax range : {era5["era5_tmax"].min():.1f}°C – {era5["era5_tmax"].max():.1f}°C')
print(f'  Heat-stress ≥35°: {era5["era5_heat_35"].describe()[["mean","max"]].to_dict()}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: CHIRPS — pentad rainfall → weekly aggregates
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 3: CHIRPS — pentad → weekly')
print('='*65)

chirps_frames = []
for fpath in sorted((RAW_DIR / 'chirps').glob('*.csv')):
    df = pd.read_csv(fpath, parse_dates=['date'], low_memory=False)
    df = df[(df['date'] >= START) & (df['date'] <= END)].copy()
    if df.empty:
        continue
    df['week_start'] = to_week_start(df['date'])
    # rain_mean_mm = spatial-mean daily rainfall (mm/day).
    # Multiply by 5 to get pentad total mm, then sum across pentads in the week.
    df['pentad_total_mm'] = df['rain_mean_mm'] * 5

    agg = df.groupby(['zone_id', 'crop', 'week_start']).agg(
        chirps_rain_mm  = ('pentad_total_mm',  'sum'),   # total mm in week
        chirps_rain_max = ('rain_max_mm',       'max'),   # max daily rainfall (mm/day)
        chirps_excess   = ('frac_excess_rain',  'mean'),  # fraction of pentads with excess rain
        chirps_n_pentad = ('date',              'count'),
    ).reset_index()
    chirps_frames.append(agg)

chirps = pd.concat(chirps_frames, ignore_index=True)
chirps['crop'] = chirps['crop'].str.lower()

print(f'  Zones processed : {chirps["zone_id"].nunique()}')
print(f'  Date range      : {chirps["week_start"].min().date()} → {chirps["week_start"].max().date()}')
print(f'  Rows            : {len(chirps):,}')
print(f'  Rain (mm/week)  : mean={chirps["chirps_rain_mm"].mean():.1f}  max={chirps["chirps_rain_mm"].max():.1f}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Sentinel-2 — scene composites → weekly (forward-filled)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 4: Sentinel-2 — scenes → weekly (forward-filled, limit=4w)')
print('='*65)

s2_frames = []
for fpath in sorted((RAW_DIR / 's2').glob('*.csv')):
    df = pd.read_csv(fpath, parse_dates=['date_start'], low_memory=False)
    df = df[(df['date_start'] >= START) & (df['date_start'] <= END)].copy()
    if df.empty:
        continue
    df['week_start'] = to_week_start(df['date_start'])
    df = df.rename(columns={'NDVI': 's2_ndvi', 'EVI': 's2_evi',
                             'valid_px_frac': 's2_valid_frac'})

    # If two composites fall in the same week, keep highest quality
    df = df.sort_values('s2_valid_frac', ascending=False)
    df = df.drop_duplicates(subset=['zone_id', 'week_start'], keep='first')
    df = df[['zone_id', 'crop', 'week_start', 's2_ndvi', 's2_evi', 's2_valid_frac']]
    s2_frames.append(df)

s2_raw = pd.concat(s2_frames, ignore_index=True)
s2_raw['crop'] = s2_raw['crop'].str.lower()

# Expand to full weekly grid with forward fill
s2 = reindex_and_ffill(
    s2_raw,
    value_cols=['s2_ndvi', 's2_evi', 's2_valid_frac'],
    meta_cols=['crop'],
    limit=4
)

# NDVI anomaly: deviation from the climatological mean for that ISO-week number
s2['iso_wk'] = s2['week_start'].dt.isocalendar().week.astype(int)
clim = (s2.groupby(['zone_id', 'iso_wk'])['s2_ndvi']
          .mean()
          .rename('s2_ndvi_clim')
          .reset_index())
s2 = s2.merge(clim, on=['zone_id', 'iso_wk'], how='left')
s2['s2_ndvi_anom'] = s2['s2_ndvi'] - s2['s2_ndvi_clim']
s2 = s2.drop(columns=['iso_wk', 's2_ndvi_clim'])

print(f'  Zones processed : {s2["zone_id"].nunique()}')
print(f'  Date range      : {s2["week_start"].min().date()} → {s2["week_start"].max().date()}')
print(f'  Rows            : {len(s2):,}')
fill_rate = s2['s2_ndvi'].notna().mean()
print(f'  S2 NDVI fill rate after forward-fill: {fill_rate:.1%}')
print(f'  NDVI range: {s2["s2_ndvi"].min():.3f} – {s2["s2_ndvi"].max():.3f}')
print(f'  NDVI anomaly range: {s2["s2_ndvi_anom"].min():.3f} – {s2["s2_ndvi_anom"].max():.3f}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: MODIS — 16-day NDVI + LST composites → weekly (forward-filled)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 5: MODIS — 16-day composites → weekly (forward-filled, limit=3w)')
print('='*65)

modis_ndvi_frames, modis_lst_frames = [], []

for fpath in sorted((RAW_DIR / 'modis').glob('*.csv')):
    df = pd.read_csv(fpath, parse_dates=['date'], low_memory=False)
    df = df[(df['date'] >= START) & (df['date'] <= END)].copy()
    if df.empty:
        continue
    df['week_start'] = to_week_start(df['date'])

    if 'NDVI' in df.columns:
        # MODIS NDVI product
        df = df[df['n_valid_px'] > 0]
        if df.empty:
            continue
        df = df.rename(columns={'NDVI': 'modis_ndvi', 'EVI': 'modis_evi'})
        df = df.sort_values('n_valid_px', ascending=False)
        df = df.drop_duplicates(subset=['zone_id', 'week_start'], keep='first')
        modis_ndvi_frames.append(
            df[['zone_id', 'crop', 'week_start', 'modis_ndvi', 'modis_evi']]
        )

    elif 'LST_mean_C' in df.columns:
        # MODIS LST product
        df = df[df['n_valid_px'] > 0].dropna(subset=['LST_mean_C'])
        if df.empty:
            continue
        df = df.rename(columns={'LST_mean_C': 'modis_lst_mean',
                                 'LST_max_C':  'modis_lst_max',
                                 'frac_above35': 'modis_lst_frac35'})
        df = df.sort_values('n_valid_px', ascending=False)
        df = df.drop_duplicates(subset=['zone_id', 'week_start'], keep='first')
        modis_lst_frames.append(
            df[['zone_id', 'crop', 'week_start',
                'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']]
        )

# Process MODIS NDVI
modis_ndvi_raw = pd.concat(modis_ndvi_frames, ignore_index=True)
modis_ndvi_raw['crop'] = modis_ndvi_raw['crop'].str.lower()
modis_ndvi = reindex_and_ffill(
    modis_ndvi_raw,
    value_cols=['modis_ndvi', 'modis_evi'],
    meta_cols=['crop'],
    limit=3
)

# Process MODIS LST
modis_lst_raw = pd.concat(modis_lst_frames, ignore_index=True)
modis_lst_raw['crop'] = modis_lst_raw['crop'].str.lower()
modis_lst = reindex_and_ffill(
    modis_lst_raw,
    value_cols=['modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35'],
    meta_cols=['crop'],
    limit=3
)

# Merge NDVI and LST on zone × week
modis = modis_ndvi.merge(
    modis_lst[['zone_id', 'week_start', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']],
    on=['zone_id', 'week_start'],
    how='outer'
)
if 'crop_x' in modis.columns:
    modis['crop'] = modis['crop_x'].fillna(modis['crop_y'])
    modis = modis.drop(columns=['crop_x', 'crop_y'], errors='ignore')

print(f'  MODIS NDVI zones: {modis_ndvi["zone_id"].nunique()}  '
      f'| MODIS LST zones: {modis_lst["zone_id"].nunique()}')
print(f'  Rows after merge: {len(modis):,}')
print(f'  MODIS NDVI fill rate: {modis["modis_ndvi"].notna().mean():.1%}')
print(f'  MODIS LST fill rate : {modis["modis_lst_mean"].notna().mean():.1%}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Merge all sources → zone_weekly_features
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 6: Merging all sources into zone_weekly_features')
print('='*65)

# Build full skeleton: all 17 zones × all 418 ISO weeks
zones_meta = era5[['zone_id', 'crop']].drop_duplicates()
skeleton = pd.merge(
    zones_meta.assign(key=1),
    pd.DataFrame({'week_start': WEEK_INDEX, 'key': 1}),
    on='key'
).drop(columns='key').sort_values(['zone_id', 'week_start']).reset_index(drop=True)
print(f'  Skeleton shape: {skeleton.shape}  ({skeleton["zone_id"].nunique()} zones × {skeleton["week_start"].nunique()} weeks)')

zone_feat = skeleton.copy()

for other, label in [(era5.drop(columns=['era5_n_days'], errors='ignore'), 'ERA5'),
                     (chirps.drop(columns=['chirps_n_pentad'], errors='ignore'), 'CHIRPS'),
                     (s2, 'S2'),
                     (modis, 'MODIS')]:
    merge_cols = [c for c in other.columns
                  if c not in ('crop', 'zone_id', 'week_start')]
    zone_feat = zone_feat.merge(
        other[['zone_id', 'week_start'] + merge_cols],
        on=['zone_id', 'week_start'],
        how='left'
    )
    if 'crop_x' in zone_feat.columns:
        zone_feat['crop'] = zone_feat['crop_x'].fillna(zone_feat['crop_y'])
        zone_feat = zone_feat.drop(columns=['crop_x', 'crop_y'], errors='ignore')
    print(f'  After merging {label}: {zone_feat.shape}')

# Drop any residual helper columns
zone_feat = zone_feat.drop(columns=['era5_n_days', 'chirps_n_pentad'], errors='ignore')

# Sort
zone_feat = zone_feat.sort_values(['crop', 'zone_id', 'week_start']).reset_index(drop=True)

# Save
zone_out = OUT_DIR / 'zone_weekly_features.csv'
zone_feat.to_csv(zone_out, index=False)

print(f'\n  zone_weekly_features.csv')
print(f'    Rows    : {len(zone_feat):,}')
print(f'    Zones   : {zone_feat["zone_id"].nunique()}')
print(f'    Weeks   : {zone_feat["week_start"].nunique()}')
print(f'    Columns : {list(zone_feat.columns)}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Crop-level average → crop_weekly_features
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 7: Crop-level average → crop_weekly_features')
print('='*65)

feat_cols = [c for c in zone_feat.columns
             if c not in ('zone_id', 'crop', 'week_start')]

crop_feat = (
    zone_feat.groupby(['crop', 'week_start'])[feat_cols]
    .mean()
    .reset_index()
)

crop_out = OUT_DIR / 'crop_weekly_features.csv'
crop_feat.to_csv(crop_out, index=False)

print(f'  crop_weekly_features.csv')
print(f'    Rows    : {len(crop_feat):,}')
print(f'    Crops   : {crop_feat["crop"].unique().tolist()}')
print(f'    Weeks   : {crop_feat["week_start"].nunique()}')
print(f'\n  Coverage per crop:')
for crop, grp in crop_feat.groupby('crop'):
    print(f'    {crop:<8}: {grp["week_start"].min().date()} → {grp["week_start"].max().date()}'
          f'  ({len(grp)} weeks)')


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Data-quality summary table
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 8: Feature completeness (% non-null in crop_weekly_features)')
print('='*65)

completeness = (crop_feat[feat_cols].notna().mean() * 100).round(1)
print(completeness.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Paper figures
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 9: Generating paper figures')
print('='*65)

CROP_COLORS = {'tomato': '#d62728', 'onion': '#9467bd', 'potato': '#8c564b'}
CROP_LABELS = {'tomato': 'Tomato', 'onion': 'Onion', 'potato': 'Potato'}

# ── Figure 1: ERA5 Weekly Temperature by Crop ────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, (crop, grp) in zip(axes, crop_feat.groupby('crop')):
    ax.plot(grp['week_start'], grp['era5_tmean'], color=CROP_COLORS[crop],
            lw=1.2, label='Mean temp', alpha=0.9)
    ax.fill_between(grp['week_start'], grp['era5_tmin'], grp['era5_tmax'],
                    alpha=0.15, color=CROP_COLORS[crop])
    ax2 = ax.twinx()
    ax2.bar(grp['week_start'], grp['era5_heat_35'], width=5,
            color='#ff7f0e', alpha=0.5, label='Heat-stress days ≥35°C')
    ax2.set_ylabel('Heat-stress\nday-equiv.', fontsize=8, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=7)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_title(f'{CROP_LABELS[crop]} — Production Zone Temperature', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
axes[-1].set_xlabel('Year', fontsize=10)
plt.suptitle('ERA5 Weekly Temperature: Production Zone Averages (2017–2024)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = FIG_DIR / 'fig_era5_temperature.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path.name}')

# ── Figure 2: CHIRPS Rainfall Heatmap ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, (crop, grp) in zip(axes, crop_feat.groupby('crop')):
    pivot = grp.copy()
    pivot['year'] = pivot['week_start'].dt.year
    pivot['week'] = pivot['week_start'].dt.isocalendar().week.astype(int)
    mat = pivot.pivot_table(index='year', columns='week', values='chirps_rain_mm',
                            aggfunc='mean').fillna(0)
    im = ax.imshow(mat.values, aspect='auto', cmap='Blues', interpolation='nearest',
                   extent=[1, 53, mat.index.max() + 0.5, mat.index.min() - 0.5])
    plt.colorbar(im, ax=ax, label='mm/week', fraction=0.03, pad=0.02)
    ax.set_ylabel('Year', fontsize=9)
    ax.set_yticks(mat.index)
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(f'{CROP_LABELS[crop]} — Weekly Rainfall (mm)', fontsize=10, fontweight='bold')
axes[-1].set_xlabel('ISO Week Number', fontsize=10)
plt.suptitle('CHIRPS Rainfall Heatmap: Crop Production Zones (2017–2024)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = FIG_DIR / 'fig_chirps_rainfall_heatmap.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path.name}')

# ── Figure 3: S2 NDVI Time Series + Anomaly ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, (crop, grp) in zip(axes, crop_feat.groupby('crop')):
    color = CROP_COLORS[crop]
    ax.plot(grp['week_start'], grp['s2_ndvi'], color=color,
            lw=1.2, alpha=0.9, label='S2 NDVI')
    # NDVI anomaly as shaded band from zero
    anom = grp.set_index('week_start')['s2_ndvi_anom']
    ax.fill_between(anom.index, 0, anom,
                    where=(anom >= 0), color='green', alpha=0.2, label='Above normal')
    ax.fill_between(anom.index, 0, anom,
                    where=(anom < 0),  color='red',   alpha=0.2, label='Below normal')
    ax.axhline(0, color='grey', lw=0.7, linestyle='--', alpha=0.5)
    ax.set_ylabel('NDVI / Anomaly', fontsize=9)
    ax.set_title(f'{CROP_LABELS[crop]} — Sentinel-2 NDVI & Anomaly', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
axes[-1].set_xlabel('Year', fontsize=10)
plt.suptitle('Sentinel-2 NDVI: Production Zone Averages and Anomalies (2017–2024)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = FIG_DIR / 'fig_s2_ndvi_anomaly.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path.name}')

# ── Figure 4: MODIS LST vs ERA5 Tmax (cross-validation) ─────────────────────
tomato_zone = crop_feat[crop_feat['crop'] == 'tomato'].copy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: LST vs ERA5 Tmax scatter
valid = tomato_zone.dropna(subset=['modis_lst_mean', 'era5_tmax'])
sc = ax1.scatter(valid['era5_tmax'], valid['modis_lst_mean'],
                 c=valid['week_start'].astype(np.int64), cmap='viridis',
                 alpha=0.5, s=10)
ax1.plot([20, 45], [20, 45], 'r--', lw=1, label='1:1 line')
plt.colorbar(sc, ax=ax1, label='Time')
ax1.set_xlabel('ERA5 Tmax (°C)', fontsize=10)
ax1.set_ylabel('MODIS LST Mean (°C)', fontsize=10)
ax1.set_title('MODIS LST vs ERA5 Tmax (Tomato zones)', fontsize=10, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
if len(valid) >= 2:
    r = np.corrcoef(valid['era5_tmax'], valid['modis_lst_mean'])[0, 1]
    ax1.text(0.05, 0.92, f'r = {r:.3f}', transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

# Panel B: MODIS NDVI vs S2 NDVI scatter
valid2 = tomato_zone.dropna(subset=['modis_ndvi', 's2_ndvi'])
sc2 = ax2.scatter(valid2['s2_ndvi'], valid2['modis_ndvi'],
                  c=valid2['week_start'].astype(np.int64), cmap='viridis',
                  alpha=0.5, s=10)
ax2.plot([0.1, 0.7], [0.1, 0.7], 'r--', lw=1, label='1:1 line')
plt.colorbar(sc2, ax=ax2, label='Time')
ax2.set_xlabel('Sentinel-2 NDVI', fontsize=10)
ax2.set_ylabel('MODIS NDVI', fontsize=10)
ax2.set_title('MODIS vs Sentinel-2 NDVI (Tomato zones)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
if len(valid2) >= 2:
    r2 = np.corrcoef(valid2['s2_ndvi'], valid2['modis_ndvi'])[0, 1]
    ax2.text(0.05, 0.92, f'r = {r2:.3f}', transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.suptitle('Satellite Cross-Validation: MODIS vs ERA5 / Sentinel-2 (Tomato)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = FIG_DIR / 'fig_satellite_cross_validation.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path.name}')

# ── Figure 5: Feature correlation heatmap ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (crop, grp) in zip(axes, crop_feat.groupby('crop')):
    corr_cols = [c for c in feat_cols
                 if grp[c].notna().sum() > 50]  # only show columns with enough data
    corr = grp[corr_cols].corr()
    labels = [c.replace('era5_', '').replace('chirps_', '').replace('s2_', '').replace('modis_', '')
              for c in corr_cols]
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f'{CROP_LABELS[crop]}', fontsize=10, fontweight='bold')
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=5, color='black' if abs(v) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

plt.suptitle('Feature Correlation: Climate & Satellite Variables (2017–2024)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = FIG_DIR / 'fig_climate_satellite_correlation.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Final summary
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('STEP 10: Final summary')
print('='*65)

print(f'\n  zone_weekly_features.csv  :  {len(zone_feat):,} rows  x  {zone_feat.shape[1]} columns')
print(f'  crop_weekly_features.csv  :  {len(crop_feat):,} rows  x  {crop_feat.shape[1]} columns')
print(f'\n  Feature columns in crop_weekly_features:')
for c in feat_cols:
    pct = crop_feat[c].notna().mean() * 100
    print(f'    {c:<30}  {pct:5.1f}% non-null')

print(f'\n  To join to market panel in Module B:')
print(f'    sat_clim = pd.read_csv("data/satellite_climate/crop_weekly_features.csv",')
print(f'                           parse_dates=["week_start"])')
print(f'    df = df.merge(sat_clim, on=["crop", "week_start"], how="left")')
print('\nScript 14 complete.')
