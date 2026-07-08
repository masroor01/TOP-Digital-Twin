# -*- coding: utf-8 -*-
"""
Script 09 — Agmarknet Weekly Panel Builder
===========================================
Input:  3 raw Agmarknet CSVs (Tomato, Onion, Potato)
Output: C:/Users/masro/Downloads/Agmarknet_Weekly/
        - tomato_weekly_panel.csv
        - onion_weekly_panel.csv
        - potato_weekly_panel.csv
        - top_weekly_panel.csv          (all three combined, long format)

Processing steps per crop:
  1. Filter to 2017-01-01 -- 2024-12-31
  2. Clip modal price to valid range (crop-specific)
  3. Drop rows with zero/null arrivals or price
  4. Assign ISO week (Monday = week start)
  5. Compute arrivals-weighted mean modal price per (market, ISO week)
  6. For Potato: balanced panel -- keep only markets present in all 8 years
  7. Output weekly panel with market metadata attached
"""

import io
import os
import sys
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
INFILES = {
    'tomato': r'C:\Users\masro\Downloads\tomato_all_india_apmcs_2000_2026.csv',
    'onion':  r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2025 (1).csv',
    'potato': r'C:\Users\masro\Downloads\potato_all_india_apmcs_2000_2026.csv',
}

OUTDIR = r'C:\Users\masro\Downloads\Agmarknet_Weekly'
os.makedirs(OUTDIR, exist_ok=True)

START_DATE = '2017-01-01'
END_DATE   = '2024-12-31'

# Price validity window per crop (Rs/quintal)
# Tomato: collapses to near-zero in glut; spikes observed ~8,000-10,000 in crisis
# Onion:  genuine crisis highs ~8,000-10,000 (2019-20 shortage); ₹460k is data error
# Potato: stable commodity; never exceeds ~3,000 in reality; ₹8,800 likely an error
PRICE_CLIP = {
    'tomato': (10,   20000),
    'onion':  (50,   12000),
    'potato': (40,    3500),
}


# ----------------------------------------------------------------
# Core processor
# ----------------------------------------------------------------

def process_crop(crop: str) -> pd.DataFrame:
    print(f'\n{"="*60}')
    print(f'Processing: {crop.upper()}')
    print('='*60)

    # 1. Load
    df = pd.read_csv(INFILES[crop], low_memory=False)
    print(f'  Loaded: {len(df):,} rows')

    # 2. Parse date & filter to study window
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = df.dropna(subset=['arrival_date'])
    df = df[(df['arrival_date'] >= START_DATE) & (df['arrival_date'] <= END_DATE)]
    print(f'  After date filter (2017-2024): {len(df):,} rows')

    # 3. Numeric coercion
    df['modal_price_rs_per_quintal'] = pd.to_numeric(df['modal_price_rs_per_quintal'], errors='coerce')
    df['arrivals_tonnes'] = pd.to_numeric(df['arrivals_tonnes'], errors='coerce')

    # 4. Drop null / zero arrivals / null price
    before = len(df)
    df = df.dropna(subset=['modal_price_rs_per_quintal', 'arrivals_tonnes'])
    df = df[df['arrivals_tonnes'] > 0]
    print(f'  After null/zero-arrivals drop: {len(df):,} rows  (removed {before-len(df):,})')

    # 5. Price outlier clip
    lo, hi = PRICE_CLIP[crop]
    before = len(df)
    df = df[(df['modal_price_rs_per_quintal'] >= lo) & (df['modal_price_rs_per_quintal'] <= hi)]
    print(f'  After price clip [{lo}, {hi}]: {len(df):,} rows  (removed {before-len(df):,} outliers)')

    # 6. ISO week assignment (Monday = start of week)
    df['iso_year']   = df['arrival_date'].dt.isocalendar().year.astype(int)
    df['iso_week']   = df['arrival_date'].dt.isocalendar().week.astype(int)
    df['week_start'] = df['arrival_date'] - pd.to_timedelta(df['arrival_date'].dt.dayofweek, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()

    # 7. Potato: balanced panel (markets with data in ALL 8 years 2017-2024)
    if crop == 'potato':
        years_per_market = df.groupby('market_id')['iso_year'].nunique()
        balanced_markets = years_per_market[years_per_market >= 8].index
        before = df['market_id'].nunique()
        df = df[df['market_id'].isin(balanced_markets)]
        print(f'  Potato balanced panel: {df["market_id"].nunique()} / {before} markets '
              f'(kept those present in all 8 years)')
        print(f'  Rows after balancing: {len(df):,}')

    # 8. Weekly aggregation: arrivals-weighted mean modal price per market per week
    def wavg(group):
        w = group['arrivals_tonnes']
        p = group['modal_price_rs_per_quintal']
        return (p * w).sum() / w.sum()

    agg = (
        df.groupby(['market_id', 'week_start'])
        .apply(lambda g: pd.Series({
            'modal_price_weighted': wavg(g),
            'arrivals_tonnes_week': g['arrivals_tonnes'].sum(),
            'trading_days':         g['arrival_date'].nunique(),
        }))
        .reset_index()
    )

    # 9. Attach market metadata (state, district, market name)
    meta_cols = ['market_id', 'market', 'district', 'state', 'state_code']
    meta = df[meta_cols].drop_duplicates('market_id').set_index('market_id')
    agg = agg.join(meta, on='market_id')

    # 10. Add crop column + calendar columns
    agg['crop']     = crop
    agg['iso_year'] = agg['week_start'].dt.isocalendar().year.astype(int)
    agg['iso_week'] = agg['week_start'].dt.isocalendar().week.astype(int)

    # 11. Column order
    out_cols = [
        'crop', 'state', 'state_code', 'district', 'market', 'market_id',
        'week_start', 'iso_year', 'iso_week',
        'modal_price_weighted', 'arrivals_tonnes_week', 'trading_days'
    ]
    agg = agg[out_cols].sort_values(['market_id', 'week_start']).reset_index(drop=True)

    # 12. Save
    outpath = os.path.join(OUTDIR, f'{crop}_weekly_panel.csv')
    agg.to_csv(outpath, index=False)
    print(f'  Saved: {outpath}')
    print(f'  Output shape: {agg.shape}')
    print(f'  Markets: {agg["market_id"].nunique()}  |  States: {agg["state"].nunique()}')
    print(f'  Week range: {agg["week_start"].min().date()} to {agg["week_start"].max().date()}')
    print(f'  Price (Rs/q) — mean: {agg["modal_price_weighted"].mean():.1f}  '
          f'min: {agg["modal_price_weighted"].min():.1f}  '
          f'max: {agg["modal_price_weighted"].max():.1f}')

    return agg


# ----------------------------------------------------------------
# Coverage report
# ----------------------------------------------------------------

def coverage_report(df: pd.DataFrame, crop: str):
    print(f'\n--- {crop.upper()} coverage by year ---')
    rpt = df.groupby('iso_year').agg(
        weeks   = ('week_start', 'nunique'),
        markets = ('market_id', 'nunique'),
        states  = ('state', 'nunique'),
        rows    = ('week_start', 'count'),
    )
    print(rpt.to_string())


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    all_dfs = []

    for crop in ['tomato', 'onion', 'potato']:
        df_crop = process_crop(crop)
        all_dfs.append(df_crop)
        coverage_report(df_crop, crop)

    # Combined long-format panel
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUTDIR, 'top_weekly_panel.csv')
    combined.to_csv(combined_path, index=False)

    print(f'\n{"="*60}')
    print('COMBINED TOP PANEL')
    print('='*60)
    print(f'  Saved: {combined_path}')
    print(f'  Total rows: {len(combined):,}')
    print(f'  Crops: {combined["crop"].unique()}')
    print(f'  Week range: {combined["week_start"].min().date()} to {combined["week_start"].max().date()}')
    print()
    print('  Rows per crop:')
    print(combined.groupby('crop').agg(
        rows    = ('week_start', 'count'),
        markets = ('market_id', 'nunique'),
        states  = ('state', 'nunique'),
    ).to_string())
    print()
    print('  Files written:')
    for f in os.listdir(OUTDIR):
        fpath = os.path.join(OUTDIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f'    {f:<35} {size_kb:>8.1f} KB')
    print()
    print('Script 09 complete.')
