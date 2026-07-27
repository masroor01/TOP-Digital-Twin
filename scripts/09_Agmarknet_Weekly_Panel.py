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
  1. Filter to START_DATE -- END_DATE (see below)
  2. Clip modal price to valid range (crop-specific)
  3. Drop rows with zero/null arrivals or price
  4. Assign ISO week (Monday = week start)
  5. Compute arrivals-weighted mean modal price per (market, ISO week)
  6. For Potato: balanced panel -- keep only markets present in >=8 of 9 years
  7. Expand to complete (market x week) grid and impute price gaps:
       <=2 weeks   : linear interpolation
       3-8 weeks   : seasonal-median fill (same market, same calendar month)
       >8 weeks    : left NaN -- excluded from model training
  8. Output complete-grid panel with market metadata and imputation flags attached
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
    'onion':  r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2026.csv',
    'potato': r'C:\Users\masro\Downloads\potato_all_india_apmcs_2000_2026.csv',
}

OUTDIR = r'C:\Users\masro\Downloads\Agmarknet_Weekly'
os.makedirs(OUTDIR, exist_ok=True)

START_DATE = '2017-01-01'
END_DATE   = '2026-07-27'

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
# Imputation helpers
# ----------------------------------------------------------------

def _gap_lengths(series: pd.Series) -> pd.Series:
    """For each NaN position return its consecutive NaN run length; 0 for non-NaN."""
    is_null = series.isna()
    block_id = (is_null != is_null.shift()).cumsum()
    return is_null.groupby(block_id).transform('sum').where(is_null, 0).astype(int)


def impute_price_gaps(agg: pd.DataFrame, grid_end: pd.Timestamp) -> pd.DataFrame:
    """
    Expand weekly aggregates to a full (market × week) grid and impute price gaps:
      <=2 weeks   : linear interpolation (price varies smoothly at short horizons)
      3-8 weeks   : seasonal-median fill (same market, same calendar month, observed years)
      >8 weeks    : left NaN -- systematic absence, excluded from modelling
    Returns grid with added columns: imputed (0/1), imputed_method.

    grid_end: the END of the grid for THIS crop specifically -- must be
    min(END_DATE, this crop's own actual max observed date), not the global
    END_DATE. Different crops' raw sources can have different real cutoffs
    (e.g. onion's scraper reaches a later date than tomato/potato's); using
    the global END_DATE for all three would silently manufacture a fully-
    imputed "phantom" tail for whichever crop's data ends earlier -- caught
    when the dashboard's "latest week" landed on such a phantom week with
    100% imputed rows across every market.
    """
    all_weeks = pd.date_range(START_DATE, grid_end, freq='W-MON')
    full = (
        pd.MultiIndex.from_product(
            [agg['market_id'].unique(), all_weeks],
            names=['market_id', 'week_start']
        )
        .to_frame(index=False)
    )

    price_col = 'modal_price_weighted'
    keep_cols = ['market_id', 'week_start', price_col, 'arrivals_tonnes_week', 'trading_days']
    full = full.merge(agg[keep_cols], on=['market_id', 'week_start'], how='left')
    full = full.sort_values(['market_id', 'week_start']).reset_index(drop=True)

    full['imputed']        = full[price_col].isna().astype(int)
    full['imputed_method'] = full['imputed'].map({0: 'observed', 1: None})

    # Gap lengths computed before any filling
    full['_gap'] = full.groupby('market_id')[price_col].transform(_gap_lengths)

    # Stage 1: <=2 week gaps — linear interpolation
    full[price_col] = (
        full.groupby('market_id')[price_col]
            .transform(lambda x: x.interpolate(method='linear', limit=2, limit_area='inside'))
    )
    s1 = full['imputed'].eq(1) & full[price_col].notna()
    full.loc[s1, 'imputed_method'] = 'linear'

    # Stage 2: 3-8 week gaps — seasonal median (from observed rows only)
    full['_month'] = full['week_start'].dt.month
    smed = (
        full[full['imputed'] == 0]
        .groupby(['market_id', '_month'])[price_col]
        .median().rename('_smed').reset_index()
    )
    full = full.merge(smed, on=['market_id', '_month'], how='left')

    s2 = full[price_col].isna() & full['_gap'].between(3, 8)
    full.loc[s2, price_col]      = full.loc[s2, '_smed']
    full.loc[s2 & full[price_col].notna(), 'imputed_method'] = 'seasonal_median'

    full = full.drop(columns=['_gap', '_month', '_smed'])
    return full


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
    print(f'  After date filter ({START_DATE} to {END_DATE}): {len(df):,} rows')

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

    # 7. Potato: balanced panel (markets with data in >=8 of the years in [START_DATE, END_DATE])
    if crop == 'potato':
        years_per_market = df.groupby('market_id')['iso_year'].nunique()
        balanced_markets = years_per_market[years_per_market >= 8].index
        before = df['market_id'].nunique()
        df = df[df['market_id'].isin(balanced_markets)]
        print(f'  Potato balanced panel: {df["market_id"].nunique()} / {before} markets '
              f'(kept those present in >=8 of 9 years)')
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

    # 9. Collect market metadata for later re-join
    meta_cols = ['market_id', 'market', 'district', 'state', 'state_code']
    meta = df[meta_cols].drop_duplicates('market_id').set_index('market_id')

    # 10. Attach metadata (needed for imputation grouping; re-attached below after grid expansion)
    agg = agg.join(meta, on='market_id')
    agg['crop'] = crop

    # 11. Expand to complete (market x week) grid and impute price gaps.
    # Cap the grid at THIS crop's own actual max observed date, not the
    # global END_DATE -- crops' raw sources can have different real
    # cutoffs (see impute_price_gaps docstring).
    crop_max_date = df['arrival_date'].max()
    grid_end = min(pd.Timestamp(END_DATE), crop_max_date)
    print(f'  Grid end for {crop}: {grid_end.date()} '
          f'(crop\'s own max observed date: {crop_max_date.date()})')
    full = impute_price_gaps(agg, grid_end)
    full = full.join(meta, on='market_id')     # re-attach state/district/market to all grid rows
    full['crop']     = crop
    full['iso_year'] = full['week_start'].dt.isocalendar().year.astype(int)
    full['iso_week'] = full['week_start'].dt.isocalendar().week.astype(int)

    # Imputation summary
    n_obs  = (full['imputed'] == 0).sum()
    n_lin  = (full['imputed_method'] == 'linear').sum()
    n_smed = (full['imputed_method'] == 'seasonal_median').sum()
    n_long = full['modal_price_weighted'].isna().sum()
    print(f'  Grid rows: {len(full):,}  '
          f'observed={n_obs:,}  linear={n_lin:,}  seasonal_median={n_smed:,}  long_gap_NaN={n_long:,}')

    # 12. Column order
    out_cols = [
        'crop', 'state', 'state_code', 'district', 'market', 'market_id',
        'week_start', 'iso_year', 'iso_week',
        'modal_price_weighted', 'arrivals_tonnes_week', 'trading_days',
        'imputed', 'imputed_method',
    ]
    full = full[out_cols].sort_values(['market_id', 'week_start']).reset_index(drop=True)

    # 13. Save
    outpath = os.path.join(OUTDIR, f'{crop}_weekly_panel.csv')
    full.to_csv(outpath, index=False)
    obs = full[full['imputed'] == 0]
    print(f'  Saved: {outpath}')
    print(f'  Output shape: {full.shape}  (observed rows: {len(obs):,})')
    print(f'  Markets: {full["market_id"].nunique()}  |  States: {full["state"].nunique()}')
    print(f'  Week range: {full["week_start"].min().date()} to {full["week_start"].max().date()}')
    print(f'  Price (Rs/q, observed) — mean: {obs["modal_price_weighted"].mean():.1f}  '
          f'min: {obs["modal_price_weighted"].min():.1f}  '
          f'max: {obs["modal_price_weighted"].max():.1f}')

    return full


# ----------------------------------------------------------------
# Coverage report
# ----------------------------------------------------------------

def coverage_report(df: pd.DataFrame, crop: str):
    print(f'\n--- {crop.upper()} coverage by year ---')
    obs = df[df['imputed'] == 0] if 'imputed' in df.columns else df
    rpt = obs.groupby('iso_year').agg(
        weeks   = ('week_start', 'nunique'),
        markets = ('market_id', 'nunique'),
        states  = ('state', 'nunique'),
        obs_rows = ('week_start', 'count'),
    )
    if 'imputed' in df.columns:
        imp_rows = df[df['imputed'] == 1].groupby('iso_year').size().rename('imputed_rows')
        rpt = rpt.join(imp_rows, how='left').fillna(0).astype({'imputed_rows': int})
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
