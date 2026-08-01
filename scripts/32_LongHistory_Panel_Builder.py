# -*- coding: utf-8 -*-
"""
Script 32 — Long-History Panel Builder (VALIDATION EXPERIMENT, not a
permanent pipeline stage)
=============================================================================
Tests whether restricting the production panel to 2017+ (Script 09) is
discarding useful signal. The raw Agmarknet source files are named
`{crop}_all_india_apmcs_2000_2026.csv` and were verified (2026-08-01) to
contain real, non-trivial data from ~2002 onward (5,646 tomato rows across
85 markets in 2002, growing smoothly to 129,297 rows / 802 markets by 2016)
-- Script 09's 2017 floor is a filter choice, not a source limitation, for
price and arrivals specifically (several macro/infrastructure layers have
real, shorter source-side floors of their own -- see paper_drafts or the
2026-08-01 layer-availability audit; this script only touches price/arrivals).

This is an EXACT copy of Script 09's processing logic (weekly aggregation,
imputation policy, coverage filter, potato balanced-panel rule) with only
START_DATE moved back from 2017-01-01 to 2003-01-01 (skipping 2000-2002's
near-empty opening years) and the output redirected to a separate
`longhistory/` subfolder -- production's `data/agmarknet_weekly/*.csv` is
never touched by this script.

Output feeds Script 33, which runs the actual validation: does a price/
arrivals-only model trained on this 23-year panel beat the current 9-year
M0/M1 baselines and naive persistence, on IDENTICAL test folds?

Run: python scripts/32_LongHistory_Panel_Builder.py
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
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFILES = {
    'tomato': r'C:\Users\masro\Downloads\tomato_all_india_apmcs_2000_2026.csv',
    'onion':  r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2026.csv',
    'potato': r'C:\Users\masro\Downloads\potato_all_india_apmcs_2000_2026.csv',
}

OUTDIR = os.path.join(BASE, 'data', 'agmarknet_weekly', 'longhistory')
os.makedirs(OUTDIR, exist_ok=True)

START_DATE = '2003-01-01'   # vs production's 2017-01-01 -- the one change under test
END_DATE   = '2026-07-27'

MIN_REAL_COVERAGE = 0.70    # identical to production, for a fair comparison

PRICE_CLIP = {
    'tomato': (10,   20000),
    'onion':  (50,   12000),
    'potato': (40,    3500),
}


# ----------------------------------------------------------------
# Imputation helpers (identical to Script 09)
# ----------------------------------------------------------------

def _gap_lengths(series: pd.Series) -> pd.Series:
    is_null = series.isna()
    block_id = (is_null != is_null.shift()).cumsum()
    return is_null.groupby(block_id).transform('sum').where(is_null, 0).astype(int)


def impute_price_gaps(agg: pd.DataFrame, grid_end: pd.Timestamp) -> pd.DataFrame:
    # Each market's grid starts at ITS OWN first real observed week, not the
    # global START_DATE -- fixed 2026-08-01, see Script 09 for the full
    # rationale. This is the change under test in this re-run: at a 23-year
    # window, scoring every market against a fixed 2003-start grid meant a
    # market that only began reporting in, say, 2015 was counted "missing"
    # for 12 years before it existed in the system, which is why the first
    # pass of this experiment lost 68-70% of otherwise-qualifying markets.
    market_starts = agg.groupby('market_id')['week_start'].min()
    all_weeks = pd.date_range(START_DATE, grid_end, freq='W-MON')
    full = (
        pd.MultiIndex.from_product(
            [agg['market_id'].unique(), all_weeks],
            names=['market_id', 'week_start']
        )
        .to_frame(index=False)
    )
    full = full.merge(market_starts.rename('_market_start'), on='market_id', how='left')
    full = full[full['week_start'] >= full['_market_start']].drop(columns='_market_start')
    full = full.reset_index(drop=True)

    price_col = 'modal_price_weighted'
    keep_cols = ['market_id', 'week_start', price_col, 'arrivals_tonnes_week', 'trading_days']
    full = full.merge(agg[keep_cols], on=['market_id', 'week_start'], how='left')
    full = full.sort_values(['market_id', 'week_start']).reset_index(drop=True)

    full['imputed']        = full[price_col].isna().astype(int)
    full['imputed_method'] = full['imputed'].map({0: 'observed', 1: None})

    full['_gap'] = full.groupby('market_id')[price_col].transform(_gap_lengths)

    full[price_col] = (
        full.groupby('market_id')[price_col]
            .transform(lambda x: x.interpolate(method='linear', limit=2, limit_area='inside'))
    )
    s1 = full['imputed'].eq(1) & full[price_col].notna()
    full.loc[s1, 'imputed_method'] = 'linear'

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
# Core processor (identical to Script 09)
# ----------------------------------------------------------------

def process_crop(crop: str) -> pd.DataFrame:
    print(f'\n{"="*60}')
    print(f'Processing: {crop.upper()}')
    print('='*60)

    df = pd.read_csv(INFILES[crop], low_memory=False)
    print(f'  Loaded: {len(df):,} rows')

    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = df.dropna(subset=['arrival_date'])
    df = df[(df['arrival_date'] >= START_DATE) & (df['arrival_date'] <= END_DATE)]
    print(f'  After date filter ({START_DATE} to {END_DATE}): {len(df):,} rows')

    df['modal_price_rs_per_quintal'] = pd.to_numeric(df['modal_price_rs_per_quintal'], errors='coerce')
    df['arrivals_tonnes'] = pd.to_numeric(df['arrivals_tonnes'], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['modal_price_rs_per_quintal', 'arrivals_tonnes'])
    df = df[df['arrivals_tonnes'] > 0]
    print(f'  After null/zero-arrivals drop: {len(df):,} rows  (removed {before-len(df):,})')

    lo, hi = PRICE_CLIP[crop]
    before = len(df)
    df = df[(df['modal_price_rs_per_quintal'] >= lo) & (df['modal_price_rs_per_quintal'] <= hi)]
    print(f'  After price clip [{lo}, {hi}]: {len(df):,} rows  (removed {before-len(df):,} outliers)')

    df['iso_year']   = df['arrival_date'].dt.isocalendar().year.astype(int)
    df['iso_week']   = df['arrival_date'].dt.isocalendar().week.astype(int)
    df['week_start'] = df['arrival_date'] - pd.to_timedelta(df['arrival_date'].dt.dayofweek, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()

    if crop == 'potato':
        years_per_market = df.groupby('market_id')['iso_year'].nunique()
        balanced_markets = years_per_market[years_per_market >= 8].index
        before = df['market_id'].nunique()
        df = df[df['market_id'].isin(balanced_markets)]
        print(f'  Potato balanced panel: {df["market_id"].nunique()} / {before} markets '
              f'(kept those present in >=8 years)')
        print(f'  Rows after balancing: {len(df):,}')

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

    meta_cols = ['market_id', 'market', 'district', 'state', 'state_code']
    meta = df[meta_cols].drop_duplicates('market_id').set_index('market_id')

    agg = agg.join(meta, on='market_id')
    agg['crop'] = crop

    crop_max_date = df['arrival_date'].max()
    grid_end = min(pd.Timestamp(END_DATE), crop_max_date)
    print(f'  Grid end for {crop}: {grid_end.date()} '
          f'(crop\'s own max observed date: {crop_max_date.date()})')
    full = impute_price_gaps(agg, grid_end)
    full = full.join(meta, on='market_id')
    full['crop']     = crop
    full['iso_year'] = full['week_start'].dt.isocalendar().year.astype(int)
    full['iso_week'] = full['week_start'].dt.isocalendar().week.astype(int)

    coverage = 1 - full.groupby('market_id')['imputed'].mean()
    keep_markets = coverage[coverage >= MIN_REAL_COVERAGE].index
    before_n = full['market_id'].nunique()
    full = full[full['market_id'].isin(keep_markets)]
    print(f'  Real-coverage filter (>= {MIN_REAL_COVERAGE:.0%} real, over each '
          f"market's OWN span back to {START_DATE}): "
          f'{full["market_id"].nunique()} / {before_n} markets kept')

    n_obs  = (full['imputed'] == 0).sum()
    n_lin  = (full['imputed_method'] == 'linear').sum()
    n_smed = (full['imputed_method'] == 'seasonal_median').sum()
    n_long = full['modal_price_weighted'].isna().sum()
    print(f'  Grid rows: {len(full):,}  '
          f'observed={n_obs:,}  linear={n_lin:,}  seasonal_median={n_smed:,}  long_gap_NaN={n_long:,}')

    out_cols = [
        'crop', 'state', 'state_code', 'district', 'market', 'market_id',
        'week_start', 'iso_year', 'iso_week',
        'modal_price_weighted', 'arrivals_tonnes_week', 'trading_days',
        'imputed', 'imputed_method',
    ]
    full = full[out_cols].sort_values(['market_id', 'week_start']).reset_index(drop=True)

    outpath = os.path.join(OUTDIR, f'{crop}_weekly_panel_longhistory.csv')
    full.to_csv(outpath, index=False)
    obs = full[full['imputed'] == 0]
    print(f'  Saved: {outpath}')
    print(f'  Output shape: {full.shape}  (observed rows: {len(obs):,})')
    print(f'  Markets: {full["market_id"].nunique()}  |  States: {full["state"].nunique()}')
    print(f'  Week range: {full["week_start"].min().date()} to {full["week_start"].max().date()}')

    return full


if __name__ == '__main__':
    print('=' * 65)
    print('SCRIPT 32: LONG-HISTORY PANEL BUILDER (VALIDATION EXPERIMENT)')
    print(f'  START_DATE={START_DATE} (vs production 2017-01-01), else identical to Script 09')
    print('=' * 65)

    all_dfs = []
    for crop in ['tomato', 'onion', 'potato']:
        df_crop = process_crop(crop)
        all_dfs.append(df_crop)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUTDIR, 'top_weekly_panel_longhistory.csv')
    combined.to_csv(combined_path, index=False)

    print(f'\n{"="*60}')
    print('COMBINED LONG-HISTORY PANEL')
    print('='*60)
    print(f'  Saved: {combined_path}')
    print(f'  Total rows: {len(combined):,}')
    print(combined.groupby('crop').agg(
        rows    = ('week_start', 'count'),
        markets = ('market_id', 'nunique'),
        states  = ('state', 'nunique'),
    ).to_string())

    # Compare market counts against the current production (2017-2026) panel
    prod_path = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
    if os.path.exists(prod_path):
        prod = pd.read_csv(prod_path, usecols=['crop', 'market_id'])
        print('\n  Market count vs production (2017-2026) panel:')
        for crop in ['tomato', 'onion', 'potato']:
            n_long = combined[combined['crop'] == crop]['market_id'].nunique()
            n_prod = prod[prod['crop'] == crop]['market_id'].nunique()
            print(f'    {crop:7s}: longhistory={n_long:4d}  production={n_prod:4d}')

    print('\nScript 32 complete.')
