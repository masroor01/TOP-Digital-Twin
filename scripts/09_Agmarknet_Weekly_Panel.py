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
  8. Real-coverage filter: drop markets below MIN_REAL_COVERAGE (80%) real
     (non-imputed) share of their own grid, all crops (added 2026-07-27)
  9. Output complete-grid panel with market metadata and imputation flags attached
"""

import io
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
DOWNLOADS = Path(os.environ.get('TOP_DOWNLOADS_DIR', r'C:\Users\masro\Downloads'))

INFILES = {
    'tomato': str(DOWNLOADS / 'tomato_all_india_apmcs_2000_2026.csv'),
    'onion':  str(DOWNLOADS / 'onion_all_india_apmcs_2000_2026.csv'),
    'potato': str(DOWNLOADS / 'potato_all_india_apmcs_2000_2026.csv'),
}

OUTDIR = str(DOWNLOADS / 'Agmarknet_Weekly')
os.makedirs(OUTDIR, exist_ok=True)

START_DATE = '2017-01-01'
# Defaults to today so the weekly automated refresh (scripts/weekly_refresh/)
# never needs a manual edit here. Each crop's actual grid end is still capped
# at that crop's own real max observed date (see impute_price_gaps's
# grid_end computation below), so leaving this at "today" can never
# manufacture data past what's genuinely on disk -- it's a safe default, not
# a promise that data reaches this date. Override with TOP_DT_END_DATE for a
# manual/reproducible run pinned to a specific historical cutoff.
END_DATE = os.environ.get('TOP_DT_END_DATE', datetime.now().strftime('%Y-%m-%d'))

# Minimum share of a market's own full grid that must be real (non-imputed)
# for the market to be retained. Added 2026-07-27, originally 0.80. Revised
# to 0.70 the same day: the market-level DM test (Script 18b) at 80% found
# onion's M0-vs-M6 result resting on only 34/189 significant markets --
# fragile. 70% adds 116 tomato / 57 onion markets (median coverage among the
# added markets is still ~90%, so this isn't admitting genuinely thin data)
# for more statistical power on that test, at the cost of a lower quality
# floor. Potato is unaffected (still West Bengal + Uttarakhand only -- that
# comes from the separate years>=8 balanced-panel filter, not this one).
MIN_REAL_COVERAGE = 0.70

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
      <=2 weeks, interior : linear interpolation (price varies smoothly at short horizons)
      <=2 weeks, trailing : forward-fill from the last real price (see note below)
      3-8 weeks            : seasonal-median fill (same market, same calendar month, observed years)
      >8 weeks             : left NaN -- systematic absence, excluded from modelling
    Returns grid with added columns: imputed (0/1), imputed_method.

    Trailing short gaps get their own fill (2026-08-21): pandas' interpolate()
    with limit_area='inside' can never fill a gap at the very end of a series
    -- there's no future anchor to interpolate toward -- so a market that is
    simply 1-2 weeks behind on reporting (the normal case at the panel's most
    recent weeks, not a data-quality problem) fell through both stages
    untouched: gap<3 excluded it from the seasonal-median stage, and
    limit_area='inside' excluded it from linear interpolation. Caught when
    potato's latest week showed 80.5% imputed, traced to West Bengal/
    Uttarakhand markets each sitting on an ordinary 1-2 week trailing lag,
    every one of them left as raw NaN with no imputed_method at all. A
    forward-fill of the last real price is the right estimate at this
    horizon -- better than a same-month historical seasonal median, since
    recent price is more informative than seasonal climatology for a gap
    this short.

    grid_end: the END of the grid for THIS crop specifically -- must be
    min(END_DATE, this crop's own actual max observed date), not the global
    END_DATE. Different crops' raw sources can have different real cutoffs
    (e.g. onion's scraper reaches a later date than tomato/potato's); using
    the global END_DATE for all three would silently manufacture a fully-
    imputed "phantom" tail for whichever crop's data ends earlier -- caught
    when the dashboard's "latest week" landed on such a phantom week with
    100% imputed rows across every market.

    Each market's grid starts at ITS OWN first real observed week, not the
    global START_DATE. Fixed 2026-08-01: building one fixed START_DATE-to-
    grid_end grid for every market (a plain cartesian product) meant a
    market that only began reporting years after START_DATE was scored as
    "missing" for every week before it existed in the system, silently
    dragging down its real-coverage ratio for a reason that has nothing to
    do with its actual data quality. Mild at a 9-year window (STUDY_START
    2017); became the dominant, market-count-collapsing effect once a
    2003-start long-history validation experiment (Script 32) extended the
    window to 23 years and lost 68-70% of otherwise-qualifying markets to
    exactly this artifact. This also brings the code in line with what the
    coverage methodology was always documented as doing (see
    paper_drafts/methods_data_section.txt Sec 3.3: "over that market's own
    reporting span").
    """
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

    # Gap lengths computed before any filling
    full['_gap'] = full.groupby('market_id')[price_col].transform(_gap_lengths)

    # Stage 1: <=2 week gaps — linear interpolation. NOTE: for an interior
    # gap this interpolates BETWEEN the surrounding real observations, i.e.
    # it can borrow a future week's real price to fill an earlier week.
    # That is an accepted design choice at this short horizon (see README),
    # but it means any lag/rolling/rolling-window feature built downstream
    # from log_price (or this price_col) on this panel MUST filter out
    # imputed==1 rows first -- otherwise a "past" feature value can
    # actually encode information from a later week (look-ahead leakage).
    # imputed_method=='linear' identifies exactly these rows.
    full[price_col] = (
        full.groupby('market_id')[price_col]
            .transform(lambda x: x.interpolate(method='linear', limit=2, limit_area='inside'))
    )
    s1 = full['imputed'].eq(1) & full[price_col].notna()
    full.loc[s1, 'imputed_method'] = 'linear'

    # Stage 1b: <=2 week gaps still unfilled after Stage 1 are trailing by
    # construction (every market's grid starts at its own first real week,
    # so there is no leading gap for interpolate() to have skipped) — carry
    # the last real price forward instead.
    s1b = full[price_col].isna() & full['_gap'].between(1, 2)
    full.loc[s1b, price_col] = (
        full.groupby('market_id')[price_col].transform(lambda x: x.ffill())
    )[s1b]
    full.loc[s1b & full[price_col].notna(), 'imputed_method'] = 'trailing_ffill'

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

    # 11b. Real-coverage filter: drop markets whose real (non-imputed) share of
    # their own full grid falls below MIN_REAL_COVERAGE. Applies to all crops
    # (potato's existing years>=8 filter already clears this bar for ~80% of
    # its markets, so this mostly affects tomato/onion, where no market-level
    # filter previously existed at all -- full retention had left most markets
    # >70-80% imputed; see 2026-07-27 coverage-gap review).
    coverage = 1 - full.groupby('market_id')['imputed'].mean()
    keep_markets = coverage[coverage >= MIN_REAL_COVERAGE].index
    before_n = full['market_id'].nunique()
    full = full[full['market_id'].isin(keep_markets)]
    print(f'  Real-coverage filter (>= {MIN_REAL_COVERAGE:.0%} real, '
          f'<= {1-MIN_REAL_COVERAGE:.0%} imputed): '
          f'{full["market_id"].nunique()} / {before_n} markets kept')

    # Imputation summary
    n_obs   = (full['imputed'] == 0).sum()
    n_lin   = (full['imputed_method'] == 'linear').sum()
    n_ffill = (full['imputed_method'] == 'trailing_ffill').sum()
    n_smed  = (full['imputed_method'] == 'seasonal_median').sum()
    n_long  = full['modal_price_weighted'].isna().sum()
    print(f'  Grid rows: {len(full):,}  observed={n_obs:,}  linear={n_lin:,}  '
          f'trailing_ffill={n_ffill:,}  seasonal_median={n_smed:,}  long_gap_NaN={n_long:,}')

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
