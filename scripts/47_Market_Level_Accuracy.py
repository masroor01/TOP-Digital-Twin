"""
Script 47 -- Hierarchical Market/State Accuracy (with shrinkage)
=============================================================================
The dashboard's "Model Accuracy" KPI (100% - MAPE) is a crop+horizon-level
statistic -- production models are trained one per (crop, horizon) across
ALL markets combined (Script 23), so it's genuinely the SAME number for
every market at a given crop/horizon. A user reasonably wanted more
granularity: a market's own track record, and a state-level figure too.

v1 of this script (2026-08-31) computed a flat per-market MAPE and hid any
market with under 10 backtested weeks. Two follow-up requests fixed here:

1. Add a STATE-level tier, pooling every market in that state -- crop-wide
   -> state -> market, each more specific but with progressively less data.
2. Replace the hard "hide if n<10" cliff with a proper shrinkage estimator,
   so every market shows a number, and thin markets show a TRUSTWORTHY one
   (pulled toward their state's tendency) instead of either a noisy raw
   figure or nothing at all.

Method -- two-level empirical-Bayes / credibility-weighted shrinkage:

  state_shrunk(crop, state, h)  = w_s * state_raw   + (1-w_s) * crop_raw
      where w_s = state_n / (state_n + K_STATE)

  market_shrunk(crop, market, h) = w_m * market_raw + (1-w_m) * state_shrunk(crop, state_of(market), h)
      where w_m = market_n / (market_n + K_MARKET)

A market shrinks toward ITS OWN STATE's (already-shrunk) estimate, not
straight to the crop-wide number -- a thin market in a well-covered state
still gets a locally-informed prior, not a blunt crop-wide average. K_STATE
and K_MARKET are pseudo-counts (in backtested weeks): K_MARKET=52 means a
market needs roughly a year of its own backtest history to weight mostly on
itself rather than its state; K_STATE=100 similarly for state vs. crop. Both
disclosed here as tunable, not hidden constants.

crop_raw (the top of the hierarchy) is computed from THIS SAME file
(dm_market_level_predictions.csv, M6, all rows for that crop+horizon) --
deliberately NOT read from model_uncertainty.json, even though the two
numbers are close, so the hierarchy is internally self-consistent (all
three levels derived from one dataset, not two different training runs).
The dashboard's main "Model Accuracy" KPI keeps using model_uncertainty.json
(the officially validated production figure) unchanged; only the
state/market tiers use this file's own crop-level mean as their shrinkage
target, to avoid a small mismatch between "crop-wide" numbers shown for
different purposes.

Every market/state gets a row now -- no hidden cells. `n` (raw backtested
weeks) is still reported alongside the shrunk figure so the UI can show a
low-confidence indicator for genuinely thin markets without hiding the
number outright.

Output: Model_Output/table_market_level_accuracy.csv
  columns: crop, market_id, state, horizon_weeks,
           market_mape_raw, market_n, market_mape_shrunk,
           state_mape_raw, state_n, state_mape_shrunk,
           crop_mape_raw
"""
import os
import sys
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'Model_Output')
PRED_FILE = os.path.join(OUT_DIR, 'dm_market_level_predictions.csv')
REF_FILE = os.path.join(OUT_DIR, 'production_models', 'reference_rows.csv')

K_STATE = 100    # pseudo-count (backtested weeks) for state -> crop shrinkage
K_MARKET = 52    # pseudo-count (backtested weeks) for market -> state shrinkage

print('=================================================================')
print('SCRIPT 47: HIERARCHICAL MARKET/STATE ACCURACY (WITH SHRINKAGE)')
print('=================================================================')

print('\n[1] Loading M6 per-market predictions + market->state lookup ...')
df = pd.read_csv(PRED_FILE, usecols=['variant', 'crop', 'market_id', 'horizon_weeks', 'y_true', 'y_pred'])
df = df[df['variant'] == 'M6'].copy()
df = df[(df['y_true'] > 0) & np.isfinite(df['y_true']) & np.isfinite(df['y_pred'])]
df['ape'] = 100 * (df['y_true'] - df['y_pred']).abs() / df['y_true']

ref = pd.read_csv(REF_FILE, usecols=['crop', 'market_id', 'state'])
ref = ref.dropna(subset=['market_id']).drop_duplicates(['crop', 'market_id'])
ref['market_id'] = ref['market_id'].astype(int)
df['market_id'] = df['market_id'].astype(int)
df = df.merge(ref, on=['crop', 'market_id'], how='left')
n_unmatched = df['state'].isna().sum()
print(f'    {len(df):,} rows, {n_unmatched:,} without a matched state (dropped)')
df = df.dropna(subset=['state'])

print('\n[2] Computing crop-level (grand mean) MAPE per horizon ...')
crop_level = df.groupby(['crop', 'horizon_weeks'])['ape'].mean().rename('crop_mape_raw').reset_index()

print('\n[3] Computing state-level MAPE + shrinkage toward crop-level ...')
state_level = (
    df.groupby(['crop', 'state', 'horizon_weeks'])
    .agg(state_mape_raw=('ape', 'mean'), state_n=('ape', 'size'))
    .reset_index()
)
state_level = state_level.merge(crop_level, on=['crop', 'horizon_weeks'], how='left')
w_s = state_level['state_n'] / (state_level['state_n'] + K_STATE)
state_level['state_mape_shrunk'] = w_s * state_level['state_mape_raw'] + (1 - w_s) * state_level['crop_mape_raw']

print('\n[4] Computing market-level MAPE + shrinkage toward its state ...')
market_level = (
    df.groupby(['crop', 'market_id', 'state', 'horizon_weeks'])
    .agg(market_mape_raw=('ape', 'mean'), market_n=('ape', 'size'))
    .reset_index()
)
market_level = market_level.merge(
    state_level[['crop', 'state', 'horizon_weeks', 'state_mape_shrunk']],
    on=['crop', 'state', 'horizon_weeks'], how='left',
)
w_m = market_level['market_n'] / (market_level['market_n'] + K_MARKET)
market_level['market_mape_shrunk'] = w_m * market_level['market_mape_raw'] + (1 - w_m) * market_level['state_mape_shrunk']

print('\n[5] Assembling final table ...')
out = market_level.merge(
    state_level[['crop', 'state', 'horizon_weeks', 'state_mape_raw', 'state_n']],
    on=['crop', 'state', 'horizon_weeks'], how='left',
)
out = out.merge(crop_level, on=['crop', 'horizon_weeks'], how='left')
out = out[[
    'crop', 'market_id', 'state', 'horizon_weeks',
    'market_mape_raw', 'market_n', 'market_mape_shrunk',
    'state_mape_raw', 'state_n', 'state_mape_shrunk',
    'crop_mape_raw',
]].round(1)

out_path = os.path.join(OUT_DIR, 'table_market_level_accuracy.csv')
out.to_csv(out_path, index=False, encoding='utf-8')
print(f'    wrote {out_path} ({len(out):,} rows, {state_level.shape[0]:,} state cells folded in)')
print(f'    market n distribution: min={out["market_n"].min()}, median={out["market_n"].median():.0f}, max={out["market_n"].max()}')
print(f'    thin markets (n<{K_MARKET}, shrinkage pulls them meaningfully toward their state): '
      f'{(out["market_n"] < K_MARKET).sum():,} ({100*(out["market_n"] < K_MARKET).mean():.1f}%)')

print('\n=== SAMPLE (tomato, Karnataka, horizon=4) ===')
sample = out[(out['crop'] == 'tomato') & (out['state'] == 'Karnataka') & (out['horizon_weeks'] == 4)]
print(sample.sort_values('market_n', ascending=False).head(10).to_string(index=False))

print('\n=================================================================')
print('DONE.')
print('=================================================================')
