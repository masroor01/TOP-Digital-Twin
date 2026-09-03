"""
Script 47 -- Hierarchical Market/State Accuracy (with shrinkage)
=============================================================================
The dashboard's "Model Accuracy" KPI is a crop+horizon-level statistic --
production models are trained one per (crop, horizon) across ALL markets
combined (Script 23), so it's genuinely the SAME number for every market at
a given crop/horizon. A user reasonably wanted more granularity: a market's
own track record, and a state-level figure too.

v1 of this script (2026-08-31) computed a flat per-market MAPE and hid any
market with under 10 backtested weeks. A follow-up added a STATE-level tier
(crop-wide -> state -> market) and replaced the hard "hide if n<10" cliff
with shrinkage, so every market shows a trustworthy number instead of a
hidden or noisy one.

SWITCHED FROM MAPE TO WAPE (2026-09-02, user-requested after an empirical
comparison -- see scripts/50_MAPE_vs_WAPE_Comparison.py): the original
metric was `mean(|y_true - y_pred| / y_true)` -- an UNWEIGHTED average of
per-row percentage errors. Two real problems follow: every (market, week)
row counts equally regardless of its actual traded value, and a row with an
unusually low true price can produce a huge percentage error for a modest
Rs miss, distorting the average at full weight. Script 50 found this is not
theoretical: e.g. Patti APMC (tomato, h=26w) showed MAPE=224% vs WAPE=38%
for the exact same predictions, driven by a handful of rows with a near-
zero recorded price (e.g. y_true=Rs 15/quintal against a Rs 2,289 model
prediction that was actually reasonable). WAPE --
`sum(|y_true-y_pred|) / sum(y_true)` -- uses the same per-row errors but
weighted by each row's own actual value instead of counted equally, which
is both more robust to this failure mode and a more economically meaningful
aggregate (dominated by what actually traded, not by row count).

Method -- two-level empirical-Bayes / credibility-weighted shrinkage,
unchanged in structure, just applied to WAPE instead of MAPE:

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
disclosed here as tunable, not hidden constants. `n` (raw backtested weeks
underlying that cell's WAPE) is unchanged in meaning by the metric switch.

crop_raw (the top of the hierarchy) is computed from THIS SAME file
(dm_market_level_predictions.csv, M6, all rows for that crop+horizon) --
deliberately NOT read from model_uncertainty.json, even though the two
numbers are close, so the hierarchy is internally self-consistent (all
three levels derived from one dataset, not two different training runs).
The dashboard's main "Model Accuracy" KPI keeps using model_uncertainty.json
(also now WAPE-based, see Script 23) unchanged; only the state/market tiers
use this file's own crop-level WAPE as their shrinkage target, to avoid a
small mismatch between "crop-wide" numbers shown for different purposes.

Every market/state gets a row now -- no hidden cells.

Output: Model_Output/table_market_level_accuracy.csv
  columns: crop, market_id, state, horizon_weeks,
           market_wape_raw, market_n, market_wape_shrunk,
           state_wape_raw, state_n, state_wape_shrunk,
           crop_wape_raw
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


def wape(abs_err_sum, y_true_sum):
    return 100 * abs_err_sum / y_true_sum


print('=================================================================')
print('SCRIPT 47: HIERARCHICAL MARKET/STATE ACCURACY (WAPE, WITH SHRINKAGE)')
print('=================================================================')

print('\n[1] Loading M6 per-market predictions + market->state lookup ...')
df = pd.read_csv(PRED_FILE, usecols=['variant', 'crop', 'market_id', 'horizon_weeks', 'y_true', 'y_pred'])
df = df[df['variant'] == 'M6'].copy()
df = df[(df['y_true'] > 0) & np.isfinite(df['y_true']) & np.isfinite(df['y_pred'])]
df['abs_err'] = (df['y_true'] - df['y_pred']).abs()

ref = pd.read_csv(REF_FILE, usecols=['crop', 'market_id', 'state'])
ref = ref.dropna(subset=['market_id']).drop_duplicates(['crop', 'market_id'])
ref['market_id'] = ref['market_id'].astype(int)
df['market_id'] = df['market_id'].astype(int)
df = df.merge(ref, on=['crop', 'market_id'], how='left')
n_unmatched = df['state'].isna().sum()
print(f'    {len(df):,} rows, {n_unmatched:,} without a matched state (dropped)')
df = df.dropna(subset=['state'])

print('\n[2] Computing crop-level WAPE per horizon ...')
crop_agg = df.groupby(['crop', 'horizon_weeks']).agg(
    _abs_err=('abs_err', 'sum'), _y_true=('y_true', 'sum')).reset_index()
crop_agg['crop_wape_raw'] = wape(crop_agg['_abs_err'], crop_agg['_y_true'])
crop_level = crop_agg[['crop', 'horizon_weeks', 'crop_wape_raw']]

print('\n[3] Computing state-level WAPE + shrinkage toward crop-level ...')
state_agg = (
    df.groupby(['crop', 'state', 'horizon_weeks'])
    .agg(_abs_err=('abs_err', 'sum'), _y_true=('y_true', 'sum'), state_n=('abs_err', 'size'))
    .reset_index()
)
state_agg['state_wape_raw'] = wape(state_agg['_abs_err'], state_agg['_y_true'])
state_level = state_agg[['crop', 'state', 'horizon_weeks', 'state_wape_raw', 'state_n']]
state_level = state_level.merge(crop_level, on=['crop', 'horizon_weeks'], how='left')
w_s = state_level['state_n'] / (state_level['state_n'] + K_STATE)
state_level['state_wape_shrunk'] = w_s * state_level['state_wape_raw'] + (1 - w_s) * state_level['crop_wape_raw']

print('\n[4] Computing market-level WAPE + shrinkage toward its state ...')
market_agg = (
    df.groupby(['crop', 'market_id', 'state', 'horizon_weeks'])
    .agg(_abs_err=('abs_err', 'sum'), _y_true=('y_true', 'sum'), market_n=('abs_err', 'size'))
    .reset_index()
)
market_agg['market_wape_raw'] = wape(market_agg['_abs_err'], market_agg['_y_true'])
market_level = market_agg[['crop', 'market_id', 'state', 'horizon_weeks', 'market_wape_raw', 'market_n']]
market_level = market_level.merge(
    state_level[['crop', 'state', 'horizon_weeks', 'state_wape_shrunk']],
    on=['crop', 'state', 'horizon_weeks'], how='left',
)
w_m = market_level['market_n'] / (market_level['market_n'] + K_MARKET)
market_level['market_wape_shrunk'] = w_m * market_level['market_wape_raw'] + (1 - w_m) * market_level['state_wape_shrunk']

print('\n[5] Assembling final table ...')
out = market_level.merge(
    state_level[['crop', 'state', 'horizon_weeks', 'state_wape_raw', 'state_n']],
    on=['crop', 'state', 'horizon_weeks'], how='left',
)
out = out.merge(crop_level, on=['crop', 'horizon_weeks'], how='left')
out = out[[
    'crop', 'market_id', 'state', 'horizon_weeks',
    'market_wape_raw', 'market_n', 'market_wape_shrunk',
    'state_wape_raw', 'state_n', 'state_wape_shrunk',
    'crop_wape_raw',
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
