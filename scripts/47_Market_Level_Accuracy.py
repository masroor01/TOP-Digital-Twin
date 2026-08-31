"""
Script 47 -- Per-Market Model Accuracy
=============================================================================
The dashboard's "Model Accuracy" KPI (100% - MAPE) is a crop+horizon-level
statistic: production models are trained ONE PER (crop, horizon) across ALL
markets combined (Script 23), so there is exactly one validated MAPE per
crop+horizon -- not one per market. A user comparing two markets at the same
crop/horizon correctly sees the same number; that's not a bug, it's what
"one shared model" means. See MANIFEST.md / this script for the fix: a real,
additional per-MARKET accuracy figure computed from actual per-market
backtest predictions, so a specific market's own track record can be shown
alongside the crop-wide average.

Source: dm_market_level_predictions.csv (Script 15's MARKET_LEVEL_DIAGNOSTIC
output, the same file Script 18b's DM tests and Script 46's directional
accuracy test are built on) -- M6 only (the full-feature variant, the
closest available per-market backtest proxy to the actual deployed
production models; NOT byte-identical to Script 23's models, which are
trained separately -- same feature config and cross-validation scheme,
different training run). No production model has its own stored per-market
backtest, so this is the best available real answer, not a synthetic one.

Markets with very few backtested observations get a noisy MAPE from a
handful of weeks -- reported with n so the dashboard can flag/hide low-n
cells rather than show a falsely precise number.

Output: Model_Output/table_market_level_accuracy.csv
  columns: crop, market_id, horizon_weeks, mape, n
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

print('=================================================================')
print('SCRIPT 47: PER-MARKET MODEL ACCURACY')
print('=================================================================')

print('\n[1] Loading M6 per-market predictions ...')
df = pd.read_csv(PRED_FILE, usecols=['variant', 'crop', 'market_id', 'horizon_weeks', 'y_true', 'y_pred'])
df = df[df['variant'] == 'M6'].copy()
print(f'    {len(df):,} rows')

print('\n[2] Computing per-market MAPE ...')
df = df[(df['y_true'] > 0) & np.isfinite(df['y_true']) & np.isfinite(df['y_pred'])]
df['ape'] = (df['y_true'] - df['y_pred']).abs() / df['y_true']

agg = (
    df.groupby(['crop', 'market_id', 'horizon_weeks'])
    .agg(mape=('ape', lambda x: round(100 * x.mean(), 1)), n=('ape', 'size'))
    .reset_index()
)
agg['market_id'] = agg['market_id'].astype(int)

out_path = os.path.join(OUT_DIR, 'table_market_level_accuracy.csv')
agg.to_csv(out_path, index=False, encoding='utf-8')
print(f'    wrote {out_path} ({len(agg):,} rows)')
print(f'    n distribution: min={agg["n"].min()}, median={agg["n"].median():.0f}, max={agg["n"].max()}')
print(f'    rows with n<10 (noisy -- flag/hide in UI): {(agg["n"] < 10).sum():,} ({100*(agg["n"]<10).mean():.1f}%)')

print('\n=== SAMPLE (tomato, horizon=4) ===')
print(agg[(agg['crop'] == 'tomato') & (agg['horizon_weeks'] == 4)].sort_values('n', ascending=False).head(10).to_string(index=False))

print('\n=================================================================')
print('DONE.')
print('=================================================================')
