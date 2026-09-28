"""
Script 43 -- Live Forecast Validation: Onion, 1-Week Horizon
=============================================================================
Checks the EXISTING, already-trained onion_1w production model's forecast
(generated from whatever reference row is currently in reference_rows.csv,
the panel's cutoff at training time) against REAL, newly-arrived observed
prices for the target week (reference week_start + 7 days) -- obtained from
a fresh Agmarknet pull that was NOT used to train the model.

Deliberately does NOT retrain on the new data before this check -- doing so
would let the model see the future it's being asked to predict, invalidating
the test. Retraining (if wanted) should happen only AFTER this validation.

TARGET_WEEK is computed dynamically from reference_rows.csv's own week_start
(see [1] below) rather than hardcoded, so this script can be re-run at any
time -- after any future Script 23 retrain -- without manually editing a date
constant first. The only manual step left is placing a fresh raw pull at
NEW_RAW_FILE that actually covers the (now dynamic) target week; see [2].
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
OUT_DIR = os.path.join(BASE, 'Model_Output')
DOWNLOADS = os.environ.get('TOP_DOWNLOADS_DIR', 'C:\\Users\\masro\\Downloads')
NEW_RAW_FILE = os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2026_new.csv')
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')

PRICE_CLIP = (50, 12000)          # same as Script 09's PRICE_CLIP['onion']

print('=================================================================')
print('SCRIPT 43: LIVE ONION FORECAST VALIDATION (1-WEEK HORIZON)')
print('=================================================================')

# ---------------------------------------------------------------------------
# [1] Load the EXISTING, unmodified production model + reference row + features
# ---------------------------------------------------------------------------
print('\n[1] Loading existing onion_1w production model and reference row ...')
with open(os.path.join(MODEL_DIR, 'feature_columns.json')) as f:
    feature_columns = json.load(f)
cols_1w = feature_columns['onion_1w']

ref = pd.read_csv(os.path.join(MODEL_DIR, 'reference_rows.csv'))
ref = ref[ref['crop'] == 'onion'].copy()
print(f'  Reference rows: {len(ref)} onion markets, week_start = {ref["week_start"].iloc[0]}')

# TARGET_WEEK is derived from reference_rows.csv itself, not hardcoded -- it
# always tracks whatever cutoff Script 23 last trained on, so this script
# stays "live" across every future retrain without manual editing.
TARGET_WEEK = pd.Timestamp(ref['week_start'].iloc[0]) + pd.Timedelta(weeks=1)

model = joblib.load(os.path.join(MODEL_DIR, 'onion_1w.joblib'))
X = pd.DataFrame([{c: row.get(c, 0) for c in cols_1w} for _, row in ref.iterrows()])
log_pred = model.predict(X)
ref['forecast_price'] = np.expm1(log_pred)
ref['naive_price'] = ref['last_observed_price']
print(f'  Forecast generated for target week: {TARGET_WEEK.date()}')

# ---------------------------------------------------------------------------
# [2] Build the REAL actual weekly price for the target week from the fresh pull
#     -- replicating Script 09's exact cleaning + weekly-aggregation logic
# ---------------------------------------------------------------------------
print('\n[2] Computing real actual weekly price from the fresh onion pull ...')
if not os.path.exists(NEW_RAW_FILE):
    print(f'\n  ERROR: no fresh raw pull found at {NEW_RAW_FILE}')
    print(f'  This script validates the onion_1w model against REAL observed prices for the')
    print(f'  target week {TARGET_WEEK.date()} (reference_rows.csv cutoff + 1 week). That requires')
    print(f'  a manually-placed fresh Agmarknet scrape -- same raw column format as Script 09\'s')
    print(f'  onion input -- covering at least through {TARGET_WEEK.date()}, saved at the path above')
    print(f'  (or set TOP_DOWNLOADS_DIR to point at wherever it lives).')
    print(f'\n  Pull a fresh export and re-run. Nothing else in this script is stale.')
    sys.exit(1)
raw = pd.read_csv(NEW_RAW_FILE, usecols=[
    'arrival_date', 'market_id', 'market', 'state', 'modal_price_rs_per_quintal', 'arrivals_tonnes'
])
raw['arrival_date'] = pd.to_datetime(raw['arrival_date'])
raw['modal_price_rs_per_quintal'] = pd.to_numeric(raw['modal_price_rs_per_quintal'], errors='coerce')
raw['arrivals_tonnes'] = pd.to_numeric(raw['arrivals_tonnes'], errors='coerce')
before = len(raw)
raw = raw.dropna(subset=['modal_price_rs_per_quintal', 'arrivals_tonnes'])
raw = raw[raw['arrivals_tonnes'] > 0]
lo, hi = PRICE_CLIP
raw = raw[(raw['modal_price_rs_per_quintal'] >= lo) & (raw['modal_price_rs_per_quintal'] <= hi)]
print(f'  Cleaned rows: {len(raw):,} (removed {before - len(raw):,})')

raw['week_start'] = (raw['arrival_date'] - pd.to_timedelta(raw['arrival_date'].dt.dayofweek, unit='D')).dt.normalize()
wk = raw[raw['week_start'] == TARGET_WEEK]
print(f'  Rows in target week {TARGET_WEEK.date()}: {len(wk)}')

if len(wk) == 0:
    max_raw_week = raw['week_start'].max() if len(raw) else None
    print(f'\n  ERROR: {NEW_RAW_FILE} has zero rows for the target week {TARGET_WEEK.date()}.')
    if pd.notna(max_raw_week):
        print(f'  Its latest coverage after cleaning is {max_raw_week.date()}, which is')
        print(f'  {(TARGET_WEEK - max_raw_week).days} day(s) short of the target week -- the file is')
        print(f'  stale relative to reference_rows.csv\'s current cutoff '
              f'({ref["week_start"].iloc[0]}).')
    else:
        print(f'  The file is empty after cleaning (see removed-row count above) -- re-check the export.')
    print(f'  Pull a fresher Agmarknet scrape covering through at least {TARGET_WEEK.date()} and re-run.')
    sys.exit(1)

def wavg(g):
    w = g['arrivals_tonnes']
    p = g['modal_price_rs_per_quintal']
    return (p * w).sum() / w.sum()

actual = (
    wk.groupby('market_id')
    .apply(lambda g: pd.Series({'actual_price': wavg(g), 'trading_days': g['arrival_date'].nunique()}))
    .reset_index()
)
print(f'  Markets with a real observed price in the target week: {len(actual)}')

# ---------------------------------------------------------------------------
# [3] Bridge market_id (raw file) -> market/state (reference_rows) via the
#     already-ingested panel, which carries both keys
# ---------------------------------------------------------------------------
print('\n[3] Bridging market_id -> market/state via the ingested panel ...')
panel = pd.read_csv(PANEL_FILE, usecols=['crop', 'market_id', 'market', 'state'])
bridge = panel[panel['crop'] == 'onion'][['market_id', 'market', 'state']].drop_duplicates('market_id')
actual = actual.merge(bridge, on='market_id', how='left')
n_unmatched = actual['market'].isna().sum()
print(f'  Bridged {len(actual) - n_unmatched} / {len(actual)} real-price markets to a known market/state '
      f'({n_unmatched} not in the ingested panel -- likely newly-onboarded markets, dropped from this check)')
actual = actual.dropna(subset=['market'])

# ---------------------------------------------------------------------------
# [4] Merge forecast vs actual vs naive, compute error metrics
# ---------------------------------------------------------------------------
print('\n[4] Merging forecast vs actual vs naive-persistence baseline ...')
merged = ref[['market_id', 'market', 'state', 'forecast_price', 'naive_price', 'imputed',
              'sufficient_history', 'stale_reference']].merge(
    actual[['market_id', 'actual_price', 'trading_days']], on=['market_id'], how='inner'
)
print(f'  Matched markets (forecast + real actual both available): {len(merged)}')
n_excluded = (~merged['sufficient_history']).sum()
if n_excluded:
    print(f'  Excluding {n_excluded} markets flagged sufficient_history=False '
          f'(insufficient price-lag history -- see Script 23 Sec 3) from the comparison below.')
    merged = merged[merged['sufficient_history']].copy()

# stale_reference: market HAS history, but its reference-row anchor is a
# long-gap fallback (e.g. seasonal-median), not a recent real trade -- both
# the forecast and the naive comparison are anchored on the same stale
# guess in these cases, so including them isn't a fair test of either
# method. Excluded separately from sufficient_history since it's a distinct
# failure mode (see Script 23 Sec 3 / README Known Gotchas).
n_stale = merged['stale_reference'].sum()
if n_stale:
    print(f'  Excluding {n_stale} additional markets flagged stale_reference=True '
          f'(reference price is a long-gap fallback, not a recent real trade) from the comparison below.')
    merged = merged[~merged['stale_reference']].copy()

merged['model_ape'] = (merged['forecast_price'] - merged['actual_price']).abs() / merged['actual_price'] * 100
merged['naive_ape'] = (merged['naive_price'] - merged['actual_price']).abs() / merged['actual_price'] * 100
merged['model_beats_naive'] = merged['model_ape'] < merged['naive_ape']

out_path = os.path.join(OUT_DIR, 'table_live_onion_1w_validation.csv')
merged.to_csv(out_path, index=False)

print('\n' + '=' * 65)
print(f'RESULTS -- Onion, 1-week-ahead, target week {TARGET_WEEK.date()}')
print('=' * 65)
print(f'  Markets checked:              {len(merged)}')
print(f'  Model  MAPE (all markets):    {merged["model_ape"].mean():6.2f}%   (median {merged["model_ape"].median():6.2f}%)')
print(f'  Naive  MAPE (all markets):    {merged["naive_ape"].mean():6.2f}%   (median {merged["naive_ape"].median():6.2f}%)')
print(f'  Model beats naive:            {merged["model_beats_naive"].mean()*100:5.1f}% of markets')

clean = merged[merged['imputed'] == False] if merged['imputed'].dtype == bool else merged[merged['imputed'] == 0]
print(f'\n  Restricting to reference rows that were NOT themselves imputed at cutoff ({len(clean)} markets):')
print(f'  Model  MAPE: {clean["model_ape"].mean():6.2f}%   Naive MAPE: {clean["naive_ape"].mean():6.2f}%   '
      f'Model beats naive: {clean["model_beats_naive"].mean()*100:5.1f}%')

# national aggregate (arrivals-unweighted mean across markets, both series)
nat_forecast = merged['forecast_price'].mean()
nat_actual = merged['actual_price'].mean()
nat_naive = merged['naive_price'].mean()
print(f'\n  National average price -- forecast: Rs {nat_forecast:,.0f}/q   '
      f'actual: Rs {nat_actual:,.0f}/q   naive: Rs {nat_naive:,.0f}/q')
print(f'  National-average forecast error: {abs(nat_forecast-nat_actual)/nat_actual*100:.2f}%   '
      f'(naive: {abs(nat_naive-nat_actual)/nat_actual*100:.2f}%)')

worst = merged.sort_values('model_ape', ascending=False).head(10)
print('\n  10 largest per-market forecast misses:')
print(worst[['market', 'state', 'forecast_price', 'actual_price', 'model_ape']].to_string(index=False))

print(f'\n  Saved: {out_path}')
print('\nScript 43 complete.')
