# -*- coding: utf-8 -*-
"""
Script 26 — Weekly-to-Daily Temporal Disaggregation
=======================================================
Produces daily-resolution price curves FROM the validated weekly M6
model's forecasts, without training a new daily model (see this script's
predecessor, deleted 2026-07-30 -- daily-native models lost to naive
persistence even more decisively than weekly ones, and the daily coverage
filter collapsed market counts by 3-6x, so training new daily ML/DL models
was abandoned).

Method: temporal disaggregation (the same family of technique used in
economics to convert quarterly/monthly series to daily/weekly, e.g.
Chow-Lin/Denton methods) -- tried and HONESTLY EVALUATED, not assumed:
  1. From REAL historical daily Agmarknet data, computed each crop's
     average day-of-week price pattern relative to its own weekly mean.
  2. Backtested it: disaggregate each historical week's REAL average
     price using ONLY the day-of-week factor, compare to that week's
     REAL observed daily prices, against a flat-interpolation baseline.
  RESULT: the day-of-week factor is negligible (0.996-1.004, essentially
  1.0) and the backtest shows it is marginally WORSE than flat
  interpolation for all 3 crops -- modal APMC prices do not have a
  systematic weekday cycle. Kept in this script for the record (and
  because the negative result is itself useful to know), but NOT used
  in the final daily curve below.
  3. What IS used: a smooth (monotonic PCHIP) interpolant through the
     weekly model's validated forecast points (h=1w/4w/13w/26w) as the
     daily central estimate, plus an honest uncertainty band from the
     historical daily residual std-dev (11-16% for tomato/onion, 7-8%
     for potato -- 10-30x larger than the day-of-week effect ever was).
     This band says plainly: day-to-day movement within a week is real
     and large, but it is NOT predictable from the calendar, only
     bounded by history -- the chart does not claim to know which
     specific day will spike.

Outputs (Model_Output/):
  table_dow_pattern.csv           -- day-of-week factor per crop
  table_disagg_backtest.csv       -- backtest: disaggregation vs flat interpolation
  fig_disagg_example.png          -- example daily curve for one market/crop
  fig_dow_pattern.png             -- day-of-week factor by crop

Run: python scripts/26_Weekly_To_Daily_Disaggregation.py
"""

import io, os, sys, warnings
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILES = {
    'tomato': r'C:\Users\masro\Downloads\tomato_all_india_apmcs_2000_2026.csv',
    'onion':  r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2026.csv',
    'potato': r'C:\Users\masro\Downloads\potato_all_india_apmcs_2000_2026.csv',
}
WEEKLY_DIR = os.path.join(BASE, 'data', 'agmarknet_weekly')
OUT_DIR    = os.path.join(BASE, 'Model_Output')
CROPS      = ['tomato', 'onion', 'potato']
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}
PRICE_CLIP = {'tomato': (10, 20000), 'onion': (50, 12000), 'potato': (40, 3500)}
START_DATE, END_DATE = '2017-01-01', '2026-07-27'

print('=' * 65)
print('SCRIPT 26: WEEKLY-TO-DAILY TEMPORAL DISAGGREGATION')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw daily data, compute day-of-week factor per crop
# ─────────────────────────────────────────────────────────────────────────────
def load_daily(crop):
    df = pd.read_csv(RAW_FILES[crop], low_memory=False)
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = df.dropna(subset=['arrival_date'])
    df = df[(df['arrival_date'] >= START_DATE) & (df['arrival_date'] <= END_DATE)]
    df['modal_price_rs_per_quintal'] = pd.to_numeric(df['modal_price_rs_per_quintal'], errors='coerce')
    df['arrivals_tonnes'] = pd.to_numeric(df['arrivals_tonnes'], errors='coerce')
    df = df.dropna(subset=['modal_price_rs_per_quintal', 'arrivals_tonnes'])
    df = df[df['arrivals_tonnes'] > 0]
    lo, hi = PRICE_CLIP[crop]
    df = df[(df['modal_price_rs_per_quintal'] >= lo) & (df['modal_price_rs_per_quintal'] <= hi)]
    df['week_start'] = (df['arrival_date'] - pd.to_timedelta(df['arrival_date'].dt.dayofweek, unit='D')).dt.normalize()
    df['dow'] = df['arrival_date'].dt.dayofweek  # 0=Mon
    return df

print('\n[1] Loading raw daily data + computing day-of-week pattern ...')
daily_raw = {}
dow_rows = []
for crop in CROPS:
    d = load_daily(crop)
    daily_raw[crop] = d
    # Weekly mean per (market, week) for normalization
    wk_mean = (d.groupby(['market_id', 'week_start'])['modal_price_rs_per_quintal']
               .transform('mean'))
    d = d.assign(rel_price=d['modal_price_rs_per_quintal'] / wk_mean)
    daily_raw[crop] = d
    factor = d.groupby('dow')['rel_price'].agg(['mean', 'std', 'count']).reset_index()
    factor['crop'] = crop
    dow_rows.append(factor)
    print(f'  {crop:8s}: day-of-week factors = ' +
          ', '.join(f'{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(r.dow)]}={r["mean"]:.3f}'
                     for _, r in factor.iterrows()))

dow_table = pd.concat(dow_rows, ignore_index=True)
dow_table.columns = ['dow', 'factor_mean', 'factor_std', 'n_obs', 'crop']
dow_table.to_csv(os.path.join(OUT_DIR, 'table_dow_pattern.csv'), index=False)
print(f'  Saved: table_dow_pattern.csv')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backtest: does the day-of-week pattern beat flat interpolation?
#    For real historical weeks, disaggregate the week's REAL average price
#    using ONLY the day-of-week factor, compare to REAL observed daily prices.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Backtesting disaggregation vs flat interpolation ...')

backtest_rows = []
for crop in CROPS:
    d = daily_raw[crop]
    factor_map = dow_table[dow_table['crop'] == crop].set_index('dow')['factor_mean']

    wk = (d.groupby(['market_id', 'week_start'])
          .agg(week_mean=('modal_price_rs_per_quintal', 'mean'))
          .reset_index())
    d2 = d.merge(wk, on=['market_id', 'week_start'])
    d2['pred_flat']  = d2['week_mean']                                    # flat interpolation baseline
    d2['pred_disagg']= d2['week_mean'] * d2['dow'].map(factor_map)        # day-of-week disaggregation
    actual = d2['modal_price_rs_per_quintal']

    for label, pred in [('Flat interpolation', d2['pred_flat']), ('Day-of-week disaggregation', d2['pred_disagg'])]:
        err = actual - pred
        mape = (err.abs() / actual).mean() * 100
        rmse = np.sqrt((err ** 2).mean())
        backtest_rows.append({'crop': crop, 'method': label, 'RMSE': round(rmse, 1),
                               'MAPE': round(mape, 2), 'N': len(actual)})
    print(f'  {crop:8s}: flat MAPE={backtest_rows[-2]["MAPE"]:.2f}%  '
          f'disagg MAPE={backtest_rows[-1]["MAPE"]:.2f}%  '
          f'(improvement: {backtest_rows[-2]["MAPE"]-backtest_rows[-1]["MAPE"]:.2f} pts)')

backtest_df = pd.DataFrame(backtest_rows)
backtest_df.to_csv(os.path.join(OUT_DIR, 'table_disagg_backtest.csv'), index=False)
print(f'  Saved: table_disagg_backtest.csv')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Apply to a forward forecast: smooth PCHIP through weekly horizon points,
#    multiply by day-of-week factor, add residual uncertainty band.
#    Demonstrated on one example market per crop using the production
#    dashboard's own ticker logic (baseline prediction, current market).
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Building example disaggregated daily forecast per crop ...')

import joblib, json
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
with open(os.path.join(MODEL_DIR, 'feature_columns.json'), encoding='utf-8') as f:
    feature_columns = json.load(f)
reference = pd.read_csv(os.path.join(MODEL_DIR, 'reference_rows.csv'), parse_dates=['week_start'])
HORIZONS = [1, 4, 13, 26]

def predict(crop, h, feature_row, models_cache={}):
    key = (crop, h)
    if key not in models_cache:
        models_cache[key] = joblib.load(os.path.join(MODEL_DIR, f'{crop}_{h}w.joblib'))
    cols = feature_columns[f'{crop}_{h}w']
    X = pd.DataFrame([{c: feature_row.get(c, 0) for c in cols}])
    return float(np.expm1(models_cache[key].predict(X)[0]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, crop in zip(axes, CROPS):
    crop_ref = reference[reference['crop'] == crop].dropna(subset=['log_price'])
    # Pick a market with a real (non-NaN) baseline price for a clean demo
    market = crop_ref.iloc[0]['market']
    base_row = crop_ref[crop_ref['market'] == market].iloc[0].to_dict()
    as_of = pd.Timestamp(base_row['week_start'])

    pts_date = [as_of] + [as_of + pd.Timedelta(weeks=h) for h in HORIZONS]
    pts_price = [base_row.get('log_price')]
    for h in HORIZONS:
        pts_price.append(np.log1p(predict(crop, h, base_row)))
    pts_num = [(d - as_of).days for d in pts_date]

    pchip = PchipInterpolator(pts_num, pts_price)
    daily_offsets = np.arange(0, HORIZONS[-1] * 7 + 1)
    daily_dates = [as_of + pd.Timedelta(days=int(o)) for o in daily_offsets]
    smooth_trend = np.expm1(pchip(daily_offsets))

    # Day-of-week correction dropped -- backtest confirmed it's negligible
    # and net-negative (see docstring). Uncertainty band uses the overall
    # daily residual std-dev (real, historically-observed day-to-day
    # variation), applied uniformly since it doesn't depend on weekday.
    overall_std = dow_table[dow_table['crop'] == crop]['factor_std'].mean()
    band = smooth_trend * overall_std

    ax.plot(daily_dates, smooth_trend, color=CROP_COLORS[crop], linewidth=1.6, label='Smooth daily trend')
    ax.fill_between(daily_dates, smooth_trend - band, smooth_trend + band, color=CROP_COLORS[crop], alpha=0.15,
                     label='Historical day-to-day noise (±1 std)')
    ax.scatter(pts_date, np.expm1(pts_price), color=CROP_COLORS[crop], zorder=5, s=40, label='Validated weekly forecast')
    ax.set_title(f'{crop.capitalize()} — {market}', fontsize=10)
    ax.set_ylabel('Price (Rs/quintal)')
    ax.tick_params(axis='x', rotation=30)
    if crop == 'tomato':
        ax.legend(fontsize=7, loc='upper left')

fig.suptitle('Weekly-to-Daily Disaggregation — Smooth Trend + Honest Daily Noise Band (not a validated daily forecast)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_disagg_example.png'), dpi=150)
print('  Saved: fig_disagg_example.png')

# Day-of-week pattern figure
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for crop in CROPS:
    sub = dow_table[dow_table['crop'] == crop].sort_values('dow')
    ax2.plot(dow_names, sub['factor_mean'], marker='o', color=CROP_COLORS[crop], label=crop.capitalize())
ax2.axhline(1.0, color='#999', linestyle=':', linewidth=1)
ax2.set_ylabel('Price relative to week average')
ax2.set_title('Day-of-Week Price Pattern (2017-2026)')
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig_dow_pattern.png'), dpi=150)
print('  Saved: fig_dow_pattern.png')

print('\nScript 26 complete.')
