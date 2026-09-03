# -*- coding: utf-8 -*-
"""
Script 50 -- MAPE vs. WAPE: Does the Accuracy Metric Choice Matter Here?
=============================================================================
Prompted by a direct question: `compute_metrics()` (Script 15, and every
downstream script that reports accuracy, including Script 47's per-market/
state figures) uses plain MAPE:

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

This is an UNWEIGHTED mean of per-row (market, week) percentage errors.
Two consequences: (1) every row counts equally regardless of that market/
week's actual traded value -- a thin, low-volume week weighs the same as a
major mandi's high-volume week; (2) a row with an unusually low true price
can produce a huge percentage error for a modest Rs miss, and that outlier
is averaged in at full weight.

WAPE (Weighted Absolute Percentage Error) uses the same per-row errors but
weighted by each row's own actual value instead of counted equally:

    wape = sum(|y_true - y_pred|) / sum(y_true) * 100

This script does NOT change any production metric -- it computes both on
the SAME already-existing backtest predictions (no retraining) to see
empirically how much they actually diverge for this data, at both the
crop-wide level (what model_uncertainty.json reports) and the per-market
level (what Script 47's hierarchical accuracy reports, the more
consequential place for thin-market noise to show up).

Input: Model_Output/dm_market_level_predictions.csv (M6 variant only --
same file Script 47 already uses for per-market accuracy).

Outputs:
  Model_Output/table_mape_vs_wape_cropwide.csv    per (crop, horizon) both metrics
  Model_Output/table_mape_vs_wape_market.csv       per (crop, market, horizon) both metrics
  Model_Output/fig_mape_vs_wape_market_scatter.png market-level MAPE vs WAPE, sized by n

Run: python scripts/50_MAPE_vs_WAPE_Comparison.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_FILE = os.path.join(BASE, 'Model_Output', 'dm_market_level_predictions.csv')
OUT_DIR = os.path.join(BASE, 'Model_Output')

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}

print('=' * 78)
print('SCRIPT 50: MAPE vs. WAPE COMPARISON (same predictions, two metrics)')
print('=' * 78)

print('\n[1] Loading M6 per-market backtest predictions ...')
df = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
df = df[df['variant'] == 'M6'].copy()
df = df[(df['y_true'] > 0) & np.isfinite(df['y_true']) & np.isfinite(df['y_pred'])]
df['abs_err'] = (df['y_true'] - df['y_pred']).abs()
df['ape'] = 100 * df['abs_err'] / df['y_true']
print(f'  {len(df):,} rows (crop x market x fold x horizon x week)')


def mape_of(g):
    return g['ape'].mean()


def wape_of(g):
    return 100 * g['abs_err'].sum() / g['y_true'].sum()


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROP-WIDE COMPARISON (what model_uncertainty.json reports)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Crop-wide comparison (pooled across every market/fold/week) ...\n')
rows = []
for (crop, h), g in df.groupby(['crop', 'horizon_weeks']):
    mape = mape_of(g)
    wape = wape_of(g)
    rows.append({'crop': crop, 'horizon_weeks': h, 'n': len(g),
                 'mape': round(mape, 2), 'wape': round(wape, 2),
                 'diff_pts': round(mape - wape, 2),
                 'accuracy_mape': round(max(0, 100 - mape), 1),
                 'accuracy_wape': round(max(0, 100 - wape), 1)})
cropwide = pd.DataFrame(rows).sort_values(['crop', 'horizon_weeks'])
cropwide_path = os.path.join(OUT_DIR, 'table_mape_vs_wape_cropwide.csv')
cropwide.to_csv(cropwide_path, index=False)
for _, r in cropwide.iterrows():
    print(f'  {r["crop"]:7s} h={int(r["horizon_weeks"]):>2}w  n={int(r["n"]):>7,}  '
          f'MAPE={r["mape"]:>6.2f}%  WAPE={r["wape"]:>6.2f}%  '
          f'(diff={r["diff_pts"]:>+6.2f}pts)  '
          f'"Accuracy": MAPE-based {r["accuracy_mape"]:>5.1f}% vs WAPE-based {r["accuracy_wape"]:>5.1f}%')
print(f'\n  Saved: {cropwide_path}')
print(f'  Mean |MAPE - WAPE| across all 12 cells: {cropwide["diff_pts"].abs().mean():.2f} percentage points')
print(f'  Max |MAPE - WAPE|: {cropwide["diff_pts"].abs().max():.2f} pts '
      f'({cropwide.loc[cropwide["diff_pts"].abs().idxmax(), "crop"]} '
      f'h={int(cropwide.loc[cropwide["diff_pts"].abs().idxmax(), "horizon_weeks"])}w)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET-LEVEL COMPARISON (what Script 47's per-market accuracy reports --
# the more consequential place for thin-market noise to show up, since a
# single market's backtest can be as few as ~10-240 weeks, not the tens of
# thousands of pooled rows the crop-wide figure above has)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Market-level comparison (per crop, market, horizon) ...')
mrows = []
for (crop, market_id, h), g in df.groupby(['crop', 'market_id', 'horizon_weeks']):
    if len(g) < 10:
        continue
    mape = mape_of(g)
    wape = wape_of(g)
    mrows.append({'crop': crop, 'market_id': market_id, 'market': g['market'].iloc[0],
                   'horizon_weeks': h, 'n': len(g), 'mape': round(mape, 2), 'wape': round(wape, 2),
                   'diff_pts': round(mape - wape, 2)})
market_df = pd.DataFrame(mrows)
market_path = os.path.join(OUT_DIR, 'table_mape_vs_wape_market.csv')
market_df.to_csv(market_path, index=False)
print(f'  {len(market_df):,} (crop, market, horizon) cells with n>=10 weeks')
print(f'  Saved: {market_path}')

print(f'\n  Mean |MAPE - WAPE| across all market cells: {market_df["diff_pts"].abs().mean():.2f} pts')
print(f'  Median |MAPE - WAPE|: {market_df["diff_pts"].abs().median():.2f} pts')
print(f'  90th percentile |MAPE - WAPE|: {market_df["diff_pts"].abs().quantile(0.9):.2f} pts')
print(f'  Correlation (MAPE, WAPE) across market cells: {market_df["mape"].corr(market_df["wape"]):.4f}')

print('\n  Top 15 market cells where MAPE and WAPE disagree most:')
top_diverge = market_df.reindex(market_df['diff_pts'].abs().sort_values(ascending=False).index).head(15)
print(top_diverge[['crop', 'market', 'horizon_weeks', 'n', 'mape', 'wape', 'diff_pts']].to_string(index=False))

# How many markets would look MEANINGFULLY different (>10 accuracy points)
# to a dashboard user depending on which metric is shown?
flip = market_df[market_df['diff_pts'].abs() >= 10]
print(f'\n  {len(flip):,} / {len(market_df):,} market cells ({100*len(flip)/len(market_df):.1f}%) '
      f'differ by >=10 accuracy points between MAPE and WAPE -- '
      f'a dashboard user would see a meaningfully different "This market" figure.')


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE -- market-level MAPE vs WAPE scatter, point size = backtest weeks
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Generating scatter figure ...')
fig, ax = plt.subplots(figsize=(7, 7))
for crop, color in CROP_COLORS.items():
    sub = market_df[market_df['crop'] == crop]
    ax.scatter(sub['mape'], sub['wape'], s=(sub['n'] / sub['n'].max() * 120 + 8),
               alpha=0.45, color=color, label=crop, edgecolors='none')
lims = [0, max(market_df['mape'].quantile(0.99), market_df['wape'].quantile(0.99))]
ax.plot(lims, lims, '--', color='#888', linewidth=1, label='MAPE = WAPE (no difference)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('MAPE (%)'); ax.set_ylabel('WAPE (%)')
ax.set_title('Per-market MAPE vs. WAPE (M6, same predictions)\nPoint size = backtested weeks; points below the line are markets\nwhere WAPE is more forgiving than MAPE',
             fontsize=10)
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_mape_vs_wape_market_scatter.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

print('\n' + '=' * 78)
print('Script 50 complete. No production metric changed -- comparison only.')
print('=' * 78)
