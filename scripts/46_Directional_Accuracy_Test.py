"""
Script 46 -- Directional Accuracy Test (all 3 crops, all 4 horizons)
=============================================================================
Every evaluation metric used elsewhere in this project (RMSE, MAE, MAPE, R^2,
MASE) is magnitude-based: how far off was the predicted PRICE LEVEL. None of
them ask the question a policy user actually cares about day to day: did the
model correctly call the DIRECTION of the move (price up vs. down from where
it stood when the forecast was made)? This script closes that gap.

Primary analysis: per-market, market_id-keyed (not market-name -- see the
2026-08-14 collision-bug entry in MANIFEST.md), using
dm_market_level_predictions.csv (Script 15's MARKET_LEVEL_DIAGNOSTIC output,
already the granular ground truth Script 18b's DM tests are built on). Only
M0 and M6 exist in that file, which is exactly the headline ablation
comparison anyway.

For each (variant, crop, horizon, market, fold, target week) row we have the
target actual price (y_true) and the model's predicted price (y_pred), but
NOT the price at the moment the forecast was made (the "origin"). That has
to be looked up separately: origin_date = target_week - horizon weeks, price
from the raw weekly panel (data/agmarknet_weekly/top_weekly_panel.csv), keyed
on (crop, market_id, week_start) same as every other script in this project.

  actual_change    = y_true - origin_price
  predicted_change = y_pred - origin_price
  correct           = sign(actual_change) == sign(predicted_change)

Ties (predicted_change == 0, i.e. the model predicted exactly no move) are
counted and reported separately, not silently folded into "wrong" -- they are
genuinely a different kind of outcome (a punt, not a wrong call).

Secondary context only (NOT the primary result, much coarser): B1_Naive's
directional accuracy, computed from ablation_predictions.csv's crop-level
weekly-mean predictions. Naive's forecast IS the origin price by
construction (Script 15 line ~695: "today's price repeated forward"), so its
predicted_change is identically 0 for every row -- it never calls a
direction. This is reported as a sanity-check baseline, explicitly labeled
crop-level (not market-level) so it is never mistaken for an apples-to-apples
comparison with the M0/M6 numbers above it.

Also runs a two-sided binomial test against a 50% null (pure coin-flip) for
every (variant, crop, horizon) cell, since "62% directional accuracy" is only
meaningful once you know whether 62% is actually distinguishable from chance
at that sample size.

Outputs:
  Model_Output/table_directional_accuracy.csv        -- per (variant, crop, horizon) summary
  Model_Output/table_directional_accuracy_naive.csv   -- naive baseline, crop-level, context only
  Model_Output/fig_directional_accuracy.png           -- grouped bar chart, M0 vs M6 vs 50% line
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'Model_Output')
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
MARKET_PRED_FILE = os.path.join(OUT_DIR, 'dm_market_level_predictions.csv')
ABLATION_PRED_FILE = os.path.join(OUT_DIR, 'ablation_predictions.csv')

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]

print('=================================================================')
print('SCRIPT 46: DIRECTIONAL ACCURACY TEST (all 3 crops, all 4 horizons)')
print('=================================================================')

# ---------------------------------------------------------------------------
# [1] Load per-market predictions (M0, M6) and the raw panel for origin prices
# ---------------------------------------------------------------------------
print('\n[1] Loading per-market predictions and raw weekly panel ...')
preds = pd.read_csv(MARKET_PRED_FILE, parse_dates=['week_start'])
print(f'    {len(preds):,} rows: {sorted(preds["variant"].unique())} x {sorted(preds["crop"].unique())} x h={sorted(preds["horizon_weeks"].unique())}')

panel = pd.read_csv(PANEL_FILE, usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted'],
                     parse_dates=['week_start'])
panel = panel.dropna(subset=['market_id'])
panel['market_id'] = panel['market_id'].astype(int)
preds['market_id'] = preds['market_id'].astype(int)

# ---------------------------------------------------------------------------
# [2] Look up the origin price (price at the week the forecast was actually
#     made, i.e. horizon weeks before the target week) for every prediction row
# ---------------------------------------------------------------------------
print('\n[2] Computing origin dates and looking up origin prices ...')
preds['origin_date'] = preds['week_start'] - pd.to_timedelta(preds['horizon_weeks'] * 7, unit='D')

panel_idx = panel.set_index(['crop', 'market_id', 'week_start'])['modal_price_weighted']
key = pd.MultiIndex.from_frame(preds[['crop', 'market_id', 'origin_date']].rename(columns={'origin_date': 'week_start'}))
preds['origin_price'] = panel_idx.reindex(key).values

n_before = len(preds)
preds = preds.dropna(subset=['origin_price'])
print(f'    matched origin price for {len(preds):,} / {n_before:,} rows '
      f'({100 * len(preds) / n_before:.1f}%) -- unmatched rows (origin date outside panel coverage) dropped')

# ---------------------------------------------------------------------------
# [3] Direction of actual vs. predicted change relative to the origin price
# ---------------------------------------------------------------------------
print('\n[3] Computing directional calls ...')
preds['actual_change'] = preds['y_true'] - preds['origin_price']
preds['predicted_change'] = preds['y_pred'] - preds['origin_price']
preds['actual_dir'] = np.sign(preds['actual_change'])
preds['predicted_dir'] = np.sign(preds['predicted_change'])
preds['is_tie'] = preds['predicted_dir'] == 0
preds['correct'] = (preds['actual_dir'] == preds['predicted_dir']) & ~preds['is_tie']

# ---------------------------------------------------------------------------
# [4] Aggregate per (variant, crop, horizon) with a binomial test vs. 50%
# ---------------------------------------------------------------------------
print('\n[4] Aggregating and running binomial tests vs. 50% null ...')
rows = []
for (variant, crop, h), g in preds.groupby(['variant', 'crop', 'horizon_weeks']):
    n_total = len(g)
    n_ties = int(g['is_tie'].sum())
    n_scored = n_total - n_ties          # denominator excludes ties (no directional call made)
    n_correct = int(g['correct'].sum())
    acc = n_correct / n_scored if n_scored else np.nan
    bt = binomtest(n_correct, n_scored, p=0.5, alternative='two-sided') if n_scored else None
    rows.append({
        'variant': variant, 'crop': crop, 'horizon_weeks': h,
        'n_total': n_total, 'n_ties': n_ties, 'n_scored': n_scored, 'n_correct': n_correct,
        'directional_accuracy_pct': round(100 * acc, 1) if n_scored else np.nan,
        'p_value_vs_50pct': bt.pvalue if bt else np.nan,
        'significant_at_05': bool(bt.pvalue < 0.05) if bt else False,
    })
summary = pd.DataFrame(rows).sort_values(['crop', 'horizon_weeks', 'variant'])
summary_path = os.path.join(OUT_DIR, 'table_directional_accuracy.csv')
summary.to_csv(summary_path, index=False, encoding='utf-8')
print(f'    wrote {summary_path}')
print()
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# [5] Naive baseline (context only, crop-level -- see module docstring)
# ---------------------------------------------------------------------------
print('\n[5] Naive baseline (crop-level, context only) ...')
abl = pd.read_csv(ABLATION_PRED_FILE)
naive = abl[abl['variant'] == 'B1_Naive'].copy()
# Naive's own predicted price IS the origin price by construction (Script 15:
# "today's price repeated forward") -- so predicted_change is identically 0
# for every row and the "prediction" is always exactly "no move".
naive['actual_change'] = naive['y_true'] - naive['y_pred']
naive['actual_dir'] = np.sign(naive['actual_change'])
naive['is_tie'] = True   # predicted_dir is always 0 (flat) by definition

naive_rows = []
for (crop, h), g in naive.groupby(['crop', 'horizon_weeks']):
    n_total = len(g)
    n_flat_actual = int((g['actual_dir'] == 0).sum())
    naive_rows.append({
        'crop': crop, 'horizon_weeks': h, 'n_weeks': n_total,
        'n_weeks_actual_flat': n_flat_actual,
        'directional_accuracy_pct': 0.0 if n_total > n_flat_actual else np.nan,
        'note': 'Naive always predicts "no change" by construction -- it never calls a direction, so its '
                'directional accuracy is ~0% whenever the actual price moves at all. Crop-level weekly-mean '
                'data (coarser than the M0/M6 per-market numbers above), reported as context only.',
    })
naive_summary = pd.DataFrame(naive_rows).sort_values(['crop', 'horizon_weeks'])
naive_path = os.path.join(OUT_DIR, 'table_directional_accuracy_naive.csv')
naive_summary.to_csv(naive_path, index=False, encoding='utf-8')
print(f'    wrote {naive_path}')
print()
print(naive_summary.drop(columns=['note']).to_string(index=False))

# ---------------------------------------------------------------------------
# [6] Figure: grouped bars, M0 vs M6 vs 50% reference line, one panel per crop
# ---------------------------------------------------------------------------
print('\n[6] Building figure ...')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
colors = {'M0': '#94A3B8', 'M6': '#1B4332'}
for ax, crop in zip(axes, CROPS):
    sub = summary[summary['crop'] == crop]
    x = np.arange(len(HORIZONS))
    width = 0.35
    for i, variant in enumerate(['M0', 'M6']):
        vals = [sub[(sub['horizon_weeks'] == h) & (sub['variant'] == variant)]['directional_accuracy_pct'].values for h in HORIZONS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=variant, color=colors[variant])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.0f}%', ha='center', fontsize=8)
    ax.axhline(50, color='#B45309', linestyle='--', linewidth=1, label='50% (coin flip)' if crop == CROPS[0] else None)
    ax.set_title(crop.capitalize(), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}W' for h in HORIZONS])
    ax.set_ylim(0, 100)
    if crop == CROPS[0]:
        ax.set_ylabel('Directional Accuracy (%)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
fig.suptitle('Directional Accuracy: M0 (price-only) vs. M6 (full model), per-market', y=1.14, fontweight='bold')
fig_path = os.path.join(OUT_DIR, 'fig_directional_accuracy.png')
fig.tight_layout()
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'    wrote {fig_path}')

print('\n=================================================================')
print('DONE.')
print('=================================================================')
