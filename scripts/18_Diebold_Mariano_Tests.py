# -*- coding: utf-8 -*-
"""
Script 18 — Diebold-Mariano Statistical Significance Tests
=============================================================
Converts the ablation study's MAPE/R² differences (Script 15) into formal
p-values, testing whether each added data layer (M0→M1→M2→M3→M4) produces a
statistically significant improvement in forecast accuracy — not just a
numerically larger metric.

Method: Diebold & Mariano (1995) test on squared-error loss differentials,
with the Harvey-Leybourne-Newbold (1997) small-sample correction (t-distribution
critical values instead of standard normal; degrees-of-freedom and variance
adjustment for finite T) and Newey-West-style autocovariance truncated at
lag h-1, since an h-step-ahead forecast error series is MA(h-1) under
correct model specification.

Test series: for each (crop, horizon), the 4 folds' test-period weeks
(2022, 2023, 2024, 2025 — non-overlapping) are concatenated in chronological
order into one continuous crop-level weekly time series, using the
crop-level average predictions Script 15 saves per variant.

Comparisons:
  Headline:      M0 (price only) vs M4 (full pipeline)
  Layer-by-layer: M0 vs M1, M1 vs M2, M2 vs M3, M3 vs M4 — isolates whether
                   each individual data layer's addition is significant

Inputs:
  Model_Output/ablation_predictions.csv   (crop-level weekly y_true/y_pred
                                            per variant × crop × fold × horizon,
                                            produced by Script 15)

Outputs (Model_Output/):
  table_diebold_mariano.csv   DM stat, p-value, significance per comparison
  fig_dm_pvalues.png          heatmap of -log10(p) for headline + layer tests

Run: python scripts/18_Diebold_Mariano_Tests.py
Estimated runtime: <1 minute (no model fitting — pure statistical test on
already-computed predictions)
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE      = r'C:\Users\masro\Documents\TOP_Digital_Twin'
PRED_FILE = os.path.join(BASE, 'Model_Output', 'ablation_predictions.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
ALPHA    = 0.05

HEADLINE_PAIR = ('M0', 'M4')
LAYER_PAIRS   = [('M0', 'M1'), ('M1', 'M2'), ('M2', 'M3'), ('M3', 'M4')]
ALL_PAIRS     = [HEADLINE_PAIR] + LAYER_PAIRS

VARIANT_LABELS = {
    'M0': 'Price only', 'M1': '+ Arrivals', 'M2': '+ Macro',
    'M3': '+ Climate', 'M4': '+ Satellite',
}

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIEBOLD-MARIANO TEST
# ─────────────────────────────────────────────────────────────────────────────
def diebold_mariano_test(y_true, pred_baseline, pred_richer, h):
    """
    DM test comparing squared-error loss of pred_baseline vs pred_richer.

    d_t = L(e_baseline_t) - L(e_richer_t), loss = squared error.
    mean_d > 0  ⇒  richer model has LOWER average loss (more accurate).

    h-step forecast errors are MA(h-1) under correct specification, so the
    long-run variance of mean(d) sums autocovariances up to lag h-1
    (Newey-West / Bartlett-style, unweighted as in the original DM paper).
    Harvey-Leybourne-Newbold (1997) small-sample correction is applied:
    a finite-sample variance adjustment plus a Student-t reference
    distribution (df = T-1) instead of the asymptotic standard normal.

    Returns dict with DM_stat, p_value, mean_d, n. NaN fields if T is too
    small or the long-run variance estimate is non-positive.
    """
    e_baseline = y_true - pred_baseline
    e_richer   = y_true - pred_richer
    d = e_baseline ** 2 - e_richer ** 2
    T = len(d)

    if T < 10:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=np.nan, n=T)

    d_mean = d.mean()
    max_lag = max(h - 1, 0)
    gamma0  = np.mean((d - d_mean) ** 2)
    var_d   = gamma0
    for k in range(1, min(max_lag, T - 1) + 1):
        cov_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
        var_d += 2 * cov_k

    var_mean_d = var_d / T
    if not np.isfinite(var_mean_d) or var_mean_d <= 0:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=round(d_mean, 6), n=T)

    dm_raw = d_mean / np.sqrt(var_mean_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    if not np.isfinite(hln_factor) or hln_factor <= 0:
        dm_adj = dm_raw
    else:
        dm_adj = dm_raw * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_adj), df=max(T - 1, 1)))

    return dict(DM_stat=round(float(dm_adj), 4), p_value=round(float(p_value), 4),
                mean_d=round(float(d_mean), 6), n=T)


def sig_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 65)
print('SCRIPT 18: DIEBOLD-MARIANO STATISTICAL TESTS')
print('=' * 65)

if not os.path.exists(PRED_FILE):
    print(f'ERROR: {PRED_FILE} not found.')
    print('Run scripts/15_Ablation_Study_M0_M4.py first (it now saves')
    print('crop-level weekly predictions needed for this test).')
    sys.exit(1)

preds = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
preds = preds.sort_values(['crop', 'horizon_weeks', 'variant', 'fold', 'week_start'])
print(f'  Loaded: {len(preds):,} rows')
print(f'  Variants: {sorted(preds["variant"].unique())}')
print(f'  Crops: {sorted(preds["crop"].unique())}')
print(f'  Horizons: {sorted(preds["horizon_weeks"].unique())}\n')


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN DM TESTS
# ─────────────────────────────────────────────────────────────────────────────
print('[1] Running DM tests: headline (M0 vs M4) + layer-by-layer ...\n')

rows = []
for crop in CROPS:
    for h in HORIZONS:
        sub = preds[(preds['crop'] == crop) & (preds['horizon_weeks'] == h)]
        if sub.empty:
            continue

        # Concatenate all 4 folds chronologically per variant
        series = {}
        for variant in ['M0', 'M1', 'M2', 'M3', 'M4']:
            vsub = (sub[sub['variant'] == variant]
                    .sort_values(['fold', 'week_start'])
                    .drop_duplicates(subset='week_start'))
            series[variant] = vsub.set_index('week_start')[['y_true', 'y_pred']]

        for baseline, richer in ALL_PAIRS:
            if baseline not in series or richer not in series:
                continue
            a, b = series[baseline], series[richer]
            merged = a.join(b, how='inner', lsuffix='_a', rsuffix='_b')
            if len(merged) < 10:
                continue

            y_true = merged['y_true_a'].values  # identical to y_true_b by construction
            result = diebold_mariano_test(
                y_true, merged['y_pred_a'].values, merged['y_pred_b'].values, h)

            better = 'richer' if (not pd.isna(result['mean_d']) and result['mean_d'] > 0) \
                     else ('baseline' if not pd.isna(result['mean_d']) else 'n/a')

            row = {
                'crop': crop, 'horizon_weeks': h,
                'baseline': baseline, 'richer': richer,
                'comparison': f'{baseline} vs {richer}',
                'is_headline': (baseline, richer) == HEADLINE_PAIR,
                **result,
                'better_model': better,
                'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < ALPHA,
            }
            rows.append(row)

            sig = sig_stars(result['p_value'])
            print(f'  {crop:7s} h={h:>2}w  {baseline} vs {richer:2s} | '
                  f'DM={result["DM_stat"]:>7.3f}  p={result["p_value"]:.4f}{sig:<4s} '
                  f'n={result["n"]:>3}  mean_d={result["mean_d"]:>10.2f}')

dm_results = pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE TABLE
# ─────────────────────────────────────────────────────────────────────────────
table_path = os.path.join(OUT_DIR, 'table_diebold_mariano.csv')
dm_results.to_csv(table_path, index=False)
print(f'\n[2] Saved: {table_path}  ({len(dm_results)} rows)')

n_sig = dm_results['significant_5pct'].sum()
n_total = len(dm_results)
print(f'\n  {n_sig}/{n_total} comparisons significant at p<0.05')

headline = dm_results[dm_results['is_headline']]
n_sig_headline = headline['significant_5pct'].sum()
print(f'  Headline (M0 vs M4): {n_sig_headline}/{len(headline)} significant')
for _, r in headline.iterrows():
    sig = sig_stars(r['p_value'])
    direction = 'M4 better' if r['better_model'] == 'richer' else 'M0 better' if r['better_model'] == 'baseline' else 'n/a'
    print(f'    {r["crop"]:7s} h={r["horizon_weeks"]:>2}w: p={r["p_value"]:.4f}{sig:<4s} ({direction})')


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE — heatmap of -log10(p) for headline comparison
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Generating figure ...')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: headline M0 vs M4, crops x horizons
pivot_headline = headline.pivot(index='crop', columns='horizon_weeks', values='p_value')
pivot_headline = pivot_headline.reindex(index=CROPS, columns=HORIZONS)
neglog_p = -np.log10(pivot_headline.clip(lower=1e-10))

ax = axes[0]
im = ax.imshow(neglog_p.values, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels([f'{h}w' for h in HORIZONS])
ax.set_yticks(range(len(CROPS)))
ax.set_yticklabels([c.capitalize() for c in CROPS])
ax.set_title('Headline: M0 vs M4\n(−log₁₀ p-value; darker green = more significant)',
             fontsize=10, fontweight='bold')
for i in range(len(CROPS)):
    for j in range(len(HORIZONS)):
        val = pivot_headline.values[i, j]
        if not np.isnan(val):
            stars = sig_stars(val)
            ax.text(j, i, f'p={val:.3f}\n{stars}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='−log₁₀(p)', fraction=0.046, pad=0.04)

# Panel B: layer-by-layer, averaged across crops
ax = axes[1]
layer_labels = [f'{a}→{b}' for a, b in LAYER_PAIRS]
layer_data = np.full((len(CROPS), len(LAYER_PAIRS)), np.nan)
for i, crop in enumerate(CROPS):
    for j, (a, b) in enumerate(LAYER_PAIRS):
        sub = dm_results[(dm_results['crop'] == crop) &
                         (dm_results['baseline'] == a) & (dm_results['richer'] == b)]
        if not sub.empty:
            layer_data[i, j] = -np.log10(np.clip(sub['p_value'].mean(), 1e-10, 1))

im2 = ax.imshow(layer_data, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
ax.set_xticks(range(len(LAYER_PAIRS)))
ax.set_xticklabels(layer_labels)
ax.set_yticks(range(len(CROPS)))
ax.set_yticklabels([c.capitalize() for c in CROPS])
ax.set_title('Layer-by-layer significance\n(mean −log₁₀ p across horizons)',
             fontsize=10, fontweight='bold')
for i in range(len(CROPS)):
    for j in range(len(LAYER_PAIRS)):
        val = layer_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im2, ax=ax, label='mean −log₁₀(p)', fraction=0.046, pad=0.04)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_dm_pvalues.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

print('\n' + '=' * 65)
print('Script 18 complete.')
print(f'\nKey outputs:')
for fname in ['table_diebold_mariano.csv', 'fig_dm_pvalues.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<35} {os.path.getsize(fpath)/1024:>7.1f} KB')
