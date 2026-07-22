# -*- coding: utf-8 -*-
"""
Script 18b — Market-Level Diebold-Mariano Check (diagnostic)
================================================================
Script 18 ran the DM test on crop-level weekly-averaged predictions and
found the M0 (price-only) vs M4 (full pipeline) headline comparison was
significant in only 3/12 crop×horizon combinations — and in 2 of those 3,
M0 was significantly *better* than M4.

This script checks whether crop-level averaging was hiding a real
per-market effect: it runs the same DM test independently for every
individual market (using Script 15's MARKET_LEVEL_DIAGNOSTIC output, which
retrained just M0 and M4 on the full market panel and saved per-market
predictions instead of crop-averaged ones), then summarizes how many
markets show a significant M4 improvement vs how many show M0 better.

If the crop-level result reflects a genuine lack of signal, the per-market
results should look similarly mixed/insignificant. If crop-level averaging
was washing out a real effect, a much larger share of individual markets
should show significant, consistently-directed results than the 3/12
crop-level headline suggested.

Input:
  Model_Output/dm_market_level_predictions.csv   (from Script 15 with
                                                   MARKET_LEVEL_DIAGNOSTIC=True)

Outputs (Model_Output/):
  table_dm_market_level_detail.csv    per-market DM stat/p-value
  table_dm_market_level_summary.csv   per crop×horizon: % significant,
                                       direction breakdown, Fisher combined p

Run: python scripts/18b_Market_Level_DM_Check.py
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE      = r'C:\Users\masro\Documents\TOP_Digital_Twin'
PRED_FILE = os.path.join(BASE, 'Model_Output', 'dm_market_level_predictions.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
ALPHA    = 0.05
MIN_OBS  = 30   # minimum concatenated weeks needed to trust a per-market DM test


def diebold_mariano_test(y_true, pred_baseline, pred_richer, h):
    """Same DM test as Script 18 — squared-error loss, HLN small-sample
    correction, MA(h-1) autocovariance. mean_d > 0 => richer model better."""
    e_baseline = y_true - pred_baseline
    e_richer   = y_true - pred_richer
    d = e_baseline ** 2 - e_richer ** 2
    T = len(d)
    if T < MIN_OBS:
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
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=round(float(d_mean), 6), n=T)

    dm_raw = d_mean / np.sqrt(var_mean_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_adj = dm_raw * hln_factor if (np.isfinite(hln_factor) and hln_factor > 0) else dm_raw
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_adj), df=max(T - 1, 1)))

    return dict(DM_stat=round(float(dm_adj), 4), p_value=round(float(p_value), 4),
                mean_d=round(float(d_mean), 6), n=T)


def fisher_combined_p(pvals):
    """Fisher's method: combine independent p-values into one meta p-value.
    Markets aren't fully independent (shared macro/climate shocks), so this
    is directional evidence, not a rigorous joint test."""
    pvals = np.array([p for p in pvals if not pd.isna(p) and p > 0])
    if len(pvals) < 2:
        return np.nan
    chi2_stat = -2 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1 - stats.chi2.cdf(chi2_stat, df))


print('=' * 65)
print('SCRIPT 18b: MARKET-LEVEL DIEBOLD-MARIANO CHECK')
print('=' * 65)

if not os.path.exists(PRED_FILE):
    print(f'ERROR: {PRED_FILE} not found.')
    print('Run scripts/15_Ablation_Study_M0_M4.py with MARKET_LEVEL_DIAGNOSTIC=True first.')
    sys.exit(1)

preds = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
print(f'  Loaded: {len(preds):,} rows')
print(f'  Markets: {preds["market"].nunique()}  Variants: {sorted(preds["variant"].unique())}\n')

detail_rows = []
for crop in CROPS:
    csub = preds[preds['crop'] == crop]
    markets = sorted(csub['market'].unique())
    for h in HORIZONS:
        hsub = csub[csub['horizon_weeks'] == h]
        if hsub.empty:
            continue
        for market in markets:
            msub = hsub[hsub['market'] == market]
            m0 = (msub[msub['variant'] == 'M0']
                  .sort_values(['fold', 'week_start'])
                  .drop_duplicates(subset='week_start')
                  .set_index('week_start')[['y_true', 'y_pred']])
            m4 = (msub[msub['variant'] == 'M4']
                  .sort_values(['fold', 'week_start'])
                  .drop_duplicates(subset='week_start')
                  .set_index('week_start')[['y_true', 'y_pred']])
            merged = m0.join(m4, how='inner', lsuffix='_m0', rsuffix='_m4')
            if len(merged) < MIN_OBS:
                continue

            result = diebold_mariano_test(
                merged['y_true_m0'].values, merged['y_pred_m0'].values,
                merged['y_pred_m4'].values, h)
            better = ('M4' if (not pd.isna(result['mean_d']) and result['mean_d'] > 0)
                      else ('M0' if not pd.isna(result['mean_d']) else 'n/a'))
            detail_rows.append({
                'crop': crop, 'horizon_weeks': h, 'market': market,
                **result, 'better_model': better,
                'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < ALPHA,
            })

detail = pd.DataFrame(detail_rows)
detail_path = os.path.join(OUT_DIR, 'table_dm_market_level_detail.csv')
detail.to_csv(detail_path, index=False)
print(f'[1] Saved: {detail_path}  ({len(detail):,} market-level tests)')

print('\n[2] Summary: % of markets with significant M0-vs-M4 difference\n')
summary_rows = []
for crop in CROPS:
    for h in HORIZONS:
        sub = detail[(detail['crop'] == crop) & (detail['horizon_weeks'] == h)]
        if sub.empty:
            continue
        n_tested = len(sub)
        n_sig = sub['significant_5pct'].sum()
        n_sig_m4_better = ((sub['significant_5pct']) & (sub['better_model'] == 'M4')).sum()
        n_sig_m0_better = ((sub['significant_5pct']) & (sub['better_model'] == 'M0')).sum()
        combined_p = fisher_combined_p(sub['p_value'].values)

        summary_rows.append({
            'crop': crop, 'horizon_weeks': h,
            'n_markets_tested': n_tested,
            'pct_significant': round(100 * n_sig / n_tested, 1) if n_tested else np.nan,
            'n_sig_M4_better': n_sig_m4_better,
            'n_sig_M0_better': n_sig_m0_better,
            'fisher_combined_p': round(combined_p, 6) if not pd.isna(combined_p) else np.nan,
        })

        combined_p_str = f'{combined_p:.2e}' if not pd.isna(combined_p) else 'N/A'
        print(f'  {crop:7s} h={h:>2}w | {n_tested:>4} markets tested | '
              f'{n_sig:>4} sig ({100*n_sig/n_tested:.1f}%) | '
              f'M4 better: {n_sig_m4_better:>3}  M0 better: {n_sig_m0_better:>3} | '
              f'Fisher combined p={combined_p_str}')

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'table_dm_market_level_summary.csv')
summary.to_csv(summary_path, index=False)
print(f'\n[3] Saved: {summary_path}')

print('\n' + '=' * 65)
print('Script 18b complete.')
print('\nInterpretation guide:')
print('  - If pct_significant is still low (~5-10%, matching the false-')
print('    positive rate expected at alpha=0.05) and M4/M0-better counts')
print('    are roughly balanced, the crop-level finding (weak/no signal)')
print('    holds up: M4 genuinely does not reliably beat M0.')
print('  - If a large majority of significant markets favor M4, crop-level')
print('    averaging was hiding a real effect — worth revisiting the')
print('    ablation methodology or reporting market-level results instead.')
