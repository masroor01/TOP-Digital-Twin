# -*- coding: utf-8 -*-
"""
Script 37 - Diebold-Mariano Test: M6 (production) vs Phase-1-alone (longhistory)
====================================================================================
Ad-hoc follow-up to the two-phase architecture evaluation. Phase 1 (Script 35)
was built as a COMPONENT of the (rejected) two-phase residual stack, but its
own MAPE numbers beat M6 on several crop/horizon cells -- notably onion, all
four horizons. This was never a designed comparison, so this script tests it
formally on the only window where both models have predictions: the 4
test years shared by M6's rolling-origin folds (2022-2025) and Phase 1's
folds 5-8 (same years, different fold indexing since Phase 1 uses 9 annual
expanding folds starting fold 0 = 2017).

Method: identical DM test (Harvey-Leybourne-Newbold small-sample correction)
as Script 18, reused directly for methodological consistency.

Inputs:
  Model_Output/ablation_predictions.csv                  (M6 crop-level, expm1 scale)
  Model_Output/table_baseline_phase_oof_predictions.csv   (Phase 1 market-level, log1p scale)

Output:
  Model_Output/table_dm_m6_vs_phase1alone.csv

Run: python scripts/37_DM_Test_M6_vs_Phase1Alone.py
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'Model_Output')

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
ALPHA = 0.05


def diebold_mariano_test(y_true, pred_baseline, pred_richer, h):
    """Same implementation as Script 18 -- squared-error loss, HLN correction."""
    e_baseline = y_true - pred_baseline
    e_richer = y_true - pred_richer
    d = e_baseline ** 2 - e_richer ** 2
    T = len(d)
    if T < 10:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=np.nan, n=T)
    d_mean = d.mean()
    max_lag = max(h - 1, 0)
    gamma0 = np.mean((d - d_mean) ** 2)
    var_d = gamma0
    for k in range(1, min(max_lag, T - 1) + 1):
        cov_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
        var_d += 2 * cov_k
    var_mean_d = var_d / T
    if not np.isfinite(var_mean_d) or var_mean_d <= 0:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=round(d_mean, 6), n=T)
    dm_raw = d_mean / np.sqrt(var_mean_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_adj = dm_raw * hln_factor if np.isfinite(hln_factor) and hln_factor > 0 else dm_raw
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


print('=' * 70)
print('SCRIPT 37: DM TEST -- M6 (production) vs PHASE-1-ALONE (longhistory)')
print('=' * 70)

# ── Load M6 (production, already crop-level, expm1/price scale) ──
m6 = pd.read_csv(os.path.join(OUT_DIR, 'ablation_predictions.csv'), parse_dates=['week_start'])
m6 = m6[m6['variant'] == 'M6'][['crop', 'horizon_weeks', 'week_start', 'y_true', 'y_pred']]
m6 = m6.drop_duplicates(subset=['crop', 'horizon_weeks', 'week_start'])
print(f'\n[1] M6 loaded: {len(m6):,} crop-week rows, {m6["week_start"].min().date()} to {m6["week_start"].max().date()}')

# ── Load Phase 1 (market-level, log1p scale) -> aggregate to crop-level price scale ──
p1_raw = pd.read_csv(os.path.join(OUT_DIR, 'experiments', 'two_phase', 'table_baseline_phase_oof_predictions.csv'),
                      parse_dates=['week_start'],
                      usecols=['crop', 'week_start', 'horizon_weeks', 'log_price_actual', 'log_price_baseline_pred'])
p1_raw['y_true'] = np.expm1(p1_raw['log_price_actual'])
p1_raw['y_pred'] = np.expm1(p1_raw['log_price_baseline_pred'])
p1 = (p1_raw.groupby(['crop', 'horizon_weeks', 'week_start'])
      .agg(y_true=('y_true', 'mean'), y_pred=('y_pred', 'mean'))
      .reset_index())
print(f'[2] Phase 1 aggregated to crop-level: {len(p1):,} crop-week rows, '
      f'{p1["week_start"].min().date()} to {p1["week_start"].max().date()}')

# ── Run DM test per crop x horizon, restricted to overlapping weeks ──
print('\n[3] Running DM tests (M6 = baseline, Phase1-alone = richer) ...\n')
rows = []
for crop in CROPS:
    for h in HORIZONS:
        a = (m6[(m6['crop'] == crop) & (m6['horizon_weeks'] == h)]
             .set_index('week_start')[['y_true', 'y_pred']])
        b = (p1[(p1['crop'] == crop) & (p1['horizon_weeks'] == h)]
             .set_index('week_start')[['y_true', 'y_pred']])
        merged = a.join(b, how='inner', lsuffix='_m6', rsuffix='_p1')
        if len(merged) < 10:
            print(f'  {crop:7s} h={h:>2}w  SKIPPED (only {len(merged)} overlapping weeks)')
            continue

        y_true = merged['y_true_m6'].values
        result = diebold_mariano_test(y_true, merged['y_pred_m6'].values, merged['y_pred_p1'].values, h)
        better = 'Phase1-alone' if (not pd.isna(result['mean_d']) and result['mean_d'] > 0) \
            else ('M6' if not pd.isna(result['mean_d']) else 'n/a')

        row = {'crop': crop, 'horizon_weeks': h, **result, 'better_model': better,
               'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < ALPHA}
        rows.append(row)

        sig = sig_stars(result['p_value'])
        print(f'  {crop:7s} h={h:>2}w  DM={result["DM_stat"]:>7.3f}  p={result["p_value"]:.4f}{sig:<4s} '
              f'n={result["n"]:>3}  better={better}')

dm_results = pd.DataFrame(rows)
table_path = os.path.join(OUT_DIR, 'table_dm_m6_vs_phase1alone.csv')
dm_results.to_csv(table_path, index=False)

n_sig = dm_results['significant_5pct'].sum()
print(f'\n[4] Saved: {table_path}')
print(f'\n  {n_sig}/{len(dm_results)} comparisons significant at p<0.05')
n_p1_wins = ((dm_results['better_model'] == 'Phase1-alone') & dm_results['significant_5pct']).sum()
n_m6_wins = ((dm_results['better_model'] == 'M6') & dm_results['significant_5pct']).sum()
print(f'  Significant Phase1-alone wins: {n_p1_wins}')
print(f'  Significant M6 wins: {n_m6_wins}')

print('\n' + '=' * 70)
print('Script 37 complete.')
