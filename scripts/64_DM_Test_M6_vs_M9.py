# -*- coding: utf-8 -*-
"""
Script 64 — Market-Level Diebold-Mariano Test: M6 vs M9 (NOAA ONI)
================================================================
Formal significance test of the M9 (NOAA ONI) layer added via Scripts
56/62, mirroring Script 55/60's exact methodology but for M6 (pre-ONI
full model) vs M9 (M6 + ONI anomaly + 3/4-month lags) instead of M6 vs
M7/M8.

Unlike M7 (and possibly unlike M8, pending the detrending follow-up),
ONI's pre-test diagnostic (Script 63) was genuinely encouraging rather
than a caution flag: real split usage in the top quarter of ~81 features,
median split counts 196-276 (higher than either M7's drought or M8's
fertilizer got), and no meaningful correlation (|r|<0.3) with any
existing macro/WPI/trend feature -- unlike fertilizer's 0.6-0.89
collinearity with year_trend. ONI also has no fold-1-predates-the-data
problem: continuous monthly coverage back to 1950, decades before any CV
fold's training window starts.

Input:
  Model_Output/dm_market_level_predictions.csv   (from Script 15 with
                                                   MARKET_LEVEL_DIAGNOSTIC=True,
                                                   DIAGNOSTIC_PAIR=('M6','M9'))

Outputs (Model_Output/):
  table_dm_m6_vs_m9_detail.csv    per-market DM stat/p-value
  table_dm_m6_vs_m9_summary.csv   per crop×horizon: % significant,
                                   direction breakdown, Fisher combined p

Run: python scripts/64_DM_Test_M6_vs_M9.py
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_FILE = os.path.join(BASE, 'Model_Output', 'dm_market_level_predictions.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
ALPHA    = 0.05
MIN_OBS  = 30
BASELINE, RICHER = 'M6', 'M9'   # must match Script 15's DIAGNOSTIC_PAIR


def diebold_mariano_test(y_true, pred_baseline, pred_richer, h):
    """Identical to Script 18b/55/60's DM test -- squared-error loss, HLN
    small-sample correction, MA(h-1) autocovariance. mean_d > 0 => richer
    model (M9) better."""
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
    Markets aren't fully independent (shared macro/climate shocks), so
    this is directional evidence, not a rigorous joint test."""
    pvals = np.array([p for p in pvals if not pd.isna(p) and p > 0])
    if len(pvals) < 2:
        return np.nan
    chi2_stat = -2 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1 - stats.chi2.cdf(chi2_stat, df))


print('=' * 65)
print('SCRIPT 64: MARKET-LEVEL DIEBOLD-MARIANO TEST -- M6 vs M9 (NOAA ONI)')
print('=' * 65)

if not os.path.exists(PRED_FILE):
    print(f'ERROR: {PRED_FILE} not found.')
    print('Run scripts/15_Ablation_Study_M0_M4.py with MARKET_LEVEL_DIAGNOSTIC=True '
          f'and DIAGNOSTIC_PAIR=({BASELINE!r}, {RICHER!r}) first.')
    sys.exit(1)

preds = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
print(f'  Loaded: {len(preds):,} rows')
variants_present = sorted(preds['variant'].unique())
print(f'  Markets: {preds["market_id"].nunique()}  Variants: {variants_present}\n')
if set(variants_present) != {BASELINE, RICHER}:
    print(f'  WARNING: expected exactly [{BASELINE!r}, {RICHER!r}] in the predictions '
          f'file, found {variants_present} -- Script 15 may have been run with a '
          f'different DIAGNOSTIC_PAIR. Proceeding, but double-check before trusting output.')

market_label = preds.drop_duplicates('market_id').set_index('market_id')['market']

# Fold-level sanity check (the check that caught M7's real problem). ONI
# has continuous coverage back to 1950, so this is expected to come back
# clean, but checked directly rather than assumed.
print('[0] Fold-level identity check (per-fold, are M6 and M9 predictions distinct?) ...')
for fold in sorted(preds['fold'].unique()):
    fsub = preds[preds['fold'] == fold]
    base_p = fsub[fsub['variant'] == BASELINE].sort_values(['crop', 'market_id', 'horizon_weeks', 'week_start'])['y_pred'].values
    rich_p = fsub[fsub['variant'] == RICHER].sort_values(['crop', 'market_id', 'horizon_weeks', 'week_start'])['y_pred'].values
    if len(base_p) == len(rich_p) and len(base_p) > 0:
        pct_identical = 100 * np.mean(np.isclose(base_p, rich_p, atol=1e-9))
        print(f'  Fold {fold}: {pct_identical:.2f}% of predictions byte-identical between {BASELINE}/{RICHER}')
    else:
        print(f'  Fold {fold}: row-count mismatch ({BASELINE}={len(base_p)}, {RICHER}={len(rich_p)}) -- cannot compare directly')

detail_rows = []
for crop in CROPS:
    csub = preds[preds['crop'] == crop]
    market_ids = sorted(csub['market_id'].unique())
    for h in HORIZONS:
        hsub = csub[csub['horizon_weeks'] == h]
        if hsub.empty:
            continue
        for market_id in market_ids:
            msub = hsub[hsub['market_id'] == market_id]
            base = (msub[msub['variant'] == BASELINE]
                    .sort_values(['fold', 'week_start'])
                    .drop_duplicates(subset='week_start')
                    .set_index('week_start')[['y_true', 'y_pred']])
            rich = (msub[msub['variant'] == RICHER]
                    .sort_values(['fold', 'week_start'])
                    .drop_duplicates(subset='week_start')
                    .set_index('week_start')[['y_true', 'y_pred']])
            merged = base.join(rich, how='inner', lsuffix='_base', rsuffix='_rich')
            if len(merged) < MIN_OBS:
                continue

            result = diebold_mariano_test(
                merged['y_true_base'].values, merged['y_pred_base'].values,
                merged['y_pred_rich'].values, h)
            better = (RICHER if (not pd.isna(result['mean_d']) and result['mean_d'] > 0)
                      else (BASELINE if not pd.isna(result['mean_d']) else 'n/a'))
            detail_rows.append({
                'crop': crop, 'horizon_weeks': h, 'market_id': market_id,
                'market': market_label.get(market_id, ''),
                **result, 'better_model': better,
                'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < ALPHA,
            })

detail = pd.DataFrame(detail_rows)
detail_path = os.path.join(OUT_DIR, 'table_dm_m6_vs_m9_detail.csv')
detail.to_csv(detail_path, index=False)
print(f'\n[1] Saved: {detail_path}  ({len(detail):,} market-level tests)')

print(f'\n[2] Summary: % of markets with significant {BASELINE}-vs-{RICHER} difference\n')
summary_rows = []
for crop in CROPS:
    for h in HORIZONS:
        sub = detail[(detail['crop'] == crop) & (detail['horizon_weeks'] == h)]
        if sub.empty:
            continue
        n_tested = len(sub)
        n_sig = sub['significant_5pct'].sum()
        n_sig_rich_better = ((sub['significant_5pct']) & (sub['better_model'] == RICHER)).sum()
        n_sig_base_better = ((sub['significant_5pct']) & (sub['better_model'] == BASELINE)).sum()
        combined_p = fisher_combined_p(sub['p_value'].values)

        summary_rows.append({
            'crop': crop, 'horizon_weeks': h,
            'n_markets_tested': n_tested,
            'pct_significant': round(100 * n_sig / n_tested, 1) if n_tested else np.nan,
            f'n_sig_{RICHER}_better': n_sig_rich_better,
            f'n_sig_{BASELINE}_better': n_sig_base_better,
            'fisher_combined_p': round(combined_p, 6) if not pd.isna(combined_p) else np.nan,
        })

        combined_p_str = f'{combined_p:.2e}' if not pd.isna(combined_p) else 'N/A'
        print(f'  {crop:7s} h={h:>2}w | {n_tested:>4} markets tested | '
              f'{n_sig:>4} sig ({100*n_sig/n_tested:.1f}%) | '
              f'{RICHER} better: {n_sig_rich_better:>3}  {BASELINE} better: {n_sig_base_better:>3} | '
              f'Fisher combined p={combined_p_str}')

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'table_dm_m6_vs_m9_summary.csv')
summary.to_csv(summary_path, index=False)
print(f'\n[3] Saved: {summary_path}')

print('\n' + '=' * 65)
print('Script 64 complete.')
print('\nInterpretation guide:')
print(f'  - If the fold-level check above shows near-0% identical predictions in')
print(f'    every fold, M9 genuinely trained on real ONI signal throughout.')
print(f'  - If pct_significant is low (~5-10%, the false-positive rate expected')
print(f'    at alpha=0.05) and {RICHER}/{BASELINE}-better counts are roughly balanced,')
print(f'    ONI has no reliable market-level effect.')
print(f'  - If a real majority of significant markets favor {RICHER} for a given')
print('    crop/horizon, that cell has a genuine, checkable improvement -- remember')
print('    to run the fold-level pooled-MSE follow-up (Scripts 55/60\'s lesson)')
print('    before trusting this per-market table alone.')
