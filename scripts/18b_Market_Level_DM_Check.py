# -*- coding: utf-8 -*-
"""
Script 18b — Market-Level Diebold-Mariano Check (diagnostic)
================================================================
Runs a market-level DM test on Script 15's MARKET_LEVEL_DIAGNOSTIC output
(currently DIAGNOSTIC_PAIR = ('M0', 'M6') — the price-only baseline vs the
full M0-M6 model, now that Layers 5/6 exist). Script 18 tests the same
comparison on crop-level weekly-averaged predictions, which has much lower
statistical power; this script checks whether that aggregation is hiding a
real per-market effect by running the DM test independently for every
individual market and summarizing how many show a significant improvement
for the richer model vs how many show the baseline better.

If the crop-level result reflects a genuine lack of signal, the per-market
results should look similarly mixed/insignificant. If crop-level averaging
was washing out a real effect, a much larger share of individual markets
should show significant, consistently-directed results than the crop-level
headline suggested.

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
BASELINE, RICHER = 'M0', 'M6'   # must match Script 15's DIAGNOSTIC_PAIR


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
    print('Run scripts/15_Ablation_Study_M0_M4.py with MARKET_LEVEL_DIAGNOSTIC=True '
          f'and DIAGNOSTIC_PAIR=({BASELINE!r}, {RICHER!r}) first.')
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
                'crop': crop, 'horizon_weeks': h, 'market': market,
                **result, 'better_model': better,
                'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < ALPHA,
            })

detail = pd.DataFrame(detail_rows)
detail_path = os.path.join(OUT_DIR, 'table_dm_market_level_detail.csv')
detail.to_csv(detail_path, index=False)
print(f'[1] Saved: {detail_path}  ({len(detail):,} market-level tests)')

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
summary_path = os.path.join(OUT_DIR, 'table_dm_market_level_summary.csv')
summary.to_csv(summary_path, index=False)
print(f'\n[3] Saved: {summary_path}')

print('\n' + '=' * 65)
print('Script 18b complete.')
print('\nInterpretation guide:')
print(f'  - If pct_significant is still low (~5-10%, matching the false-')
print(f'    positive rate expected at alpha=0.05) and {RICHER}/{BASELINE}-better counts')
print(f'    are roughly balanced, the crop-level finding (weak/no signal)')
print(f'    holds up: {RICHER} genuinely does not reliably beat {BASELINE}.')
print(f'  - If a large majority of significant markets favor {RICHER}, crop-level')
print('    averaging was hiding a real effect — worth revisiting the')
print('    ablation methodology or reporting market-level results instead.')
