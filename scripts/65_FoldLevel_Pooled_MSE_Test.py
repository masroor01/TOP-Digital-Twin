# -*- coding: utf-8 -*-
"""
Script 65 -- Fold-level pooled-MSE binomial test (conservative layer significance)
==================================================================================
Market-level Diebold-Mariano tests (Scripts 18b/55/60/64) score hundreds of
markets from ONE fitted model per fold, so they are not independent trials:
a small shift in one model gets replicated across thousands of market-weeks
and can look decisive (e.g. "75.6 % of markets favour drought for potato at
1 week" turned out to be 1 of 4 real folds). This script is the conservative
check: for each crop x horizon, pool every market-week in a fold, compute the
pooled MSE of each variant, and count in how many independent FOLDS variant A
beats variant B. The p-value is the exact one-sided binomial P(X >= wins | n,
0.5); with 5 folds only a 5-of-5 sweep reaches p = 0.031.

Input   Model_Output/dm_market_level_predictions.csv, produced by Script 15
        with MARKET_LEVEL_DIAGNOSTIC=True and DIAGNOSTIC_PAIR=(B, A).
Usage   python scripts/65_FoldLevel_Pooled_MSE_Test.py A B [out_csv]
        e.g. A=M6 B=M0 tests "does the full stack beat price-only?"
Output  Model_Output/table_foldlevel_<a>_vs_<b>.csv (or out_csv):
        crop, horizon_weeks, <A>_wins, n_folds, binom_p
"""
import os, sys
import pandas as pd
from scipy.stats import binom

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')

def fold_level(preds, a, b):
    rows = []
    for (crop, h), g in preds.groupby(['crop', 'horizon_weeks']):
        wins = n = 0
        for fold, gf in g.groupby('fold'):
            pa = gf[gf.variant == a]; pb = gf[gf.variant == b]
            if pa.empty or pb.empty:
                continue
            mse_a = ((pa.y_true - pa.y_pred) ** 2).mean()
            mse_b = ((pb.y_true - pb.y_pred) ** 2).mean()
            n += 1; wins += int(mse_a < mse_b)
        if n:
            rows.append(dict(crop=crop, horizon_weeks=int(h), **{f'{a}_wins': wins}, n_folds=n,
                             binom_p=round(float(binom.sf(wins - 1, n, 0.5)), 4)))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    src = os.path.join(OUT, 'dm_market_level_predictions.csv')
    preds = pd.read_csv(src, usecols=['variant', 'crop', 'fold', 'horizon_weeks', 'y_true', 'y_pred'])
    have = set(preds.variant.unique())
    assert {a, b} <= have, f'predictions file holds {sorted(have)}; need {a} and {b} (re-run Script 15 with DIAGNOSTIC_PAIR)'
    res = fold_level(preds, a, b)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, f'table_foldlevel_{a.lower()}_vs_{b.lower()}.csv')
    res.to_csv(out, index=False)
    print(res.to_string(index=False)); print('wrote', out)
