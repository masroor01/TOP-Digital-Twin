# -*- coding: utf-8 -*-
"""
Script 80 -- Probe: does the origin-week price (p_t) as a feature close the short-horizon gap? (2026-10-27)
Finding that prompted it: Script 15 / 23 feature sets contain price lags from 1 week BEFORE the origin week
(price_lag_1 = p_(t-1)) and never the origin-week price p_t itself (log_price is in _EXCLUDE), so a forecast labelled
"h weeks ahead" sees prices only up to t-1, while the persistence baseline uses p_t.
Probe: M0 (price-only) with and without log_price as an extra feature, same folds and settings, one seed, all test rows;
skill vs persistence = (1 - MAE_model / MAE_persistence) x 100.
Output (printed): table_origin_price_probe.csv in Model_Output (written by the helper block below).
"""
import sys, io, os, importlib.util
import numpy as np, pandas as pd
BASE = r'C:\Users\masro\Documents\TOP_Digital_Twin'
spec = importlib.util.spec_from_file_location('s75', os.path.join(BASE, 'scripts', '75_Onion_Short_Horizon_Test.py'))
s75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s75)
g = s75.load_g(); feat = g['feat']; FS = g['MODEL_FEATURE_SETS']; FOLDS = g['FOLDS']
rows = []
for crop in ['onion', 'tomato', 'potato']:
    df0 = feat[crop].sort_values(['market_id', 'week_start']).copy()
    for h in (1, 4, 13):
        df = df0.copy(); df['target'] = df.groupby('market_id')['log_price'].shift(-h); df = df.dropna(subset=['target', 'price_lag_1'])
        for name, extra in (('M0', []), ('M0+p_t', ['log_price'])):
            cols = [c for c in FS['M0'] if c in df.columns] + extra
            num = {'model': 0.0, 'naive': 0.0}
            res = []
            for fi in FOLDS:
                t_end, v_s, v_e = pd.Timestamp(fi['train_end']), pd.Timestamp(fi['val_start']), pd.Timestamp(fi['val_end'])
                te_s, te_e = pd.Timestamp(fi['test_start']), pd.Timestamp(fi['test_end'])
                tr = df[df['week_start'] + pd.Timedelta(weeks=h) <= t_end]
                va = df[(df['week_start'] + pd.Timedelta(weeks=h)).between(v_s, v_e)]
                te = df[(df['week_start'] >= te_s) & (df['week_start'] <= te_e)].dropna(subset=['log_price'])
                if len(tr) < 100 or len(te) < 10: continue
                _, p_te = s75.fit_predict(g, tr[cols].fillna(0), tr['target'], va[cols].fillna(0), va['target'], te[cols].fillna(0), cols)
                yt = np.expm1(te['target'].to_numpy()); yp = np.expm1(p_te); yn = np.expm1(te['log_price'].to_numpy())
                res.append(pd.DataFrame({'yt': yt, 'yp': yp, 'yn': yn}))
            r = pd.concat(res); sk = 100 * (1 - (r.yp - r.yt).abs().sum() / (r.yn - r.yt).abs().sum())
            rows.append((crop, h, name, round(sk, 1), len(r))); print(crop, h, name, 'skill vs persistence %.1f' % sk, 'n', len(r), flush=True)
R = pd.DataFrame(rows, columns=['crop', 'horizon_weeks', 'variant', 'skill_vs_persistence_pct', 'n'])
R.to_csv(os.path.join(BASE, 'Model_Output', 'table_origin_price_probe.csv'), index=False)
print(R.pivot(index=['crop', 'horizon_weeks'], columns='variant', values='skill_vs_persistence_pct').to_string())
