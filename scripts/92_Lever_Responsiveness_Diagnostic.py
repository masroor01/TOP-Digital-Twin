"""Script 92: lever responsiveness diagnostic (2026-11-23). READ-ONLY: it trains nothing and changes no model or table in production.

Question: which what-if levers actually move each production model INSIDE their observed range?
Method: for each of the 12 production models and each lever, set the lever to 10 % and to 90 % of its observed range
(Model_Output/production_models/feature_ranges.json), keep every other input at the market's baseline row (300 sampled
markets), and take the larger absolute change in predicted log price. Report the mean over markets.
Output: Model_Output/lever_responsiveness/table_lever_diagnostic.csv (by lever) and ..._cells.csv (by crop x horizon).
Run from anywhere: python scripts/92_Lever_Responsiveness_Diagnostic.py
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(ROOT, 'Model_Output', 'production_models')
OUT = os.path.join(ROOT, 'Model_Output', 'lever_responsiveness')
os.makedirs(OUT, exist_ok=True)

ref = pd.read_csv(os.path.join(PM, 'reference_rows.csv'))
RANGES = json.load(open(os.path.join(PM, 'feature_ranges.json')))
config = open(os.path.join(ROOT, 'web', 'backend', 'src', 'config.js'), encoding='utf8').read()


def js_list(name):
    m = re.search(r'export const %s = \[(.*?)\];' % name, config, re.S)
    return re.findall(r"'([^']+)'", m.group(1)) if m else []


LEVERS = {k: js_list(k) for k in ('POLICY_FIELDS', 'CLIMATE_FIELDS', 'MACRO_FIELDS', 'INFRA_FIELDS')}
rows = []
for crop in ('tomato', 'onion', 'potato'):
    d = ref[ref.crop == crop]
    for h in (1, 4, 13, 26):
        model = joblib.load(os.path.join(PM, f'{crop}_{h}w.joblib'))
        feats = list(model.feature_name_)
        X0 = d.sample(min(300, len(d)), random_state=7)[feats].astype(float)
        base = model.predict(X0)
        splits = dict(zip(feats, model.booster_.feature_importance('split')))
        for group, names in LEVERS.items():
            for f in names:
                if f not in feats:
                    continue
                r = RANGES.get(f)
                if not r or r['max'] <= r['min']:
                    rows.append((crop, h, group, f, 0.0, 0.0, int(splits.get(f, 0)))); continue
                span = r['max'] - r['min']
                moves = []
                for v in (r['min'] + 0.1 * span, r['min'] + 0.9 * span):
                    X1 = X0.copy(); X1[f] = v
                    moves.append(np.abs(model.predict(X1) - base))
                mv = np.maximum(*moves)
                rows.append((crop, h, group, f, float(mv.mean()), float((mv > 1e-9).mean()), int(splits.get(f, 0))))
cells = pd.DataFrame(rows, columns=['crop', 'h', 'group', 'lever', 'mean_abs_dlogp', 'share_rows_moved', 'splits'])
cells.to_csv(os.path.join(OUT, 'table_lever_diagnostic_cells.csv'), index=False)
agg = cells.groupby(['group', 'lever']).agg(mean_dlogp=('mean_abs_dlogp', 'mean'), moved=('share_rows_moved', 'mean'),
                                            splits=('splits', 'mean')).reset_index()
agg['pct_effect'] = (np.expm1(agg.mean_dlogp) * 100).round(2)
agg = agg.sort_values('mean_dlogp', ascending=False).round(4)
agg.to_csv(os.path.join(OUT, 'table_lever_diagnostic.csv'), index=False)
print(agg.to_string(index=False))
