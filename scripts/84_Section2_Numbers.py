# -*- coding: utf-8 -*-
"""
Script 84 -- every number quoted in manuscript Section 2 ("Study area and data"), recomputed from the current per-crop weekly panels
(data/agmarknet_weekly/{tomato,onion,potato}_weekly_panel.csv, tracked and refreshed by the weekly workflow) and the layer files.
Kerala and Keralam are one state (the source renamed it in 2026).
Output: Model_Output/table_section2_numbers.json and a printed summary (read this when editing the text so figures and prose agree).
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
CROPS = ['tomato', 'onion', 'potato']
PANEL_DIR = os.path.join(BASE, 'data', 'agmarknet_weekly')


def load_panel():
    fr = []
    for c in CROPS:
        d = pd.read_csv(os.path.join(PANEL_DIR, f'{c}_weekly_panel.csv'), parse_dates=['week_start'])
        d['crop'] = c
        fr.append(d)
    p = pd.concat(fr, ignore_index=True)
    p['state'] = p['state'].replace({'Keralam': 'Kerala'})
    p['market_id'] = p['market_id'].astype(float).astype(int)
    return p


def main():
    p = load_panel()
    N = {'panel_first_week': str(p.week_start.min().date()), 'panel_last_week': str(p.week_start.max().date()),
         'n_weeks': int(p.week_start.nunique())}
    N['imputed_method_values'] = {k: int(v) for k, v in p['imputed_method'].fillna('NA').value_counts().items()}
    N['rows'] = int(len(p))
    N['distinct_markets_all'] = int(p[['crop', 'market_id']].drop_duplicates().assign(k=lambda d: d.market_id).k.nunique())
    N['states_all'] = int(p.state.nunique())
    per = {}
    for c in CROPS:
        d = p[p.crop == c]
        obs = d[d.imputed == 0]
        last52 = d[d.week_start > d.week_start.max() - pd.Timedelta(weeks=52)]
        steps = obs.groupby('market_id').week_start.min().value_counts().sort_values(ascending=False).head(4)
        g = obs.groupby('week_start').modal_price_weighted
        thr = 25 if c == 'potato' else 100
        s = g.median()[g.count() >= thr]
        per[c] = {
            'markets': int(d.market_id.nunique()), 'states': int(d.state.nunique()), 'districts': int(d.district.nunique()),
            'market_weeks': int(len(d)),
            'observed_share_pct': round(100 * len(obs) / len(d), 1),
            'by_method_pct': {k: round(100 * v / len(d), 1) for k, v in d['imputed_method'].fillna('NA').value_counts().items()},
            'markets_observed_2017': int(obs[obs.week_start.dt.year == 2017].market_id.nunique()),
            'markets_observed_latest52': int(last52[last52.imputed == 0].market_id.nunique()),
            'median_price': round(float(obs.modal_price_weighted.median())),
            'iqr_price': [round(float(obs.modal_price_weighted.quantile(.25))), round(float(obs.modal_price_weighted.quantile(.75)))],
            'median_weekly_arrivals_t': round(float(obs.arrivals_tonnes_week.median())),
            'top_states': {k: int(v) for k, v in d.groupby('state').market_id.nunique().sort_values(ascending=False).head(5).items()},
            'first_report_steps': {str(k.date()): int(v) for k, v in steps.items()},
            'national_median_min': [round(float(s.min())), str(s.idxmin().date())],
            'national_median_max': [round(float(s.max())), str(s.idxmax().date())],
            'national_median_cv': round(float(s.std() / s.mean()), 2),
            'national_median_last': [round(float(s.iloc[-1])), str(s.index[-1].date())],
        }
    N['per_crop'] = per
    # share of market-weeks by method, all crops
    N['all_by_method'] = {k: [int(v), round(100 * v / len(p), 1)] for k, v in p['imputed_method'].fillna('NA').value_counts().items()}
    # market -> nearest production zone (Script 78 writes table_market_locations.csv)
    loc_f = os.path.join(OUT, 'table_market_locations.csv')
    if os.path.exists(loc_f):
        loc = pd.read_csv(loc_f)
        N['zone_assignment'] = {c: {'markets': int((loc.crop == c).sum()), 'assigned_within_500km': int(loc[(loc.crop == c)].zone.notna().sum()),
                                    'median_dist_km': round(float(loc[loc.crop == c].dist_km.median())),
                                    'mean_dist_km': round(float(loc[loc.crop == c].dist_km.mean()))} for c in CROPS}
    json.dump(N, open(os.path.join(OUT, 'table_section2_numbers.json'), 'w'), indent=1)
    print(json.dumps(N, indent=1))


if __name__ == '__main__':
    main()
