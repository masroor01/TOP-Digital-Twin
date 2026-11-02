# -*- coding: utf-8 -*-
"""
Script 88 -- price-regime outcomes and outcome bands for the dashboard (2026-11-02).

Why: the forecasting model has learned strong mean reversion (after a price run-up it forecasts a fall, after a slump a rise). That has been right on
average but is based on few price cycles and the spread of outcomes is wide, while the dashboard shows a single number with a symmetric +- RMSE band.
This script turns the backtest into (a) a regime notice (what happened after similar situations) and (b) an asymmetric outcome band, and CHECKS the band.

Regime of a market-week = price / median price of the previous 52 weeks ("level"), in buckets <0.8, 0.8-1.25, 1.25-1.75, 1.75-2.5, >2.5 (thin buckets at the
ends are merged until they hold at least MIN_WEEKS distinct origin weeks).
Band: the 10th and 90th percentile of log(actual / forecast) among past test forecasts of the same crop, horizon and regime, applied to the forecast.
Check (leave-one-fold-out): for each fold the band is built from the other four folds only and its coverage on the held-out fold is compared with the coverage
of the current +- RMSE band. The band is adopted by the dashboard only if the held-out coverage is close to its nominal 80 %.

Input   Model_Output/dm_market_level_predictions.csv (Script 15 diagnostic, variant M6; test years 2022-2026), weekly panels.
Output  web/data/regime_outcomes.json (served to the dashboard), Model_Output/table_regime_outcomes.csv, table_outcome_band_coverage.csv
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(BASE, 'web', 'data', 'regime_outcomes.json')
EDGES = [0, 0.8, 1.25, 1.75, 2.5, np.inf]
LABELS = ['<0.8', '0.8-1.25', '1.25-1.75', '1.75-2.5', '>2.5']
MIN_WEEKS = 30
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]


def load():
    pred = pd.read_csv(os.path.join(BASE, 'Model_Output', 'dm_market_level_predictions.csv'), parse_dates=['week_start'])
    pred = pred[pred['variant'] == 'M6'].copy()
    pred['market_id'] = pred['market_id'].astype(float).astype(int)
    fr = []
    for c in CROPS:
        p = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', f'{c}_weekly_panel.csv'), parse_dates=['week_start'],
                        usecols=['market_id', 'week_start', 'modal_price_weighted'])
        p['market_id'] = p['market_id'].astype(float).astype(int)
        p = p.sort_values(['market_id', 'week_start'])
        g = p.groupby('market_id')['modal_price_weighted']
        p['p0'] = p['modal_price_weighted']
        p['med52'] = g.transform(lambda s: s.shift(1).rolling(52, min_periods=26).median())
        p['crop'] = c
        fr.append(p[['crop', 'market_id', 'week_start', 'p0', 'med52']])
    P = pd.concat(fr)
    d = pred.merge(P, on=['crop', 'market_id', 'week_start'], how='left').dropna(subset=['p0', 'med52'])
    d = d[(d['y_true'] > 0) & (d['y_pred'] > 0)].copy()
    d['level'] = d['p0'] / d['med52']
    d['e'] = np.log(d['y_true'] / d['y_pred'])
    d['r_true'] = d['y_true'] / d['p0'] - 1
    d['r_pred'] = d['y_pred'] / d['p0'] - 1
    d['bucket'] = pd.cut(d['level'], EDGES, labels=LABELS, right=False)
    return d


def merge_thin(sub):
    """Merge the thin extreme buckets of one (crop, horizon) into their neighbour until each holds >= MIN_WEEKS distinct origin weeks."""
    labels = [l for l in LABELS if (sub['bucket'] == l).any()]
    groups = [[l] for l in labels]

    def weeks(g):
        return sub.loc[sub['bucket'].isin(g), 'week_start'].nunique()
    changed = True
    while changed and len(groups) > 1:
        changed = False
        if weeks(groups[-1]) < MIN_WEEKS:
            groups[-2] += groups.pop(); changed = True
        elif weeks(groups[0]) < MIN_WEEKS:
            groups[1] = groups[0] + groups[1]; groups.pop(0); changed = True
    return groups


def name(g):
    lo = g[0]; hi = g[-1]
    if len(g) == 1:
        return lo
    a = lo.replace('<', '').split('-')[0] if not lo.startswith('<') else '0'
    b = hi.split('-')[-1] if not hi.startswith('>') else 'inf'
    return ('<%s' % b) if a == '0' else (('>%s' % a) if b == 'inf' else '%s-%s' % (a, b))


def main():
    d = load()
    rows, cover, J = [], [], {'edges': [e if np.isfinite(e) else None for e in EDGES], 'minOriginWeeks': MIN_WEEKS, 'crops': {}}
    folds = sorted(d['fold'].unique())
    rmse = {}
    for (c, h), x in d.groupby(['crop', 'horizon_weeks']):
        rmse[(c, h)] = float(np.sqrt(((x['y_true'] - x['y_pred']) ** 2).mean()))
    for c in CROPS:
        J['crops'][c] = {}
        for h in HORIZONS:
            sub = d[(d['crop'] == c) & (d['horizon_weeks'] == h)]
            if len(sub) == 0:
                continue
            groups = merge_thin(sub)
            J['crops'][c][str(h)] = []
            for g in groups:
                x = sub[sub['bucket'].isin(g)]
                qe = x['e'].quantile(QS).to_dict(); qa = x['r_true'].quantile(QS).to_dict()
                yrs = sorted(x['week_start'].dt.year.unique().tolist())
                rec = dict(bucket=name(g), levelLow=float(EDGES[LABELS.index(g[0])]), levelHigh=(float(EDGES[LABELS.index(g[-1]) + 1]) if np.isfinite(EDGES[LABELS.index(g[-1]) + 1]) else None),
                           n=int(len(x)), originWeeks=int(x['week_start'].nunique()), years=yrs,
                           forecastChangeMedian=float(x['r_pred'].median()), actualChange={('q%02d' % int(100 * k)): float(v) for k, v in qa.items()},
                           residLog={('q%02d' % int(100 * k)): float(v) for k, v in qe.items()}, shareFall25=float((x['r_true'] <= -0.25).mean()), shareRise25=float((x['r_true'] >= 0.25).mean()))
                J['crops'][c][str(h)].append(rec)
                rows.append(dict(crop=c, horizon_weeks=h, **{k: v for k, v in rec.items() if not isinstance(v, (dict, list))},
                                 **{'actual_' + k: v for k, v in rec['actualChange'].items()}))
                # leave-one-fold-out coverage of the outcome band vs the +-RMSE band
                for k in folds:
                    held = x[x['fold'] == k]; rest = x[x['fold'] != k]
                    if len(held) < 50 or rest['week_start'].nunique() < 20:
                        continue
                    lo, hi = rest['e'].quantile(0.10), rest['e'].quantile(0.90)
                    inband = ((held['e'] >= lo) & (held['e'] <= hi)).mean()
                    rm = rmse[(c, h)]
                    inrmse = (((held['y_true'] - held['y_pred']).abs()) <= rm).mean()
                    cover.append(dict(crop=c, horizon_weeks=h, bucket=rec['bucket'], fold=int(k), n=int(len(held)), coverage_outcome_band=inband, coverage_rmse_band=inrmse))
    R = pd.DataFrame(rows); R.round(4).to_csv(os.path.join(BASE, 'Model_Output', 'table_regime_outcomes.csv'), index=False)
    C = pd.DataFrame(cover); C.round(4).to_csv(os.path.join(BASE, 'Model_Output', 'table_outcome_band_coverage.csv'), index=False)
    w = lambda g, col: np.average(g[col], weights=g['n'])
    summ = C.groupby(['crop', 'horizon_weeks']).apply(lambda g: pd.Series({'outcome_band_80%_coverage': w(g, 'coverage_outcome_band'), 'rmse_band_coverage': w(g, 'coverage_rmse_band'), 'folds_buckets': len(g)}), include_groups=False)
    print('Held-out coverage (nominal 80 % for the outcome band; the +- RMSE band is nominally ~68 % if errors were normal):')
    print((100 * summ[['outcome_band_80%_coverage', 'rmse_band_coverage']]).round(0).assign(cells=summ['folds_buckets']).to_string())
    print('overall: outcome band %.0f %%, +-RMSE band %.0f %%' % (100 * w(C, 'coverage_outcome_band'), 100 * w(C, 'coverage_rmse_band')))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(J, f)
    print('wrote', OUT_JSON)


if __name__ == '__main__':
    main()
