# -*- coding: utf-8 -*-
"""
Script 74 -- Live track record of the deployed dashboard forecasts.
====================================================================
Every weekly refresh commits the production models (Script 23) and the reference rows (the forecast origin) to git.
That archive lets us replay exactly what the dashboard would have shown at each refresh and score it against the
weekly prices that arrived afterwards. Nothing is retrained and nothing is tuned here.

For each archived refresh commit (market_id-keyed era, from 2026-08-14):
  * load that commit's models, feature_columns.json and reference_rows.csv from git;
  * forecast price at origin + h weeks for every market (h = 1, 4, 13, 26);
  * keep a forecast only once its target week has an OBSERVED (not imputed) price in the current panel;
  * a (crop, market, origin week, horizon) forecast is counted once, from the earliest commit that issued it.
Markets with sufficient_history = False or stale_reference = True are excluded (same rule as Scripts 43/45).

Metrics per crop x horizon: WAPE of the forecast vs persistence (last observed price at origin), skill =
(1 - WAPE_model / WAPE_persistence) x 100, median APE, bias, and directional accuracy = sign(forecast - last observed)
equals sign(actual - last observed) (flat actuals and exact ties dropped), with a two-sided binomial p-value vs 50 %.
Standard errors are not adjusted for the clustering of markets within a week; with few weeks this is descriptive only.

Outputs (Model_Output/): table_live_track_record_forecasts.csv, table_live_track_record_summary.csv
Run: python scripts/74_Live_Track_Record.py
"""
import os, io, sys, json, subprocess, tempfile, shutil
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
MD = 'Model_Output/production_models'
FIRST_COMMIT = 'eec34e9'          # 2026-08-14: reference rows keyed by market_id, feature-engineering fix
CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]


def git(*args, binary=False):
    r = subprocess.run(['git', '-C', BASE] + list(args), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace')[:300])
    return r.stdout if binary else r.stdout.decode('utf-8')


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    log = git('log', '--reverse', '--format=%h %ad', '--date=short', f'{FIRST_COMMIT}^..HEAD', '--', f'{MD}/reference_rows.csv').strip().splitlines()
    commits = [(l.split()[0], l.split()[1]) for l in log]
    print('[74] archived refresh commits:', len(commits), commits[0], '->', commits[-1])

    panels = []
    for c in CROPS:
        p = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', f'{c}_weekly_panel.csv'),
                        usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted', 'imputed'])
        panels.append(p[p['imputed'] == 0])
    act = pd.concat(panels, ignore_index=True)
    act['week_start'] = pd.to_datetime(act['week_start']); act['market_id'] = act['market_id'].astype(float)
    act = act.rename(columns={'modal_price_weighted': 'actual_price'})[['crop', 'market_id', 'week_start', 'actual_price']]
    print('[74] latest observed week in panels:', act['week_start'].max().date())

    rows = []
    tmp = tempfile.mkdtemp()
    try:
        for sha, day in commits:
            try:
                feat = json.loads(git('show', f'{sha}:{MD}/feature_columns.json'))
                ref = pd.read_csv(io.StringIO(git('show', f'{sha}:{MD}/reference_rows.csv')))
            except RuntimeError:
                continue
            if 'market_id' not in ref.columns:
                continue
            ref['week_start'] = pd.to_datetime(ref['week_start']); ref['market_id'] = ref['market_id'].astype(float)
            for crop in CROPS:
                r = ref[ref['crop'] == crop]
                if r.empty:
                    continue
                for h in HORIZONS:
                    key = f'{crop}_{h}w'
                    mp = os.path.join(tmp, f'{sha}_{key}.joblib')
                    try:
                        with open(mp, 'wb') as f:
                            f.write(git('show', f'{sha}:{MD}/{key}.joblib', binary=True))
                        model = joblib.load(mp)
                    except Exception:
                        continue
                    cols = feat.get(key)
                    if not cols:
                        continue
                    X = pd.DataFrame([{c: row.get(c, 0) for c in cols} for _, row in r.iterrows()])
                    pred = np.expm1(model.predict(X))
                    d = r[['crop', 'market_id', 'market', 'state', 'week_start', 'last_observed_price', 'sufficient_history', 'stale_reference']].copy()
                    d = d.rename(columns={'week_start': 'origin_week'})
                    d['horizon_weeks'] = h; d['forecast_price'] = pred; d['commit'] = sha; d['commit_date'] = day
                    rows.append(d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    f = pd.concat(rows, ignore_index=True)
    f['target_week'] = f['origin_week'] + pd.to_timedelta(f['horizon_weeks'] * 7, unit='D')
    n_issued = len(f)
    f = f.sort_values(['commit_date', 'commit']).drop_duplicates(['crop', 'market_id', 'origin_week', 'horizon_weeks'], keep='first')
    print('[74] forecasts issued (all commits): %d; unique (market, origin, horizon): %d' % (n_issued, len(f)))
    f = f.merge(act.rename(columns={'week_start': 'target_week'}), on=['crop', 'market_id', 'target_week'], how='inner')
    print('[74] with an observed target-week price:', len(f))
    f = f[(f['sufficient_history'] == True) & (f['stale_reference'] == False)].copy()
    f['abs_err'] = (f['forecast_price'] - f['actual_price']).abs()
    f['naive_abs_err'] = (f['last_observed_price'] - f['actual_price']).abs()
    f['ape'] = f['abs_err'] / f['actual_price'] * 100
    f['dir_actual'] = np.sign(f['actual_price'] - f['last_observed_price'])
    f['dir_pred'] = np.sign(f['forecast_price'] - f['last_observed_price'])
    f.to_csv(os.path.join(OUT, 'table_live_track_record_forecasts.csv'), index=False)

    out = []
    for (crop, h), g in f.groupby(['crop', 'horizon_weeks']):
        wape = g['abs_err'].sum() / g['actual_price'].sum() * 100
        wape_n = g['naive_abs_err'].sum() / g['actual_price'].sum() * 100
        dd = g[(g['dir_actual'] != 0) & (g['dir_pred'] != 0)]
        k = int((dd['dir_actual'] == dd['dir_pred']).sum()); n = len(dd)
        out.append(dict(crop=crop, horizon_weeks=h, n_forecasts=len(g), n_origin_weeks=g['origin_week'].nunique(), n_markets=g['market_id'].nunique(),
                        first_origin=g['origin_week'].min().date(), last_target=g['target_week'].max().date(),
                        WAPE_model_pct=round(wape, 2), WAPE_persistence_pct=round(wape_n, 2), skill_vs_persistence_pct=round((1 - wape / wape_n) * 100, 1),
                        median_APE_pct=round(g['ape'].median(), 2), bias_pct=round((g['forecast_price'] - g['actual_price']).sum() / g['actual_price'].sum() * 100, 2),
                        dir_n=n, dir_correct=k, directional_accuracy_pct=round(100 * k / n, 1) if n else np.nan,
                        dir_p_vs_50=round(binomtest(k, n, 0.5).pvalue, 4) if n else np.nan))
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(OUT, 'table_live_track_record_summary.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(S.to_string(index=False))


if __name__ == '__main__':
    main()
