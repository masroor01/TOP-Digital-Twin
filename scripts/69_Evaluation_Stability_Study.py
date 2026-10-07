# -*- coding: utf-8 -*-
"""
Script 69 -- Evaluation stability study: how much of a cell-level result is noise?
==================================================================================
On 2026-10-07 the validation window in Scripts 12/15/15b/48/49 was corrected (selected by forecast-target
date).  Re-running moved single-cell results a lot (e.g. fold wins by 2-3 folds, M0 onion 26 wk MSE +65 %),
although the change only affects early stopping.  This script quantifies the instability so that cell-level
claims can be judged against it.

Design (M0 vs M6, all crops/horizons/folds, CPU LightGBM, via Script 15's MARKET_LEVEL_DIAGNOSTIC mode):
  * seeds      : LightGBM random_state (bagging / feature sampling) varied;
  * rule       : validation window `new` (target-date, current code) or `old` (origin-week, pre-2026-10-06);
  * ensemble   : predictions averaged over the seeds of one rule (a cheap stabilised estimator).
Per cell (crop x horizon) it reports the spread across seeds of the M6-vs-M0 pooled-MSE gap and of the
fold-level win count, whether the sign of the gap is consistent across seeds, how large the old-vs-new rule
shift is relative to seed noise, and the same quantities for the seed-averaged ensemble.

Usage
  python scripts/69_Evaluation_Stability_Study.py run <new|old> <seed>   # one Script-15 diagnostic run (~4 min)
  python scripts/69_Evaluation_Stability_Study.py analyze                # tables from all saved runs
  (optional pair: `run new 43 M6 M9`, `analyze M6 M9` -- `gap` is then richer-vs-base pooled MSE, e.g. M9 vs M6)
Runs are saved compactly to Model_Output/_stability_runs/ (git-ignored); the shared
dm_market_level_predictions.csv is backed up before and restored after every run.

Outputs (Model_Output/): table_stability_runs.csv, table_stability_cells.csv, table_stability_summary.csv
"""
import io, os, sys, glob, shutil
import numpy as np
import pandas as pd
from scipy.stats import binom

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
RUNS = os.path.join(OUT, '_stability_runs')
PRED = os.path.join(OUT, 'dm_market_level_predictions.csv')
RAW = os.path.join(OUT, 'dm_market_level_raw_results.csv')
KEY = ['variant', 'crop', 'fold', 'horizon_weeks', 'market_id', 'week_start']


def run(rule, seed, pair=('M0', 'M6')):
    assert rule in ('new', 'old')
    os.makedirs(RUNS, exist_ok=True)
    sys.path.insert(0, os.path.join(BASE, 'scripts'))
    import gpu_utils
    gpu_utils._cache['lgbm'] = {}
    path = os.path.join(BASE, 'scripts', '15_Ablation_Study_M0_M4.py')
    src = io.open(path, encoding='utf-8').read()
    for a, b in [("MARKET_LEVEL_DIAGNOSTIC = False", "MARKET_LEVEL_DIAGNOSTIC = True"), ("DIAGNOSTIC_PAIR = ('M0', 'M6')", "DIAGNOSTIC_PAIR = (%r, %r)" % tuple(pair)),
                 ("SEED     = 42", "SEED     = %d" % seed)]:
        assert a in src, a
        src = src.replace(a, b)
    new_val = "val   = df_h[(df_h['week_start'] + pd.Timedelta(weeks=h)).between(v_start, v_end)]"
    old_val = "val   = df_h[(df_h['week_start'] >= v_start) & (df_h['week_start'] <= v_end)]"
    assert new_val in src
    if rule == 'old':
        src = src.replace(new_val, old_val)
    bak = [(PRED, PRED + '.stabbak'), (RAW, RAW + '.stabbak')]
    for a, b in bak:
        shutil.copy2(a, b)
    try:
        print('[69] run rule=%s seed=%d' % (rule, seed), flush=True)
        try:
            exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
        except SystemExit:
            pass   # Script 15's diagnostic mode ends with sys.exit()
        p = pd.read_csv(PRED, parse_dates=['week_start'])
        assert set(p['variant'].unique()) == set(pair)
        p[['crop', 'fold', 'horizon_weeks']] = p[['crop', 'fold', 'horizon_weeks']]
        for c in ('y_true', 'y_pred'):
            p[c] = p[c].astype('float32')
        tag = '' if tuple(pair) == ('M0', 'M6') else '_%s_%s' % tuple(pair)
        p.drop(columns=['market']).to_parquet(os.path.join(RUNS, '%s_%d%s.parquet' % (rule, seed, tag)), index=False)
        print('[69] saved %s_%d%s.parquet (%d rows)' % (rule, seed, tag, len(p)))
    finally:
        for a, b in bak:
            shutil.copy2(b, a); os.remove(b)


def cell_metrics(p, base='M0', rich='M6'):
    """Per (crop, horizon): pooled MSE per variant, M6-vs-M0 gap %, fold wins of M6."""
    p = p.assign(se=(p['y_true'].astype(float) - p['y_pred'].astype(float)) ** 2)
    rows = []
    for (crop, h), g in p.groupby(['crop', 'horizon_weeks']):
        m = g.groupby('variant')['se'].mean()
        f = g.groupby(['fold', 'variant'])['se'].mean().unstack()
        wins = int((f[rich] < f[base]).sum())
        rows.append(dict(crop=crop, horizon_weeks=int(h), mse_M0=m[base], mse_M6=m[rich], gap_pct=100 * (m[rich] / m[base] - 1), m6_fold_wins=wins, n_folds=len(f)))
    return pd.DataFrame(rows)


def analyze(pair=('M0', 'M6')):
    tag = '' if tuple(pair) == ('M0', 'M6') else '_%s_%s' % tuple(pair)
    files = sorted(f for f in glob.glob(os.path.join(RUNS, '*.parquet')) if (os.path.basename(f)[:-8].count('_') == 1) == (tag == '') and (tag == '' or os.path.basename(f)[:-8].endswith(tag)))
    assert files, 'no runs found; run the study first'
    per_run, preds = [], {}
    for f in files:
        rule, seed = os.path.basename(f)[:-8].split('_')[:2]
        p = pd.read_parquet(f)
        cm = cell_metrics(p, *pair); cm['rule'] = rule; cm['seed'] = int(seed); per_run.append(cm)
        preds.setdefault(rule, []).append(p.set_index(KEY)['y_pred'].astype('float64').rename(seed))
        if 'y_true_' + rule not in preds:
            preds['y_true_' + rule] = p.set_index(KEY)['y_true']
    runs = pd.concat(per_run, ignore_index=True)
    runs.to_csv(os.path.join(OUT, 'table_stability_runs%s.csv' % tag), index=False)
    # ensemble (seed-average) per rule
    ens = []
    for rule in ('new', 'old'):
        if rule not in preds: continue
        df = pd.concat(preds[rule], axis=1)
        e = pd.DataFrame({'y_pred': df.mean(axis=1), 'y_true': preds['y_true_' + rule].reindex(df.index)}).reset_index()
        cm = cell_metrics(e, *pair); cm['rule'] = rule; cm['n_seeds'] = df.shape[1]; ens.append(cm)
    ens = pd.concat(ens, ignore_index=True)
    # per-cell summary across seeds
    out = []
    for (rule, crop, h), g in runs.groupby(['rule', 'crop', 'horizon_weeks']):
        e = ens[(ens.rule == rule) & (ens.crop == crop) & (ens.horizon_weeks == h)].iloc[0]
        out.append(dict(rule=rule, crop=crop, horizon_weeks=h, n_seeds=len(g), gap_mean=g.gap_pct.mean(), gap_sd=g.gap_pct.std(ddof=1) if len(g) > 1 else np.nan,
                        gap_min=g.gap_pct.min(), gap_max=g.gap_pct.max(), share_seeds_m6_better=(g.gap_pct < 0).mean(), sign_flips=bool((g.gap_pct < 0).any() and (g.gap_pct > 0).any()),
                        wins_min=g.m6_fold_wins.min(), wins_max=g.m6_fold_wins.max(), mse_M6_cv_pct=100 * g.mse_M6.std(ddof=1) / g.mse_M6.mean() if len(g) > 1 else np.nan,
                        mse_M0_cv_pct=100 * g.mse_M0.std(ddof=1) / g.mse_M0.mean() if len(g) > 1 else np.nan,
                        ens_gap_pct=e.gap_pct, ens_m6_fold_wins=e.m6_fold_wins, ens_binom_p=binom.sf(e.m6_fold_wins - 1, int(e.n_folds), 0.5)))
    cells = pd.DataFrame(out).round(3)
    # rule shift vs seed noise
    sh = []
    for (crop, h), g in cells.groupby(['crop', 'horizon_weeks']):
        if len(g) == 2:
            n_, o_ = g[g.rule == 'new'].iloc[0], g[g.rule == 'old'].iloc[0]
            noise = np.nanmean([n_.gap_sd, o_.gap_sd])
            sh.append(dict(crop=crop, horizon_weeks=h, rule_shift_gap_pts=n_.gap_mean - o_.gap_mean, seed_sd_gap_pts=noise,
                           shift_over_noise=(n_.gap_mean - o_.gap_mean) / noise if noise else np.nan, ens_gap_new=n_.ens_gap_pct, ens_gap_old=o_.ens_gap_pct))
    shift = pd.DataFrame(sh).round(3)
    cells.to_csv(os.path.join(OUT, 'table_stability_cells%s.csv' % tag), index=False)
    if len(shift): shift.to_csv(os.path.join(OUT, 'table_stability_summary%s.csv' % tag), index=False)
    pd.set_option('display.width', 220)
    print(cells[cells.rule == 'new'][['crop', 'horizon_weeks', 'n_seeds', 'gap_mean', 'gap_sd', 'gap_min', 'gap_max', 'share_seeds_m6_better', 'sign_flips', 'wins_min', 'wins_max', 'ens_gap_pct', 'ens_m6_fold_wins', 'ens_binom_p']].to_string(index=False))
    print(shift.to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        run(sys.argv[2], int(sys.argv[3]), tuple(sys.argv[4:6]) if len(sys.argv) > 5 else ('M0', 'M6'))
    else:
        analyze(tuple(sys.argv[2:4]) if len(sys.argv) > 3 else ('M0', 'M6'))
