# -*- coding: utf-8 -*-
"""
Script 93 -- Protocol E: can the model's response to macro levers be enlarged without losing accuracy?
=====================================================================================================
Frozen protocol: Model_Output/lever_responsiveness/PROTOCOL_E_lever_specification.md (committed and pushed 2026-11-23, before any
estimate).  Run once against that text; every deviation is reported as a deviation.

Specifications (fixed): L0 = current M6 (control); L1 = M6 with the macro levels replaced by their deviation from the mean of the
previous 52 weeks (origin-week information only); L2 = L1 plus non-decreasing price in the diesel and crude-oil deviations.
Evaluation: Script 15's own diagnostic mode (five expanding folds, test years 2022-2026, same panel, hyperparameters and
validation-window rule), patched IN MEMORY (Script 15 is not modified), five seeds, pairs (M6, L1) and (M6, L2) per seed.

Usage
  python scripts/93_Lever_Specification_Test.py run all          # 2 pairs x 5 seeds (skips runs already saved)
  python scripts/93_Lever_Specification_Test.py run L1 42        # one run: pair (M6, L1), seed 42
  python scripts/93_Lever_Specification_Test.py analyze          # tables + decision from all saved runs
Runs are saved to Model_Output/lever_responsiveness/_runs/ (git-ignored); the shared dm_market_level_*.csv files are backed up
before and restored after every run (as in Script 69).

Outputs (Model_Output/lever_responsiveness/): table_spec_skill.csv, table_spec_lever_response.csv, table_spec_decision.csv
"""
import glob
import io
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTM = os.path.join(BASE, 'Model_Output')
OUT = os.path.join(OUTM, 'lever_responsiveness')
RUNS = os.path.join(OUT, '_runs')
PRED = os.path.join(OUTM, 'dm_market_level_predictions.csv')
RAW = os.path.join(OUTM, 'dm_market_level_raw_results.csv')
SEEDS = [42, 43, 44, 45, 46]
CANDS = ['L1', 'L2']
COST_PUSH = ('diesel_4city_rs_litre', 'crude_oil_usd_bbl')

# ------------------------------------------------------------------------------ patches applied to Script 15's source (in memory)
INJECT_FEATURES = '''feat = build_features(df)
# ---- Script 93 injection: macro deviation features (mean of the PREVIOUS 52 weeks, origin-week information only) ----
LEVER_RESP = []
_present = [c for c in MACRO_COLS if c in feat['tomato'].columns]
MACRO_DEV = []
for _crop in list(feat.keys()):
    _sub = feat[_crop]
    _wk = _sub.groupby('week_start')[_present].first().sort_index()
    _trm = _wk.shift(1).rolling(52, min_periods=26).mean()
    for _c in _present:
        _sub[_c + '_trm'] = _sub['week_start'].map(_trm[_c])
        _sub[_c + '_dev'] = _sub[_c] - _sub[_c + '_trm']
    feat[_crop] = _sub
MACRO_DEV = [c + '_dev' for c in _present]
print('   [93] macro deviation features:', len(MACRO_DEV))
'''
INJECT_VARIANTS = '''MODEL_FEATURE_SETS['L1'] = PRICE_FEATS + ARR_FEATS + MACRO_DEV + CLIMATE_FEATS + SAT_FEATS + INFRA_FEATS + POLICY_FEATS
MODEL_FEATURE_SETS['L2'] = list(MODEL_FEATURE_SETS['L1'])
VARIANT_LABELS['L1'] = 'M6 with macro deviations'
VARIANT_LABELS['L2'] = 'M6 with macro deviations + fuel-cost monotone'
if MARKET_LEVEL_DIAGNOSTIC:
    a, b = DIAGNOSTIC_PAIR'''
INJECT_MONO = '''                mono = build_monotone(fcols)
                if variant == 'L2' and mono is not None:
                    mono = [(1 if c in ('diesel_4city_rs_litre_dev', 'crude_oil_usd_bbl_dev') else m_) for c, m_ in zip(fcols, mono)]
                if mono is not None:'''
INJECT_LEVER = '''                y_pred  = model.predict(X_te)
                if variant in ('M6', 'L1', 'L2'):
                    _s = X_te.sample(min(300, len(X_te)), random_state=7)
                    _base = model.predict(_s)
                    for _c in MACRO_COLS:
                        if _c not in train.columns or (variant == 'M6' and _c not in _s.columns):
                            continue
                        _lv = train[_c].astype(float)
                        _lo, _hi = _lv.min(), _lv.max()
                        if not (_hi > _lo):
                            continue
                        _preds = []
                        for _v in (_lo + 0.1 * (_hi - _lo), _lo + 0.9 * (_hi - _lo)):
                            _x = _s.copy()
                            if variant == 'M6':
                                _x[_c] = _v
                            else:
                                _x[_c + '_dev'] = _v - test.loc[_s.index, _c + '_trm'].values
                            _preds.append(model.predict(_x))
                        LEVER_RESP.append(dict(variant=variant, crop=crop, fold=fold, horizon_weeks=h, lever=_c,
                                               abs_effect=float(np.maximum(np.abs(_preds[0] - _base), np.abs(_preds[1] - _base)).mean()),
                                               signed_effect=float((_preds[1] - _preds[0]).mean())))
'''
PATCHES = [
    ("MARKET_LEVEL_DIAGNOSTIC = False", "MARKET_LEVEL_DIAGNOSTIC = True"),
    ("feat = build_features(df)\n", INJECT_FEATURES),
    ("if MARKET_LEVEL_DIAGNOSTIC:\n    a, b = DIAGNOSTIC_PAIR", INJECT_VARIANTS),
    ("                mono = build_monotone(fcols)\n                if mono is not None:", INJECT_MONO),
    ("                    'y_pred': np.expm1(y_pred),\n                })", "                    'y_pred': np.expm1(y_pred),\n                    'y_naive': np.expm1(test['log_price'].values),\n                })"),
    ("'market', 'market_id', 'week_start', 'y_true', 'y_pred']]", "'market', 'market_id', 'week_start', 'y_true', 'y_pred', 'y_naive']]"),
    ("                y_pred  = model.predict(X_te)\n", INJECT_LEVER),
]


def run(cand, seed):
    assert cand in CANDS
    os.makedirs(RUNS, exist_ok=True)
    fp = os.path.join(RUNS, '%s_%d_pred.parquet' % (cand, seed))
    if os.path.exists(fp):
        print('[93] %s seed %d already saved' % (cand, seed)); return
    sys.path.insert(0, os.path.join(BASE, 'scripts'))
    import gpu_utils
    gpu_utils._cache['lgbm'] = {}
    path = os.path.join(BASE, 'scripts', '15_Ablation_Study_M0_M4.py')
    src = io.open(path, encoding='utf-8').read()
    for a, b in PATCHES:
        # the prediction-frame anchor also occurs in the persistence block (identical lines); the FIRST occurrence is the model block
        assert src.count(a) == 1 or (a.startswith("                    'y_pred': np.expm1") and src.count(a) == 2), 'patch anchor not found/unique: %r (%d)' % (a[:60], src.count(a))
        src = src.replace(a, b, 1)
    for a, b in [("DIAGNOSTIC_PAIR = ('M0', 'M6')", "DIAGNOSTIC_PAIR = ('M6', %r)" % cand), ("SEED     = 42", "SEED     = %d" % seed)]:
        assert a in src, a
        src = src.replace(a, b)
    bak = [(PRED, PRED + '.l93bak'), (RAW, RAW + '.l93bak')]
    for a, b in bak:
        shutil.copy2(a, b)
    ns = {'__name__': '__main__', '__file__': path}
    try:
        print('[93] run cand=%s seed=%d' % (cand, seed), flush=True)
        try:
            exec(compile(src, path, 'exec'), ns)
        except SystemExit:
            pass
        p = pd.read_csv(PRED, parse_dates=['week_start'])
        assert set(p['variant'].unique()) == {'M6', cand}, p['variant'].unique()
        for c in ('y_true', 'y_pred', 'y_naive'):
            p[c] = p[c].astype('float32')
        lev = pd.DataFrame(ns['LEVER_RESP'])
        lev.to_parquet(os.path.join(RUNS, '%s_%d_lever.parquet' % (cand, seed)), index=False)
        p.drop(columns=['market']).to_parquet(fp, index=False)
        print('[93] saved %s seed %d (%d prediction rows, %d lever rows)' % (cand, seed, len(p), len(lev)))
    finally:
        for a, b in bak:
            shutil.copy2(b, a); os.remove(b)


# ------------------------------------------------------------------------------ analysis
KEY = ['variant', 'crop', 'fold', 'horizon_weeks', 'market_id', 'week_start']


def skill_table(p):
    """Headline definition: per-fold pooled MAE, averaged over folds, divided by the persistence MAE -> skill = (1 - MASE) x 100."""
    d = p.assign(ae=(p.y_true.astype(float) - p.y_pred.astype(float)).abs(), an=(p.y_true.astype(float) - p.y_naive.astype(float)).abs())
    f = d.groupby(['variant', 'crop', 'horizon_weeks', 'fold']).agg(mae=('ae', 'mean'), mae_naive=('an', 'mean')).reset_index()
    g = f.groupby(['variant', 'crop', 'horizon_weeks']).agg(mae=('mae', 'mean'), mae_naive=('mae_naive', 'mean')).reset_index()
    g['skill'] = (1 - g.mae / g.mae_naive) * 100
    return g, f


def dm_hln(d, h):
    """Diebold-Mariano on a loss differential series with Newey-West variance (h-1 lags) and the HLN small-sample correction."""
    d = np.asarray(d, float); T = len(d)
    dbar = d.mean(); dc = d - dbar
    L = max(h - 1, 0)
    lrv = (dc @ dc) / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / T
    lrv = max(lrv, 1e-12)
    stat = dbar / np.sqrt(lrv / T)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(stat), float(2 * stats.t.sf(abs(stat), T - 1)), float(dbar)


def analyze():
    rows, lever_rows, ens_inputs = [], [], {}
    for cand in CANDS:
        files = sorted(glob.glob(os.path.join(RUNS, '%s_*_pred.parquet' % cand)))
        assert files, 'no saved runs for %s' % cand
        for f in files:
            seed = int(os.path.basename(f).split('_')[1])
            p = pd.read_parquet(f)
            g, _ = skill_table(p)
            g['cand'] = cand; g['seed'] = seed
            rows.append(g)
            ens_inputs.setdefault(cand, []).append(p.set_index(KEY)[['y_true', 'y_pred', 'y_naive']].assign(seed=seed))
            lv = pd.read_parquet(os.path.join(RUNS, '%s_%d_lever.parquet' % (cand, seed)))
            lv['cand'] = cand; lv['seed'] = seed
            lever_rows.append(lv)
    skills = pd.concat(rows, ignore_index=True)
    # --- primary criterion 1: per-seed skill difference (candidate - M6, same run) ---
    piv = skills.pivot_table(index=['cand', 'seed', 'crop', 'horizon_weeks'], columns='variant', values='skill').reset_index()
    out = []
    for cand in CANDS:
        s = piv[piv.cand == cand].copy()
        s['cand_skill'] = s[cand]; s['diff'] = s[cand] - s['M6']
        for (crop, h), g in s.groupby(['crop', 'horizon_weeks']):
            # M6 seed spread over all runs of this cell (the control's noise)
            m6 = piv[(piv.crop == crop) & (piv.horizon_weeks == h)]['M6']
            tol = 2.0 if crop in ('tomato', 'onion') else max(float(m6.std(ddof=1)), 2.0)
            # ensemble over seeds for the fold-level and DM criteria
            ens = pd.concat(ens_inputs[cand])
            ens = ens.reset_index()
            e = ens[(ens.crop == crop) & (ens.horizon_weeks == h)]
            e = e.groupby(['variant', 'fold', 'market_id', 'week_start']).agg(y_true=('y_true', 'mean'), y_pred=('y_pred', 'mean'), y_naive=('y_naive', 'mean')).reset_index()
            eg, ef = skill_table(e.assign(crop=crop, horizon_weeks=h))
            fold = ef.pivot_table(index='fold', columns='variant', values='mae')
            worse_all = bool((fold[cand] > fold['M6']).all()) and len(fold) == 5
            wk = e.groupby(['variant', 'week_start']).agg(y_true=('y_true', 'mean'), y_pred=('y_pred', 'mean')).reset_index()
            a = wk[wk.variant == cand].set_index('week_start'); b = wk[wk.variant == 'M6'].set_index('week_start')
            idx = a.index.intersection(b.index)
            d = ((a.loc[idx, 'y_true'] - a.loc[idx, 'y_pred']) ** 2 - (b.loc[idx, 'y_true'] - b.loc[idx, 'y_pred']) ** 2).values
            stat, pval, dbar = dm_hln(d, int(h))
            out.append(dict(cand=cand, crop=crop, horizon_weeks=int(h), n_seeds=len(g), m6_skill=g['M6'].mean(), cand_skill=g['cand_skill'].mean(),
                            diff_mean=g['diff'].mean(), diff_min=g['diff'].min(), diff_max=g['diff'].max(), tolerance=tol,
                            crit1_ok=bool(g['diff'].mean() >= -tol), crit2_ok=not worse_all, fold_worse=int((fold[cand] > fold['M6']).sum()),
                            dm_stat=stat, dm_p=pval, crit3_ok=not (dbar > 0 and pval < 0.05),
                            ens_skill_M6=eg[eg.variant == 'M6'].skill.iloc[0], ens_skill_cand=eg[eg.variant == cand].skill.iloc[0]))
    cells = pd.DataFrame(out).round(4)
    cells.to_csv(os.path.join(OUT, 'table_spec_skill.csv'), index=False)
    # --- secondary: lever response ---
    lev = pd.concat(lever_rows, ignore_index=True)
    per_seed = lev.groupby(['cand', 'variant', 'seed', 'lever']).agg(abs_effect=('abs_effect', 'mean'), signed=('signed_effect', 'mean')).reset_index()
    resp = []
    for cand in CANDS:
        for lever, g in per_seed[(per_seed.cand == cand)].groupby('lever'):
            c = g[g.variant == cand]; m = g[g.variant == 'M6']
            eff_c, eff_m = np.expm1(c.abs_effect.mean()) * 100, np.expm1(m.abs_effect.mean()) * 100
            pos = int((c.signed > 0).sum()); neg = int((c.signed < 0).sum()); n = len(c)
            consistent = max(pos, neg) >= 4 if n >= 5 else max(pos, neg) == n
            sign_ok = (pos >= neg) if lever in COST_PUSH else True
            resp.append(dict(cand=cand, lever=lever, effect_pct_M6=eff_m, effect_pct_cand=eff_c, ratio=eff_c / eff_m if eff_m > 0 else np.nan,
                             seeds_pos=pos, seeds_neg=neg, sign_consistent=consistent, sign_ok=sign_ok,
                             qualifies=bool(eff_c >= 1.0 and eff_c >= 1.5 * eff_m and consistent and sign_ok)))
    resp = pd.DataFrame(resp).round(4)
    resp.to_csv(os.path.join(OUT, 'table_spec_lever_response.csv'), index=False)
    # --- decision ---
    dec = []
    for cand in CANDS:
        c = cells[cells.cand == cand]
        primary = bool(c.crit1_ok.all() and c.crit2_ok.all() and c.crit3_ok.all())
        nq = int(resp[resp.cand == cand].qualifies.sum())
        dec.append(dict(cand=cand, primary_all_ok=primary, crit1_cells_ok=int(c.crit1_ok.sum()), crit2_cells_ok=int(c.crit2_ok.sum()),
                        crit3_cells_ok=int(c.crit3_ok.sum()), levers_qualifying=nq, secondary_ok=nq >= 3, adopt=bool(primary and nq >= 3)))
    dec = pd.DataFrame(dec)
    dec.to_csv(os.path.join(OUT, 'table_spec_decision.csv'), index=False)
    pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)
    print(cells[['cand', 'crop', 'horizon_weeks', 'm6_skill', 'cand_skill', 'diff_mean', 'diff_min', 'diff_max', 'tolerance', 'crit1_ok', 'fold_worse', 'crit2_ok', 'dm_p', 'crit3_ok']].to_string(index=False))
    print(resp.sort_values(['cand', 'effect_pct_cand'], ascending=[True, False]).to_string(index=False))
    print(dec.to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if sys.argv[2] == 'all':
            # one process per run: Script 15 re-wraps sys.stdout, which breaks when it is exec'd twice in one process
            import subprocess
            for sd in SEEDS:
                for cd in CANDS:
                    if os.path.exists(os.path.join(RUNS, '%s_%d_pred.parquet' % (cd, sd))):
                        print('[93] %s seed %d already saved' % (cd, sd)); continue
                    r = subprocess.run([sys.executable, os.path.abspath(__file__), 'run', cd, str(sd)], env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                    if r.returncode != 0:
                        print('[93] run %s seed %d FAILED (exit %d); stopping' % (cd, sd, r.returncode)); sys.exit(1)
        else:
            run(sys.argv[2], int(sys.argv[3]))
    else:
        analyze()
