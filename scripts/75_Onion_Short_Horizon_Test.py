# -*- coding: utf-8 -*-
"""
Script 75 -- Onion short-horizon test: level target vs change target vs persistence shrinkage.
Protocol frozen before running: docs/ONION_SHORT_HORIZON_PROTOCOL.md.

Variants (onion, production M6 features, Script 15's LightGBM settings, same five rolling-origin folds):
  A  level target (current production method)
  B  change target: log price(t+h) - log price(t); forecast = log price(t) + predicted change
  C  A shrunk toward persistence, weight a in {0, .25, .5, .75, 1} chosen on the validation window only
  P  persistence (reference)

Usage:  python scripts/75_Onion_Short_Horizon_Test.py [h ...]      (default 1 4 13 26)
Output: Model_Output/_onion_sh_runs/onion_sh_h<h>.parquet (git-ignored) and, after all horizons exist,
        table_onion_short_horizon_{summary,folds,tests}.csv
"""
import io, os, sys, time, glob
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, 'Model_Output', '_onion_sh_runs')
OUT = os.path.join(BASE, 'Model_Output')
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HORIZONS = [1, 4, 13, 26]
NBOOT, BLOCK, SEED = 2000, 13, 20261010


def load_g():
    sys.path.insert(0, os.path.join(BASE, 'scripts'))
    import gpu_utils
    gpu_utils._cache['lgbm'] = {}
    path = os.path.join(BASE, 'scripts', '15_Ablation_Study_M0_M4.py')
    src = io.open(path, encoding='utf-8').read()
    cut = src.rfind('print(', 0, src.index('Running ablation: M0'))
    g = {'__name__': '__not_main__', '__file__': path}
    exec(compile(src[:cut], path, 'exec'), g)
    return g


def fit_predict(g, Xtr, ytr, Xva, yva, Xte, fcols):
    params = dict(g['LGBM_PARAMS'])
    mono = g['build_monotone'](fcols)
    if mono is not None:
        params['monotone_constraints'] = mono
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict(Xva), m.predict(Xte)


def run(h, g):
    os.makedirs(RUNS, exist_ok=True)
    feat = g['feat']; FS = g['MODEL_FEATURE_SETS']; FOLDS = g['FOLDS']
    df = feat['onion'].sort_values(['market_id', 'week_start']).copy()
    fcols = [c for c in FS['M6'] if c in df.columns]
    df['target'] = df.groupby('market_id')['log_price'].shift(-h)
    df = df.dropna(subset=['target', 'price_lag_1'])
    df['ret4'] = df['log_price'] - df['price_lag_4']
    df['nat_ret4'] = df.groupby('week_start')['ret4'].transform('median')   # known at the origin week
    rows = []
    for fi in FOLDS:
        t_end, v_s, v_e = pd.Timestamp(fi['train_end']), pd.Timestamp(fi['val_start']), pd.Timestamp(fi['val_end'])
        te_s, te_e = pd.Timestamp(fi['test_start']), pd.Timestamp(fi['test_end'])
        tr = df[df['week_start'] + pd.Timedelta(weeks=h) <= t_end]
        va = df[(df['week_start'] + pd.Timedelta(weeks=h)).between(v_s, v_e)]
        te = df[(df['week_start'] >= te_s) & (df['week_start'] <= te_e)]
        if len(tr) < 100 or len(te) < 10:
            continue
        Xtr, Xva, Xte = tr[fcols].fillna(0), va[fcols].fillna(0), te[fcols].fillna(0)
        t0 = time.time()
        a_va, a_te = fit_predict(g, Xtr, tr['target'], Xva, va['target'], Xte, fcols)                       # A level
        b_va, b_te = fit_predict(g, Xtr, tr['target'] - tr['log_price'], Xva, va['target'] - va['log_price'], Xte, fcols)  # B change
        b_te = te['log_price'].to_numpy() + b_te
        # C: shrink A toward persistence; a chosen on validation only
        err = {a: np.mean(np.abs(va['target'].to_numpy() - (va['log_price'].to_numpy() + a * (a_va - va['log_price'].to_numpy())))) for a in ALPHAS}
        a_best = min(err, key=err.get)
        c_te = te['log_price'].to_numpy() + a_best * (a_te - te['log_price'].to_numpy())
        o = te[['market_id', 'week_start', 'nat_ret4']].copy()
        o['horizon_weeks'] = h; o['fold'] = fi['fold']; o['alpha_C'] = a_best
        o['y_true'] = np.expm1(te['target'].to_numpy()); o['y_last'] = np.expm1(te['log_price'].to_numpy())
        o['y_A'] = np.expm1(a_te); o['y_B'] = np.expm1(b_te); o['y_C'] = np.expm1(c_te)
        rows.append(o)
        print('[75] h=%d fold%d  n_test=%d  alpha_C=%.2f  [%.0fs]' % (h, fi['fold'], len(te), a_best, time.time() - t0), flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(RUNS, 'onion_sh_h%d.parquet' % h), index=False)


def wape(d, col):
    return (d[col] - d['y_true']).abs().sum() / d['y_true'].sum() * 100


def skill(d, col):
    return (1 - (d[col] - d['y_true']).abs().sum() / (d['y_last'] - d['y_true']).abs().sum()) * 100


def direction(d, col):
    a = np.sign(d['y_true'] - d['y_last']); p = np.sign(d[col] - d['y_last']); m = (a != 0) & (p != 0)
    return 100 * (a[m] == p[m]).mean()


def summarise():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RUNS, 'onion_sh_h*.parquet')))], ignore_index=True)
    # rows whose origin-week price is missing have no persistence/change forecast; drop them for ALL variants (same sample)
    n0 = len(d); d = d.dropna(subset=['y_last', 'y_B', 'y_C']); print('[75] dropped %d of %d rows with no origin-week price' % (n0 - len(d), n0))
    d['week_start'] = pd.to_datetime(d['week_start']); d['regime'] = np.where(d['nat_ret4'] >= 0.15, 'rising', 'calm')
    d['realised'] = np.where(d['y_true'] / d['y_last'] - 1 >= 0.30, 'spike', 'other')
    V = ['A', 'B', 'C']
    for v in V + ['last']:
        pass
    summ, folds, tests = [], [], []
    rng = np.random.default_rng(SEED)
    for h, gh in d.groupby('horizon_weeks'):
        def row(sub, label, strat):
            r = dict(horizon_weeks=h, stratum=strat, level=label, n=len(sub))
            for v in V:
                r['WAPE_' + v] = round(wape(sub, 'y_' + v), 2); r['skill_' + v] = round(skill(sub, 'y_' + v), 1)
                r['bias_' + v] = round((sub['y_' + v] - sub['y_true']).sum() / sub['y_true'].sum() * 100, 2); r['dir_' + v] = round(direction(sub, 'y_' + v), 1)
            r['WAPE_persist'] = round(wape(sub.assign(y_P=sub['y_last']), 'y_P'), 2)
            return r
        summ.append(row(gh, 'all', 'all'))
        for k, sub in gh.groupby('regime'): summ.append(row(sub, k, 'regime_at_origin'))
        for k, sub in gh.groupby('realised'): summ.append(row(sub, k, 'realised_outcome_hindsight'))
        for f, sub in gh.groupby('fold'): folds.append({**row(sub, 'fold%d' % f, 'fold'), 'fold': f})
        # bootstrap over calendar weeks: WAPE(A) - WAPE(variant)
        weeks = np.sort(gh['week_start'].unique()); widx = {w: i for i, w in enumerate(weeks)}
        gi = gh['week_start'].map(widx).to_numpy(); W = len(weeks)
        den = np.bincount(gi, weights=gh['y_true'].to_numpy(), minlength=W)
        ae = {v: np.bincount(gi, weights=(gh['y_' + v] - gh['y_true']).abs().to_numpy(), minlength=W) for v in V}
        diffs = {v: [] for v in ('B', 'C')}
        for _ in range(NBOOT):
            st = rng.integers(0, W, size=int(np.ceil(W / BLOCK)))
            sel = np.concatenate([np.arange(s, min(s + BLOCK, W)) for s in st])[:W]
            dn = den[sel].sum()
            for v in ('B', 'C'):
                diffs[v].append((ae['A'][sel].sum() - ae[v][sel].sum()) / dn * 100)
        fold_a = {f: skill(s, 'y_A') for f, s in gh.groupby('fold')}
        calm = gh[gh['regime'] == 'calm']
        for v in ('B', 'C'):
            lo, hi = np.percentile(diffs[v], [5, 95])
            wins = sum(1 for f, s in gh.groupby('fold') if skill(s, 'y_' + v) > fold_a[f])
            calm_rel = (wape(calm, 'y_' + v) / wape(calm, 'y_A') - 1) * 100 if len(calm) else np.nan
            ok = bool(wins >= 4 and lo > 0 and (np.isnan(calm_rel) or calm_rel <= 5))
            tests.append(dict(horizon_weeks=h, variant=v, WAPE_A=round(wape(gh, 'y_A'), 2), WAPE_variant=round(wape(gh, 'y_' + v), 2),
                              diff_pp=round(wape(gh, 'y_A') - wape(gh, 'y_' + v), 2), ci90_lo=round(lo, 2), ci90_hi=round(hi, 2),
                              folds_beating_A=wins, n_folds=gh['fold'].nunique(), calm_WAPE_rel_change_pct=round(calm_rel, 1), passes_standard=ok))
    S, F, T = pd.DataFrame(summ), pd.DataFrame(folds), pd.DataFrame(tests)
    S.to_csv(os.path.join(OUT, 'table_onion_short_horizon_summary.csv'), index=False)
    F.to_csv(os.path.join(OUT, 'table_onion_short_horizon_folds.csv'), index=False)
    T.to_csv(os.path.join(OUT, 'table_onion_short_horizon_tests.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(S[S.stratum.isin(['all', 'regime_at_origin'])].to_string(index=False))
    print(T.to_string(index=False))
    print('horizons with a pass (B or C):', sorted(T[T.passes_standard].horizon_weeks.unique().tolist()))


if __name__ == '__main__':
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if '--summarise' in sys.argv:
        summarise()
    else:
        G = load_g()          # Script 15 is exec'd once per process (it rewraps stdout)
        for h in (args or HORIZONS):
            run(h, G)
