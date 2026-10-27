# -*- coding: utf-8 -*-
"""
Script 81 -- Origin-week price fix test. Protocol frozen in docs/ORIGIN_PRICE_FIX_PROTOCOL.md.
Variants: M0, M6 (current; predictions read from Model_Output/dm_market_level_predictions.csv, produced by Script 15),
M0P = M0 + log_price (price in the origin week), M6P = M6 + log_price (trained here, same folds / settings / seed).

Usage:  python scripts/81_Origin_Price_Fix_Test.py [tomato|onion|potato ...] [h ...]     (train M0P and M6P)
        python scripts/81_Origin_Price_Fix_Test.py --summarise                           (score everything)
Outputs (Model_Output/): table_origin_price_fix_summary.csv, _folds.csv, _tests.csv; raw predictions in _origin_fix_runs/ (git-ignored)
"""
import io, os, sys, glob, time, importlib.util
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output'); RUNS = os.path.join(OUT, '_origin_fix_runs')
CROPS = ['tomato', 'onion', 'potato']; HORIZONS = [1, 4, 13, 26]
NBOOT, BLOCK, SEED = 2000, 13, 20261027
spec = importlib.util.spec_from_file_location('s75', os.path.join(BASE, 'scripts', '75_Onion_Short_Horizon_Test.py'))
s75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s75)


def run(crop, h, g):
    os.makedirs(RUNS, exist_ok=True)
    feat = g['feat']; FS = g['MODEL_FEATURE_SETS']; FOLDS = g['FOLDS']
    df = feat[crop].sort_values(['market_id', 'week_start']).copy()
    df['target'] = df.groupby('market_id')['log_price'].shift(-h)
    df = df.dropna(subset=['target', 'price_lag_1'])
    sets = {'M0P': [c for c in FS['M0'] if c in df.columns] + ['log_price'], 'M6P': [c for c in FS['M6'] if c in df.columns] + ['log_price']}
    rows = []
    for fi in FOLDS:
        t_end, v_s, v_e = pd.Timestamp(fi['train_end']), pd.Timestamp(fi['val_start']), pd.Timestamp(fi['val_end'])
        te_s, te_e = pd.Timestamp(fi['test_start']), pd.Timestamp(fi['test_end'])
        tr = df[df['week_start'] + pd.Timedelta(weeks=h) <= t_end]
        va = df[(df['week_start'] + pd.Timedelta(weeks=h)).between(v_s, v_e)]
        te = df[(df['week_start'] >= te_s) & (df['week_start'] <= te_e)]
        if len(tr) < 100 or len(te) < 10:
            continue
        o = te[['market_id', 'week_start']].copy()
        o['crop'] = crop; o['horizon_weeks'] = h; o['fold'] = fi['fold']; o['y_true'] = np.expm1(te['target'].to_numpy()); o['y_last'] = np.expm1(te['log_price'].to_numpy())
        for name, cols in sets.items():
            t0 = time.time()
            _, p = s75.fit_predict(g, tr[cols].fillna(0), tr['target'], va[cols].fillna(0), va['target'], te[cols].fillna(0), cols)
            o['y_' + name] = np.expm1(p)
            print('[81] %s h=%d fold%d %s  n_test=%d  [%.0fs]' % (crop, h, fi['fold'], name, len(te), time.time() - t0), flush=True)
        rows.append(o)
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(RUNS, 'fix_%s_h%d.parquet' % (crop, h)), index=False)


def mae(d, col): return (d[col] - d['y_true']).abs().sum()
def skill(d, col): return (1 - mae(d, col) / (d['y_last'] - d['y_true']).abs().sum()) * 100
def acc(d, col): return 100 - mae(d, col) / d['y_true'].sum() * 100
def direction(d, col):
    a = np.sign(d['y_true'] - d['y_last']); p = np.sign(d[col] - d['y_last']); m = (a != 0) & (p != 0)
    return 100 * (a[m] == p[m]).mean()


def summarise():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RUNS, 'fix_*.parquet')))], ignore_index=True)
    d['week_start'] = pd.to_datetime(d['week_start']); d['market_id'] = d['market_id'].astype(int)
    ex = pd.read_csv(os.path.join(OUT, 'dm_market_level_predictions.csv'), parse_dates=['week_start'])
    ex['market_id'] = ex['market_id'].astype(int)
    for v in ('M0', 'M6'):
        e = ex[ex['variant'] == v][['crop', 'horizon_weeks', 'fold', 'market_id', 'week_start', 'y_pred']].rename(columns={'y_pred': 'y_' + v})
        d = d.merge(e, on=['crop', 'horizon_weeks', 'fold', 'market_id', 'week_start'], how='inner')
    d = d.dropna(subset=['y_last', 'y_true', 'y_M0', 'y_M6', 'y_M0P', 'y_M6P'])
    print('[81] rows scored (all four variants, same rows): %d' % len(d))
    V = ['M0', 'M0P', 'M6', 'M6P']
    rng = np.random.default_rng(SEED)
    summ, folds, tests = [], [], []
    for (crop, h), g in d.groupby(['crop', 'horizon_weeks']):
        r = dict(crop=crop, horizon_weeks=h, n=len(g))
        for v in V:
            r['skill_' + v] = round(skill(g, 'y_' + v), 1); r['acc_' + v] = round(acc(g, 'y_' + v), 1); r['dir_' + v] = round(direction(g, 'y_' + v), 1)
        summ.append(r)
        for f, gf in g.groupby('fold'):
            folds.append({'crop': crop, 'horizon_weeks': h, 'fold': f, **{'skill_' + v: round(skill(gf, 'y_' + v), 1) for v in V}})
        weeks = np.sort(g['week_start'].unique()); widx = {w: i for i, w in enumerate(weeks)}; gi = g['week_start'].map(widx).to_numpy(); W = len(weeks)
        den = np.bincount(gi, weights=g['y_true'].to_numpy(), minlength=W)
        ae = {v: np.bincount(gi, weights=(g['y_' + v] - g['y_true']).abs().to_numpy(), minlength=W) for v in V}
        fold_sk = {f: {v: skill(gf, 'y_' + v) for v in V} for f, gf in g.groupby('fold')}
        def boot(a, b):
            out = []
            for _ in range(NBOOT):
                st = rng.integers(0, W, size=int(np.ceil(W / BLOCK))); sel = np.concatenate([np.arange(s_, min(s_ + BLOCK, W)) for s_ in st])[:W]
                out.append((ae[a][sel].sum() - ae[b][sel].sum()) / den[sel].sum() * 100)
            return np.percentile(out, [5, 95])
        for lab, base, new in (('M0P vs M0', 'M0', 'M0P'), ('M6P vs M6', 'M6', 'M6P'), ('layers: M6P vs M0P', 'M0P', 'M6P')):
            lo, hi = boot(base, new); wins = sum(1 for f in fold_sk if fold_sk[f][new] > fold_sk[f][base]); n_f = len(fold_sk)
            diff = (ae[base].sum() - ae[new].sum()) / den.sum() * 100
            need = 5 if lab.startswith('layers') else 4
            tests.append(dict(crop=crop, horizon_weeks=h, comparison=lab, skill_base=round(skill(g, 'y_' + base), 1), skill_new=round(skill(g, 'y_' + new), 1), wape_gain_pp=round(diff, 2),
                              ci90_lo=round(lo, 2), ci90_hi=round(hi, 2), folds_new_better=wins, n_folds=n_f, passes=bool(wins >= need and lo > 0)))
        for v in ('M0P', 'M6P'):
            pos = sum(1 for f in fold_sk if fold_sk[f][v] > 0)
            tests.append(dict(crop=crop, horizon_weeks=h, comparison='beats no-change: ' + v, skill_base=0.0, skill_new=round(skill(g, 'y_' + v), 1), wape_gain_pp=np.nan, ci90_lo=np.nan, ci90_hi=np.nan,
                              folds_new_better=pos, n_folds=len(fold_sk), passes=bool(skill(g, 'y_' + v) > 0 and pos >= 4)))
    S, F, T = pd.DataFrame(summ), pd.DataFrame(folds), pd.DataFrame(tests)
    S.to_csv(os.path.join(OUT, 'table_origin_price_fix_summary.csv'), index=False)
    F.to_csv(os.path.join(OUT, 'table_origin_price_fix_folds.csv'), index=False)
    T.to_csv(os.path.join(OUT, 'table_origin_price_fix_tests.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(S.to_string(index=False)); print(T.to_string(index=False))
    for lab in ('M0P vs M0', 'M6P vs M6', 'layers: M6P vs M0P', 'beats no-change: M0P', 'beats no-change: M6P'):
        print('%-26s passes in %d of %d cells' % (lab, int(T[T.comparison == lab].passes.sum()), int((T.comparison == lab).sum())))


if __name__ == '__main__':
    if '--summarise' in sys.argv:
        summarise()
    else:
        crops = [a for a in sys.argv[1:] if a in CROPS] or CROPS
        hs = [int(a) for a in sys.argv[1:] if a.isdigit()] or HORIZONS
        G = s75.load_g()
        for c in crops:
            for h in hs:
                run(c, h, G)
