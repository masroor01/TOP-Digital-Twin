# -*- coding: utf-8 -*-
"""
Script 82 -- V2 data fix tests (T1 potato coverage, T2 spatial weather, T3 observed-only features).
Protocol frozen in docs/V2_DATA_FIXES_PROTOCOL.md. Baseline P = M0P / M6P of Script 81 (origin-week price included).

Usage (one test per process: Script 15 can be loaded only once per process):
  python scripts/82_V2_Data_Fix_Tests.py t1        potato on the expanded panel (needs Model_Output/_v2_experiment/potato_expanded, built by Script 09 with TOP_POTATO_MIN_YEARS=0)
  python scripts/82_V2_Data_Fix_Tests.py t2        spatial weather, M6S
  python scripts/82_V2_Data_Fix_Tests.py t3        observed-only features and rows, M0O / M6O
  python scripts/82_V2_Data_Fix_Tests.py --summarise
Outputs: Model_Output/_v2_runs/*.parquet (git-ignored); table_v2_fix_{summary,tests}.csv
"""
import io, os, sys, glob, time, importlib.util
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output'); RUNS = os.path.join(OUT, '_v2_runs'); R81 = os.path.join(OUT, '_origin_fix_runs')
HORIZONS = [1, 4, 13, 26]; NBOOT, BLOCK, SEED = 2000, 13, 20261028
spec = importlib.util.spec_from_file_location('s75', os.path.join(BASE, 'scripts', '75_Onion_Short_Horizon_Test.py'))
s75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s75)
EXP_PANEL = os.path.join(OUT, '_v2_experiment', 'potato_expanded', 'potato_weekly_panel.csv')


def load_g(potato_panel=None):
    """Script 15 up to the ablation loop; optionally with the potato rows of the panel replaced by an experimental panel."""
    sys.path.insert(0, os.path.join(BASE, 'scripts'))
    import gpu_utils
    gpu_utils._cache['lgbm'] = {}
    path = os.path.join(BASE, 'scripts', '15_Ablation_Study_M0_M4.py')
    src = io.open(path, encoding='utf-8').read()
    cut = src.rfind('print(', 0, src.index('Running ablation: M0'))
    code = src[:cut]
    if potato_panel:
        tmp = os.path.join(OUT, '_v2_experiment', 'panel_t1.csv')
        base = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv'))
        pot = pd.read_csv(potato_panel)
        pd.concat([base[base['crop'] != 'potato'], pot], ignore_index=True).to_csv(tmp, index=False)
        line = "AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')"
        assert line in code
        code = code.replace(line, "AGM_FILE = %r" % tmp)
    g = {'__name__': '__not_main__', '__file__': path}
    exec(compile(code, path, 'exec'), g)
    return g


def folds_iter(g, df, h, need):
    for fi in g['FOLDS']:
        t_end, v_s, v_e = pd.Timestamp(fi['train_end']), pd.Timestamp(fi['val_start']), pd.Timestamp(fi['val_end'])
        te_s, te_e = pd.Timestamp(fi['test_start']), pd.Timestamp(fi['test_end'])
        tr = df[df['week_start'] + pd.Timedelta(weeks=h) <= t_end]
        va = df[(df['week_start'] + pd.Timedelta(weeks=h)).between(v_s, v_e)]
        te = df[(df['week_start'] >= te_s) & (df['week_start'] <= te_e)]
        if len(tr) < 100 or len(te) < 10:
            continue
        yield fi['fold'], tr, va, te


def train_variants(g, feat_crop, crop, h, variants, tag, need=('target', 'price_lag_1'), fill0=True):
    df = feat_crop.sort_values(['market_id', 'week_start']).copy()
    df['target'] = df.groupby('market_id')['log_price'].shift(-h)
    df = df.dropna(subset=list(need))
    rows = []
    for fold, tr, va, te in folds_iter(g, df, h, need):
        o = te[['market_id', 'week_start']].copy(); o['market_id'] = o['market_id'].astype(int)
        o['crop'] = crop; o['horizon_weeks'] = h; o['fold'] = fold; o['y_true'] = np.expm1(te['target'].to_numpy()); o['y_last'] = np.expm1(te['log_price'].to_numpy())
        for name, cols in variants.items():
            t0 = time.time()
            f = (lambda x: x.fillna(0)) if fill0 else (lambda x: x)      # observed-only variant keeps missing lags as NaN (LightGBM handles them)
            _, p = s75.fit_predict(g, f(tr[cols]), tr['target'], f(va[cols]), va['target'], f(te[cols]), cols)
            o['y_' + name] = np.expm1(p)
            print('[82] %s %s h=%d fold%d %s n_test=%d [%.0fs]' % (tag, crop, h, fold, name, len(te), time.time() - t0), flush=True)
        rows.append(o)
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(RUNS, '%s_%s_h%d.parquet' % (tag, crop, h)), index=False)


def feature_sets(g, df, suffix=''):
    FS = g['MODEL_FEATURE_SETS']
    return {'M0' + suffix: [c for c in FS['M0'] if c in df.columns] + ['log_price'], 'M6' + suffix: [c for c in FS['M6'] if c in df.columns] + ['log_price']}


def zone_overlay(g):
    """Replace the crop-level weather/satellite columns by each market's own production-zone series where one is assigned."""
    mz = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'market_zone_features.csv'), parse_dates=['week_start'])
    mz['market_id'] = mz['market_id'].astype(int)
    mz = mz.sort_values(['crop', 'market_id', 'week_start']).reset_index(drop=True)
    roll_specs = [('era5_heat_35', 'sum', [4, 8]), ('chirps_rain_mm', 'sum', [4, 8]), ('s2_ndvi', 'mean', [4]), ('s2_ndvi_anom', 'mean', [4]), ('modis_lst_mean', 'mean', [4])]
    for col, func, windows in roll_specs:
        if col not in mz.columns:
            continue
        for w in windows:
            gb = mz.groupby(['crop', 'market_id'])[col]
            mz['%s_roll%d' % (col, w)] = gb.transform(lambda x: x.shift(1).rolling(w, min_periods=2).sum() if func == 'sum' else x.shift(1).rolling(w, min_periods=2).mean())
    df = g['df'].copy(); df['market_id'] = df['market_id'].astype(int)
    cols = [c for c in (g['CLIMATE_FEATS'] + g['SAT_FEATS']) if c in mz.columns and c in df.columns]
    m = df.merge(mz[['crop', 'market_id', 'week_start'] + cols], on=['crop', 'market_id', 'week_start'], how='left', suffixes=('', '__mz'))
    n_over = 0
    for c in cols:
        has = m[c + '__mz'].notna()
        n_over += int(has.sum()); m[c] = m[c + '__mz'].where(has, m[c]); m = m.drop(columns=[c + '__mz'])
    print('[82] spatial overlay: %d columns, %d cell-values replaced' % (len(cols), n_over), flush=True)
    return m


def run_t1():
    os.makedirs(RUNS, exist_ok=True)
    g = load_g(potato_panel=EXP_PANEL)
    feat = g['feat']['potato']; vs = feature_sets(g, feat, 'X1')
    for h in HORIZONS:
        train_variants(g, feat, 'potato', h, vs, 't1')


def run_t2():
    os.makedirs(RUNS, exist_ok=True)
    g = load_g()
    df2 = zone_overlay(g)
    feat2 = g['build_features'](df2)
    for crop in ('tomato', 'onion', 'potato'):
        vs = {k: v for k, v in feature_sets(g, feat2[crop], 'S').items() if k.startswith('M6')}
        for h in HORIZONS:
            train_variants(g, feat2[crop], crop, h, vs, 't2')


def run_t3():
    os.makedirs(RUNS, exist_ok=True)
    g = load_g()
    df = g['df'].copy()
    df.loc[df['imputed'] == 1, 'modal_price_weighted'] = np.nan
    feat3 = g['build_features'](df)
    for crop in ('tomato', 'onion', 'potato'):
        vs = feature_sets(g, feat3[crop], 'O')
        for h in HORIZONS:
            train_variants(g, feat3[crop], crop, h, vs, 't3', need=('target', 'log_price', 'price_lag_1'), fill0=False)


def mae(d, c): return (d[c] - d['y_true']).abs().sum()
def skill(d, c): return (1 - mae(d, c) / (d['y_last'] - d['y_true']).abs().sum()) * 100


def compare(g, base, new, ci_rng, fold_need):
    """Compare two prediction columns on the same rows: pooled skill, folds new better, WAPE gain with block-bootstrap CI."""
    weeks = np.sort(g['week_start'].unique()); widx = {w: i for i, w in enumerate(weeks)}; gi = g['week_start'].map(widx).to_numpy(); W = len(weeks)
    den = np.bincount(gi, weights=g['y_true'].to_numpy(), minlength=W)
    ae = {v: np.bincount(gi, weights=(g[v] - g['y_true']).abs().to_numpy(), minlength=W) for v in (base, new)}
    out = []
    for _ in range(NBOOT):
        st = ci_rng.integers(0, W, size=int(np.ceil(W / BLOCK))); sel = np.concatenate([np.arange(s_, min(s_ + BLOCK, W)) for s_ in st])[:W]
        out.append((ae[base][sel].sum() - ae[new][sel].sum()) / den[sel].sum() * 100)
    lo, hi = np.percentile(out, [5, 95])
    wins = sum(1 for _, gf in g.groupby('fold') if skill(gf, new) > skill(gf, base)); nf = g['fold'].nunique()
    return dict(skill_base=round(skill(g, base), 1), skill_new=round(skill(g, new), 1), wape_gain_pp=round((ae[base].sum() - ae[new].sum()) / den.sum() * 100, 2),
                ci90_lo=round(lo, 2), ci90_hi=round(hi, 2), folds_new_better=wins, n_folds=nf, passes=bool(wins >= fold_need and lo > 0))


def load_runs(tag):
    fs = sorted(glob.glob(os.path.join(RUNS, tag + '_*.parquet')))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); d['week_start'] = pd.to_datetime(d['week_start']); return d


def p_runs():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(R81, 'fix_*.parquet')))], ignore_index=True)
    d['week_start'] = pd.to_datetime(d['week_start']); d['market_id'] = d['market_id'].astype(int)
    return d.rename(columns={'y_M0P': 'y_M0P', 'y_M6P': 'y_M6P'})[['crop', 'horizon_weeks', 'fold', 'market_id', 'week_start', 'y_true', 'y_last', 'y_M0P', 'y_M6P']]


def summarise():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    rng = np.random.default_rng(SEED); P = p_runs(); K = ['crop', 'horizon_weeks', 'fold', 'market_id', 'week_start']
    res = []
    d1 = load_runs('t1')
    if d1 is not None:
        d1['market_id'] = d1['market_id'].astype(int)
        orig = P[P.crop == 'potato'][K].assign(orig=1)
        m = d1.merge(P[P.crop == 'potato'][K + ['y_M0P', 'y_M6P']], on=K, how='left'); m['orig'] = m['y_M6P'].notna()
        for h, gh in m.groupby('horizon_weeks'):
            go = gh[gh['orig']]
            for v in ('M0', 'M6'):
                r = compare(go, 'y_%sP' % v, 'y_%sX1' % v, rng, 4); r.update(test='T1 potato coverage (original 83 markets)', crop='potato', horizon_weeks=h, comparison='%sP vs %sX1' % (v, v)); res.append(r)
            gn = gh[~gh['orig']]
            for v in ('M0', 'M6'):
                res.append(dict(test='T1 potato coverage (new markets only, descriptive)', crop='potato', horizon_weeks=h, comparison='%sX1 skill on %d new-market rows' % (v, len(gn)), skill_base=np.nan, skill_new=round(skill(gn, 'y_%sX1' % v), 1), wape_gain_pp=np.nan, ci90_lo=np.nan, ci90_hi=np.nan, folds_new_better=np.nan, n_folds=gn.fold.nunique(), passes=np.nan))
    d2 = load_runs('t2')
    if d2 is not None:
        d2['market_id'] = d2['market_id'].astype(int)
        m = d2.merge(P[K + ['y_M0P', 'y_M6P']], on=K, how='inner')
        for (crop, h), g in m.groupby(['crop', 'horizon_weeks']):
            r = compare(g, 'y_M6P', 'y_M6S', rng, 4); r.update(test='T2 spatial weather', crop=crop, horizon_weeks=h, comparison='M6S vs M6P'); res.append(r)
            r = compare(g, 'y_M0P', 'y_M6S', rng, 5); r.update(test='T2 spatial weather', crop=crop, horizon_weeks=h, comparison='layers: M6S vs M0P'); res.append(r)
    d3 = load_runs('t3')
    if d3 is not None:
        d3['market_id'] = d3['market_id'].astype(int)
        m = d3.merge(P[K + ['y_M0P', 'y_M6P']], on=K, how='inner')
        for (crop, h), g in m.groupby(['crop', 'horizon_weeks']):
            for v in ('M0', 'M6'):
                r = compare(g, 'y_%sP' % v, 'y_%sO' % v, rng, 4); r.update(test='T3 observed-only', crop=crop, horizon_weeks=h, comparison='%sP vs %sO (same rows)' % (v, v)); res.append(r)
            r = compare(g, 'y_M0O', 'y_M6O', rng, 5); r.update(test='T3 observed-only', crop=crop, horizon_weeks=h, comparison='layers: M6O vs M0O'); res.append(r)
    T = pd.DataFrame(res)
    cols = ['test', 'crop', 'horizon_weeks', 'comparison', 'skill_base', 'skill_new', 'wape_gain_pp', 'ci90_lo', 'ci90_hi', 'folds_new_better', 'n_folds', 'passes']
    T = T[cols]; T.to_csv(os.path.join(OUT, 'table_v2_fix_tests.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 300)
    print(T.to_string(index=False))
    for t, g in T[T.passes.notna()].groupby(['test', 'comparison']):
        print('%-60s passes in %d of %d cells' % (' | '.join(t), int(g.passes.sum()), len(g)))


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    {'t1': run_t1, 't2': run_t2, 't3': run_t3, '--summarise': summarise}[a]()
