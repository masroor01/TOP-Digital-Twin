# -*- coding: utf-8 -*-
"""
Script 73 -- Onion forward confirmatory test: do the data layers improve early-warning decisions? (Arm D3)
=========================================================================================================
Retrospective tests (Scripts 70-72) did not show that the layers help the early-warning decision, except possibly for onion
under the calibrated-probability rule (exploratory, wide intervals).  This script runs the CONFIRMATORY test on weeks whose
outcomes are not yet known.  Everything is frozen before any outcome exists; protocol: Model_Output/forward_test_onion/PROTOCOL.md.

Modes
  freeze    train and save the final models once (price-only M0E vs layered LAY; spike and crash; h = 4, 13, 26; 5 seeds; isotonic
            calibration) on all data whose targets are known; write a SHA-256 manifest (git-tracked).  Refuses to run twice.
  score     (run after each weekly refresh) log calibrated probabilities for every onion market at every origin week >= FIRST_ORIGIN whose
            outcome is NOT yet observable (origin + h > latest price week).  Append-only; existing keys are never overwritten.
  status    progress toward the pre-specified evaluation.
  evaluate  single pre-specified analysis once 26 origin weeks have matured at h = 13; otherwise prints progress only.

Usage: python scripts/73_Forward_Test_Onion.py <freeze|score|status|evaluate>
"""
import os, sys, io, json, hashlib, subprocess, importlib.util, datetime
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, 'Model_Output', 'forward_test_onion')
MODELS = os.path.join(D, 'frozen_models')
MANIFEST = os.path.join(D, 'freeze_manifest.json')
LOG = os.path.join(D, 'forecast_log.csv')
RESULTS = os.path.join(BASE, 'Model_Output', 'table_forward_test_onion_results.csv')
PANEL = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
FIRST_ORIGIN = pd.Timestamp('2026-07-13')
HORIZONS = [4, 13, 26]; SEEDS = [42, 43, 44, 45, 46]
UP, DN = 0.30, -0.25
ALARM_P = 0.20; CL = 0.2; REQUIRED_WEEKS = 26; BLOCK = 4; NBOOT = 5000; SEED = 20261008
CROP = 'onion'


def s71():
    spec = importlib.util.spec_from_file_location('s71', os.path.join(BASE, 'scripts', '71_Event_Models.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def features():
    m = s71(); g = m.load_features()
    df, price_cols, layer_cols = m.add_event_features(g['feat'][CROP])
    FS = g['MODEL_FEATURE_SETS']
    m0 = [c for c in FS['M0'] if c in df.columns]; m6 = [c for c in FS['M6'] if c in df.columns]
    extra = [c for c in (g['DROUGHT_FEATS'] + g['ONI_FEATS']) if c in df.columns]
    sets = {'M0E': m0 + price_cols, 'LAY': list(dict.fromkeys(m0 + price_cols + m6 + extra + layer_cols))}
    return m, df, sets


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def git_commit():
    try: return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE, capture_output=True, text=True).stdout.strip()
    except Exception: return 'unknown'


def freeze():
    if os.path.exists(MANIFEST):
        print('[73] already frozen (%s). Refusing to retrain.' % MANIFEST); return
    os.makedirs(MODELS, exist_ok=True)
    m, df, sets = features()
    data_through = df['week_start'].max()
    manifest = dict(frozen_at_utc=datetime.datetime.utcnow().isoformat() + 'Z', data_through=str(data_through.date()), first_origin=str(FIRST_ORIGIN.date()), code_commit=git_commit(),
                    params=m.PARAMS, seeds=SEEDS, thresholds=dict(spike=UP, crash=DN, alarm_probability=ALARM_P, cost_loss=CL), feature_sets=sets, files={})
    for h in HORIZONS:
        d = df.copy(); d['target'] = d.groupby('market_id')['log_price'].shift(-h); d = d.dropna(subset=['target', 'price_lag_1'])
        d['ret_true'] = np.expm1(d['target']) / np.expm1(d['log_price']) - 1
        d['target_week'] = d['week_start'] + pd.Timedelta(weeks=h)
        val_start = data_through - pd.Timedelta(weeks=26)
        tr = d[d['target_week'] <= val_start]; va = d[(d['target_week'] > val_start) & (d['target_week'] <= data_through)]
        for side, fn in (('spike', lambda r: r >= UP), ('crash', lambda r: r <= DN)):
            ytr, yva = fn(tr['ret_true']).astype(int).to_numpy(), fn(va['ret_true']).astype(int).to_numpy()
            for name, cols in sets.items():
                Xtr, Xva = tr[cols].fillna(0), va[cols].fillna(0)
                mods, isos = [], []
                for s in SEEDS:
                    mdl = lgb.LGBMClassifier(**m.PARAMS, random_state=s)
                    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                    pv = mdl.predict_proba(Xva)[:, 1]
                    mods.append(mdl); isos.append(IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip').fit(pv, yva))
                fn_ = 'h%d_%s_%s.joblib' % (h, side, name)
                joblib.dump(dict(models=mods, isos=isos, features=cols), os.path.join(MODELS, fn_))
                manifest['files'][fn_] = sha(os.path.join(MODELS, fn_))
                print('[73] frozen %s | train %d (events %d) val %d (events %d)' % (fn_, len(tr), ytr.sum(), len(va), yva.sum()), flush=True)
    json.dump(manifest, open(MANIFEST, 'w'), indent=1)
    print('[73] manifest written; data_through %s' % manifest['data_through'])


def load_frozen():
    man = json.load(open(MANIFEST))
    for fn, h in man['files'].items():
        assert sha(os.path.join(MODELS, fn)) == h, 'frozen model file changed: ' + fn
    return man


def score():
    man = load_frozen()
    m, df, sets = features()
    data_through = df['week_start'].max()
    old = pd.read_csv(LOG, parse_dates=['origin_week']) if os.path.exists(LOG) else pd.DataFrame(columns=['origin_week', 'h'])
    done = set(zip(old['origin_week'], old['h']))
    new_rows = []
    for h in HORIZONS:
        for ow in sorted(df.loc[df['week_start'] >= FIRST_ORIGIN, 'week_start'].unique()):
            ow = pd.Timestamp(ow)
            if ow + pd.Timedelta(weeks=h) <= data_through: continue        # outcome already observable: not a forward forecast
            if (ow, h) in done: continue
            rows = df[df['week_start'] == ow]
            out = pd.DataFrame({'scored_at_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'data_through': str(data_through.date()), 'code_commit': git_commit(),
                                'origin_week': ow, 'h': h, 'market_id': rows['market_id'].to_numpy()})
            for side in ('spike', 'crash'):
                for name, cols in sets.items():
                    pk = joblib.load(os.path.join(MODELS, 'h%d_%s_%s.joblib' % (h, side, name)))
                    X = rows[pk['features']].fillna(0)
                    pr = np.mean([mdl.predict_proba(X)[:, 1] for mdl in pk['models']], axis=0)
                    pc = np.mean([iso.predict(mdl.predict_proba(X)[:, 1]) for mdl, iso in zip(pk['models'], pk['isos'])], axis=0)
                    out['p_raw_%s_%s' % (name, side)] = pr; out['p_cal_%s_%s' % (name, side)] = pc
            new_rows.append(out)
    if not new_rows:
        print('[73] nothing new to log (data through %s)' % data_through.date()); return
    new = pd.concat(new_rows, ignore_index=True)
    os.makedirs(D, exist_ok=True)
    new.round(5).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
    print('[73] logged %d rows (%d origin-horizon pairs); data through %s' % (len(new), new.groupby(['origin_week', 'h']).ngroups, data_through.date()))


def matured(h, data_through):
    lg = pd.read_csv(LOG, parse_dates=['origin_week']); lg = lg[lg.h == h]
    return lg[lg['origin_week'] + pd.Timedelta(weeks=h) <= data_through]


def panel():
    p = pd.read_csv(PANEL, parse_dates=['week_start'], usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted'])
    p = p[p.crop == CROP].set_index(['market_id', 'week_start'])['modal_price_weighted']
    return p


def status():
    p = panel(); dt = p.index.get_level_values(1).max()
    lg = pd.read_csv(LOG, parse_dates=['origin_week'])
    for h in HORIZONS:
        mat = matured(h, dt)
        print('h=%2d: %d origin weeks logged, %d matured (data through %s)' % (h, lg[lg.h == h].origin_week.nunique(), mat.origin_week.nunique(), dt.date()))
    final_origin = FIRST_ORIGIN + pd.Timedelta(weeks=REQUIRED_WEEKS - 1); ready = final_origin + pd.Timedelta(weeks=13)
    print('primary evaluation (h = 13) needs %d matured origin weeks: last origin %s, evaluable from %s' % (REQUIRED_WEEKS, final_origin.date(), ready.date()))


def value(c, cl):
    hits, miss, fa, cn = c; n = c.sum(); s = (hits + miss) / n
    e_clim, e_perf, e_rule = min(cl, s), s * cl, (cl * (hits + fa) + miss) / n
    return (e_clim - e_rule) / (e_clim - e_perf) if e_clim > e_perf else np.nan


def evaluate():
    p = panel(); dt = p.index.get_level_values(1).max()
    rng = np.random.default_rng(SEED); res = []
    for h in HORIZONS:
        mat = matured(h, dt)
        weeks_all = sorted(mat.origin_week.unique())
        final = (h == 13)
        if len(weeks_all) < (REQUIRED_WEEKS if final else 1):
            print('h=%d: %d matured origin weeks; primary analysis (h = 13) not yet evaluable (%d required).' % (h, len(weeks_all), REQUIRED_WEEKS)); continue
        weeks = weeks_all[:REQUIRED_WEEKS] if final else weeks_all
        mat = mat[mat.origin_week.isin(weeks)].copy()
        p0 = p.reindex(pd.MultiIndex.from_arrays([mat.market_id, mat.origin_week])).to_numpy()
        p1 = p.reindex(pd.MultiIndex.from_arrays([mat.market_id, mat.origin_week + pd.Timedelta(weeks=h)])).to_numpy()
        mat['ret'] = p1 / p0 - 1; mat = mat.dropna(subset=['ret'])
        widx = np.searchsorted(np.array(weeks, dtype='datetime64[ns]'), mat.origin_week.to_numpy()); W = len(weeks)
        for side in ('spike', 'crash'):
            e = (mat['ret'] >= UP).to_numpy() if side == 'spike' else (mat['ret'] <= DN).to_numpy()
            cm = {}
            for name in ('M0E', 'LAY'):
                a = (mat['p_cal_%s_%s' % (name, side)] >= ALARM_P).to_numpy()
                cm[name] = np.stack([np.bincount(widx[mask], minlength=W) for mask in (e & a, e & ~a, ~e & a, ~e & ~a)], axis=1).astype(float)
            vobs = {n: value(c.sum(axis=0), CL) for n, c in cm.items()}
            nb = int(np.ceil(W / BLOCK)); diffs = np.zeros(NBOOT)
            for b in range(NBOOT):
                sel = ((rng.integers(0, W, size=nb)[:, None] + np.arange(BLOCK)[None, :]) % W).ravel()[:W]
                diffs[b] = value(cm['LAY'][sel].sum(axis=0), CL) - value(cm['M0E'][sel].sum(axis=0), CL)
            lo, hi = np.nanpercentile(diffs, [5, 95])
            res.append(dict(horizon_weeks=h, side=side, origin_weeks=W, n=len(mat), base_rate=float(e.mean()), V_LAY=vobs['LAY'], V_M0E=vobs['M0E'], diff=vobs['LAY'] - vobs['M0E'], ci90_lo=lo, ci90_hi=hi, ci_above_zero=bool(lo > 0), primary=final))
    if not res: return
    R = pd.DataFrame(res); R.round(4).to_csv(RESULTS, index=False)
    print(R.round(3).to_string(index=False))
    pr = R[R.primary]
    if len(pr) == 2:
        print('CONFIRMATORY VERDICT (onion, h = 13, both sides required): %s' % ('layers help the decision: CONFIRMED' if pr.ci_above_zero.all() else 'NOT confirmed'))


if __name__ == '__main__':
    mode = sys.argv[1]
    {'freeze': freeze, 'score': score, 'status': status, 'evaluate': evaluate}[mode]()
