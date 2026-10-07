# -*- coding: utf-8 -*-
"""
Script 70 -- Decision-value backtest (Track B, Arm D)
=====================================================
Does an early-warning rule driven by the platform's out-of-sample forecasts beat simple rules at the
decision a State marketing board / Early-Warning Console user takes each week?  Specification and
evidence standard: docs/ARM_D_DECISION_RULE_DRAFT.md, parameters frozen 2026-10-07 BEFORE any scoring
(defaults approved by Dr. Masroor: event +30 % within 13 weeks, alarm at +20 %, cost-loss ratio 0.2 with
0.1-0.3 sensitivity, crash side secondary, M9 as a diagnostic row only).

Unit: market-week.  Decision date t = forecast origin (`week_start`); target = price at t+h (`y_true`).
Event (spike):  y_true / p_t - 1 >= +30 %.      Crash event: y_true / p_t - 1 <= -25 %.
Rules (same thresholds, frozen):
  never / always alarm; momentum (p_t / p_{t-13} - 1 >= alarm); seasonal naive (p_{t-52+13} / p_{t-52} - 1 >= alarm);
  M0 and M6 (forecast / p_t - 1 >= alarm), M6_ens and M0_ens (average of the 5 seeds in Script 69's runs), M9 (diagnostic).
  Alarm threshold: spike +20 %; crash -15 % (mirror of the 20/30 ratio).
Metrics: contingency table, POD, FAR, CSI, share of M6 hits not also flagged by momentum, and relative economic
value V(C/L) = (E_clim - E_rule) / (E_clim - E_perfect), E_clim = min(C, s*L), E_perfect = s*C (L = 1).
Inference: 13-week block bootstrap over calendar weeks (2,000 resamples) for paired differences in V; per-fold V.
Standard for "adds decision value" (per crop, primary setting h = 13, spike, alarm +20 %, C/L = 0.2): M6 V > 0 and above
momentum, seasonal naive and M0, each with a 90 % bootstrap interval above zero, and M6 beats momentum in >= 4 of 5 folds.

Run: python scripts/70_Decision_Value_Backtest.py        (minutes; no model training)
Outputs (Model_Output/): table_decision_value_summary.csv, table_decision_value_tests.csv, table_decision_value_folds.csv
"""
import os, io, glob, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
RUNS = os.path.join(OUT, '_stability_runs')
PANEL = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
SEEDS = [42, 43, 44, 45, 46]
EVENT_UP, EVENT_DN = 0.30, -0.25
ALARM_UP_PRIMARY, ALARM_DN = 0.20, -0.15
ALARM_UP_GRID = [0.10, 0.20, 0.30]
CL_GRID = [0.1, 0.2, 0.3]; CL_PRIMARY = 0.2
HORIZONS = [4, 13, 26]; PRIMARY_H = 13
BLOCK, NBOOT, SEED = 13, 2000, 20261007
CROPS = ['tomato', 'onion', 'potato']
RULES = ['never', 'always', 'momentum', 'seasonal', 'M0', 'M6', 'M0_ens', 'M6_ens', 'M9']


def load_predictions():
    """Return dict model -> DataFrame indexed by (crop, horizon, market_id, week_start) with y_pred; plus y_true, fold."""
    def read(f):
        return pd.read_parquet(f).assign(week_start=lambda d: pd.to_datetime(d['week_start']))
    keycols = ['crop', 'horizon_weeks', 'market_id', 'week_start']
    base = read(os.path.join(RUNS, 'new_42.parquet'))
    truth = base[base.variant == 'M6'].set_index(keycols)[['y_true', 'fold']]
    out = {}
    for m in ('M0', 'M6'):
        out[m] = base[base.variant == m].set_index(keycols)['y_pred'].astype('float64')
    ens = {m: [] for m in ('M0', 'M6')}
    for s in SEEDS:
        d = read(os.path.join(RUNS, 'new_%d.parquet' % s))
        for m in ens:
            ens[m].append(d[d.variant == m].set_index(keycols)['y_pred'].astype('float64').rename(s))
    for m in ens:
        out[m + '_ens'] = pd.concat(ens[m], axis=1).mean(axis=1)
    e9 = []
    for s in SEEDS:
        f = os.path.join(RUNS, 'new_%d_M6_M9.parquet' % s)
        if os.path.exists(f):
            d = read(f); e9.append(d[d.variant == 'M9'].set_index(keycols)['y_pred'].astype('float64').rename(s))
    if e9:
        out['M9'] = pd.concat(e9, axis=1).mean(axis=1)
    return out, truth


def panel_prices():
    p = pd.read_csv(PANEL, parse_dates=['week_start'], usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted'])
    return p.set_index(['crop', 'market_id', 'week_start'])['modal_price_weighted'].astype('float64')


def lagged(price, df, weeks):
    idx = pd.MultiIndex.from_arrays([df['crop'], df['market_id'], df['week_start'] + pd.Timedelta(weeks=weeks)])
    return price.reindex(idx).to_numpy()


def build():
    preds, truth = load_predictions()
    price = panel_prices()
    frames = []
    for h in HORIZONS:
        t = truth.reset_index(); t = t[t.horizon_weeks == h].copy()
        t['p0'] = lagged(price, t, 0)
        t['p_m13'] = lagged(price, t, -13)
        t['p_m52'] = lagged(price, t, -52)
        t['p_m39'] = lagged(price, t, -39)       # t-52+13
        key = pd.MultiIndex.from_frame(t[['crop', 'horizon_weeks', 'market_id', 'week_start']])
        for m, s in preds.items():
            t['f_' + m] = s.reindex(key).to_numpy()
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    d['ret_true'] = d['y_true'] / d['p0'] - 1
    for m in preds:
        d['ret_' + m] = d['f_' + m] / d['p0'] - 1
    d['ret_mom'] = d['p0'] / d['p_m13'] - 1
    d['ret_seas'] = d['p_m39'] / d['p_m52'] - 1
    need = ['p0', 'p_m13', 'p_m39', 'p_m52', 'y_true'] + ['f_' + m for m in preds]
    n0 = len(d); d = d.dropna(subset=need)
    d = d[(d['p0'] > 0)]
    print('[70] common sample: %d of %d rows (rules need price history and all forecasts)' % (len(d), n0))
    return d, list(preds)


def alarms(d, side, up, models):
    """Boolean alarm columns per rule for the given side."""
    a = {}
    if side == 'spike':
        a['never'] = np.zeros(len(d), bool); a['always'] = np.ones(len(d), bool)
        a['momentum'] = (d['ret_mom'] >= up).to_numpy(); a['seasonal'] = (d['ret_seas'] >= up).to_numpy()
        for m in models: a[m] = (d['ret_' + m] >= up).to_numpy()
    else:
        dn = ALARM_DN
        a['never'] = np.zeros(len(d), bool); a['always'] = np.ones(len(d), bool)
        a['momentum'] = (d['ret_mom'] <= dn).to_numpy(); a['seasonal'] = (d['ret_seas'] <= dn).to_numpy()
        for m in models: a[m] = (d['ret_' + m] <= dn).to_numpy()
    return a


def counts(ev, al):
    return np.array([(ev & al).sum(), (ev & ~al).sum(), (~ev & al).sum(), (~ev & ~al).sum()], float)   # hits, misses, FA, CN


def value(c, cl):
    hits, miss, fa, cn = c; n = c.sum()
    if n == 0: return np.nan
    s = (hits + miss) / n
    e_clim = min(cl, s); e_perf = s * cl
    e_rule = (cl * (hits + fa) + miss) / n
    return (e_clim - e_rule) / (e_clim - e_perf) if e_clim > e_perf else np.nan


def skill_scores(c):
    hits, miss, fa, cn = c
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + miss + fa) if hits + miss + fa else np.nan
    return pod, far, csi


def block_boot_weeks(weeks, rng):
    """Indices into the unique-week array, resampled in blocks of BLOCK consecutive weeks (circular)."""
    n = len(weeks); nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]) % n
    return idx.ravel()[:n]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    d, models = build()
    rng = np.random.default_rng(SEED)
    summ, tests, folds = [], [], []
    for side in ('spike', 'crash'):
        for up in (ALARM_UP_GRID if side == 'spike' else [None]):
            for crop in CROPS:
                for h in HORIZONS:
                    g = d[(d.crop == crop) & (d.horizon_weeks == h)]
                    if g.empty: continue
                    ev = (g['ret_true'] >= EVENT_UP).to_numpy() if side == 'spike' else (g['ret_true'] <= EVENT_DN).to_numpy()
                    al = alarms(g, side, up, models)
                    wk = g['week_start'].to_numpy(); weeks = np.sort(np.unique(wk)); widx = np.searchsorted(weeks, wk)
                    # per-week count matrices for the bootstrap: shape (W, rules, 4)
                    W = len(weeks)
                    cm = {r: np.zeros((W, 4)) for r in al}
                    for r, a in al.items():
                        for j, mask in enumerate([ev & a, ev & ~a, ~ev & a, ~ev & ~a]):
                            cm[r][:, j] = np.bincount(widx[mask], minlength=W)
                    for cl in CL_GRID:
                        for r in al:
                            c = cm[r].sum(axis=0); pod, far, csi = skill_scores(c)
                            hits_not_mom = np.nan
                            if r == 'M6':
                                hm = ev & al['M6']; hits_not_mom = (hm & ~al['momentum']).sum() / hm.sum() if hm.sum() else np.nan
                            summ.append(dict(side=side, alarm_threshold=up if up is not None else ALARM_DN, crop=crop, horizon_weeks=h, rule=r, cl_ratio=cl, n=int(c.sum()), base_rate=(c[0] + c[1]) / c.sum(),
                                             hits=int(c[0]), misses=int(c[1]), false_alarms=int(c[2]), POD=pod, FAR=far, CSI=csi, V=value(c, cl), m6_hits_not_flagged_by_momentum=hits_not_mom))
                    # tests at the primary setting only (spike, +20 %, C/L 0.2, h = 13) and the same for crash
                    if h == PRIMARY_H and ((side == 'spike' and up == ALARM_UP_PRIMARY) or side == 'crash'):
                        cl = CL_PRIMARY
                        boots = np.zeros((NBOOT, len(al)))
                        names = list(al)
                        for b in range(NBOOT):
                            sel = block_boot_weeks(weeks, rng)
                            for k, r in enumerate(names):
                                boots[b, k] = value(cm[r][sel].sum(axis=0), cl)
                        vobs = {r: value(cm[r].sum(axis=0), cl) for r in names}
                        fold_v = {}
                        for f in sorted(g['fold'].unique()):
                            m_ = (g['fold'] == f).to_numpy()
                            fold_v[f] = {r: value(counts(ev[m_], al[r][m_]), cl) for r in ('M6', 'momentum', 'seasonal', 'M0')}
                            folds.append(dict(side=side, crop=crop, fold=int(f), **{('V_' + r): v for r, v in fold_v[f].items()}, n=int(m_.sum()), base_rate=float(ev[m_].mean())))
                        wins_mom = sum(1 for f in fold_v if fold_v[f]['M6'] > fold_v[f]['momentum'])
                        for rival in ('momentum', 'seasonal', 'M0'):
                            diff_obs = vobs['M6'] - vobs[rival]
                            bd = boots[:, names.index('M6')] - boots[:, names.index(rival)]
                            lo, hi = np.nanpercentile(bd, [5, 95])
                            tests.append(dict(side=side, crop=crop, horizon_weeks=h, comparison='M6 minus ' + rival, V_M6=vobs['M6'], V_rival=vobs[rival], diff=diff_obs, ci90_lo=lo, ci90_hi=hi,
                                              ci_above_zero=bool(lo > 0), m6_beats_momentum_folds=wins_mom, n_folds=len(fold_v)))
    S = pd.DataFrame(summ); T = pd.DataFrame(tests); F = pd.DataFrame(folds)
    # verdict per crop / side
    ver = []
    for (side, crop), g in T.groupby(['side', 'crop']):
        s6 = S[(S.side == side) & (S.crop == crop) & (S.horizon_weeks == PRIMARY_H) & (S.rule == 'M6') & (S.cl_ratio == CL_PRIMARY) & (S.alarm_threshold == (ALARM_UP_PRIMARY if side == 'spike' else ALARM_DN))].iloc[0]
        ok = bool(s6.V > 0 and g.ci_above_zero.all() and g.m6_beats_momentum_folds.iloc[0] >= 4)
        ver.append(dict(side=side, crop=crop, V_M6=s6.V, all_three_intervals_above_zero=bool(g.ci_above_zero.all()), m6_beats_momentum_folds=int(g.m6_beats_momentum_folds.iloc[0]), adds_decision_value=ok))
    V = pd.DataFrame(ver)
    S.round(4).to_csv(os.path.join(OUT, 'table_decision_value_summary.csv'), index=False)
    T.round(4).merge(V[['side', 'crop', 'adds_decision_value']], on=['side', 'crop']).to_csv(os.path.join(OUT, 'table_decision_value_tests.csv'), index=False)
    F.round(4).to_csv(os.path.join(OUT, 'table_decision_value_folds.csv'), index=False)
    pd.set_option('display.width', 220)
    prim = S[(S.horizon_weeks == PRIMARY_H) & (S.cl_ratio == CL_PRIMARY) & (((S.side == 'spike') & (S.alarm_threshold == ALARM_UP_PRIMARY)) | (S.side == 'crash'))]
    print(prim[['side', 'crop', 'rule', 'base_rate', 'POD', 'FAR', 'CSI', 'V']].round(3).to_string(index=False))
    print(T.round(3).to_string(index=False))
    print(V.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
