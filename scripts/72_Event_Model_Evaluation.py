# -*- coding: utf-8 -*-
"""
Script 72 -- Scoring of the event models (Arm D2). Protocol frozen in docs/EVENT_MODELS_PROTOCOL.md.
===============================================================================================
Do the data layers help the early-warning DECISION?  Compares, on identical market-weeks and under the same
capacity rule, the price-only event model (M0E), the layered event model (LAY), momentum, seasonal naive and the
Script 70 regression forecasts (M0, M6).

Primary setting (h = 13): alarm the top 20 % of markets each week and crop, C/L = 0.2.  Sensitivity: capacity 10/30 %,
C/L 0.1/0.3, h = 4/26, probability rule (calibrated p >= 0.2), persistence-2.
Pass for a crop x side: V(LAY) - V(M0E) has a 90 % 13-week block-bootstrap interval above zero, LAY beats M0E in >= 4 of 5
folds, and V(LAY) exceeds the better of momentum / seasonal naive with an interval above zero.  General claim needs >= 3 of
the 6 crop x side passes; a crop claim needs both sides.

Run: python scripts/72_Event_Model_Evaluation.py     Outputs (Model_Output/): table_event_models_summary.csv,
     table_event_models_tests.csv, table_event_models_folds.csv, table_event_models_verdict.csv
"""
import os, io, sys, glob, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output'); EV = os.path.join(OUT, '_event_runs')
spec = importlib.util.spec_from_file_location('s70', os.path.join(BASE, 'scripts', '70_Decision_Value_Backtest.py'))
s70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s70)
CAPS = [0.10, 0.20, 0.30]; CAP_PRIMARY = 0.20
CLS = [0.1, 0.2, 0.3]; CL_PRIMARY = 0.2
PRIMARY_H = 13; NBOOT = 2000; BLOCK = 13; SEED = 20261008
RIVALS = ['momentum', 'seasonal', 'M0reg', 'M6reg']


def capacity(score, wk_codes, q):
    """Top q of markets within each week (by score). Returns boolean array."""
    s = pd.Series(score)
    r = s.groupby(wk_codes).rank(pct=True, method='first')
    return (r > 1 - q).to_numpy()


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    reg, models = s70.build()
    ev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(EV, 'event_probs_h*_*.parquet')))], ignore_index=True)
    ev['week_start'] = pd.to_datetime(ev['week_start'])
    key = ['crop', 'horizon_weeks', 'market_id', 'week_start']
    d = ev.merge(reg[key + ['ret_true', 'ret_mom', 'ret_seas', 'ret_M0', 'ret_M6']], on=key, how='inner')
    print('[72] merged sample: %d rows (events %d)' % (len(d), len(ev)))
    rng = np.random.default_rng(SEED)
    summ, tests, folds_out = [], [], []
    for (crop, h, side), g in d.groupby(['crop', 'horizon_weeks', 'side']):
        sgn = 1 if side == 'spike' else -1
        g = g.sort_values(['week_start', 'market_id']).reset_index(drop=True)
        e = g['event'].to_numpy().astype(bool)
        wk = g['week_start'].to_numpy(); weeks = np.sort(np.unique(wk)); widx = np.searchsorted(weeks, wk); W = len(weeks)
        scores = {'M0E': g['p_raw_M0E'].to_numpy(), 'LAY': g['p_raw_LAY'].to_numpy(), 'momentum': sgn * g['ret_mom'].to_numpy(), 'seasonal': sgn * g['ret_seas'].to_numpy(),
                  'M0reg': sgn * g['ret_M0'].to_numpy(), 'M6reg': sgn * g['ret_M6'].to_numpy()}
        for q in CAPS:
            al = {r: capacity(scores[r], widx, q) for r in scores}
            al['never'] = np.zeros(len(g), bool); al['always'] = np.ones(len(g), bool)
            if q == CAP_PRIMARY:
                al['M0E_prob'] = (g['p_cal_M0E'] >= 0.2).to_numpy(); al['LAY_prob'] = (g['p_cal_LAY'] >= 0.2).to_numpy()
                # persistence-2: also in the top group the previous week
                for r in ('M0E', 'LAY'):
                    prev = g[['market_id', 'week_start']].copy(); prev['a'] = al[r]; prev['week_start'] = prev['week_start'] + pd.Timedelta(weeks=1)
                    m_ = g[['market_id', 'week_start']].merge(prev, on=['market_id', 'week_start'], how='left')['a'].fillna(False).to_numpy().astype(bool)
                    al[r + '_persist2'] = al[r] & m_
            cm = {r: np.zeros((W, 4)) for r in al}
            for r, a in al.items():
                for j, mask in enumerate([e & a, e & ~a, ~e & a, ~e & ~a]):
                    cm[r][:, j] = np.bincount(widx[mask], minlength=W)
            for cl in CLS:
                for r in al:
                    c = cm[r].sum(axis=0); pod, far, csi = s70.skill_scores(c)
                    summ.append(dict(crop=crop, horizon_weeks=h, side=side, capacity=q, rule=r, cl_ratio=cl, n=int(c.sum()), base_rate=float(e.mean()), hits=int(c[0]), misses=int(c[1]), false_alarms=int(c[2]),
                                     POD=pod, FAR=far, CSI=csi, V=s70.value(c, cl)))
            if h == PRIMARY_H and q == CAP_PRIMARY:
                cl = CL_PRIMARY; names = list(al)
                vobs = {r: s70.value(cm[r].sum(axis=0), cl) for r in names}
                rival = max(('momentum', 'seasonal'), key=lambda r: vobs[r])
                boots = np.zeros((NBOOT, len(names)))
                for b in range(NBOOT):
                    sel = s70.block_boot_weeks(weeks, rng)
                    for k, r in enumerate(names):
                        boots[b, k] = s70.value(cm[r][sel].sum(axis=0), cl)
                col = {r: names.index(r) for r in names}
                fold_v = []
                for f in sorted(g['fold'].unique()):
                    m_ = (g['fold'] == f).to_numpy()
                    v = {r: s70.value(s70.counts(e[m_], al[r][m_]), cl) for r in ('LAY', 'M0E', 'momentum', 'seasonal', 'LAY_prob', 'M0E_prob')}
                    fold_v.append(v); folds_out.append(dict(crop=crop, side=side, fold=int(f), **{'V_' + r: x for r, x in v.items()}, n=int(m_.sum()), base_rate=float(e[m_].mean())))
                wins = sum(1 for v in fold_v if v['LAY'] > v['M0E'])
                wins_prob = sum(1 for v in fold_v if v['LAY_prob'] > v['M0E_prob'])
                for lab, a_, b_ in (('LAY minus M0E', 'LAY', 'M0E'), ('LAY minus best naive (%s)' % rival, 'LAY', rival), ('LAY minus M6reg', 'LAY', 'M6reg'), ('M0E minus M6reg', 'M0E', 'M6reg')):
                    bd = boots[:, col[a_]] - boots[:, col[b_]]; lo, hi = np.nanpercentile(bd, [5, 95])
                    tests.append(dict(crop=crop, side=side, horizon_weeks=h, comparison=lab, V_a=vobs[a_], V_b=vobs[b_], diff=vobs[a_] - vobs[b_], ci90_lo=lo, ci90_hi=hi, ci_above_zero=bool(lo > 0), lay_beats_m0e_folds=wins, n_folds=len(fold_v)))
                for lab, a_, b_ in (('EXPLORATORY probability rule: LAY_prob minus M0E_prob', 'LAY_prob', 'M0E_prob'), ('EXPLORATORY probability rule: LAY_prob minus M6reg (capacity)', 'LAY_prob', 'M6reg')):
                    bd = boots[:, col[a_]] - boots[:, col[b_]]; lo, hi = np.nanpercentile(bd, [5, 95])
                    tests.append(dict(crop=crop, side=side, horizon_weeks=h, comparison=lab, V_a=vobs[a_], V_b=vobs[b_], diff=vobs[a_] - vobs[b_], ci90_lo=lo, ci90_hi=hi, ci_above_zero=bool(lo > 0), lay_beats_m0e_folds=wins_prob, n_folds=len(fold_v)))
        # discrimination metrics (threshold-free)
        for h_, nm, sc in [(h, 'M0E', g['p_raw_M0E']), (h, 'LAY', g['p_raw_LAY']), (h, 'M0reg', sgn * g['ret_M0']), (h, 'M6reg', sgn * g['ret_M6']), (h, 'momentum', sgn * g['ret_mom'])]:
            summ.append(dict(crop=crop, horizon_weeks=h, side=side, capacity=np.nan, rule='AUC_' + nm, cl_ratio=np.nan, n=len(g), base_rate=float(e.mean()), V=roc_auc_score(e, sc)))
        for nm in ('M0E', 'LAY'):
            summ.append(dict(crop=crop, horizon_weeks=h, side=side, capacity=np.nan, rule='Brier_' + nm, cl_ratio=np.nan, n=len(g), base_rate=float(e.mean()), V=brier_score_loss(e, g['p_cal_' + nm])))
    S = pd.DataFrame(summ); T = pd.DataFrame(tests); F = pd.DataFrame(folds_out)
    ver = []
    for (crop, side), t in T[T.horizon_weeks == PRIMARY_H].groupby(['crop', 'side']):
        a = t[t.comparison == 'LAY minus M0E'].iloc[0]; b = t[t.comparison.str.startswith('LAY minus best naive')].iloc[0]
        ok = bool(a.ci_above_zero and a.lay_beats_m0e_folds >= 4 and b.ci_above_zero)
        ver.append(dict(crop=crop, side=side, V_LAY=a.V_a, V_M0E=a.V_b, diff_vs_M0E=a['diff'], ci_lo=a.ci90_lo, ci_hi=a.ci90_hi, folds_LAY_beats_M0E=int(a.lay_beats_m0e_folds), beats_best_naive=bool(b.ci_above_zero), layers_help=ok))
    V = pd.DataFrame(ver)
    S.round(4).to_csv(os.path.join(OUT, 'table_event_models_summary.csv'), index=False)
    T.round(4).to_csv(os.path.join(OUT, 'table_event_models_tests.csv'), index=False)
    F.round(4).to_csv(os.path.join(OUT, 'table_event_models_folds.csv'), index=False)
    V.round(4).to_csv(os.path.join(OUT, 'table_event_models_verdict.csv'), index=False)
    pd.set_option('display.width', 230)
    prim = S[(S.horizon_weeks == PRIMARY_H) & (S.capacity == CAP_PRIMARY) & (S.cl_ratio == CL_PRIMARY)]
    print(prim.pivot_table(index=['crop', 'side'], columns='rule', values='V').round(2).to_string())
    print(S[S.rule.str.startswith(('AUC_', 'Brier_')) & (S.horizon_weeks == PRIMARY_H)].pivot_table(index=['crop', 'side'], columns='rule', values='V').round(3).to_string())
    print(V.round(3).to_string(index=False))
    print('layers_help passes: %d of %d (general claim needs >= 3 of 6)' % (int(V.layers_help.sum()), len(V)))


if __name__ == '__main__':
    main()
