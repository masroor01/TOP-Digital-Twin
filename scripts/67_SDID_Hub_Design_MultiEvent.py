# -*- coding: utf-8 -*-
"""
Script 67 -- Within-India export-hub SDID design across multiple policy events (Track B, Arm A')
================================================================================================
WHY: the tomato/potato-donor SDID (Scripts 31/39/66) cannot separate the onion effect from
onion's own volatility (placebo-in-time sd ~30 %). A within-onion design removes the
cross-crop problem: treated = onion markets in the Nashik export belt (Script 31 Part B list),
donors = other onion markets, so both share the same crop, season and national shocks.
Script 31 Part B did this for the 2023 ban only, with one fake date. This script extends it.

PRE-REGISTERED DESIGN (written before any result was looked at)
  Events (independent regimes used for the pooled claim, signed so + means "as expected"):
    R1  2019-09-29  export ban imposed           restrictive  expected hub price DOWN vs non-hub
    R2  2020-03-15  ban lifted                   supportive   expected hub price UP
    R3  2023-08-19  first 2023 measure (duty)    restrictive  expected hub price DOWN
    R4  2024-05-04  ban lifted                   supportive   expected hub price UP
    R5  2025-04-01  export duty removed          supportive   expected hub price UP
  Supplementary, reported but NOT pooled (overlap with a regime above): 2023-12-08 ban,
  2024-09-13 minimum-export-price removal.  Not estimable under the 95 % coverage rule:
  2020-09-14 ban (only 2 hub markets qualify).
  Rationale for the sign: an export restriction removes export demand, which falls
  hardest on export-hub mandis; lifting restores it.  Arrivals: sign not pre-registered
  (reported, not pooled).
  Panel: standard onion panel, 104 weeks before to 34 weeks after each event; markets need
  >= 95 % real coverage and no gaps in that window; >= 3 hub markets required.
  Treated = mean log price (or log1p arrivals) of qualifying hub markets; donors = all other
  qualifying onion markets (primary); sensitivity: also drop all other Maharashtra markets.
  Statistic: mean dynamic SDID effect (%) over weeks 0-12 and 0-26 after the event.
  Placebo distributions (both required):
    in-space : K random blocks of the same size drawn from the NON-hub markets (crop-coherent,
               donors = the rest), same estimator.
    in-time  : the hub block at fake dates in clean periods, same estimator.
  p-values = share of placebo |ATT| >= real |ATT| (+1 correction), two-sided.
  EVIDENCE STANDARD for a claim "export policy moved export-hub prices": expected sign and
  p <= 0.05 under BOTH placebo distributions in at least two independent regimes, and the
  signed pooled effect robust to leaving out any one regime.  Otherwise reported as not shown.

Outputs (Model_Output/):
  table_sdid_hub_multievent_<outcome>.csv        real ATT, both placebo p-values, per event/window
  table_sdid_hub_multievent_pooled_<outcome>.csv signed pooled effect, leave-one-regime-out
  table_sdid_hub_multievent_draws_<outcome>.csv  placebo ATTs

Run: python scripts/67_SDID_Hub_Design_MultiEvent.py [price|arrivals] [K_in_space] [donor_set: all|no_mh]
"""
import io, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR = os.path.join(BASE, 'Model_Output')

NASHIK_HUB_MARKETS = [
    'Chandvad APMC', 'Devala APMC', 'Dindori(Vani) APMC', 'Kalvan APMC',
    'Lasalgaon APMC', 'Lasalgaon(Niphad) APMC', 'Manmad APMC', 'Nandgaon APMC',
    'Nasik APMC', 'Pimpalgaon APMC', 'Pimpalgaon Baswant(Saykheda) APMC',
    'Satana APMC', 'Sinner APMC', 'Yeola APMC',
]
COVERAGE = 0.95
INTERP_LIMIT = 4
PRE_WEEKS, POST_FIT_WEEKS, POST_TOTAL = 104, 26, 34
WIN_SHORT, WIN_LONG = 12, 26
MIN_HUB = 3
SEED = 20261006
# (id, date, expected sign for hub price: -1 restrictive / +1 supportive, pooled?)
EVENTS = [
    ('R1_2019_ban', '2019-09-29', -1, True), ('R2_2020_lift', '2020-03-15', +1, True),
    ('R3_2023_duty', '2023-08-19', -1, True), ('R4_2024_lift', '2024-05-04', +1, True),
    ('R5_2025_dutyoff', '2025-04-01', +1, True),
    ('S_2023_ban', '2023-12-08', -1, False), ('S_2024_MEPoff', '2024-09-13', +1, False),
]


def solve_simplex_regression(X, y, ridge_penalty, maxiter=500):
    n_vars = X.shape[1]

    def obj(p):
        return float(np.sum((y - p[0] - X @ p[1:]) ** 2) + ridge_penalty * np.sum(p[1:] ** 2))
    x0 = np.concatenate([[float(np.mean(y))], np.full(n_vars, 1.0 / n_vars)])
    res = minimize(obj, x0, method='SLSQP', bounds=[(None, None)] + [(0.0, 1.0)] * n_vars,
                   constraints=[{'type': 'eq', 'fun': lambda p: np.sum(p[1:]) - 1.0}],
                   options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x[0], res.x[1:]


def sdid_dynamic_pct(treated, donors, pre_mask, post_mask):
    pre_diffs = donors.loc[pre_mask].diff().dropna().values
    sigma = float(np.std(pre_diffs, ddof=1)) if pre_diffs.size else 0.0
    zeta = (int(post_mask.sum()) ** 0.25) * sigma if sigma > 0 else 1e-6
    J = donors.shape[1]
    _, tw = solve_simplex_regression(donors.loc[pre_mask].values.T, donors.loc[post_mask].mean(axis=0).values, J * zeta ** 2)
    w0, uw = solve_simplex_regression(donors.loc[pre_mask].values, treated.loc[pre_mask].values, int(pre_mask.sum()) * zeta ** 2)
    gap = treated - pd.Series(w0 + donors.values @ uw, index=treated.index)
    base = float(np.sum(tw * gap.loc[pre_mask].values))
    return (gap - base).apply(lambda v: float(np.expm1(v) * 100))


def att_windows(dyn, weeks_idx, d):
    ws = (weeks_idx - d).dt.days / 7
    return float(dyn[(ws >= 0) & (ws <= WIN_SHORT)].mean()), float(dyn[(ws >= 0) & (ws <= WIN_LONG)].mean())


def build_onion_panel(df_o, start, end, outcome):
    w = df_o[(df_o['week_start'] >= start) & (df_o['week_start'] <= end)]
    weeks = sorted(w['week_start'].unique()); nw = len(weeks)
    c = w.groupby('market')['imputed'].agg(['mean', 'count']); c['real'] = 1 - c['mean']
    q = c[(c['count'] == nw) & (c['real'] >= COVERAGE)].index
    pp = w.pivot_table(index='week_start', columns='market', values='modal_price_weighted').reindex(weeks)
    q = [m for m in q if pp[m].notna().all()]
    if outcome == 'price':
        piv = np.log(pp[q])
    else:
        ap = w.pivot_table(index='week_start', columns='market', values='arrivals_tonnes_week').reindex(weeks)[q]
        cand = ap.columns[ap.notna().mean(axis=0) >= COVERAGE]
        ai = ap[cand].interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
        piv = np.log1p(ai[ai.columns[ai.notna().all(axis=0)]])
    return piv, pd.Series(weeks, index=piv.index)


def fit_block(piv, weeks_idx, tr_cols, dn_cols, d):
    pre = pd.Series((weeks_idx < d), index=piv.index)
    post = pd.Series((weeks_idx >= d) & (weeks_idx <= d + pd.Timedelta(weeks=POST_FIT_WEEKS)), index=piv.index)
    dyn = sdid_dynamic_pct(piv[tr_cols].mean(axis=1), piv[dn_cols], pre, post)
    return att_windows(dyn, weeks_idx, d)


def inspace_draw(k, piv, weeks_idx, nonhub, n_block, d):
    rng = np.random.default_rng(SEED + k)
    pick = set(rng.choice(len(nonhub), size=n_block, replace=False).tolist())
    tr = [m for i, m in enumerate(nonhub) if i in pick]; dn = [m for i, m in enumerate(nonhub) if i not in pick]
    a, b = fit_block(piv, weeks_idx, tr, dn, d)
    return dict(draw=k, att_0_12=a, att_0_26=b)


def intime_draw(piv, weeks_idx, hub, nonhub, d):
    a, b = fit_block(piv, weeks_idx, hub, nonhub, d)
    return dict(placebo_date=str(d.date()), att_0_12=a, att_0_26=b)


def pval(v, a):
    return (1 + np.sum(np.abs(np.asarray(v)) >= abs(a))) / (len(v) + 1)


def main():
    outcome = sys.argv[1] if len(sys.argv) > 1 else 'price'
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    donor_set = sys.argv[3] if len(sys.argv) > 3 else 'all'
    assert outcome in ('price', 'arrivals') and donor_set in ('all', 'no_mh')
    tag = outcome + ('' if donor_set == 'all' else '_noMH')
    print('=' * 70); print('SCRIPT 67: HUB-vs-NONHUB SDID, MULTI-EVENT -- outcome=%s donors=%s K=%d' % (outcome, donor_set, K)); print('=' * 70)
    df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
    df = df[df['crop'] == 'onion'].copy()
    col = df.groupby('market')['market_id'].transform('nunique') > 1
    df.loc[col, 'market'] = df.loc[col, 'market'] + ' (' + df.loc[col, 'state'] + ')'
    mh = set(df[df['state'] == 'Maharashtra']['market'])
    rows, draws = [], []
    real = {}
    for eid, ds, sign, pooled in EVENTS:
        d = pd.Timestamp(ds); t0 = time.time()
        piv, wk = build_onion_panel(df, d - pd.Timedelta(weeks=PRE_WEEKS), d + pd.Timedelta(weeks=POST_TOTAL), outcome)
        hub = [m for m in piv.columns if m in NASHIK_HUB_MARKETS]
        nonhub = [m for m in piv.columns if m not in NASHIK_HUB_MARKETS and not (donor_set == 'no_mh' and m in mh)]
        if len(hub) < MIN_HUB or len(nonhub) < 3 * len(hub):
            print('  [%s] skipped: hub=%d nonhub=%d' % (eid, len(hub), len(nonhub))); continue
        a12, a26 = fit_block(piv, wk, hub, nonhub, d)
        res = Parallel(n_jobs=-1)(delayed(inspace_draw)(k, piv, wk, nonhub, len(hub), d) for k in range(K))
        sp = pd.DataFrame(res); sp.insert(0, 'event', eid); sp.insert(1, 'kind', 'in_space'); draws.append(sp)
        real[eid] = dict(a12=a12, a26=a26, sign=sign, pooled=pooled, hub=hub, nonhub=nonhub, d=d)
        print('  [%s %s] hub n=%d, donors=%d | ATT 0-12 %+.1f %%, 0-26 %+.1f %% | in-space placebo sd %.1f, p(0-12)=%.3f | %.0fs' % (
            eid, ds, len(hub), len(nonhub), a12, a26, sp['att_0_12'].std(), pval(sp['att_0_12'], a12), time.time() - t0), flush=True)
    # in-time placebo for the hub block: common panel, fake dates in clean windows
    piv, wk = build_onion_panel(df, pd.Timestamp('2017-01-02'), pd.Timestamp('2023-08-14'), outcome)
    hub = [m for m in piv.columns if m in NASHIK_HUB_MARKETS]
    nonhub = [m for m in piv.columns if m not in NASHIK_HUB_MARKETS and not (donor_set == 'no_mh' and m in mh)]
    dates = [w for lo, hi in ((pd.Timestamp('2018-01-01'), pd.Timestamp('2019-03-04')), (pd.Timestamp('2021-02-01'), pd.Timestamp('2023-02-13')))
             for w in wk if lo <= w <= hi][::2]
    print('  in-time placebo: hub n=%d, donors=%d, %d fake dates' % (len(hub), len(nonhub), len(dates)), flush=True)
    it = pd.DataFrame(Parallel(n_jobs=-1)(delayed(intime_draw)(piv, wk, hub, nonhub, d) for d in dates))
    it.insert(0, 'event', 'placebo_in_time'); it.insert(1, 'kind', 'in_time')
    for eid, r in real.items():
        for win, a in (('att_0_12', r['a12']), ('att_0_26', r['a26'])):
            sp = pd.concat(draws)
            spv = sp[sp['event'] == eid][win].values
            rows.append(dict(event=eid, outcome=tag, window=win, pooled_regime=r['pooled'], expected_sign=r['sign'], n_hub=len(r['hub']), n_donors=len(r['nonhub']),
                             att_pct=round(a, 2), signed_att=round(a * r['sign'], 2), sign_as_expected=bool(a * r['sign'] > 0),
                             inspace_sd=round(float(spv.std(ddof=1)), 2), p_inspace=round(pval(spv, a), 4),
                             intime_sd=round(float(it[win].std(ddof=1)), 2), p_intime=round(pval(it[win].values, a), 4)))
    out = pd.DataFrame(rows)
    out['meets_standard'] = (out['sign_as_expected']) & (out['p_inspace'] <= 0.05) & (out['p_intime'] <= 0.05)
    prow = []
    for win in ('att_0_12', 'att_0_26'):
        s = out[(out['window'] == win) & (out['pooled_regime'])]
        regs = list(s['event'])
        for label, use in [('all', regs)] + [('leave_out_%s' % e, [x for x in regs if x != e]) for e in regs]:
            if not use: continue
            prow.append(dict(outcome=tag, window=win, set=label, n_regimes=len(use), signed_pooled_att=round(float(s[s['event'].isin(use)]['signed_att'].mean()), 2),
                             n_meeting_standard=int(s[s['event'].isin(use)]['meets_standard'].sum())))
    pooled = pd.DataFrame(prow)
    out.to_csv(os.path.join(OUT_DIR, 'table_sdid_hub_multievent_%s.csv' % tag), index=False)
    pooled.to_csv(os.path.join(OUT_DIR, 'table_sdid_hub_multievent_pooled_%s.csv' % tag), index=False)
    pd.concat(draws + [it], ignore_index=True).to_csv(os.path.join(OUT_DIR, 'table_sdid_hub_multievent_draws_%s.csv' % tag), index=False)
    print(out[['event', 'window', 'att_pct', 'sign_as_expected', 'inspace_sd', 'p_inspace', 'intime_sd', 'p_intime', 'meets_standard']].to_string(index=False))
    print(pooled.to_string(index=False))


if __name__ == '__main__':
    main()
