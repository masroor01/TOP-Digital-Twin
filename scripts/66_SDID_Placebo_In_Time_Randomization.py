# -*- coding: utf-8 -*-
"""
Script 66 -- SDID placebo-in-time randomization inference (Track B, Arm A)
==========================================================================
Scripts 31/38/39 judge the onion block's SDID effect against only TWO placebo
blocks (tomato, potato). The 2026-10-06 re-read showed why that is not enough:
the post-ban onion price effect (+14.5 %) equals the tomato (+14.5 %) and potato
(+15.2 %) placebos. A first attempt here with random mixed tomato+potato blocks
gave an absurdly tight placebo band (sd 1-2 %), because averaging random markets
washes out CROP-level shocks, which are exactly the noise onion carries. It was
discarded as anti-conservative.

The relevant null is "what does the same estimator find for the onion block on
dates when no export measure happened?" So:
  1. Real effect: the onion block's SDID ATT (mean dynamic effect, weeks 0-12 and
     0-26 after the ban date) for each ban episode, exactly as in Script 39.
  2. Placebo distribution: the identical estimator for the onion block at fake ban
     dates every `step` weeks inside clean periods (2018-01..2019-03, 2021-02..
     2023-02), pre-window = all weeks since 2017-01, post window 26 weeks, never
     overlapping a real onion measure.
  3. Randomization p-value = share of placebo ATTs with |ATT| >= the real |ATT|,
     with the +1 correction. Pooled across episodes by comparing the mean of the
     real episode ATTs with means of random triples of placebo ATTs; plus
     leave-one-episode-out.
Caveat: the placebo dates share one panel (markets qualifying through 2023-08)
and overlapping windows, so they are not independent; the distribution is a
noise gauge, not an exact reference law. Each real episode uses its own
qualifying panel, as in Script 39.

Evidence standard (docs/TRACK_B_SCOPING_NOTE.md): a causal claim needs the onion
effect outside the placebo distribution in at least two independent episodes and
robust to leaving any one episode out.

Outputs (Model_Output/):
  table_sdid_placebo_in_time_draws_<outcome>.csv     one row per fake date
  table_sdid_placebo_in_time_summary_<outcome>.csv   real ATT vs placebo distribution

Run: python scripts/66_SDID_Placebo_In_Time_Randomization.py [price|arrivals] [step_weeks]
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

PANEL_START = pd.Timestamp('2017-01-02')
COVERAGE_THRESHOLD = 0.95
INTERP_LIMIT = 4
POST_WEEKS_FOR_FIT = 26
WIN_SHORT, WIN_LONG = 12, 26
SEED = 20261006
EPISODES = {'2019': pd.Timestamp('2019-09-29'), '2020': pd.Timestamp('2020-09-14'), '2023': pd.Timestamp('2023-12-08')}


# --- SDID estimator (identical to Script 31/38/39) ---------------------------------------
def solve_simplex_regression(X, y, ridge_penalty, x0=None, maxiter=500):
    n_obs, n_vars = X.shape

    def objective(params):
        w0, w = params[0], params[1:]
        resid = y - w0 - X @ w
        return float(np.sum(resid ** 2) + ridge_penalty * np.sum(w ** 2))
    if x0 is None:
        x0 = np.concatenate([[float(np.mean(y))], np.full(n_vars, 1.0 / n_vars)])
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[1:]) - 1.0}]
    bounds = [(None, None)] + [(0.0, 1.0)] * n_vars
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x[0], res.x[1:]


def sdid_dynamic_pct(treated, donors, pre_mask, post_mask):
    pre_diffs = donors.loc[pre_mask].diff().dropna().values
    sigma_hat = float(np.std(pre_diffs, ddof=1)) if pre_diffs.size else 0.0
    n_post = int(post_mask.sum())
    zeta = (n_post ** 0.25) * sigma_hat if sigma_hat > 0 else 1e-6
    J = donors.shape[1]
    _, time_w = solve_simplex_regression(donors.loc[pre_mask].values.T, donors.loc[post_mask].mean(axis=0).values,
                                         ridge_penalty=J * zeta ** 2)
    n_pre = int(pre_mask.sum())
    w0_u, unit_w = solve_simplex_regression(donors.loc[pre_mask].values, treated.loc[pre_mask].values,
                                            ridge_penalty=n_pre * zeta ** 2)
    synthetic = pd.Series(w0_u + donors.values @ unit_w, index=treated.index)
    gap = treated - synthetic
    baseline = float(np.sum(time_w * gap.loc[pre_mask].values))
    return (gap - baseline).apply(lambda v: float(np.expm1(v) * 100))


def att_windows(dyn, weeks_s, ban_start):
    ws = (pd.Series(weeks_s.values, index=dyn.index) - ban_start).dt.days / 7
    return (float(dyn[(ws >= 0) & (ws <= WIN_SHORT)].mean()), float(dyn[(ws >= 0) & (ws <= WIN_LONG)].mean()))


def one_placebo_date(d, treated, donors, weeks_idx, pre_start):
    """Fake 'ban' at date d: same SDID, onion block vs tomato+potato donors, pre = all weeks before d."""
    wk = weeks_idx
    pre = pd.Series((wk >= pre_start) & (wk < d), index=treated.index)
    post = pd.Series((wk >= d) & (wk <= d + pd.Timedelta(weeks=POST_WEEKS_FOR_FIT)), index=treated.index)
    dyn = sdid_dynamic_pct(treated, donors, pre, post)
    a, b = att_windows(dyn, wk, d)
    return dict(placebo_date=str(d.date()), att_0_12=a, att_0_26=b)


def build_panel(df, win_end, outcome):
    w = df[(df['week_start'] >= PANEL_START) & (df['week_start'] <= win_end)].copy()
    weeks = sorted(w['week_start'].unique()); nw = len(weeks)
    cov = w.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
    cov['real_cov'] = 1 - cov['mean']
    q = cov[(cov['count'] == nw) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()
    pp = w.pivot_table(index='week_start', columns=['crop', 'market'], values='modal_price_weighted').reindex(weeks)
    ok = pp.columns[pp.notna().all(axis=0)]
    q = q[q.apply(lambda r: (r['crop'], r['market']) in ok, axis=1)]
    if outcome == 'price':
        piv = np.log(pp)
        keep = [(r['crop'], r['market']) for _, r in q.iterrows()]
    else:
        ap = w.pivot_table(index='week_start', columns=['crop', 'market'], values='arrivals_tonnes_week').reindex(weeks)
        qc = [(r['crop'], r['market']) for _, r in q.iterrows()]
        apq = ap.reindex(columns=[c for c in qc if c in ap.columns])
        cand = apq.notna().mean(axis=0); cand = cand[cand >= COVERAGE_THRESHOLD].index
        ai = apq[cand].interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
        full = ai.columns[ai.notna().all(axis=0)]
        piv = np.log1p(ai[full]); keep = list(full)
    return piv, keep, pd.Series(weeks)


def main():
    outcome = sys.argv[1] if len(sys.argv) > 1 else 'price'
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2   # fake-date spacing in weeks
    assert outcome in ('price', 'arrivals')
    print('=' * 65); print('SCRIPT 66: SDID PLACEBO-IN-TIME RANDOMIZATION -- outcome=%s, spacing=%dwk' % (outcome, step)); print('=' * 65)
    df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
    col = df.groupby(['crop', 'market'])['market_id'].transform('nunique') > 1
    df.loc[col, 'market'] = df.loc[col, 'market'] + ' (' + df.loc[col, 'state'] + ')'

    # 1. real episodes (each on its own qualifying panel, as in Script 39)
    real = {}
    for ep, ban in EPISODES.items():
        piv, keep, weeks_s = build_panel(df, ban + pd.Timedelta(weeks=34), outcome)
        onion = [c for c in keep if c[0] == 'onion']; pool = [c for c in keep if c[0] in ('tomato', 'potato')]
        weeks_idx = pd.Series(weeks_s.values, index=piv.index)
        pre = pd.Series((weeks_idx >= PANEL_START) & (weeks_idx < ban), index=piv.index)
        post = pd.Series((weeks_idx >= ban) & (weeks_idx <= ban + pd.Timedelta(weeks=POST_WEEKS_FOR_FIT)), index=piv.index)
        donors = piv[pool]; donors.columns = ['%s__%s' % c for c in pool]
        dyn = sdid_dynamic_pct(piv[onion].mean(axis=1), donors, pre, post)
        a12, a26 = att_windows(dyn, weeks_idx, ban)
        real[ep] = (a12, a26, len(onion), len(pool))
        print('  [%s] onion n=%d, donors=%d, pre=%dwk | ATT 0-12 = %+.1f %%, 0-26 = %+.1f %%' % (ep, len(onion), len(pool), int(pre.sum()), a12, a26), flush=True)

    # 2. placebo dates: one common panel up to 2023-08-14, dates away from every real measure
    end = pd.Timestamp('2023-08-14')
    piv, keep, weeks_s = build_panel(df, end, outcome)
    onion = [c for c in keep if c[0] == 'onion']; pool = [c for c in keep if c[0] in ('tomato', 'potato')]
    weeks_idx = pd.Series(weeks_s.values, index=piv.index)
    treated = piv[onion].mean(axis=1)
    donors = piv[pool]; donors.columns = ['%s__%s' % c for c in pool]
    print('  placebo panel: onion n=%d, donors=%d, weeks=%d' % (len(onion), len(pool), len(piv)))
    clean = [(pd.Timestamp('2018-01-01'), pd.Timestamp('2019-03-04')), (pd.Timestamp('2021-02-01'), pd.Timestamp('2023-02-13'))]
    dates = []
    for lo, hi in clean:
        dates += [w for w in weeks_idx if lo <= w <= hi][::step]
    print('  %d placebo dates in clean periods (%s)' % (len(dates), ', '.join('%s to %s' % (a.date(), b.date()) for a, b in clean)), flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=-1)(delayed(one_placebo_date)(d, treated, donors, weeks_idx, PANEL_START) for d in dates)
    pl = pd.DataFrame(res); print('  placebo fits done in %.0fs' % (time.time() - t0))

    rows = []
    for win, idx in (('att_0_12', 0), ('att_0_26', 1)):
        v = pl[win].values
        for ep, r in real.items():
            a = r[idx]
            rows.append(dict(episode=ep, outcome=outcome, window=win, onion_att_pct=round(a, 2), n_onion=r[2], n_donors=r[3], n_placebo=len(v),
                             placebo_mean=round(float(v.mean()), 2), placebo_sd=round(float(v.std(ddof=1)), 2),
                             placebo_p05=round(float(np.quantile(v, .05)), 2), placebo_p95=round(float(np.quantile(v, .95)), 2),
                             p_two_sided=round((1 + np.sum(np.abs(v) >= abs(a))) / (len(v) + 1), 4),
                             p_one_sided=round((1 + np.sum(v >= a if a >= 0 else v <= a)) / (len(v) + 1), 4), z=round((a - v.mean()) / v.std(ddof=1), 2)))
        # pooled: mean of 3 episodes vs mean of 3 placebo dates drawn at random (10,000 triples)
        rng = np.random.default_rng(SEED)
        trip = rng.choice(v, size=(10000, len(real))).mean(axis=1)
        for use in [list(real)] + [[e for e in real if e != d] for d in real]:
            on = float(np.mean([real[e][idx] for e in use]))
            t = rng.choice(v, size=(10000, len(use))).mean(axis=1)
            rows.append(dict(episode='pooled(%s)' % '+'.join(use) if len(use) == len(real) else 'leave_out_%s' % [e for e in real if e not in use][0],
                             outcome=outcome, window=win, onion_att_pct=round(on, 2), n_placebo=len(v), placebo_sd=round(float(t.std(ddof=1)), 2),
                             p_two_sided=round((1 + np.sum(np.abs(t) >= abs(on))) / 10001, 4)))
    summ = pd.DataFrame(rows)
    pl.to_csv(os.path.join(OUT_DIR, 'table_sdid_placebo_in_time_draws_%s.csv' % outcome), index=False)
    summ.to_csv(os.path.join(OUT_DIR, 'table_sdid_placebo_in_time_summary_%s.csv' % outcome), index=False)
    print(summ[['episode', 'window', 'onion_att_pct', 'placebo_mean', 'placebo_sd', 'placebo_p05', 'placebo_p95', 'p_two_sided']].to_string(index=False))


if __name__ == '__main__':
    main()
