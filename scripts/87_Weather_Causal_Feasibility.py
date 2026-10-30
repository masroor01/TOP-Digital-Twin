# -*- coding: utf-8 -*-
"""
Script 87 -- weather-as-instrument feasibility test for a causal supply-shock effect on prices.
Protocol (frozen 2026-10-30 before any estimate): Model_Output/weather_causal_feasibility/PROTOCOL.md. Run once; deviations are reported.

  python scripts/87_Weather_Causal_Feasibility.py selftest   # simulated data with a known beta: checks the estimators only
  python scripts/87_Weather_Causal_Feasibility.py run        # the real test

All estimators are written out with numpy (no IV package installed): HAC (Newey-West, Bartlett, 13 lags) OLS, 2SLS, Anderson-Rubin confidence set
over a grid, two-step GMM Hansen J.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output', 'weather_causal_feasibility')
CROPS = ['tomato', 'onion', 'potato']
LAGS = 13
GRID = np.round(np.arange(-6.0, 2.0001, 0.01), 2)
BLOCKS = {'b1': (1, 8), 'b2': (9, 17), 'b3': (18, 26)}          # weeks before t, inclusive
LEAD_BLOCKS = {'l1': (-8, -1), 'l2': (-17, -9), 'l3': (-26, -18)}  # weeks after t


# ------------------------------------------------------------------ estimators
def hac_meat(U, L=LAGS):
    """Newey-West long-run covariance of the rows of U (T x k), Bartlett kernel."""
    T = U.shape[0]
    S = U.T @ U
    for l in range(1, L + 1):
        G = U[l:].T @ U[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    return S


def ols_hac(y, X):
    """OLS with a Newey-West covariance scaled by T/(T-k) (finite-sample correction added 2026-10-30 after a simulation showed the uncorrected
    Anderson-Rubin set covering the truth in only 78-82 % of samples; see PROTOCOL.md, addendum)."""
    T, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    V = XtXi @ hac_meat(X * u[:, None]) @ XtXi * T / (T - k)
    return b, V, u


def wald(b, V, idx, df2=None):
    """Wald statistic and p value: F(q, df2) reference when df2 is given (small-sample), else chi-square."""
    bb = b[idx]
    W = float(bb @ np.linalg.inv(V[np.ix_(idx, idx)]) @ bb)
    if df2 is None:
        return W, 1 - stats.chi2.cdf(W, len(idx))
    return W, 1 - stats.f.cdf(W / len(idx), len(idx), df2)


def tsls(y, x, Zi, C):
    """beta on x (one endogenous variable), instruments Zi (T x q), exogenous controls C (T x c, includes constant)."""
    Z = np.column_stack([Zi, C]); X = np.column_stack([x, C])
    Pz_X = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    b = np.linalg.solve(Pz_X.T @ X, Pz_X.T @ y)
    u = y - X @ b
    A = np.linalg.inv(Pz_X.T @ Pz_X)
    V = A @ hac_meat(Pz_X * u[:, None]) @ A * len(y) / (len(y) - X.shape[1])
    return b, V, u, Z, X


def anderson_rubin(y, x, Zi, C, level=0.10):
    q = Zi.shape[1]
    Z = np.column_stack([Zi, C]); idx = list(range(q))
    keep = []
    for b0 in GRID:
        _, V, _ = None, None, None
        bz, Vz, _ = ols_hac(y - b0 * x, Z)
        _, p = wald(bz, Vz, idx, len(y) - Z.shape[1])
        if p >= level:
            keep.append(b0)
    return np.array(keep)


def hansen_j(y, x, Zi, C):
    T = len(y)
    b, _, u, Z, X = tsls(y, x, Zi, C)
    S = hac_meat(Z * u[:, None]) / T
    W = np.linalg.inv(S)
    ZtX = Z.T @ X / T; Zty = Z.T @ y / T
    b2 = np.linalg.solve(ZtX.T @ W @ ZtX, ZtX.T @ W @ Zty)
    g = Z.T @ (y - X @ b2) / T
    J = float(T * g @ W @ g)
    df = Z.shape[1] - X.shape[1]
    return J, 1 - stats.chi2.cdf(J, df), df


K_PERM = 999
MIN_SHIFT = 52


def _perm_setup(y, x, Zi, C, K, seed):
    """Frisch-Waugh: residualise on the controls once; circular shifts of the instrument rows give the null distribution (no HAC needed)."""
    T = len(y)
    M = np.eye(T) - C @ np.linalg.solve(C.T @ C, C.T)
    rng = np.random.default_rng(seed)
    shifts = np.concatenate([[0], rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=K)])
    return M, M @ y, M @ x, shifts


AR_LEVEL = 0.08   # test level of the "AR 90 % set": calibrated by simulation (level 0.10 gave 86-88 % coverage, 0.06 gave 93-96 %; see PROTOCOL.md addendum)


def perm_ar(y, x, Zi, C, level=AR_LEVEL, K=K_PERM, seed=1):
    """Anderson-Rubin set by circular-shift permutation: for each beta0 the statistic is the F of the instruments in (y - beta0 x); its null
    distribution is formed by shifting the instrument rows in time (preserves the autocorrelation of instruments and outcome)."""
    T, q = Zi.shape
    c = C.shape[1]
    M, yt, xt, shifts = _perm_setup(y, x, Zi, C, K, seed)
    d, e, f = yt @ yt, yt @ xt, xt @ xt
    cnt = np.zeros(len(GRID))
    F0 = None
    for i, sh in enumerate(shifts):
        Zs = np.roll(Zi, sh, axis=0)
        Q, _ = np.linalg.qr(M @ Zs)
        qy, qx = Q.T @ yt, Q.T @ xt
        a, b, cc = qy @ qy, qy @ qx, qx @ qx
        num = a - 2 * GRID * b + GRID ** 2 * cc
        tot = d - 2 * GRID * e + GRID ** 2 * f
        F = (num / q) / np.maximum(tot - num, 1e-12) * (T - q - c)
        if i == 0:
            F0 = F
        else:
            cnt += (F >= F0)
    pvals = (1 + cnt) / (K + 1)
    keep = GRID[pvals >= level]
    return keep, pvals


def perm_pvalue_F(y, Zi, C, K=K_PERM, seed=2):
    """Permutation p value (circular shifts of the instrument rows) of the F statistic of Zi in a regression of y on [Zi, C]."""
    T, q = Zi.shape
    c = C.shape[1]
    M, yt, _, shifts = _perm_setup(y, y, Zi, C, K, seed)
    d = yt @ yt
    stats_ = []
    for sh in shifts:
        Q, _ = np.linalg.qr(M @ np.roll(Zi, sh, axis=0))
        a = float((Q.T @ yt) @ (Q.T @ yt))
        stats_.append((a / q) / max(d - a, 1e-12) * (T - q - c))
    stats_ = np.array(stats_)
    return float(stats_[0]), float((1 + (stats_[1:] >= stats_[0]).sum()) / (len(stats_)))


def analyse(y_price, y_arr, Zi, C, Zlead=None):
    q = Zi.shape[1]
    out = {'n': len(y_price)}
    # first stage
    Z = np.column_stack([Zi, C])
    bf, Vf, _ = ols_hac(y_arr, Z)
    W, p = wald(bf, Vf, list(range(q)), len(y_arr) - Z.shape[1])
    out.update(F_first=W / q, p_first=p)
    # placebo (lead) first stage is run by the caller on its own sample
    # 2SLS
    b, V, _, _, _ = tsls(y_price, y_arr, Zi, C)
    out.update(beta=float(b[0]), se=float(np.sqrt(V[0, 0])))
    # Anderson-Rubin set by permutation (decisive); the HAC version is kept only as a descriptive column
    keep, _ = perm_ar(y_price, y_arr, Zi, C)
    out.update(ar_lo=float(keep.min()) if len(keep) else np.nan, ar_hi=float(keep.max()) if len(keep) else np.nan, ar_n=int(len(keep)))
    _, out['p_first_perm'] = perm_pvalue_F(y_arr, Zi, C)
    _, out['p_reduced_perm'] = perm_pvalue_F(y_price, Zi, C)
    # overidentification
    J, pJ, df = hansen_j(y_price, y_arr, Zi, C)
    out.update(J=J, p_J=pJ, J_df=df)
    # reduced form
    br, Vr, _ = ols_hac(y_price, Z)
    Wr, pr = wald(br, Vr, list(range(q)), len(y_price) - Z.shape[1])
    out.update(p_reduced=pr)
    return out


def passes(r, p_placebo):
    below_zero = (r['ar_n'] > 0) and (r['ar_hi'] < 0) and (r['ar_lo'] > GRID[0])
    return {'F>=10 and perm p<0.05': bool(r['F_first'] >= 10 and r['p_first_perm'] < 0.05),
            'placebo perm p>=0.10': (p_placebo >= 0.10) if p_placebo == p_placebo else False,
            'AR set non-empty': r['ar_n'] > 0, 'AR set below 0': bool(below_zero)}


# ------------------------------------------------------------------ self-test
def _block_instruments(rng, T):
    n = T + 40
    cols = []
    for _ in range(2):
        a = rng.normal(size=n)
        for lo, hi in ((1, 8), (9, 17), (18, 26)):
            cols.append(np.array([a[t - hi:t - lo + 1].mean() for t in range(30, 30 + T)]))
    return np.column_stack(cols)


def selftest(reps=100, K=399):
    """Simulated data shaped like the real design: six moving-average block instruments, AR(1) errors, endogenous regressor, known beta.
    Checks (i) 2SLS recovers beta, (ii) the permutation Anderson-Rubin 90 % set covers beta about 90 % of the time, (iii) the permutation
    first-stage test has about the nominal size when the instruments are irrelevant."""
    rng = np.random.default_rng(1)
    T, beta = 500, -0.8
    ok = True
    for rho in (0.0, 0.6, 0.85):
        cover, bhat, size = [], [], []
        for _ in range(reps):
            Zi = _block_instruments(rng, T)
            xi = rng.normal(size=T); eta = rng.normal(size=T)
            v = np.zeros(T); e = np.zeros(T)
            for t in range(1, T):
                v[t] = rho * v[t - 1] + xi[t]
                e[t] = rho * e[t - 1] + 0.8 * xi[t] + 0.6 * eta[t]
            x = Zi @ np.array([0.5, 0.4, 0.3, 0.5, 0.4, 0.3]) + v
            y = beta * x + e
            C = np.ones((T, 1))
            b, _, _, _, _ = tsls(y, x, Zi, C)
            bhat.append(b[0])
            keep, _ = perm_ar(y, x, Zi, C, K=K)
            cover.append(len(keep) > 0 and keep.min() <= beta <= keep.max())
            # irrelevant instruments (null) for the first-stage permutation test
            _, p0 = perm_pvalue_F(v + 0.0, Zi, C, K=K)
            size.append(p0 < 0.05)
        cov, sz = float(np.mean(cover)), float(np.mean(size))
        print('  rho %.2f | mean 2SLS beta %.3f | perm-AR 90%% coverage %.2f | perm first-stage size at 5%% %.2f' % (rho, np.mean(bhat), cov, sz))
        # 2SLS is biased towards OLS when the instruments are weak (persistent noise), which is why the permutation Anderson-Rubin set is the decisive
        # inference; the bias check applies to the well-identified case only
        ok &= (rho > 0 or abs(np.mean(bhat) - beta) < 0.12) and cov >= 0.82 and sz <= 0.12
    print('SELFTEST', 'OK' if ok else 'FAILED')
    return ok


# ------------------------------------------------------------------ data
def week_grid():
    return pd.date_range('2017-01-02', '2026-10-26', freq='W-MON')


def fixed_set(crop):
    p = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', f'{crop}_weekly_panel.csv'), parse_dates=['week_start'])
    p['market_id'] = p['market_id'].astype(float).astype(int)
    p = p[(p['imputed'] == 0) & (p['modal_price_weighted'] > 0) & p['arrivals_tonnes_week'].notna()]
    n_weeks = len(week_grid())
    obs = p.groupby('market_id')['week_start'].nunique()
    keep = obs[obs >= 0.8 * n_weeks].index
    return p[p['market_id'].isin(keep)], len(keep)


def national_series(p, markets=None, min_share=0.8):
    q = p if markets is None else p[p['market_id'].isin(markets)]
    n_set = q['market_id'].nunique()
    g = q.groupby('week_start')
    price = g['modal_price_weighted'].median()
    arr = g['arrivals_tonnes_week'].sum()
    cnt = g['market_id'].nunique()
    d = pd.DataFrame({'price': price, 'arr': arr, 'cnt': cnt}).reindex(week_grid())
    d = d[d['cnt'] >= min_share * n_set].copy()
    d['lp'] = np.log(d['price']); d['la'] = np.log(d['arr'])
    return d[['lp', 'la']], n_set


def anomalies(zone_ids=None, crop=None):
    z = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'zone_weekly_features.csv'), parse_dates=['week_start'])
    z = z[['zone_id', 'crop', 'week_start', 'era5_tmax', 'chirps_rain_mm']]
    if crop is not None:
        z = z[z['crop'] == crop]
    if zone_ids is not None:
        z = z[z['zone_id'].isin(zone_ids)]
    z['iso'] = z['week_start'].dt.isocalendar().week.astype(int).clip(upper=52)
    out = {}
    for var, name in (('chirps_rain_mm', 'rain'), ('era5_tmax', 'tmax')):
        base = z[z['week_start'] < '2017-01-01'].groupby(['zone_id', 'iso'])[var].agg(['mean', 'std'])
        zz = z.join(base, on=['zone_id', 'iso'])
        zz['a'] = (zz[var] - zz['mean']) / zz['std']
        out[name] = zz.groupby('week_start')['a'].mean()
    return pd.DataFrame(out)


def blocks(A, blocks_def):
    """mean of anomaly series A (indexed by Monday) over the weeks t-hi..t-lo (lo, hi = block bounds before t; negative = after t)."""
    cols = {}
    idx = pd.date_range(A.index.min(), A.index.max() + pd.Timedelta(weeks=30), freq='W-MON')
    A = A.reindex(idx)
    for bname, (lo, hi) in blocks_def.items():
        for v in A.columns:
            parts = [A[v].shift(k) for k in range(lo, hi + 1)]
            cols[f'{v}_{bname}'] = pd.concat(parts, axis=1).mean(axis=1, skipna=False)
    return pd.DataFrame(cols)


def controls(index, policy=None):
    t = np.arange(len(index))
    iso = pd.Series(index).dt.isocalendar().week.astype(int).values
    cols = [np.ones(len(index))]
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * iso / 52.18), np.cos(2 * np.pi * k * iso / 52.18)]
    cols.append((index - index.min()).days.values / 3650.0)
    cols.append(((index >= '2020-03-23') & (index <= '2020-06-30')).astype(float))
    C = np.column_stack(cols)
    if policy is not None:
        C = np.column_stack([C, policy.reindex(index).values])
    return C


def assemble(series, A, policy=None):
    B = blocks(A, BLOCKS); Lb = blocks(A, LEAD_BLOCKS)
    d = series.join(B, how='inner').join(Lb, how='left')
    inst = [c for c in B.columns]; lead = [c for c in Lb.columns]
    main = d.dropna(subset=['lp', 'la'] + inst)
    # contiguous-week requirement is not needed for HAC with this many gaps only in lockdown; rows are used in calendar order
    pol = policy
    Cm = controls(main.index, pol)
    return main, inst, lead, Cm


def policy_controls():
    p = pd.read_csv(os.path.join(BASE, 'data', 'policy_trade', 'policy_weekly_features.csv'), parse_dates=['week_start'])
    p = p[p['crop'] == 'onion'].set_index('week_start')[['export_banned', 'export_duty_pct', 'mep_usd_per_tonne']]
    return p


def run():
    os.makedirs(OUT, exist_ok=True)
    rows, zrows, rob = [], [], []
    loc = pd.read_csv(os.path.join(BASE, 'Model_Output', 'table_market_locations.csv'))
    for crop in CROPS:
        print('=== %s' % crop)
        p, n_fixed = fixed_set(crop)
        series, n_set = national_series(p)
        A = anomalies(crop=crop)
        main, inst, lead, C = assemble(series, A)
        Zi = main[inst].values; y = main['lp'].values; x = main['la'].values
        r = analyse(y, x, Zi, C)
        # placebo: lead blocks on the sample where they exist
        pl = main.dropna(subset=lead)
        Cp = controls(pl.index)
        _, p_placebo = perm_pvalue_F(pl['la'].values, pl[lead].values, Cp, seed=3)
        ok = passes(r, p_placebo)
        r.update(crop=crop, n_markets_fixed=n_set, n_placebo=len(pl), p_placebo=p_placebo, **{'pass_' + k: v for k, v in ok.items()}, passes_all=all(ok.values()))
        rows.append(r)
        print('  fixed markets %d | weeks %d | first-stage F %.1f | placebo p %.3f | Hansen p %.3f (df %d) | beta %.3f (se %.3f) | AR90 [%.2f, %.2f] (%d grid pts) | reduced-form p %.3f'
              % (n_set, r['n'], r['F_first'], p_placebo, r['p_J'], r['J_df'], r['beta'], r['se'], r['ar_lo'], r['ar_hi'], r['ar_n'], r['p_reduced']))
        print('  checks:', ok, '=> PASS' if all(ok.values()) else '=> not passed')
        # onion robustness with policy controls
        if crop == 'onion':
            pol = policy_controls()
            main2, inst2, lead2, C2 = assemble(series, A, policy=pol)
            C2 = C2[:, :]
            r2 = analyse(main2['lp'].values, main2['la'].values, main2[inst2].values, C2)
            r2.update(crop='onion_policy_controls'); rob.append(r2)
            print('  [robustness, onion with export-policy controls] F %.1f | Hansen p %.3f | beta %.3f (se %.3f) | AR90 [%.2f, %.2f]' % (r2['F_first'], r2['p_J'], r2['beta'], r2['se'], r2['ar_lo'], r2['ar_hi']))
        # zone-level descriptive check
        L = loc[(loc['crop'] == crop) & loc['zone'].notna() & (loc['dist_km'] <= 150)]
        fixed_ids = set(p['market_id'].unique())
        for zid, g in L.groupby('zone'):
            mk = [int(m) for m in g['market_id'] if int(m) in fixed_ids]
            if len(mk) < 5:
                continue
            s_z, nz = national_series(p, markets=mk)
            Az = anomalies(zone_ids=[zid])
            try:
                mz, iz, lz, Cz = assemble(s_z, Az)
                rz = analyse(mz['lp'].values, mz['la'].values, mz[iz].values, Cz)
                rz.update(crop=crop, zone=zid, n_markets=nz)
                zrows.append(rz)
                print('  zone %-18s markets %3d | weeks %3d | F %.1f | beta %.2f (se %.2f)' % (zid, nz, rz['n'], rz['F_first'], rz['beta'], rz['se']))
            except Exception as e:
                print('  zone', zid, 'skipped:', e)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'table_results.csv'), index=False)
    pd.DataFrame(zrows).to_csv(os.path.join(OUT, 'table_zone_level.csv'), index=False)
    pd.DataFrame(rob).to_csv(os.path.join(OUT, 'table_onion_policy_robustness.csv'), index=False)
    print('\nDECISION: crops passing all four checks ->', [r['crop'] for r in rows if r['passes_all']] or 'none')
    print('wrote', OUT)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('selftest', 'run'):
        print(__doc__); sys.exit(1)
    if sys.argv[1] == 'selftest':
        sys.exit(0 if selftest() else 2)
    run()
