# -*- coding: utf-8 -*-
"""
Script 77 -- Onion spike onset: leading-indicator event study. Protocol frozen in docs/ONION_SPIKE_EVENT_STUDY_PROTOCOL.md.
Run: python scripts/77_Onion_Spike_Event_Study.py
Outputs (Model_Output/): table_onion_spike_episodes.csv, table_onion_spike_drivers.csv
"""
import os, io, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
FWD, UP, EXCESS, CALM, MINMKT, SEP = 8, 0.30, 0.20, 0.10, 100, 12
LEAD_LO, LEAD_HI = 2, 8
NPERM, SEED = 5000, 20261010


def national():
    p = pd.read_csv(os.path.join(BASE, 'data', 'master_weekly_panel_all_layers.csv'), parse_dates=['week_start'],
                    usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted', 'imputed', 'arrivals_tonnes_week', 'chirps_rain_mm',
                             'era5_heat_35', 's2_ndvi_anom', 'diesel_4city_rs_litre'])
    o = p[p['crop'] == 'onion'].copy()
    obs = o[o['imputed'] == 0]
    g = obs.groupby('week_start')
    nat = pd.DataFrame({'price': g['modal_price_weighted'].median(), 'arr': g['arrivals_tonnes_week'].sum(), 'n': g['market_id'].nunique()})
    nat = nat[nat['n'] >= MINMKT]
    w = o['arrivals_tonnes_week'].fillna(0).clip(lower=0) + 1e-9
    for c in ('chirps_rain_mm', 'era5_heat_35', 's2_ndvi_anom'):
        v = o[c]; ok = v.notna()
        nat[c] = (v.where(ok, 0) * w).groupby(o['week_start']).sum() / w.where(ok, 0).groupby(o['week_start']).sum().replace(0, np.nan)
    nat['diesel'] = o.groupby('week_start')['diesel_4city_rs_litre'].mean()
    idx = pd.date_range(nat.index.min(), nat.index.max(), freq='W-MON')
    nat = nat.reindex(idx)
    nat['price'] = nat['price'].interpolate(limit=2); nat['arr'] = nat['arr'].interpolate(limit=2)
    for c in ('chirps_rain_mm', 'era5_heat_35', 's2_ndvi_anom', 'diesel'):
        nat[c] = nat[c].interpolate(limit=4)
    return nat


def z_seasonal(s, min_years=3):
    """z of s versus the same iso-week (+/- 1) in PRIOR years only; NaN if fewer than min_years prior years."""
    iso = s.index.isocalendar(); wk = iso['week'].to_numpy(); yr = iso['year'].to_numpy(); val = s.to_numpy(float)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        if np.isnan(val[i]): continue
        m = (yr < yr[i]) & (np.abs(wk - wk[i]) <= 1) & ~np.isnan(val)
        if len(np.unique(yr[m])) >= min_years:
            ref = val[m]; sd = ref.std()
            if sd > 0: out[i] = (val[i] - ref.mean()) / sd
    return pd.Series(out, index=s.index)


def z_expanding(s, min_n=52):
    mu = s.expanding(min_periods=min_n).mean().shift(1); sd = s.expanding(min_periods=min_n).std().shift(1)
    return (s - mu) / sd


def episodes(nat):
    s = nat['price']
    fwd = s.shift(-FWD) / s - 1; back = s / s.shift(FWD) - 1
    iso = s.index.isocalendar(); wk = iso['week'].to_numpy(); yr = iso['year'].to_numpy(); fv = fwd.to_numpy()
    seas = np.full(len(s), np.nan)
    for i in range(len(s)):
        m = (yr != yr[i]) & (np.abs(wk - wk[i]) <= 1) & ~np.isnan(fv)
        if m.sum() >= 3: seas[i] = np.median(fv[m])
    ok = (fwd >= UP) & ((fwd - seas) >= EXCESS) & (back < CALM)
    weeks = list(s.index[ok]); eps = []
    for w in weeks:
        if not eps or (w - eps[-1]['last']).days > SEP * 7: eps.append({'t0': w, 'last': w})
        else: eps[-1]['last'] = w
    rows = []
    for e in eps:
        t0 = e['t0']; rows.append(dict(onset_week=t0.date(), fwd8_return_pct=round(100 * fwd[t0], 1), seasonal_norm_pct=round(100 * seas[list(s.index).index(t0)], 1),
                                         peak_week=e['last'].date()))
    return [e['t0'] for e in eps], pd.DataFrame(rows)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    nat = national(); print('[77] national weeks: %d  (%s -> %s)' % (nat['price'].notna().sum(), nat.index.min().date(), nat.index.max().date()))
    t0s, E = episodes(nat); print('[77] episodes:', len(t0s)); print(E.to_string(index=False))
    E.to_csv(os.path.join(OUT, 'table_onion_spike_episodes.csv'), index=False)
    arr4 = nat['arr'].rolling(4, min_periods=3).mean()
    D = {
        '1 arrivals shortfall (4-wk mean)':        (z_seasonal(arr4), -1),
        '2 arrivals 4-wk change':                  (z_seasonal(arr4 / arr4.shift(4) - 1), -1),
        '3 stock proxy: 26-wk arrivals shortfall':  (z_seasonal(nat['arr'].rolling(26, min_periods=20).mean()), -1),
        '4 rainfall excess (4-wk sum)':            (z_seasonal(nat['chirps_rain_mm'].rolling(4, min_periods=3).sum()), +1),
        '5 rainfall deficit (4-wk sum)':           (z_seasonal(nat['chirps_rain_mm'].rolling(4, min_periods=3).sum()), -1),
        '6 heat-day anomaly (4-wk mean)':          (z_seasonal(nat['era5_heat_35'].rolling(4, min_periods=3).mean()), +1),
        '7 vegetation anomaly (4-wk mean)':        (z_seasonal(nat['s2_ndvi_anom'].rolling(4, min_periods=3).mean()), -1),
        '8 diesel 13-wk change':                   (z_expanding(nat['diesel'] / nat['diesel'].shift(13) - 1), +1),
        '9 price low vs season (comparator)':      (z_seasonal(nat['price']), -1),
    }
    weeks = nat.index; n = len(weeks); pos = {w: i for i, w in enumerate(weeks)}
    onset_idx = [pos[t] for t in t0s]
    rng = np.random.default_rng(SEED)
    # eligible pseudo-onset weeks: not within 12 weeks before / 8 after any real onset
    bad = np.zeros(n, bool)
    for i in onset_idx: bad[max(0, i - SEP): i + LEAD_HI + 1] = True
    res = []
    for name, (z, sg) in D.items():
        fire = ((z * sg) >= 1).astype(float).where(z.notna()).to_numpy()          # NaN where z undefined
        def hit(i):
            lo, hi = i - LEAD_HI, i - LEAD_LO
            if lo < 0: return np.nan
            w = fire[lo:hi + 1]
            return np.nan if np.isnan(w).any() else float(w.max())
        hv = [hit(i) for i in onset_idx]; used = [i for i, h in zip(onset_idx, hv) if not np.isnan(h)]; hv = [h for h in hv if not np.isnan(h)]
        elig = [i for i in range(n) if not bad[i] and not np.isnan(hit(i))]
        fa = float(np.mean([hit(i) for i in elig])) if elig else np.nan
        k = len(hv); hr = float(np.mean(hv)) if k else np.nan
        perm = np.array([np.mean([hit(i) for i in rng.choice(elig, size=k, replace=False)]) for _ in range(NPERM)]) if k and len(elig) > k else np.array([np.nan])
        p = float((perm >= hr).mean()) if k else np.nan
        res.append(dict(driver=name, episodes_testable=k, episodes_hit=int(sum(hv)) if k else 0, hit_rate=round(hr, 2) if k else np.nan, false_alarm_rate=round(fa, 2), lift=round(hr / fa, 2) if fa and k else np.nan,
                        p_perm=round(p, 4), p_bonferroni=round(min(1, p * len(D)), 4) if k else np.nan))
    R = pd.DataFrame(res)
    R['candidate'] = (R['hit_rate'] >= 2 / 3) & (R['lift'] >= 2) & (R['p_bonferroni'] < 0.05)
    R.to_csv(os.path.join(OUT, 'table_onion_spike_drivers.csv'), index=False)
    pd.set_option('display.width', 220); print(R.to_string(index=False))
    print('candidates:', R.loc[R['candidate'], 'driver'].tolist())


if __name__ == '__main__':
    main()
