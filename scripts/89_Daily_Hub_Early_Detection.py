# -*- coding: utf-8 -*-
"""
Script 89 -- daily hub early-detection test for national onion spikes. Protocol (frozen 2026-11-02 before any estimate):
Model_Output/spike_signals/PROTOCOL_B_daily_hub.md. Run once; deviations are reported.
Inputs : the daily Agmarknet history file and the portal report (paths below, override with TOP_ONION_DAILY / TOP_ONION_REPORT), the weekly onion panel,
         Model_Output/table_onion_spike_episodes.csv.
Outputs: Model_Output/spike_signals/table_hub_episode_timing.csv, table_hub_alerts.csv, table_hub_summary.csv
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output', 'spike_signals')
DAILY = os.environ.get('TOP_ONION_DAILY', r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2025.csv')
REPORT = os.environ.get('TOP_ONION_REPORT', r'C:\Users\masro\Downloads\Daily Price Arrival Report-07-11-2025 to 28-10-2026 (1).csv')
START, SWITCH = pd.Timestamp('2016-11-01'), pd.Timestamp('2025-11-07')
THR, THR2 = 1.15, 1.25
MIN_HUB, MIN_FOLLOW = 3, 30
WIN_BEFORE, WIN_AFTER, ALERT_GAP, HORIZON = 56, 28, 14, 56
NPERM = 5000


def load_daily():
    d = pd.read_csv(DAILY, usecols=['state', 'district', 'market', 'arrival_date', 'arrivals_tonnes', 'modal_price_rs_per_quintal'], parse_dates=['arrival_date'])
    d = d.rename(columns={'arrival_date': 'date', 'arrivals_tonnes': 'arr', 'modal_price_rs_per_quintal': 'modal'})
    d = d[(d['date'] >= START) & (d['date'] <= '2025-12-31')]
    r = pd.read_csv(REPORT, skiprows=1, thousands=',')
    r = r.rename(columns={'State/UT': 'state', 'District': 'district', 'Market': 'market', 'Modal Price': 'modal', 'Arrival Quantity': 'arr'})
    r['date'] = pd.to_datetime(r['Arrival Date'], format='%d-%m-%Y', errors='coerce')
    r = r[['state', 'district', 'market', 'date', 'arr', 'modal']].dropna(subset=['date'])
    return d, r


def market_days(x):
    x = x[(x['modal'] >= 50) & (x['modal'] <= 12000) & (x['arr'] > 0)].copy()
    x['w'] = x['arr']
    x['pw'] = x['modal'] * x['w']
    g = x.groupby(['state', 'district', 'market', 'date']).agg(pw=('pw', 'sum'), w=('w', 'sum')).reset_index()
    g['price'] = g['pw'] / g['w']
    return g[['state', 'district', 'market', 'date', 'price']]


def group_series(md):
    dist = md['district'].astype(str).str.lower()
    hub = dist.str.contains('nashik|nasik')
    delhi = md['state'].astype(str).str.contains('Delhi', case=False)
    out = {}
    for name, sel, mn in (('hub', hub, MIN_HUB), ('follower', ~hub, MIN_FOLLOW), ('delhi', delhi, 1)):
        g = md[sel].groupby('date')['price'].agg(['median', 'size'])
        s = g['median'].where(g['size'] >= mn)
        out[name] = s
    return out


def signal(s, thr):
    idx = pd.date_range(s.index.min(), s.index.max(), freq='D')
    s = s.reindex(idx)
    tr = s.rolling(7, min_periods=4).median()
    base = s.shift(14).rolling(22, min_periods=10).median()
    return (tr >= thr * base).fillna(False), tr, base


def weekly_reference():
    p = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', 'onion_weekly_panel.csv'), parse_dates=['week_start'],
                    usecols=['market_id', 'week_start', 'modal_price_weighted', 'imputed'])
    p = p[(p['imputed'] == 0) & (p['modal_price_weighted'] > 0)]
    g = p.groupby('week_start')['modal_price_weighted'].agg(['median', 'size'])
    w = g['median'].where(g['size'] >= 100)
    base = w.shift(3).rolling(3, min_periods=2).median()
    sig = (w >= THR * base)
    prior8 = w / w.shift(8) - 1
    return w, sig, prior8


def first_on(sig, lo, hi):
    s = sig[(sig.index >= lo) & (sig.index <= hi)]
    on = s[s]
    return on.index[0] if len(on) else pd.NaT


def main():
    os.makedirs(OUT, exist_ok=True)
    d, r = load_daily()
    # comparability of the two sources on their overlap (2025-11-07 to 2025-12-31)
    ov = lambda x: group_series(market_days(x[(x['date'] >= SWITCH) & (x['date'] <= '2025-12-31')]))['follower']
    a, b = ov(d), ov(r)
    j = pd.concat([a, b], axis=1, keys=['file', 'report']).dropna()
    comp = dict(days=len(j), corr=float(j['file'].corr(j['report'])), median_ratio=float((j['report'] / j['file']).median()))
    print('source overlap 2025-11-07..2025-12-31: %d days, correlation of daily follower median %.3f, median ratio report/file %.3f' % (comp['days'], comp['corr'], comp['median_ratio']))
    main = pd.concat([d[d['date'] < SWITCH], r[r['date'] >= SWITCH]], ignore_index=True)
    md = market_days(main)
    G = group_series(md)
    sig = {k: signal(v, THR)[0] for k, v in G.items()}
    sig2 = {k: signal(v, THR2)[0] for k, v in G.items()}
    w, wsig, prior8 = weekly_reference()
    wsig_day = pd.Series(wsig.values, index=wsig.index + pd.Timedelta(days=6))          # signal date = end of the week

    ep = pd.read_csv(os.path.join(BASE, 'Model_Output', 'table_onion_spike_episodes.csv'), parse_dates=['onset_week'])
    rows = []
    for _, e in ep.iterrows():
        t0 = e['onset_week']; lo, hi = t0 - pd.Timedelta(days=WIN_BEFORE), t0 + pd.Timedelta(days=WIN_AFTER)
        rec = {'onset': t0.date()}
        for k in ('hub', 'follower', 'delhi'):
            rec['first_' + k] = first_on(sig[k], lo, hi)
            rec['first25_' + k] = first_on(sig2[k], lo, hi)
        rec['first_national_weekly'] = first_on(wsig_day, lo, hi)
        rec['already_on_hub'] = bool(sig['hub'].get(lo, False))
        for k in ('follower', 'delhi', 'national_weekly'):
            rec['lead_vs_' + k + '_days'] = (rec['first_' + k] - rec['first_hub']).days if pd.notna(rec['first_hub']) and pd.notna(rec['first_' + k]) else np.nan
        for k in ('follower', 'delhi'):
            rec['lead25_vs_' + k + '_days'] = (rec['first25_' + k] - rec['first25_hub']).days if pd.notna(rec['first25_hub']) and pd.notna(rec['first25_' + k]) else np.nan
        rows.append(rec)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'table_hub_episode_timing.csv'), index=False)

    # hub-only alerts
    days = sig['hub'].index.intersection(sig['follower'].index)
    wk = pd.Series(days - pd.to_timedelta(days.dayofweek, unit='D'), index=days)
    calm = wk.map(lambda m: (prior8.get(m, np.nan) < 0.10) if m in prior8.index and prior8.get(m, np.nan) == prior8.get(m, np.nan) else False)
    hubonly = sig['hub'].reindex(days).fillna(False) & ~sig['follower'].reindex(days).fillna(False) & calm
    on = hubonly[hubonly].index
    alerts, last = [], None
    for dd in on:
        if last is None or (dd - last).days >= ALERT_GAP:
            alerts.append(dd)
        last = dd
    alerts = pd.DatetimeIndex(alerts)
    onsets = pd.DatetimeIndex(ep['onset_week'])
    recall_hit = [bool(((alerts >= t - pd.Timedelta(days=WIN_BEFORE)) & (alerts < t)).any()) for t in onsets]
    alert_hit = [bool(((onsets > a) & (onsets <= a + pd.Timedelta(days=HORIZON))).any()) for a in alerts]
    recall = float(np.mean(recall_hit)); precision = float(np.mean(alert_hit)) if len(alerts) else np.nan
    pd.DataFrame({'alert_date': alerts, 'followed_by_onset_within_56d': alert_hit}).to_csv(os.path.join(OUT, 'table_hub_alerts.csv'), index=False)

    # permutation test of the recall (shuffle the onset dates among eligible Mondays)
    rng = np.random.default_rng(20261102)
    mondays = pd.date_range('2017-03-06', '2026-08-31', freq='W-MON')
    ok = np.array([not ((abs((onsets - m).days) < 56) .any()) for m in mondays])
    elig = mondays[ok]
    cnt = 0
    for _ in range(NPERM):
        pick = pd.DatetimeIndex(rng.choice(elig, size=len(onsets), replace=False))
        rp = np.mean([bool(((alerts >= t - pd.Timedelta(days=WIN_BEFORE)) & (alerts < t)).any()) for t in pick])
        cnt += rp >= recall
    p_perm = (1 + cnt) / (NPERM + 1)

    lead = T['lead_vs_follower_days']
    both = lead.notna()
    hub_first = int((lead > 0).sum())
    med_lead = float(lead.median()) if both.any() else np.nan
    S = pd.Series({
        'episodes': len(T), 'episodes_both_fire': int(both.sum()), 'hub_first_n': hub_first, 'hub_first_>=7d_n': int((lead >= 7).sum()), 'median_lead_days_vs_follower': med_lead,
        'median_lead_days_vs_delhi': float(T['lead_vs_delhi_days'].median()), 'median_lead_days_vs_national_weekly': float(T['lead_vs_national_weekly_days'].median()),
        'median_lead25_days_vs_follower (descriptive)': float(T['lead25_vs_follower_days'].median()),
        'n_alerts': int(len(alerts)), 'recall_episodes': int(sum(recall_hit)), 'recall': recall, 'precision': precision, 'perm_p_recall': p_perm,
        'source_overlap_corr': comp['corr'], 'source_overlap_median_ratio': comp['median_ratio'],
    })
    S.to_csv(os.path.join(OUT, 'table_hub_summary.csv'), header=['value'])
    pd.set_option('display.width', 200)
    print(T[['onset', 'first_hub', 'first_follower', 'first_delhi', 'first_national_weekly', 'lead_vs_follower_days', 'lead_vs_delhi_days', 'lead_vs_national_weekly_days', 'already_on_hub']].to_string(index=False))
    print('\n' + S.to_string())
    crit = {'(i) median lead >= 7 d and hub first in >= 6 episodes': bool(med_lead >= 7 and hub_first >= 6), '(ii) recall >= 6 of 12': sum(recall_hit) >= 6,
            '(iii) precision >= 0.30': bool(precision >= 0.30), '(iv) permutation p < 0.05': bool(p_perm < 0.05)}
    print('\nDECISION RULE:', crit, '=> ' + ('PASS (proceed to B2)' if all(crit.values()) else 'NOT PASSED (stop)'))


if __name__ == '__main__':
    main()
