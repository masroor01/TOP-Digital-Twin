# -*- coding: utf-8 -*-
"""
Script 91 -- news-volume early-warning test for national onion spikes. Protocol (frozen 2026-11-02 before any news data was pulled):
Model_Output/spike_signals/PROTOCOL_D_news_signal.md. Run once; deviations are reported.
Input : data/news_gdelt/onion_news_daily.csv (Script 90), weekly onion panel, Model_Output/table_onion_spike_episodes.csv.
Output: Model_Output/spike_signals/table_news_episode_timing.csv, table_news_alerts.csv, table_news_summary.csv
Reuses the weekly national reference and the alert / permutation conventions of Script 89 (Protocol B).
"""
import importlib.util
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output', 'spike_signals')
spec = importlib.util.spec_from_file_location('s89', os.path.join(BASE, 'scripts', '89_Daily_Hub_Early_Detection.py'))
s89 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s89)
WIN_BEFORE, WIN_AFTER, ALERT_GAP, HORIZON, NPERM = 56, 28, 14, 56, 5000


def news_signal(s, k):
    """7-day trailing mean vs median + k * MAD of days t-90..t-30 (raw MAD, not scaled). Missing days stay missing."""
    idx = pd.date_range(s.index.min(), s.index.max(), freq='D')
    s = s.reindex(idx)
    tr = s.rolling(7, min_periods=4).mean()
    med = s.shift(30).rolling(61, min_periods=30).median()
    mad = s.shift(30).rolling(61, min_periods=30).apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
    return (tr > med + k * mad).fillna(False), mad


def main():
    d = pd.read_csv(os.path.join(BASE, 'data', 'news_gdelt', 'onion_news_daily.csv'), parse_dates=['date'])
    w, wsig, prior8 = s89.weekly_reference()
    wsig_day = pd.Series(wsig.values, index=wsig.index + pd.Timedelta(days=6))
    ep = pd.read_csv(os.path.join(BASE, 'Model_Output', 'table_onion_spike_episodes.csv'), parse_dates=['onset_week'])
    onsets = pd.DatetimeIndex(ep['onset_week'])
    rng = np.random.default_rng(20261106)
    mondays = pd.date_range('2017-04-03', '2026-08-31', freq='W-MON')    # needs 90 days of news history
    elig = mondays[np.array([not (abs((onsets - m).days) < 56).any() for m in mondays])]
    summ, tabs, alert_rows = {}, [], []
    for q, g in d.groupby('query'):
        s = g.set_index('date')['value']
        for k in (2, 3):
            sig, mad = news_signal(s, k)
            share_on = float(sig[sig.index >= '2017-04-01'].mean())
            zero_mad = float((mad[mad.index >= '2017-04-01'] == 0).mean())
            rows = []
            for t0 in onsets:
                lo, hi = t0 - pd.Timedelta(days=WIN_BEFORE), t0 + pd.Timedelta(days=WIN_AFTER)
                fn = s89.first_on(sig, lo, hi); fp = s89.first_on(wsig_day, lo, hi)
                rows.append(dict(onset=t0.date(), first_news=fn, first_price_weekly=fp, lead_days=(fp - fn).days if pd.notna(fn) and pd.notna(fp) else np.nan))
            T = pd.DataFrame(rows)
            days = sig.index
            wk = pd.Series(days - pd.to_timedelta(days.dayofweek, unit='D'), index=days)
            calm = wk.map(lambda m: bool(m in prior8.index and prior8.get(m, np.nan) == prior8.get(m, np.nan) and prior8.get(m) < 0.10))
            on = (sig & calm & (days >= '2017-04-01'))
            on = on[on].index
            alerts, last = [], None
            for dd in on:
                if last is None or (dd - last).days >= ALERT_GAP:
                    alerts.append(dd)
                last = dd
            alerts = pd.DatetimeIndex(alerts)
            rec_hit = [bool(((alerts >= t - pd.Timedelta(days=WIN_BEFORE)) & (alerts < t)).any()) for t in onsets]
            a_hit = [bool(((onsets > a) & (onsets <= a + pd.Timedelta(days=HORIZON))).any()) for a in alerts]
            recall = float(np.mean(rec_hit)); prec = float(np.mean(a_hit)) if len(alerts) else np.nan
            cnt = 0
            for _ in range(NPERM):
                pick = pd.DatetimeIndex(rng.choice(elig, size=len(onsets), replace=False))
                cnt += np.mean([bool(((alerts >= t - pd.Timedelta(days=WIN_BEFORE)) & (alerts < t)).any()) for t in pick]) >= recall
            p = (1 + cnt) / (NPERM + 1)
            lead = T['lead_days']; both = lead.notna()
            med = float(lead.median()) if both.any() else np.nan
            first_n = int((lead > 0).sum())
            crit = {'i': bool(med >= 7 and first_n >= 6), 'ii': sum(rec_hit) >= 6, 'iii': bool(prec >= 0.30), 'iv': bool(p < 0.05)}
            name = '%s_k%d%s' % (q, k, '' if k == 2 else '_descriptive')
            summ[name] = dict(days_signal_on=share_on, share_days_mad_zero=zero_mad, episodes_both_fire=int(both.sum()), news_first_n=first_n, median_lead_days=med,
                              n_alerts=int(len(alerts)), recall_episodes=int(sum(rec_hit)), precision=prec, perm_p=p, pass_i=crit['i'], pass_ii=crit['ii'], pass_iii=crit['iii'], pass_iv=crit['iv'],
                              PASS_ALL=all(crit.values()))
            T['query'] = q; T['k'] = k; tabs.append(T)
            alert_rows += [dict(query=q, k=k, alert_date=a, followed_by_onset_within_56d=h) for a, h in zip(alerts, a_hit)]
    S = pd.DataFrame(summ).T
    S.to_csv(os.path.join(OUT, 'table_news_summary.csv'))
    pd.concat(tabs).to_csv(os.path.join(OUT, 'table_news_episode_timing.csv'), index=False)
    pd.DataFrame(alert_rows).to_csv(os.path.join(OUT, 'table_news_alerts.csv'), index=False)
    pd.set_option('display.width', 220)
    print(S.to_string())
    print('\nPRIMARY (k=2):', {k: bool(v['PASS_ALL']) for k, v in summ.items() if k.endswith('_k2')})


if __name__ == '__main__':
    main()
