# -*- coding: utf-8 -*-
"""
Script 68 -- Pre-trend check for the export-hub design (Track B, Arm A')
========================================================================
Script 67 found hub-vs-non-hub price effects with the OPPOSITE sign to the pre-registered
expectation after the 2019 ban (+21 %) and the 2020 lift (-12.6 %). The leading explanation
is that the hub was already moving before each event (policy is triggered by a supply shock
concentrated at the hub, e.g. Nashik floods before the 2020 ban). Model-free test, no SDID:

  relative gap_t = mean log price(hub markets) - mean log price(non-hub markets)
  baseline       = mean gap over weeks -104..-27 before the event
  pre_26 / pre_12 = mean (gap - baseline) over weeks -26..-1 / -12..-1
  slope_26       = OLS slope of the gap over weeks -26..-1 (log points per week)
  post_12        = mean (gap - baseline) over weeks 0..12 (for comparison)
Reported in % (expm1) and compared with the same statistics for K random same-size blocks of
non-hub markets (crop-coherent placebo) at the same date: p = share of placebo |stat| >= hub |stat|.
If |pre_26| is already about as large as |post_12| and unusual relative to the placebo
blocks, the design's no-anticipation / parallel-trend assumption fails for that event.

Outputs: Model_Output/table_hub_pretrend_check.csv
Run: python scripts/68_Hub_PreTrend_Check.py [K]
"""
import io, os, sys, importlib.util
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('s67', os.path.join(HERE, '67_SDID_Hub_Design_MultiEvent.py'))
s67 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s67)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 20261007


def gap_stats(piv, wk, tr, dn, d):
    g = piv[tr].mean(axis=1) - piv[dn].mean(axis=1)
    rel = (wk - d).dt.days / 7
    base = g[(rel >= -104) & (rel <= -27)].mean()
    gg = g - base
    pre26 = gg[(rel >= -26) & (rel <= -1)]; pre12 = gg[(rel >= -12) & (rel <= -1)]
    post12 = gg[(rel >= 0) & (rel <= 12)]
    x = rel[(rel >= -26) & (rel <= -1)].values
    slope = np.polyfit(x, pre26.values, 1)[0]
    return dict(pre_26=pre26.mean(), pre_12=pre12.mean(), slope_26=slope, post_12=post12.mean())


def main():
    df = pd.read_csv(s67.AGM_FILE, parse_dates=['week_start'])
    df = df[df['crop'] == 'onion'].copy()
    col = df.groupby('market')['market_id'].transform('nunique') > 1
    df.loc[col, 'market'] = df.loc[col, 'market'] + ' (' + df.loc[col, 'state'] + ')'
    rows = []
    for eid, ds, sign, pooled in s67.EVENTS:
        d = pd.Timestamp(ds)
        piv, wk = s67.build_onion_panel(df, d - pd.Timedelta(weeks=s67.PRE_WEEKS), d + pd.Timedelta(weeks=s67.POST_TOTAL), 'price')
        hub = [m for m in piv.columns if m in s67.NASHIK_HUB_MARKETS]
        nonhub = [m for m in piv.columns if m not in s67.NASHIK_HUB_MARKETS]
        if len(hub) < s67.MIN_HUB:
            continue
        real = gap_stats(piv, wk, hub, nonhub, d)
        rng = np.random.default_rng(SEED)
        pl = []
        for _ in range(K):
            pick = set(rng.choice(len(nonhub), size=len(hub), replace=False).tolist())
            tr = [m for i, m in enumerate(nonhub) if i in pick]; dn = [m for i, m in enumerate(nonhub) if i not in pick]
            pl.append(gap_stats(piv, wk, tr, dn, d))
        pl = pd.DataFrame(pl)
        row = dict(event=eid, date=ds, expected_sign=sign, n_hub=len(hub), n_donors=len(nonhub))
        for k in ('pre_26', 'pre_12', 'post_12'):
            row[k + '_pct'] = round(float(np.expm1(real[k]) * 100), 2)
            row['p_' + k] = round((1 + np.sum(np.abs(pl[k]) >= abs(real[k]))) / (K + 1), 4)
        row['slope_26_pct_per_wk'] = round(float(np.expm1(real['slope_26']) * 100), 3)
        row['p_slope_26'] = round((1 + np.sum(np.abs(pl['slope_26']) >= abs(real['slope_26']))) / (K + 1), 4)
        row['placebo_sd_pre_26_pct'] = round(float(pl['pre_26'].std() * 100), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(s67.OUT_DIR, 'table_hub_pretrend_check.csv'), index=False)
    print(out[['event', 'n_hub', 'pre_26_pct', 'p_pre_26', 'pre_12_pct', 'p_pre_12', 'slope_26_pct_per_wk', 'p_slope_26', 'post_12_pct', 'p_post_12', 'placebo_sd_pre_26_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
