# -*- coding: utf-8 -*-
"""
Script 42 — Market Leader-Follower Network (rigorous extension of Script 29 Part B)
=====================================================================================
Script 29 Part B ran a light top-5-market Granger lead-lag test per crop and
summarized it as "most pairs significant in both directions... strong
national market co-movement rather than one market cleanly leading the
others." Recomputing the NET directionality from that same already-computed
output (session discussion) shows this undersold a real, striking pattern --
onion's Pimpalgaon APMC Granger-causes essentially every other top-5 onion
market with zero reverse causality (7 outgoing significant edges out of 8
possible, 0 incoming). Tomato and potato show smaller but real asymmetries
too, with the follower markets (Azadpur, Jangipur, Siliguri) being large
consumption/redistribution hubs rather than growing-region markets -- an
economically sensible pattern (price signals should originate where produce
is grown, not where it is redistributed).

This script validates that finding properly rather than trusting a 5-market
sample, addressing three specific risks identified before running anything:

  1. SCOPE: top-5 is too small to trust as the true national leader. Extended
     to the top 20 markets per crop by mean arrivals volume, restricted to
     markets meeting this project's standard >=70% real-coverage bar
     (Script 11's threshold) so the candidate set is drawn from the same
     quality-filtered population used everywhere else in this project.
  2. CONFOUND: a market that reports more consistently could mechanically
     appear to "Granger-cause" a spottier-reporting market, independent of
     genuine economic leadership. Checked directly: correlation between each
     market's real-coverage rate and its net Granger out-degree.
  3. STABILITY: no robustness check has been done. The full 2017-2026 window
     is split into two roughly equal halves (2017-01 to 2021-07, 2021-07 to
     2026-07) and the SAME leader-ranking procedure is run independently in
     each half; a genuine leader should rank highly in both, not just the
     full-sample average.

Method: identical to Script 29 Part B -- log price, ADF stationarity check
(first-differenced if a unit root is present), bidirectional pairwise
Granger F-tests at lags 1 and 4 weeks, Benjamini-Hochberg FDR correction
within each crop x period. Net leadership score per market = (fraction of
outgoing directed pairs significant) - (fraction of incoming directed pairs
significant), so markets are comparable regardless of how many of their
pairs had usable overlapping data.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv

Outputs (Model_Output/):
  table_market_leader_network.csv       full-sample pairwise Granger results, top-20 x crop
  table_market_leader_ranking.csv       net leadership score per market, full sample + both half-periods
  table_market_leader_confound_check.csv  coverage-vs-leadership correlation per crop
  fig_market_leader_ranking.png         leadership ranking bar chart, all 3 crops
  fig_market_leader_stability.png       full-sample vs half-period rank comparison (top candidate per crop)

Run: python scripts/42_Market_Leader_Network.py
Estimated runtime: ~5-10 minutes (adds sub-period reruns to Script 29 Part B's approach)
"""

import io, os, sys, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
TOP_N = 20
COVERAGE_THRESHOLD = 0.70   # matches Script 11's project-wide qualifying bar
NETWORK_LAGS = [1, 4]
FDR_ALPHA = 0.05
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 42: MARKET LEADER-FOLLOWER NETWORK')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PANEL, SELECT TOP-20 QUALIFYING MARKETS PER CROP
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Selecting top-20 markets per crop (by volume, >=70% real coverage) ...')
panel = pd.read_csv(AGM_FILE, parse_dates=['week_start'])

candidate_markets = {}
for crop in CROPS:
    sub = panel[panel['crop'] == crop]
    cov = 1 - sub.groupby('market')['imputed'].mean()
    qualifying = cov[cov >= COVERAGE_THRESHOLD].index
    vol = sub[sub['market'].isin(qualifying)].groupby('market')['arrivals_tonnes_week'].mean()
    top20 = vol.nlargest(TOP_N).index.tolist()
    candidate_markets[crop] = top20
    print(f'  {crop:7s}: {len(top20)} markets selected (of {len(qualifying)} qualifying at >={COVERAGE_THRESHOLD:.0%} coverage)')


def make_stationary(s):
    s = s.dropna()
    if len(s) < 30:
        return None, 'insufficient_data'
    try:
        p = adfuller(s, autolag='AIC')[1]
    except Exception:
        return None, 'adf_failed'
    if p < 0.05:
        return s, 'level'
    d = s.diff().dropna()
    if d.std() < 1e-8:
        return None, 'degenerate_after_diff'
    return d, 'diff'


def run_granger_pair(y_stat, x_stat, lags):
    merged = pd.concat([y_stat, x_stat], axis=1).dropna()
    merged.columns = ['y', 'x']
    if len(merged) < max(lags) + 15:
        return None
    try:
        res = grangercausalitytests(merged[['y', 'x']], maxlag=max(lags), verbose=False)
    except Exception:
        return None
    out = {}
    for lag in lags:
        if lag in res:
            f, p = res[lag][0]['ssr_ftest'][0], res[lag][0]['ssr_ftest'][1]
            out[lag] = (float(f), float(p))
    return out


def build_market_series(sub, markets, win_start=None, win_end=None):
    series = {}
    for m in markets:
        s = sub[sub['market'] == m].sort_values('week_start').set_index('week_start')['modal_price_weighted']
        if win_start is not None:
            s = s[(s.index >= win_start) & (s.index < win_end)]
        s = np.log(s)
        s_stat, note = make_stationary(s)
        if s_stat is not None:
            series[m] = s_stat
    return series


def run_network(series, lags):
    rows = []
    for m_from in series:
        for m_to in series:
            if m_from == m_to:
                continue
            result = run_granger_pair(series[m_to], series[m_from], lags)
            if result is None:
                continue
            for lag, (f, p) in result.items():
                rows.append({'market_from': m_from, 'market_to': m_to, 'lag_weeks': lag,
                             'F_stat': round(f, 3), 'p_value': round(p, 4)})
    return pd.DataFrame(rows)


def fdr_correct(df):
    if df.empty:
        return df
    rej, p_adj, _, _ = multipletests(df['p_value'], alpha=FDR_ALPHA, method='fdr_bh')
    df = df.copy()
    df['p_value_fdr'] = p_adj
    df['significant_fdr05'] = rej
    return df


def net_scores(df, markets):
    rows = []
    for m in markets:
        out_n = len(df[df['market_from'] == m])
        out_sig = df[df['market_from'] == m]['significant_fdr05'].sum()
        in_n = len(df[df['market_to'] == m])
        in_sig = df[df['market_to'] == m]['significant_fdr05'].sum()
        out_rate = out_sig / out_n if out_n else np.nan
        in_rate = in_sig / in_n if in_n else np.nan
        net = (out_rate - in_rate) if pd.notna(out_rate) and pd.notna(in_rate) else np.nan
        rows.append({'market': m, 'out_sig': int(out_sig), 'out_n': out_n, 'in_sig': int(in_sig),
                      'in_n': in_n, 'out_rate': out_rate, 'in_rate': in_rate, 'net_score': net})
    return pd.DataFrame(rows).sort_values('net_score', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FULL-SAMPLE NETWORK, TOP-20, ALL 3 CROPS
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Full-sample pairwise Granger network (top-20, lags 1 & 4 weeks) ...')
full_networks = {}
full_rankings = {}
for crop in CROPS:
    sub = panel[(panel['crop'] == crop) & (panel['imputed'] == 0)]
    series = build_market_series(sub, candidate_markets[crop])
    net = run_network(series, NETWORK_LAGS)
    net = fdr_correct(net)
    net.insert(0, 'crop', crop)
    full_networks[crop] = net
    ranking = net_scores(net, list(series.keys()))
    full_rankings[crop] = ranking
    print(f'\n  {crop.upper()} full-sample leadership ranking (net score = out-rate - in-rate):')
    for _, r in ranking.head(5).iterrows():
        print(f"    {r['market']:35s} out={r['out_sig']}/{r['out_n']}  in={r['in_sig']}/{r['in_n']}  "
              f"net={r['net_score']:+.3f}")

table_net_path = os.path.join(OUT_DIR, 'table_market_leader_network.csv')
pd.concat(full_networks.values(), ignore_index=True).to_csv(table_net_path, index=False)
print(f'\n  Saved: {table_net_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFOUND CHECK — does leadership just track reporting coverage?
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Confound check: does net leadership score correlate with real-coverage rate? ...')
confound_rows = []
for crop in CROPS:
    sub = panel[panel['crop'] == crop]
    cov = 1 - sub.groupby('market')['imputed'].mean()
    ranking = full_rankings[crop].copy()
    ranking['coverage'] = ranking['market'].map(cov)
    valid = ranking.dropna(subset=['net_score', 'coverage'])
    if len(valid) >= 4:
        rho, p = spearmanr(valid['coverage'], valid['net_score'])
    else:
        rho, p = np.nan, np.nan
    confound_rows.append({'crop': crop, 'n_markets': len(valid), 'spearman_rho': round(float(rho), 3),
                           'p_value': round(float(p), 4) if pd.notna(p) else None})
    verdict = 'NO evidence of coverage confound' if (pd.notna(p) and p > 0.10) else 'POSSIBLE coverage confound -- inspect'
    print(f'  {crop:7s} coverage vs. net-leadership-score: rho={rho:+.3f}  p={p:.3f}  -- {verdict}')

confound_df = pd.DataFrame(confound_rows)
confound_path = os.path.join(OUT_DIR, 'table_market_leader_confound_check.csv')
confound_df.to_csv(confound_path, index=False)
print(f'\n  Saved: {confound_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. STABILITY CHECK — split into two halves, rerun independently
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Stability check: independent rerun in two half-periods ...')
all_weeks = sorted(panel['week_start'].unique())
midpoint = all_weeks[len(all_weeks) // 2]
periods = {'early': (all_weeks[0], midpoint), 'late': (midpoint, all_weeks[-1] + pd.Timedelta(days=1))}
print(f'  early: {periods["early"][0]} to {periods["early"][1]}  |  late: {periods["late"][0]} to {periods["late"][1]}')

period_rankings = {}
for period_name, (w_start, w_end) in periods.items():
    for crop in CROPS:
        sub = panel[(panel['crop'] == crop) & (panel['imputed'] == 0)]
        series = build_market_series(sub, candidate_markets[crop], w_start, w_end)
        net = run_network(series, NETWORK_LAGS)
        net = fdr_correct(net)
        ranking = net_scores(net, list(series.keys()))
        period_rankings[(period_name, crop)] = ranking

stability_rows = []
for crop in CROPS:
    full_r = full_rankings[crop].set_index('market')['net_score']
    early_r = period_rankings[('early', crop)].set_index('market')['net_score']
    late_r = period_rankings[('late', crop)].set_index('market')['net_score']
    common = full_r.index.intersection(early_r.index).intersection(late_r.index)
    if len(common) >= 4:
        rho_early, _ = spearmanr(full_r[common], early_r[common])
        rho_late, _ = spearmanr(full_r[common], late_r[common])
    else:
        rho_early, rho_late = np.nan, np.nan
    top_market = full_rankings[crop].iloc[0]['market']
    top_rank_early = (early_r.rank(ascending=False).get(top_market))
    top_rank_late = (late_r.rank(ascending=False).get(top_market))
    stability_rows.append({'crop': crop, 'full_sample_leader': top_market,
                            'rank_in_early_period': int(top_rank_early) if pd.notna(top_rank_early) else None,
                            'rank_in_late_period': int(top_rank_late) if pd.notna(top_rank_late) else None,
                            'n_common_markets': len(common),
                            'spearman_full_vs_early': round(float(rho_early), 3) if pd.notna(rho_early) else None,
                            'spearman_full_vs_late': round(float(rho_late), 3) if pd.notna(rho_late) else None})
    print(f'  {crop:7s} full-sample leader "{top_market}": rank #{top_rank_early} in early period, '
          f'#{top_rank_late} in late period (of {len(common)} common markets)')

stability_df = pd.DataFrame(stability_rows)
stability_path = os.path.join(OUT_DIR, 'table_market_leader_stability.csv')
stability_df.to_csv(stability_path, index=False)
print(f'\n  Saved: {stability_path}')

ranking_all_path = os.path.join(OUT_DIR, 'table_market_leader_ranking.csv')
ranking_out = []
for crop in CROPS:
    r = full_rankings[crop].copy()
    r.insert(0, 'crop', crop)
    r['period'] = 'full_sample'
    ranking_out.append(r)
    for period_name in ['early', 'late']:
        r2 = period_rankings[(period_name, crop)].copy()
        r2.insert(0, 'crop', crop)
        r2['period'] = period_name
        ranking_out.append(r2)
pd.concat(ranking_out, ignore_index=True).to_csv(ranking_all_path, index=False)
print(f'  Saved: {ranking_all_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Generating figures ...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, crop in zip(axes, CROPS):
    r = full_rankings[crop].dropna(subset=['net_score']).sort_values('net_score')
    colors = [CROP_COLORS[crop] if v > 0 else '#AAAAAA' for v in r['net_score']]
    ax.barh(r['market'], r['net_score'], color=colors)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_title(f'{crop.capitalize()}: net leadership score\n(out-rate − in-rate, top-20 markets)',
                 fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Net score (+ = leader, − = follower)')
    ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, 'fig_market_leader_ranking.png')
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig1_path}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, crop in zip(axes, CROPS):
    full_r = full_rankings[crop].set_index('market')['net_score']
    early_r = period_rankings[('early', crop)].set_index('market')['net_score']
    late_r = period_rankings[('late', crop)].set_index('market')['net_score']
    common = sorted(full_r.index.intersection(early_r.index).intersection(late_r.index))
    x = np.arange(len(common))
    w = 0.27
    ax.bar(x - w, full_r[common], width=w, label='Full sample', color=CROP_COLORS[crop])
    ax.bar(x, early_r[common], width=w, label='Early period', color=CROP_COLORS[crop], alpha=0.5)
    ax.bar(x + w, late_r[common], width=w, label='Late period', color=CROP_COLORS[crop], alpha=0.2, edgecolor=CROP_COLORS[crop])
    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:12] for m in common], rotation=90, fontsize=6.5)
    ax.set_title(f'{crop.capitalize()}: stability across periods', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'fig_market_leader_stability.png')
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig2_path}')

print('\n' + '=' * 65)
print('Script 42 complete.')
print('\nRead table_market_leader_stability.csv and the confound-check table before')
print('trusting any single market as "the leader" -- a market only qualifies as a')
print('robust finding if its full-sample rank holds up in BOTH half-periods AND its')
print('leadership does not simply track its reporting coverage rate.')
