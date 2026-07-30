# -*- coding: utf-8 -*-
"""
Script 28 — Crisis Backtesting Case Studies
=============================================
The ablation study (Script 15) and MCS analysis (Script 27) answer "how
accurate is the model on average, by horizon." This script asks the
question a policy reader actually cares about: "how did the model do
during the specific price crises that mattered?"

Four episodes, chosen by requiring BOTH an empirically detected price
extreme in the real (non-imputed) weekly panel AND, where one exists, a
verified primary-source policy event from Script 19's event log
(`data/policy_trade/export_policy_events.csv`) that coincides with it —
so the case studies aren't cherry-picked, they're the panel's own largest
moves, corroborated by a documented policy response:

  TOMATO_2023   Jun-Sep 2023 nationwide spike (+454% over 4 weeks at
                peak, early Jul) then crash (-70%, early Sep) after
                unseasonal rain crop damage; NCCF/NAFED began subsidised
                retail sale 2023-07-20 (verified event).
  ONION_2023_24 Aug 2023-Jan 2024 export-restriction escalation
                (40% duty 2023-08-19 -> MEP 2023-10-28 -> export ban
                2023-12-08) tracking a +89% price spike (late Oct), then
                a -46% crash within weeks of the ban taking effect -- a
                natural before/after test of policy effectiveness.
  POTATO_SPIKE_2024  Apr 2024 nationwide spike (+45%). No matching
                policy event in the log -- unlike tomato/onion, potato
                has no active export/MEP regime, consistent with this
                project's earlier finding that potato's price dynamics
                are the least policy-driven of the three crops.
  POTATO_CRASH_2025  Jan 2025 nationwide crash (-47%). Also no matching
                policy event -- same story.

All four episodes fall inside a genuinely out-of-sample fold (fold
train_end dates are 2021/2022/2023/2024-06-30; each fold's predictions
for the following calendar year never saw that year's data in training),
so this reuses Script 15's already-computed crop-level weekly predictions
rather than retraining anything.

Inputs:
  Model_Output/ablation_predictions.csv        (Script 15)
  data/policy_trade/export_policy_events.csv   (Script 19)

Outputs (Model_Output/):
  table_crisis_backtests.csv   per-episode x horizon x variant error metrics,
                                 restricted to each episode's core crisis window
  fig_crisis_tomato_2023.png
  fig_crisis_onion_2023_24.png
  fig_crisis_potato_2024_25.png

Run: python scripts/28_Crisis_Backtesting_Case_Studies.py
Estimated runtime: <1 minute (no model fitting, pure analysis on
already-computed predictions)
"""

import io, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_FILE  = os.path.join(BASE, 'Model_Output', 'ablation_predictions.csv')
EVENT_FILE = os.path.join(BASE, 'data', 'policy_trade', 'export_policy_events.csv')
OUT_DIR    = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}
NAIVE_COLOR = '#7C8A78'

HORIZONS_TO_PLOT = [1, 4]     # shortest two horizons -- most policy-relevant for a live crisis
HORIZONS_FOR_TABLE = [1, 4, 13, 26]

# episode: display_window (for the chart, with lead-in/out context),
#          core_window (for the error table -- the actual crisis dates),
#          policy events to annotate (may be empty)
EPISODES = [
    {
        'key': 'TOMATO_2023', 'crop': 'tomato',
        'title': 'Tomato — Jun-Sep 2023 nationwide spike & crash',
        'display_start': '2023-04-01', 'display_end': '2023-11-01',
        'core_start': '2023-06-01', 'core_end': '2023-09-15',
        'event_filter': lambda ev: (ev['crop'].isin(['tomato', 'top'])) &
                                    (ev['event_date'] >= '2023-01-01') & (ev['event_date'] <= '2023-12-31'),
    },
    {
        'key': 'ONION_2023_24', 'crop': 'onion',
        'title': 'Onion — Aug 2023-Jan 2024 export-restriction escalation & post-ban crash',
        'display_start': '2023-06-01', 'display_end': '2024-03-01',
        'core_start': '2023-08-01', 'core_end': '2024-02-01',
        'event_filter': lambda ev: (ev['crop'] == 'onion') &
                                    (ev['event_date'] >= '2023-06-01') & (ev['event_date'] <= '2024-02-01'),
    },
    {
        'key': 'POTATO_SPIKE_2024', 'crop': 'potato',
        'title': 'Potato — Apr 2024 spike (empirically detected, no matching policy event)',
        'display_start': '2024-01-01', 'display_end': '2024-07-15',
        'core_start': '2024-03-01', 'core_end': '2024-06-01',
        'event_filter': lambda ev: (ev['crop'].isin(['potato', 'top'])) &
                                    (ev['event_date'] >= '2024-01-01') & (ev['event_date'] <= '2024-07-15'),
    },
    {
        'key': 'POTATO_CRASH_2025', 'crop': 'potato',
        'title': 'Potato — Jan 2025 crash (empirically detected, no matching policy event)',
        'display_start': '2024-10-01', 'display_end': '2025-04-01',
        'core_start': '2024-12-01', 'core_end': '2025-03-01',
        'event_filter': lambda ev: (ev['crop'].isin(['potato', 'top'])) &
                                    (ev['event_date'] >= '2024-10-01') & (ev['event_date'] <= '2025-04-01'),
    },
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('=' * 65)
print('SCRIPT 28: CRISIS BACKTESTING CASE STUDIES')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(PRED_FILE):
    print(f'ERROR: {PRED_FILE} not found. Run scripts/15_Ablation_Study_M0_M4.py first.')
    sys.exit(1)

preds = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
preds = preds.sort_values(['crop', 'horizon_weeks', 'variant', 'fold', 'week_start'])

events = pd.read_csv(EVENT_FILE, parse_dates=['event_date']) if os.path.exists(EVENT_FILE) else pd.DataFrame()
print(f'  Predictions loaded: {len(preds):,} rows')
print(f'  Policy events loaded: {len(events):,} rows\n')


def get_series(crop, horizon, variant):
    sub = (preds[(preds['crop'] == crop) & (preds['horizon_weeks'] == horizon) & (preds['variant'] == variant)]
           .sort_values(['fold', 'week_start'])
           .drop_duplicates(subset='week_start'))
    return sub.set_index('week_start')[['y_true', 'y_pred', 'fold']]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PER-EPISODE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
table_rows = []

for ep in EPISODES:
    crop = ep['crop']
    print(f"[{ep['key']}] {ep['title']}")

    ev_sub = events[ep['event_filter'](events)].sort_values('event_date') if not events.empty else pd.DataFrame()
    if not ev_sub.empty:
        for _, r in ev_sub.iterrows():
            print(f"    policy event: {r['event_date'].date()}  {r['policy_type']:<28s} {r['description'][:70]}")
    else:
        print('    policy event: none in the verified log for this window')

    # error metrics over the CORE crisis window, per horizon, M6 vs naive
    for h in HORIZONS_FOR_TABLE:
        m6 = get_series(crop, h, 'M6')
        nv = get_series(crop, h, 'B1_Naive')
        core = (m6.index >= ep['core_start']) & (m6.index <= ep['core_end'])
        m6c, nvc = m6.loc[core], nv.loc[nv.index.isin(m6.index[core])]
        if len(m6c) < 2:
            continue
        m6_mae = float((m6c['y_true'] - m6c['y_pred']).abs().mean())
        m6_mape = float(((m6c['y_true'] - m6c['y_pred']).abs() / m6c['y_true']).mean() * 100)
        nv_mae = float((nvc['y_true'] - nvc['y_pred']).abs().mean())
        nv_mape = float(((nvc['y_true'] - nvc['y_pred']).abs() / nvc['y_true']).mean() * 100)
        table_rows.append({
            'episode': ep['key'], 'crop': crop, 'horizon_weeks': h,
            'core_start': ep['core_start'], 'core_end': ep['core_end'], 'n_weeks': len(m6c),
            'M6_MAE': round(m6_mae, 1), 'M6_MAPE': round(m6_mape, 1),
            'Naive_MAE': round(nv_mae, 1), 'Naive_MAPE': round(nv_mape, 1),
            'M6_beats_naive_in_crisis': m6_mae < nv_mae,
        })
        print(f'    h={h:>2}w  M6 MAPE={m6_mape:>6.1f}%  Naive MAPE={nv_mape:>6.1f}%  '
              f'{"M6 wins" if m6_mae < nv_mae else "Naive wins"}')

    # ── figure ──
    fig, ax = plt.subplots(figsize=(10, 5))
    actual = get_series(crop, 1, 'M6')  # y_true identical across variants at a given horizon
    actual = actual.loc[(actual.index >= ep['display_start']) & (actual.index <= ep['display_end'])]
    ax.plot(actual.index, actual['y_true'], color='black', linewidth=2.2, label='Actual price', zorder=5)

    linestyles = {1: '-', 4: '--'}
    for h in HORIZONS_TO_PLOT:
        m6 = get_series(crop, h, 'M6')
        m6 = m6.loc[(m6.index >= ep['display_start']) & (m6.index <= ep['display_end'])]
        ax.plot(m6.index, m6['y_pred'], color=CROP_COLORS[crop], linewidth=1.6,
                 linestyle=linestyles[h], alpha=0.85, label=f'M6 forecast (h={h}w)')

    nv = get_series(crop, 1, 'B1_Naive')
    nv = nv.loc[(nv.index >= ep['display_start']) & (nv.index <= ep['display_end'])]
    ax.plot(nv.index, nv['y_pred'], color=NAIVE_COLOR, linewidth=1.3, linestyle=':',
             alpha=0.9, label='Naive persistence (h=1w)')

    ax.axvspan(pd.Timestamp(ep['core_start']), pd.Timestamp(ep['core_end']),
               color=CROP_COLORS[crop], alpha=0.06, zorder=0)

    if not ev_sub.empty:
        for _, r in ev_sub.iterrows():
            ax.axvline(r['event_date'], color='#333333', linewidth=0.9, linestyle='-', alpha=0.55)
            ax.annotate(r['policy_type'].replace('_', ' '), xy=(r['event_date'], ax.get_ylim()[1]),
                        xytext=(2, -4), textcoords='offset points', rotation=90, va='top', ha='left',
                        fontsize=7, color='#333333', alpha=0.85)

    ax.set_title(ep['title'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (Rs/quintal)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"fig_crisis_{ep['key'].lower()}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {fig_path}\n')

# rename potato figures to match the two-file naming in the docstring
os.replace(os.path.join(OUT_DIR, 'fig_crisis_potato_spike_2024.png'),
           os.path.join(OUT_DIR, 'fig_crisis_potato_2024_25_spike.png'))
os.replace(os.path.join(OUT_DIR, 'fig_crisis_potato_crash_2025.png'),
           os.path.join(OUT_DIR, 'fig_crisis_potato_2024_25_crash.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE TABLE
# ─────────────────────────────────────────────────────────────────────────────
table = pd.DataFrame(table_rows)
table_path = os.path.join(OUT_DIR, 'table_crisis_backtests.csv')
table.to_csv(table_path, index=False)
print(f'Saved: {table_path}  ({len(table)} rows)')

n_m6_wins = table['M6_beats_naive_in_crisis'].sum()
print(f'\nAcross all episodes x horizons: M6 beats naive in {n_m6_wins}/{len(table)} crisis-window comparisons')
print('(compare to the all-horizon average, where naive wins comfortably at h=1w for every crop --')
print(' this table asks whether that holds specifically DURING the crisis windows policy cares about.)')

print('\n' + '=' * 65)
print('Script 28 complete.')
print('\nKey outputs:')
for fname in ['table_crisis_backtests.csv', 'fig_crisis_tomato_2023.png',
              'fig_crisis_onion_2023_24.png', 'fig_crisis_potato_2024_25_spike.png',
              'fig_crisis_potato_2024_25_crash.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<40} {os.path.getsize(fpath)/1024:>7.1f} KB')
