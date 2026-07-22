# -*- coding: utf-8 -*-
"""
Script 19 — Layer 6: Policy/Trade Events (Export Bans, MEP, Export Duty)
===========================================================================
Compiles India's onion/potato/tomato export-policy history (2017-2025) into
a weekly panel-joinable feature set: export ban flag, Minimum Export Price
(MEP, USD/tonne), and export duty (%).

Unlike Layers 1-4 (price, arrivals, macro, climate, satellite), this is not
a bulk downloadable dataset — DGFT notifications are individual gazette
documents. The event table below was compiled from public reporting (PIB,
DGFT/APEDA notification summaries, news coverage) rather than a single
authoritative source. Where the exact effective date wasn't independently
confirmable from multiple sources, the `confidence` column is set to
'approximate' — verify against the DGFT notification archive
(https://apeda.gov.in/dgft-notifications) before treating those specific
transitions as precise in the manuscript.

Coverage found for the 2017-2025 window:
  Onion:  rich history — MEP and export-ban episodes in 2017, 2019-2021,
          and 2023-2025 (crisis years), plus a 2023-2025 export duty.
  Potato: MEP episodes (2014, 2016) predate the 2017-2025 window; no
          confirmed export restriction found inside the window.
  Tomato: no export-policy history found — India is not a significant
          tomato exporter, so this layer is likely genuinely empty/N-A
          for tomato rather than a data-collection gap.

Outputs (data/policy_trade/):
  export_policy_events.csv     raw event table (one row per policy episode)
  policy_weekly_features.csv   weekly per-crop features for panel join:
                                export_banned (0/1), mep_usd_per_tonne,
                                export_duty_pct

Run: python scripts/19_Policy_Trade_Events.py
"""

import os
import pandas as pd
import numpy as np

BASE    = r'C:\Users\masro\Documents\TOP_Digital_Twin'
OUT_DIR = os.path.join(BASE, 'data', 'policy_trade')
os.makedirs(OUT_DIR, exist_ok=True)

PANEL_START = '2017-01-01'
PANEL_END   = '2025-12-31'

# ─────────────────────────────────────────────────────────────────────────────
# RAW EVENT TABLE
# Sources: PIB press releases, DGFT/APEDA notification summaries, Reuters/
# CNBC/Deccan Herald/Business Standard reporting (compiled via web research,
# not scraped from a single dataset). See `notes` per row for the citation
# basis and `confidence` for date precision.
# ─────────────────────────────────────────────────────────────────────────────
EVENTS = [
    # --- ONION ---
    dict(crop='onion', event_type='mep_usd_per_tonne', value=850,
         start_date='2017-11-24', end_date='2017-12-31', confidence='approximate',
         notes='DGFT MEP $850/t on onion exports to boost domestic supply after '
               'exports surged; reported 2017-11-24, effective till 2017-12-31.'),

    dict(crop='onion', event_type='mep_usd_per_tonne', value=850,
         start_date='2019-09-29', end_date='2020-03-14', confidence='approximate',
         notes='Sept 2019: govt banned onion exports and imposed MEP $850/t after '
               'monsoon crop damage; escalated to full prohibition in early Oct 2019 '
               '(see export_ban row below). Free trade resumed 2020-03-15.'),
    dict(crop='onion', event_type='export_ban', value=1,
         start_date='2019-09-29', end_date='2020-03-14', confidence='approximate',
         notes='Export ban "till further orders" from ~2019-09-29 (Sunday '
               'announcement per CNBC 2019-10-02 report), lifted 2020-03-15.'),

    dict(crop='onion', event_type='export_ban', value=1,
         start_date='2020-09-14', end_date='2020-12-31', confidence='approximate',
         notes='Ministry of Commerce prohibited all onion variety exports '
               '2020-09-13/14 to curb domestic shortage; lifted effective 2021-01-01.'),

    dict(crop='onion', event_type='export_duty_pct', value=40,
         start_date='2023-08-19', end_date='2024-09-12', confidence='approximate',
         notes='40% export duty imposed 2023-08-19. Duty reduced (not fully '
               'removed) around Sept 2024 per news of "duty removal" framing '
               'that coincides with MEP removal — see 20%-duty row below for '
               'the inferred continuation. VERIFY exact reduction date via DGFT.'),
    dict(crop='onion', event_type='mep_usd_per_tonne', value=800,
         start_date='2023-10-29', end_date='2023-12-07', confidence='confirmed',
         notes='MEP $800/t effective 2023-10-29, superseded by full export '
               'prohibition from 2023-12-08.'),
    dict(crop='onion', event_type='export_ban', value=1,
         start_date='2023-12-08', end_date='2024-05-03', confidence='confirmed',
         notes='Export policy converted Free -> Prohibited; ban held from '
               '2023-12-08 to 2024-05-03 (widely reported, ~5 month duration).'),
    dict(crop='onion', event_type='mep_usd_per_tonne', value=550,
         start_date='2024-05-04', end_date='2024-09-12', confidence='approximate',
         notes='Ban lifted 2024-05-04 with MEP $550/t imposed in its place; '
               'MEP removed "with immediate effect" per Sept 2024 DGFT notice '
               '(exact day within Sept 2024 not confirmed from sources found).'),
    dict(crop='onion', event_type='export_duty_pct', value=20,
         start_date='2024-09-13', end_date='2025-03-31', confidence='approximate',
         notes='INFERRED: PIB release states the duty withdrawn effective '
               '2025-04-01 was 20% — implying the original 40% duty (Aug 2023) '
               'was reduced to 20% around the Sept 2024 MEP-removal date. '
               'Exact reduction date not independently confirmed — VERIFY.'),
    # export_duty_pct = 0 from 2025-04-01 onward (PIB, confirmed) — no row
    # needed, weekly feature builder defaults to 0 outside all duty rows.

    # --- POTATO ---
    # MEP episodes found (2014-06 to 2015-02 at $450/t; 2016-07 to 2016-12
    # at $360/t) both predate the 2017-2025 panel window. No confirmed
    # export restriction found inside the window from available research.
    # POTATO INTENTIONALLY HAS NO EVENT ROWS FOR THIS PROJECT'S TIME RANGE.

    # --- TOMATO ---
    # No export-policy history found. India is not a significant tomato
    # exporter (high domestic consumption, competes with cheaper Chinese
    # tomato paste in what little export market exists) — this is treated
    # as a genuine absence of the phenomenon, not a data gap.
    # TOMATO INTENTIONALLY HAS NO EVENT ROWS.
]

events_df = pd.DataFrame(EVENTS)
events_df['start_date'] = pd.to_datetime(events_df['start_date'])
events_df['end_date']   = pd.to_datetime(events_df['end_date'])

events_path = os.path.join(OUT_DIR, 'export_policy_events.csv')
events_df.to_csv(events_path, index=False)

print('=' * 65)
print('SCRIPT 19: LAYER 6 — POLICY/TRADE EVENTS')
print('=' * 65)
print(f'  Saved: {events_path}  ({len(events_df)} events)')
print(f'\n  By crop:')
print(events_df.groupby('crop').size().to_string())
print(f'\n  By confidence:')
print(events_df.groupby('confidence').size().to_string())
n_approx = (events_df['confidence'] == 'approximate').sum()
if n_approx:
    print(f'\n  NOTE: {n_approx}/{len(events_df)} events have approximate dates.')
    print('  Recommend verifying against https://apeda.gov.in/dgft-notifications')
    print('  before treating exact transition weeks as precise in the manuscript.')


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY PANEL-JOINABLE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Building weekly features (join key: crop, week_start) ...')

weeks = pd.date_range(PANEL_START, PANEL_END, freq='W-MON')
CROPS = ['tomato', 'onion', 'potato']

rows = []
for crop in CROPS:
    crop_events = events_df[events_df['crop'] == crop]
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)

        def active_value(event_type, default=0):
            active = crop_events[
                (crop_events['event_type'] == event_type) &
                (crop_events['start_date'] <= week_end) &
                (crop_events['end_date'] >= week_start)
            ]
            return active['value'].iloc[0] if not active.empty else default

        rows.append({
            'crop': crop,
            'week_start': week_start,
            'export_banned':      int(active_value('export_ban', 0)),
            'mep_usd_per_tonne':  float(active_value('mep_usd_per_tonne', 0)),
            'export_duty_pct':    float(active_value('export_duty_pct', 0)),
        })

weekly = pd.DataFrame(rows)
weekly_path = os.path.join(OUT_DIR, 'policy_weekly_features.csv')
weekly.to_csv(weekly_path, index=False)

print(f'  Saved: {weekly_path}  ({len(weekly):,} rows)')
print(f'\n  Weeks with an active policy event, by crop:')
for crop in CROPS:
    csub = weekly[weekly['crop'] == crop]
    n_active = ((csub['export_banned'] == 1) |
                (csub['mep_usd_per_tonne'] > 0) |
                (csub['export_duty_pct'] > 0)).sum()
    print(f'    {crop:8s}: {n_active:>4} / {len(csub)} weeks '
          f'({100*n_active/len(csub):.1f}%)')

print('\n' + '=' * 65)
print('Script 19 complete.')
print('\nNext: join policy_weekly_features.csv onto the main weekly panel')
print('(crop, week_start) — same pattern as the macro/climate/satellite joins.')
