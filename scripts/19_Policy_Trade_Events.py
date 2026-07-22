# -*- coding: utf-8 -*-
"""
Script 19 — Layer 6: Policy/Trade Events (Export Bans, MEP, Export Duty)
===========================================================================
Rebuilt on a verified primary-source event log (36 records, 2017-2026),
replacing the original hand-compiled table. Every event in the source file
is either `primary_source_verified` (checked against the primary PIB/DGFT
document) or `primary_source_retrospective` (documented via a later PIB
reference that cites the original effective date) — spot-checked against
three independent sources in-session (PIB PRID 1985229, PIB PRID 2042765,
and the DGFT Notification 28/2024-25 PDF, all confirmed genuine).

Source file: Downloads-adjacent Codex scraper output —
  TOP_policy_trade_verified_2017_2026.xlsx ("Verified TOP records" sheet)

Derived weekly panel-joinable features (join key: crop, week_start):
  export_banned            0/1 — Onion export prohibition in effect
  mep_usd_per_tonne         Minimum Export Price floor, USD/MT (0 if none)
  export_duty_pct          Export duty, % (0 if none)
  market_intervention_flag 0/1 — a buffer procurement/release, subsidised
                            retail sale, or transport subsidy was reported
                            for this crop in this exact week (point-in-time
                            flag only — the source gives no explicit
                            duration for these actions, so no follow-on
                            weeks are inferred as still "active")
  operation_greens_active  0/1 — Operation Greens (TOP value-chain scheme,
                            launched 2018-11-05) in effect, all 3 crops

Known gap the source itself doesn't resolve: the onion export duty is
confirmed at 40% when imposed (2023-08-19) and confirmed at 20% just
before its 2025-04-01 removal, but no event marks the exact 40%->20%
reduction date. This script assumes the reduction coincided with the
2024-09-13 MEP-removal date (the same approximation used in the prior
version of this script) — flagged `approximate` in the events table.

Outputs (data/policy_trade/):
  export_policy_events.csv     cleaned copy of the verified source (all
                                36 records, all crops/record types)
  policy_weekly_features.csv   weekly per-crop features for panel join

Run: python scripts/19_Policy_Trade_Events.py
"""

import os
import pandas as pd
import numpy as np

BASE      = r'C:\Users\masro\Documents\TOP_Digital_Twin'
SRC_FILE  = r'C:\Users\masro\Documents\Codex\2026-06-15\top_policy_trade_scraper\output_live\TOP_policy_trade_verified_2017_2026.xlsx'
OUT_DIR   = os.path.join(BASE, 'data', 'policy_trade')
os.makedirs(OUT_DIR, exist_ok=True)

PANEL_START = pd.Timestamp('2017-01-01')
PANEL_END   = pd.Timestamp('2025-12-31')
CROPS = ['tomato', 'onion', 'potato']

DUTY_REDUCTION_DATE = pd.Timestamp('2024-09-13')  # approximate — see docstring
OPERATION_GREENS_START = pd.Timestamp('2018-11-05')


def week_monday(d):
    """Snap a date to its Monday (matches the panel's W-MON week_start)."""
    return d - pd.Timedelta(days=d.weekday())


print('=' * 65)
print('SCRIPT 19: LAYER 6 — POLICY/TRADE EVENTS (verified source)')
print('=' * 65)

src = pd.read_excel(SRC_FILE, sheet_name='Verified TOP records')
for col in ['event_date', 'publication_date', 'effective_date']:
    src[col] = pd.to_datetime(src[col], errors='coerce')
src['crop'] = src['crop'].str.lower()

events_path = os.path.join(OUT_DIR, 'export_policy_events.csv')
src.to_csv(events_path, index=False, encoding='utf-8')
print(f'  Saved: {events_path}  ({len(src)} events)')
print(f'\n  By crop:')
print(src.groupby('crop').size().to_string())
print(f'\n  By verification status:')
print(src.groupby('verification_status').size().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 1. ONION CORE POLICY STATE (export_banned, mep, duty) — state-transition
# chronology built from the policy_event rows
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Building Onion export-policy state timeline ...')

pe = src[src['record_type'] == 'policy_event'].sort_values('effective_date')

ban_state = []   # (start, end_inclusive)
mep_state = []   # (start, end_inclusive, value)
duty_state = []  # (start, end_inclusive, value)

# The 2017-12-31 MEP event's effective_date is documented as when that
# measure's window ENDED, not started — the source explicitly states it
# does not know the start date (see notes column). Record it as a
# single-week span rather than feeding it into the state machine below,
# which would otherwise treat effective_date as a start and incorrectly
# extend MEP=$850 all the way to the next minimum_export_price event
# (Sept 2019, a 21-month phantom span).
first_mep_row = pe[(pe['policy_type'] == 'minimum_export_price') &
                    (pe['event_date'] == pd.Timestamp('2017-12-31'))]
if not first_mep_row.empty:
    mep_state.append((pd.Timestamp('2017-12-31'), pd.Timestamp('2017-12-31'),
                       float(first_mep_row.iloc[0]['price_usd_per_mt'])))
    pe = pe.drop(first_mep_row.index)

ban_start = None
mep_value, mep_start = 0, None
duty_value, duty_start = 0, None

for _, row in pe.iterrows():
    eff = row['effective_date']
    ptype = row['policy_type']

    if ptype == 'export_ban':
        ban_start = eff
        # a ban supersedes any active MEP
        if mep_start is not None:
            mep_state.append((mep_start, eff - pd.Timedelta(days=1), mep_value))
            mep_value, mep_start = 0, None
    elif ptype == 'export_ban_removal':
        if ban_start is not None:
            ban_state.append((ban_start, eff - pd.Timedelta(days=1)))
            ban_start = None
        # some removal events simultaneously set a new MEP/duty (2024-05-04)
        if pd.notna(row['price_usd_per_mt']):
            mep_value, mep_start = row['price_usd_per_mt'], eff
        if 'duty' in str(row['description']).lower() and duty_start is None:
            duty_value, duty_start = 40, eff  # reiterated, not a new imposition
    elif ptype == 'minimum_export_price':
        if mep_start is not None:
            mep_state.append((mep_start, eff - pd.Timedelta(days=1), mep_value))
        mep_value, mep_start = row['price_usd_per_mt'], eff
    elif ptype == 'minimum_export_price_removal':
        if mep_start is not None:
            mep_state.append((mep_start, eff - pd.Timedelta(days=1), mep_value))
        mep_value, mep_start = 0, None
    elif ptype == 'export_duty':
        if duty_start is not None:
            duty_state.append((duty_start, eff - pd.Timedelta(days=1), duty_value))
        duty_value, duty_start = 40, eff
    elif ptype == 'export_duty_removal':
        if duty_start is not None:
            # split at the approximate reduction date if it falls inside this span
            if duty_start < DUTY_REDUCTION_DATE < eff:
                duty_state.append((duty_start, DUTY_REDUCTION_DATE - pd.Timedelta(days=1), 40))
                duty_state.append((DUTY_REDUCTION_DATE, eff - pd.Timedelta(days=1), 20))
            else:
                duty_state.append((duty_start, eff - pd.Timedelta(days=1), duty_value))
        duty_value, duty_start = 0, None

# close any still-open spans at panel end
if ban_start is not None:
    ban_state.append((ban_start, PANEL_END))
if mep_start is not None:
    mep_state.append((mep_start, PANEL_END, mep_value))
if duty_start is not None:
    if duty_start < DUTY_REDUCTION_DATE < PANEL_END:
        duty_state.append((duty_start, DUTY_REDUCTION_DATE - pd.Timedelta(days=1), 40))
        duty_state.append((DUTY_REDUCTION_DATE, PANEL_END, 20))
    else:
        duty_state.append((duty_start, PANEL_END, duty_value))

print(f'  export_banned spans: {ban_state}')
print(f'  mep spans: {mep_state}')
print(f'  duty spans: {duty_state}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEEKLY FEATURE TABLE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Building weekly features (join key: crop, week_start) ...')

weeks = pd.date_range(PANEL_START, PANEL_END, freq='W-MON')

mi = src[src['record_type'] == 'market_intervention'].copy()
mi['week'] = mi['event_date'].apply(week_monday)
mi_weeks = set(zip(mi['crop'], mi['week']))

og_start_week = week_monday(OPERATION_GREENS_START)

rows = []
for crop in CROPS:
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)

        if crop == 'onion':
            banned = int(any(s <= week_end and e >= week_start for s, e in ban_state))
            mep = next((v for s, e, v in mep_state if s <= week_end and e >= week_start), 0)
            duty = next((v for s, e, v in duty_state if s <= week_end and e >= week_start), 0)
        else:
            banned, mep, duty = 0, 0, 0

        rows.append({
            'crop': crop,
            'week_start': week_start,
            'export_banned': banned,
            'mep_usd_per_tonne': float(mep),
            'export_duty_pct': float(duty),
            'market_intervention_flag': int((crop, week_start) in mi_weeks),
            'operation_greens_active': int(week_start >= og_start_week),
        })

weekly = pd.DataFrame(rows)
weekly_path = os.path.join(OUT_DIR, 'policy_weekly_features.csv')
weekly.to_csv(weekly_path, index=False, encoding='utf-8')

print(f'  Saved: {weekly_path}  ({len(weekly):,} rows)')
print(f'\n  Weeks with an active export policy (ban/MEP/duty), by crop:')
for crop in CROPS:
    csub = weekly[weekly['crop'] == crop]
    n_active = ((csub['export_banned'] == 1) |
                (csub['mep_usd_per_tonne'] > 0) |
                (csub['export_duty_pct'] > 0)).sum()
    n_mi = csub['market_intervention_flag'].sum()
    print(f'    {crop:8s}: {n_active:>4} / {len(csub)} weeks with export policy active, '
          f'{n_mi} weeks with a market intervention reported')

print('\n' + '=' * 65)
print('Script 19 complete.')
print('\nNext: re-run scripts/22_Master_Panel_Join.py to pick up the new')
print('market_intervention_flag / operation_greens_active columns.')
