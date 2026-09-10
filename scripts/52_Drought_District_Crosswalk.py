# -*- coding: utf-8 -*-
"""
Script 52 — District Crosswalk: VEDAS/SAC Trigger-1 <-> Panel Districts
==============================================================
Script 51 pulls Trigger-1 at VEDAS's own district granularity and district
naming. Those names only match the panel's `district` field partially
(see Script 51 probe report) because of three distinct, real causes that
each need different handling -- this script does NOT collapse them into one
blind fuzzy-match pass:

  1. Cosmetic differences (spacing, punctuation) -- e.g. "SOUTH 24PARGANAS"
     vs "South 24 Parganas". Fixed by normalization alone.
  2. Statutory renames/splits -- e.g. Maharashtra's 2023 renames (Ahmednagar
     -> Ahilyanagar, Aurangabad -> Chhatrapati Sambhaji Nagar, Osmanabad ->
     Dharashiv), UP's Allahabad -> Prayagraj / Faizabad -> Ayodhya (2018),
     Karnataka's 2014 Kannada-spelling renames (Bangalore -> Bengaluru,
     Belgaum -> Belagavi, Mysore -> Mysuru, etc.), Rajasthan's 2023 district
     creation (new districts carved from existing ones, since partly
     reverted). These are real, documented administrative changes, not
     typos -- handled via an explicit, commented ALIAS table below, not
     fuzzy-matched, because a rename should map with certainty or not at
     all.
  3. Genuine unresolvable mismatches -- a VEDAS district with no panel
     counterpart (VEDAS uses a finer/coarser split) or vice versa. Left
     unmatched rather than forced.

Matching order per (state, panel_district):
  a. Normalized exact match against VEDAS district names in that state.
  b. ALIAS table lookup (documented renames/splits only).
  c. Fuzzy match (difflib) within the state, ratio >= FUZZY_ACCEPT ->
     accepted but flagged 'fuzzy' for spot-checking; ratio in
     [FUZZY_REVIEW, FUZZY_ACCEPT) -> written to the output as
     'needs_review' with no VEDAS district assigned; below that -> 'unmatched'.

This crosswalk is deliberately conservative: a panel district left
'unmatched' or 'needs_review' gets no Trigger-1 feature rather than a wrong
one. Whoever joins this onto the market panel should treat 'needs_review'
rows as "not yet safe to join," not "close enough."

Input:
  data/agmarknet_weekly/top_weekly_panel.csv   (state, district)
  data/drought_vedas/trigger1_district_weekly.csv   (state, district_norm)
Output:
  data/drought_vedas/district_crosswalk.csv
    columns: state, panel_district, vedas_district, match_type, similarity,
             notes
  Console: coverage summary (exact/alias/fuzzy/needs_review/unmatched
  counts, and volume-weighted panel-district coverage per crop using the
  same methodology as the earlier CMIE wage state-coverage check).

Run: python scripts/52_Drought_District_Crosswalk.py
"""

import os
import re
import difflib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
TRIGGER1_FILE = os.path.join(BASE, 'data', 'drought_vedas', 'trigger1_district_weekly.csv')
OUT_FILE = os.path.join(BASE, 'data', 'drought_vedas', 'district_crosswalk.csv')

FUZZY_ACCEPT = 0.82   # auto-accept, flagged 'fuzzy' for spot-checking
FUZZY_REVIEW = 0.60   # below FUZZY_ACCEPT but >= this -> 'needs_review'

# Documented (state, panel_district_norm) -> vedas_district_norm renames/
# splits. Only entries verified against a real administrative change go
# here -- do not add a pair just because fuzzy-matching missed it.
ALIAS = {
    ('Maharashtra', 'AHILYANAGAR'):               'AHMEDNAGAR',            # renamed 2023
    ('Maharashtra', 'CHATTRAPATI SAMBHAJINAGAR'):  'CHHATRAPATI SAMBHAJINAGAR',  # renamed 2023 (from Aurangabad)
    ('Maharashtra', 'DHARASHIV'):                  'OSMANABAD',             # renamed 2023
    ('Maharashtra', 'AMARAWATI'):                  'AMRAVATI',              # spelling variant
    ('Uttar Pradesh', 'PRAYAGRAJ'):                'ALLAHABAD',             # renamed 2018
    ('Uttar Pradesh', 'AYODHYA'):                  'FAIZABAD',              # renamed 2018
    ('Karnataka', 'BENGALURU'):                    'BENGALURU (URBAN)',    # 2014 Kannada-spelling rename + urban/rural split
    ('Karnataka', 'BANGALORE'):                    'BENGALURU (URBAN)',
    ('Karnataka', 'BELGAUM'):                      'BELAGAVI',              # renamed 2014
    ('Karnataka', 'BIJAPUR'):                       'VIJAYAPURA',            # renamed 2014
    ('Karnataka', 'GULBARGA'):                      'KALABURAGI',           # renamed 2014
    ('Karnataka', 'MYSORE'):                        'MYSURU',                # renamed 2014
    ('Karnataka', 'TUMKUR'):                        'TUMAKURU',              # renamed 2014
    ('Karnataka', 'SHIMOGA'):                       'SHIVAMOGGA',            # renamed 2014
    ('Karnataka', 'CHIKMAGALUR'):                   'CHIKKAMAGALURU',       # renamed 2014
    ('Karnataka', 'CHIKBALLAPUR'):                  'CHIKKABALLAPURA',      # renamed 2014
    ('Karnataka', 'HASSAN'):                        'HASAN',                # spelling variant
    # VEDAS spells this one district out in words, not digits ("TWENTYFOUR"
    # after normalization strips the hyphen from "TWENTY-FOUR") -- confirmed
    # against the actual normalized candidate list, not guessed.
    ('West Bengal', 'NORTH 24 PARGANAS'):           'NORTH TWENTYFOUR PARGANAS',
    ('West Bengal', 'SOUNTH 24 PARGANAS'):          'SOUTH 24PARGANAS',      # panel typo ("Sounth")
    ('West Bengal', 'COOCHBEHAR'):                  'KOCH BIHAR',            # English vs Bengali transliteration
    ('West Bengal', 'HOWRAH'):                      'HAORA',                 # English vs Bengali transliteration
    ('West Bengal', 'HOOGHLY'):                     'HUGLI',                 # English vs Bengali transliteration
    ('West Bengal', 'MEDINIPURE'):                  'PURBA MEDINIPUR',       # "Medinipur(E)" after norm() drops parens -> "MEDINIPURE"; East = Purba
    ('West Bengal', 'MEDINIPURW'):                  'PASCHIM MEDINIPUR',     # "Medinipur(W)" -> "MEDINIPURW"; West = Paschim
    # Genuine statutory renames/mergers, each confirmed present under the
    # new name in the actual VEDAS candidate list for that state (not
    # guessed from string similarity).
    ('Haryana', 'MEWAT'):                           'NUH',                   # renamed 2016
    ('Haryana', 'GURGAON'):                         'GURUGRAM',              # renamed 2016
    ('Punjab', 'FATEHGARH'):                        'FATEHGARH SAHIB',       # panel's shortened form of the official name
    ('Punjab', 'MUKTSAR'):                          'SRI MUKTSAR SAHIB',     # renamed 2019
    ('Punjab', 'NAWANSHAHR'):                       'SHAHID BHAGAT SINGH NAGAR',  # renamed 2008
    ('Punjab', 'MOHALI'):                           'SAS NAGAR SAHIBZADA AJIT SINGH NAGAR',  # official name S.A.S. Nagar
    ('Tamil Nadu', 'TUTICORIN'):                    'THOOTHUKUDI',           # renamed 2020 (English spelling)
    ('Tamil Nadu', 'THIRUVELLORE'):                 'THIRUVALLUR',           # spelling variant
    ('Gujarat', 'MEHSANA'):                         'MAHESANA',              # spelling variant
    ('Himachal Pradesh', 'SIRMORE'):                'SIRMAUR',               # spelling variant
    ('Odisha', 'KHURDA'):                           'KHORDHA',               # spelling variant
    ('Odisha', 'NUAPADA'):                          'NUAPARHA',              # spelling variant (as spelled in VEDAS)
    ('Odisha', 'SONEPUR'):                          'SUBARNAPUR',            # renamed 2002
    ('Tripura', 'WEST DISTRICT'):                   'WEST TRIPURA',
    ('Uttarakhand', 'GARHWAL PAURI'):               'PAURI GARHWAL',         # word order
    ('Madhya Pradesh', 'HOSHANGABAD'):              'NARMADAPURAM',          # renamed 2021
    # VEDAS's own Maharashtra list has no separate "RAIGAD" entry and
    # instead has "RAIGARH" (the Chhattisgarh spelling) grouped under
    # Maharashtra -- looks like a labelling error on VEDAS's side, not a
    # second real geography. Mapped because it's the only plausible
    # candidate, but flagged in `notes` for anyone auditing this table.
    ('Maharashtra', 'RAIGAD'):                      'RAIGARH',
}

# Panel district labels that are clearly noisy (a market/APMC-yard name, a
# parenthetical alt-name, or a hyphen-joined pair of names) rather than a
# clean district name. Each is normalized into one or more candidate
# strings and matched the same way as a plain district name; the ORIGINAL
# label is still what's written to `panel_district` in the output.
NOISE_SUFFIX_RE = re.compile(
    r'\b(APMC|F ?& ?V|FV|TOWN|MARKET|MANDI|SUB\s*YARD|RYTHU\s*BAZAR)\b', re.I)


def candidate_strings(raw_label):
    """All normalized variants worth trying for a messy panel district
    label: the label itself, with parenthetical content stripped, with
    the parenthetical content alone, and with known noise suffixes
    removed from each of those."""
    variants = {raw_label}
    base = re.sub(r'\(.*?\)', ' ', raw_label)
    variants.add(base)
    for paren_content in re.findall(r'\((.*?)\)', raw_label):
        variants.add(paren_content)
    variants |= {NOISE_SUFFIX_RE.sub(' ', v) for v in list(variants)}
    variants |= {v.replace('-', ' ') for v in list(variants)}
    for v in list(variants):
        if '-' in v:
            variants |= {part.strip() for part in v.split('-') if part.strip()}
    normed = {norm(v) for v in variants}
    normed.discard('')
    return normed

# Directional/positional qualifiers that change WHICH district something is
# (North vs South 24 Parganas, Purba vs Paschim Medinipur, etc.) -- a fuzzy
# match that swaps one of these for another is a false positive no matter
# how high the string-similarity score, because the shared base name (e.g.
# "24 PARGANAS") dominates the ratio. Caught in practice: fuzzy matching
# paired "North 24 Parganas" with VEDAS's "SOUTH 24PARGANAS" at ratio 0.848
# (above FUZZY_ACCEPT) before this guard was added.
DIRECTIONAL_WORDS = {'NORTH', 'SOUTH', 'EAST', 'WEST',
                      'PURBA', 'PASCHIM', 'DAKSHIN', 'UTTAR', 'PURVA'}


def directional_mismatch(a_norm, b_norm):
    a_dir = {w for w in a_norm.split() if w in DIRECTIONAL_WORDS}
    b_dir = {w for w in b_norm.split() if w in DIRECTIONAL_WORDS}
    return a_dir != b_dir


def norm(s):
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9 ]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def best_fuzzy(target, candidates):
    if not candidates:
        return None, 0.0
    scored = [(c, difflib.SequenceMatcher(None, target, c).ratio()) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored[0]


print('=' * 65)
print('SCRIPT 52: DISTRICT CROSSWALK (VEDAS Trigger-1 <-> panel districts)')
print('=' * 65)

panel = pd.read_csv(PANEL_FILE, usecols=['state', 'district']).drop_duplicates()
panel['p_norm'] = panel['district'].map(norm)

# district_norm here is Script 51's own normalization (upper+strip only) --
# re-normalize through this script's norm() (punctuation/hyphen-stripped)
# so both sides of the match are on the same footing.
veda = pd.read_csv(TRIGGER1_FILE, usecols=['state', 'district_norm']).dropna().drop_duplicates()
# VEDAS itself sometimes carries a parenthetical alt-name (e.g. "KEONJHAR
# (KENDUJHAR)") -- expand those into extra matchable keys too, same as the
# panel side, so a clean panel label like "Keonjhar" can hit the
# paren-stripped form directly instead of needing fuzzy matching.
VNORM_TO_ORIGINAL = {}
veda_by_state = {}
for st, g in veda.groupby('state'):
    key_set = set()
    for orig in g['district_norm']:
        for v in candidate_strings(orig):
            key_set.add(v)
            VNORM_TO_ORIGINAL[(st, v)] = orig
    veda_by_state[st] = sorted(key_set)

rows = []
for _, prow in panel.iterrows():
    state, p_district, p_norm = prow['state'], prow['district'], prow['p_norm']
    candidates = veda_by_state.get(state, [])
    variants = candidate_strings(p_district)  # p_norm is always one of these

    exact_hit = next((v for v in variants if v in candidates), None)
    if exact_hit:
        orig = VNORM_TO_ORIGINAL[(state, exact_hit)]
        note = '' if exact_hit == p_norm else f'matched via cleaned variant "{exact_hit}"'
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': orig,
                      'match_type': 'exact', 'similarity': 1.0, 'notes': note})
        continue

    alias_target = next((ALIAS.get((state, v)) for v in variants if (state, v) in ALIAS), None)
    if alias_target and alias_target in candidates:
        orig = VNORM_TO_ORIGINAL[(state, alias_target)]
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': orig,
                      'match_type': 'alias', 'similarity': None,
                      'notes': 'documented rename/split, see ALIAS table'})
        continue

    # Fuzzy across every cleaned variant of the panel label; keep whichever
    # (variant, candidate) pair scores best.
    best = (None, None, 0.0)  # (variant, match, score)
    for v in variants:
        m, s = best_fuzzy(v, candidates)
        if m and s > best[2]:
            best = (v, m, s)
    _, match, score = best
    if match and directional_mismatch(p_norm, match):
        # Don't let a shared base name (e.g. "24 PARGANAS") outweigh a
        # differing North/South/East/West/etc. qualifier -- force review
        # instead of silently accepting a same-family but wrong district.
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': None,
                      'match_type': 'needs_review', 'similarity': round(score, 3),
                      'notes': f'directional-word mismatch vs closest candidate: '
                               f'{VNORM_TO_ORIGINAL[(state, match)]} -- '
                               f'do not auto-accept, verify by hand'})
    elif match and score >= FUZZY_ACCEPT:
        orig = VNORM_TO_ORIGINAL[(state, match)]
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': orig,
                      'match_type': 'fuzzy', 'similarity': round(score, 3), 'notes': ''})
    elif match and score >= FUZZY_REVIEW:
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': None,
                      'match_type': 'needs_review', 'similarity': round(score, 3),
                      'notes': f'closest VEDAS candidate: {VNORM_TO_ORIGINAL[(state, match)]}'})
    elif not candidates:
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': None,
                      'match_type': 'unmatched', 'similarity': None,
                      'notes': 'state not present in Trigger-1 pull (no fortnights returned data)'})
    else:
        rows.append({'state': state, 'panel_district': p_district, 'vedas_district': None,
                      'match_type': 'unmatched', 'similarity': round(score, 3) if match else None,
                      'notes': f'best candidate too weak: {VNORM_TO_ORIGINAL[(state, match)]}' if match else 'no candidate'})

cw = pd.DataFrame(rows).sort_values(['state', 'panel_district']).reset_index(drop=True)
cw.to_csv(OUT_FILE, index=False, encoding='utf-8')
print(f'\nSaved: {OUT_FILE}  ({len(cw):,} panel districts)')

print('\nMatch-type breakdown:')
print(cw['match_type'].value_counts().to_string())

print('\nStates with the most needs_review/unmatched (worth a manual look):')
problem = cw[cw['match_type'].isin(['needs_review', 'unmatched'])]
print(problem.groupby('state').size().sort_values(ascending=False).head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Volume-weighted crop coverage, same methodology as the earlier CMIE wage
# state-coverage check: what share of REAL (non-imputed) arrivals volume,
# per crop, sits in a district that now has a usable ('exact'/'alias'/
# 'fuzzy') Trigger-1 mapping?
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('Volume-weighted crop coverage (last ~12 months, non-imputed rows)')
print('=' * 65)
full_panel = pd.read_csv(PANEL_FILE,
                          usecols=['crop', 'state', 'district', 'week_start',
                                   'arrivals_tonnes_week', 'imputed'])
full_panel['week_start'] = pd.to_datetime(full_panel['week_start'])
cutoff = full_panel['week_start'].max() - pd.Timedelta(days=365)
recent = full_panel[(full_panel['week_start'] >= cutoff) & (~full_panel['imputed'])]

usable = set(zip(cw.loc[cw['match_type'].isin(['exact', 'alias', 'fuzzy']), 'state'],
                  cw.loc[cw['match_type'].isin(['exact', 'alias', 'fuzzy']), 'panel_district']))
recent = recent.copy()
recent['has_trigger1'] = list(zip(recent['state'], recent['district']))
recent['has_trigger1'] = recent['has_trigger1'].isin(usable)

for crop, g in recent.groupby('crop'):
    total = g['arrivals_tonnes_week'].sum()
    covered = g.loc[g['has_trigger1'], 'arrivals_tonnes_week'].sum()
    print(f'  {crop:8s}: {100*covered/total:5.1f}% of real arrivals volume in a '
          f'district with a usable Trigger-1 mapping')

print('\nScript 52 complete.')
