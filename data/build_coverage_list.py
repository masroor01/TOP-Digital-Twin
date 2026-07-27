# Regenerates market_coverage_list.json/.csv from the current raw Agmarknet
# source files (post mid-2026 refresh). Same methodology as the original
# 2026-07-09 build: n_weeks = distinct ISO weeks with a real (non-imputed)
# trade per market, bucketed by share of the study window's max possible
# weeks. Window grew from 2017-2024 (419 max weeks) to 2017-2026-07-27
# (~499 max weeks) with this refresh, so bucket cutoffs are recomputed as
# the SAME percentage-of-window bands, not the same raw week counts.
import pandas as pd
import json

START_DATE = '2017-01-01'
END_DATE   = '2026-07-27'

INFILES = {
    'tomato': r'C:\Users\masro\Downloads\tomato_all_india_apmcs_2000_2026.csv',
    'onion':  r'C:\Users\masro\Downloads\onion_all_india_apmcs_2000_2026.csv',
    'potato': r'C:\Users\masro\Downloads\potato_all_india_apmcs_2000_2026.csv',
}

MAX_WEEKS = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days // 7 + 1
print(f'Study window: {START_DATE} to {END_DATE}  (~{MAX_WEEKS} max weeks)')

# Same percentage-of-window bands as the original 2017-2024 build
# (Full >=87.1%, Good 55.8-86.9%, Moderate 24.8-55.6%, Short 12.4-24.3%, Thin <12.4%)
BANDS = [
    ('Full (7yr+)',       0.871, 1.01),
    ('Good (4.5-7yr)',    0.558, 0.871),
    ('Moderate (2-4.5yr)',0.248, 0.558),
    ('Short (1-2yr)',     0.124, 0.248),
    ('Thin (<1yr)',       0.0,   0.124),
]

def bucket(n_weeks):
    frac = n_weeks / MAX_WEEKS
    for label, lo, hi in BANDS:
        if lo <= frac < hi or (hi >= 1.0 and frac >= lo):
            return label
    return 'Thin (<1yr)'

rows = []
for crop, path in INFILES.items():
    df = pd.read_csv(path, low_memory=False)
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = df.dropna(subset=['arrival_date'])
    df = df[(df['arrival_date'] >= START_DATE) & (df['arrival_date'] <= END_DATE)]
    df['modal_price_rs_per_quintal'] = pd.to_numeric(df['modal_price_rs_per_quintal'], errors='coerce')
    df['arrivals_tonnes'] = pd.to_numeric(df['arrivals_tonnes'], errors='coerce')
    df = df.dropna(subset=['modal_price_rs_per_quintal', 'arrivals_tonnes'])
    df = df[df['arrivals_tonnes'] > 0]

    df['week_start'] = (df['arrival_date'] - pd.to_timedelta(df['arrival_date'].dt.dayofweek, unit='D')).dt.normalize()

    g = df.groupby(['market', 'state']).agg(
        first_week=('week_start', 'min'),
        last_week=('week_start', 'max'),
        n_weeks=('week_start', 'nunique'),
    ).reset_index()

    for _, r in g.iterrows():
        rows.append({
            'crop': crop,
            'state': r['state'],
            'market': r['market'],
            'first_week': r['first_week'].strftime('%Y-%m-%d'),
            'last_week': r['last_week'].strftime('%Y-%m-%d'),
            'n_weeks': int(r['n_weeks']),
            'coverage_category': bucket(r['n_weeks']),
        })
    print(f'  {crop}: {len(g)} markets')

rows.sort(key=lambda r: (r['crop'], r['state'], r['market']))

with open('data/market_coverage_list.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, separators=(',', ':'))

pd.DataFrame(rows).to_csv('data/market_coverage_list.csv', index=False)

print(f'Total rows: {len(rows)}')
print('Saved: data/market_coverage_list.json, data/market_coverage_list.csv')
