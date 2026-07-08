# -*- coding: utf-8 -*-
"""
Script 11 — Market Selection + Data Structure Visualisation
===========================================================
Filters Agmarknet weekly panel to markets with ≥90% temporal coverage,
generates paper-ready figures and appendix tables.

Outputs (all in Model_Output/):
  filtered_panel_top.csv              — ≥90% coverage panel (3 crops combined)
  appendix_market_selection.xlsx      — per-crop appendix tables
  fig01_data_architecture.png         — Digital Twin 6-layer architecture diagram
  fig02_market_coverage_distribution.png  — coverage histogram + tier breakdown
  fig03_markets_by_state.png          — state-wise market counts (3 crops)
  fig04_coverage_heatmap.png          — year-by-year coverage heatmap (top markets)
  fig05_panel_timeline.png            — weekly observation counts 2017-2024

Run: python scripts/11_Market_Selection_And_DataStructure.py
"""

import io, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = r'C:\Users\masro\Documents\TOP_Digital_Twin'
INFILE   = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
APPX_DIR = os.path.join(BASE, 'Market_Output')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(APPX_DIR, exist_ok=True)

COVERAGE_THRESHOLD = 90          # % of weeks required
CROPS = ['tomato', 'onion', 'potato']

CROP_COLORS = {
    'tomato': '#E63946',
    'onion':  '#F4A261',
    'potato': '#457B9D',
}

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':    150,
})


# ── 1. Load & filter ──────────────────────────────────────────────────────────
print('Loading weekly panel ...')
df = pd.read_csv(INFILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2024-12-31')]
TOTAL_WEEKS = df['week_start'].nunique()
print(f'  Total ISO weeks 2017-2024: {TOTAL_WEEKS}')

def compute_coverage(sub):
    sub = sub.copy()
    sub['year'] = sub['week_start'].dt.year
    return (sub.groupby('market')
               .agg(state         = ('state', 'first'),
                    district      = ('district', 'first'),
                    weeks_present = ('week_start', 'nunique'),
                    years_present = ('year',       'nunique'),
                    mean_price    = ('modal_price_weighted',  'mean'),
                    mean_arrivals = ('arrivals_tonnes_week',  'mean'),
                    cv_price      = ('modal_price_weighted',
                                     lambda x: x.std()/x.mean()*100 if x.mean()>0 else 0))
               .assign(pct_coverage = lambda d: (d['weeks_present']/TOTAL_WEEKS*100).round(1))
               .reset_index())

coverage = {}
for crop in CROPS:
    coverage[crop] = compute_coverage(df[df['crop'] == crop])
    coverage[crop]['crop'] = crop

all_coverage = pd.concat(coverage.values(), ignore_index=True)

# Filter
selected = {c: coverage[c][coverage[c]['pct_coverage'] >= COVERAGE_THRESHOLD]
            for c in CROPS}

print('\nMarket selection summary (≥90% coverage):')
for crop in CROPS:
    n_all = len(coverage[crop])
    n_sel = len(selected[crop])
    print(f'  {crop:8s}: {n_sel:>4} / {n_all} markets selected  '
          f'({n_sel/n_all*100:.0f}%)')

# Save filtered panel
sel_ids = {crop: set(selected[crop]['market']) for crop in CROPS}
df_filtered = df[df.apply(lambda r: r['market'] in sel_ids[r['crop']], axis=1)]
filtered_path = os.path.join(OUT_DIR, 'filtered_panel_top.csv')
df_filtered.to_csv(filtered_path, index=False)
print(f'\nFiltered panel saved: {filtered_path}  ({len(df_filtered):,} rows)')


# ── 2. Appendix Excel ─────────────────────────────────────────────────────────
appx_path = os.path.join(APPX_DIR, 'appendix_market_selection.xlsx')
with pd.ExcelWriter(appx_path, engine='openpyxl') as writer:
    # Summary sheet
    summary_rows = []
    for crop in CROPS:
        s = selected[crop]
        summary_rows.append({
            'Crop': crop.capitalize(),
            'Markets selected': len(s),
            'Markets total':    len(coverage[crop]),
            'States covered':   s['state'].nunique(),
            'Min coverage %':   s['pct_coverage'].min(),
            'Median coverage %':s['pct_coverage'].median(),
            'Mean price (Rs/q)':round(s['mean_price'].mean(), 1),
        })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    for crop in CROPS:
        out = (selected[crop][['market','state','district',
                               'weeks_present','pct_coverage',
                               'years_present','mean_price',
                               'mean_arrivals','cv_price']]
               .sort_values(['state','pct_coverage'], ascending=[True, False])
               .reset_index(drop=True))
        out.index += 1
        out.columns = ['Market', 'State', 'District', 'Weeks present',
                       'Coverage %', 'Years present',
                       'Mean price (Rs/q)', 'Mean arrivals (t/wk)', 'CV price (%)']
        out.to_excel(writer, sheet_name=crop.capitalize(), index_label='No.')
print(f'Appendix saved: {appx_path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Data Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 1: Data architecture ...')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

LAYER_COLORS = ['#2196F3','#4CAF50','#00BCD4','#FF9800','#9C27B0','#F44336']
MODULE_COLORS= ['#1565C0','#2E7D32','#6A1B9A','#BF360C']

layers = [
    ('Layer 1', 'Market',         'Agmarknet / e-NAM',
     'Modal price, arrivals,\nprice spreads, volatility'),
    ('Layer 2', 'Satellite',      'Sentinel-2 / MODIS (GEE)',
     'NDVI, EVI, NDVI anomaly,\npre-harvest vegetation slope'),
    ('Layer 3', 'Climate',        'ERA5-Land / CHIRPS / IMD',
     'Rainfall, SPI-30/90,\nTmax, heat days, excess rain'),
    ('Layer 4', 'Macro-Logistics','CMIE / RBI DBIE / PPAC',
     'Fuel prices, WPI, repo rate,\ncredit, export/import'),
    ('Layer 5', 'Infrastructure', 'CMIE States / NHB / NCCD',
     'Cold storage, road density,\nirrigation, rural wages'),
    ('Layer 6', 'Policy & Trade', 'DGFT / APEDA / NAFED',
     'Export ban, MEP,\nbuffer release, procurement'),
]
modules = [
    ('Module A', 'Supply Signal',     'Satellite + climate\n→ supply-risk score'),
    ('Module B', 'Price Forecasting', 'LightGBM multi-horizon\nforecast + spike prob.'),
    ('Module C', 'Market Network',    'Granger causality\nleader/follower nodes'),
    ('Module D', 'Stress Testing',    'Scenario perturbation\npolicy stress tests'),
]

# Draw layers (left column)
for i, (lnum, lname, lsrc, lvars) in enumerate(layers):
    y = 7.0 - i * 1.1
    rect = mpatches.FancyBboxPatch((0.2, y-0.38), 5.6, 0.76,
        boxstyle='round,pad=0.05', facecolor=LAYER_COLORS[i],
        alpha=0.15, edgecolor=LAYER_COLORS[i], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5,  y+0.12, lnum,   fontsize=7,  color=LAYER_COLORS[i], fontweight='bold')
    ax.text(0.5,  y-0.12, lname,  fontsize=9,  color='#333333', fontweight='bold')
    ax.text(3.0,  y+0.12, lsrc,   fontsize=7,  color='#555555', style='italic', ha='center')
    ax.text(3.0,  y-0.15, lvars,  fontsize=7,  color='#444444', ha='center')

# Arrow from layers to integration box
ax.annotate('', xy=(7.0, 3.5), xytext=(5.8, 3.5),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Integration box (centre)
int_rect = mpatches.FancyBboxPatch((7.0, 2.3), 1.8, 2.4,
    boxstyle='round,pad=0.08', facecolor='#FFF9C4',
    edgecolor='#F9A825', linewidth=2)
ax.add_patch(int_rect)
ax.text(7.9, 3.8, 'Master\nPanel',     ha='center', fontsize=9,  fontweight='bold', color='#333333')
ax.text(7.9, 3.3, 'market × crop\n× date',  ha='center', fontsize=7.5, color='#555555')
ax.text(7.9, 2.75,'No look-ahead\nalignment', ha='center', fontsize=7, color='#777777', style='italic')

# Arrow from integration to modules
ax.annotate('', xy=(9.0, 3.5), xytext=(8.8, 3.5),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Modules (right column)
for i, (mnum, mname, mdesc) in enumerate(modules):
    y = 6.2 - i * 1.35
    rect = mpatches.FancyBboxPatch((9.0, y-0.42), 4.7, 0.84,
        boxstyle='round,pad=0.05', facecolor=MODULE_COLORS[i],
        alpha=0.15, edgecolor=MODULE_COLORS[i], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.3,  y+0.14, mnum,  fontsize=7.5, color=MODULE_COLORS[i], fontweight='bold')
    ax.text(9.3,  y-0.12, mname, fontsize=9,   color='#333333', fontweight='bold')
    ax.text(12.0, y,      mdesc, fontsize=7.5, color='#555555', ha='center', va='center')
    ax.annotate('', xy=(9.0, y), xytext=(8.8, y),
                arrowprops=dict(arrowstyle='->', color=MODULE_COLORS[i], lw=1.2))

# Module flow arrows
for i in range(3):
    y_from = 6.2 - i*1.35 - 0.42
    y_to   = 6.2 - (i+1)*1.35 + 0.42
    ax.annotate('', xy=(11.35, y_to), xytext=(11.35, y_from),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

ax.set_title('TOP Crop Digital Twin — Data Architecture (6 Layers × 4 Modules)',
             fontsize=12, fontweight='bold', pad=12, color='#222222')

plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig01_data_architecture.png')
plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f'  Saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Market Coverage Distribution
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 2: Coverage distribution ...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bins = [0, 10, 25, 50, 75, 90, 100]
labels = ['<10', '10-25', '25-50', '50-75', '75-90', '>90']

for ax, crop in zip(axes, CROPS):
    cov = coverage[crop]['pct_coverage']
    counts, _ = np.histogram(cov, bins=bins)
    bar_colors = ['#FFCCBC','#FFCCBC','#FFE0B2','#FFF9C4','#DCEDC8', '#A5D6A7']
    bars = ax.bar(range(len(labels)), counts, color=bar_colors, edgecolor='white',
                  linewidth=0.8, width=0.7)

    # Highlight ≥90% bar
    bars[-1].set_facecolor(CROP_COLORS[crop])
    bars[-1].set_alpha(0.9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l+'%' for l in labels], fontsize=8)
    ax.set_xlabel('Coverage (% of 418 weeks)', fontsize=9)
    ax.set_ylabel('Number of markets', fontsize=9)
    n_sel = len(selected[crop])
    n_all = len(coverage[crop])
    ax.set_title(f'{crop.capitalize()}\n{n_sel} selected / {n_all} total '
                 f'({n_sel/n_all*100:.0f}%)', fontsize=10, fontweight='bold',
                 color=CROP_COLORS[crop])

    # Annotate counts
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(count), ha='center', va='bottom', fontsize=8)

    ax.axvline(x=4.65, color=CROP_COLORS[crop], linestyle='--', linewidth=1.2,
               label='≥90% threshold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, counts.max() * 1.15)

fig.suptitle('Market Coverage Distribution — Temporal Completeness (2017-2024)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig02_market_coverage_distribution.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Markets by State (selected panel)
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 3: Markets by state ...')

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, crop in zip(axes, CROPS):
    state_counts = (selected[crop].groupby('state')['market']
                    .count().sort_values(ascending=True))
    colors = [CROP_COLORS[crop]] * len(state_counts)
    bars = ax.barh(range(len(state_counts)), state_counts.values,
                   color=colors, alpha=0.8, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(state_counts)))
    ax.set_yticklabels(state_counts.index, fontsize=8)
    ax.set_xlabel('Number of markets (≥90% coverage)', fontsize=9)
    ax.set_title(f'{crop.capitalize()}\n{len(selected[crop])} markets  |  '
                 f'{selected[crop]["state"].nunique()} states',
                 fontsize=10, fontweight='bold', color=CROP_COLORS[crop])
    for bar, val in zip(bars, state_counts.values):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=8)
    ax.set_xlim(0, state_counts.max() * 1.15)

fig.suptitle('Selected Markets by State — ≥90% Temporal Coverage Panel',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig03_markets_by_state.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Coverage Heatmap (top 30 markets per crop)
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 4: Coverage heatmap ...')

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
years = list(range(2017, 2025))

cmap = LinearSegmentedColormap.from_list('cov', ['#FFEBEE','#EF5350','#1B5E20'], N=256)

for ax, crop in zip(axes, CROPS):
    sub = df[df['crop'] == crop]
    top30 = (selected[crop].nlargest(30, 'pct_coverage')['market'].tolist())
    sub30 = sub[sub['market'].isin(top30)]

    sub30 = sub30.copy()
    sub30['year'] = sub30['week_start'].dt.year
    matrix = pd.pivot_table(sub30, values='week_start',
                             index='market', columns='year',
                             aggfunc='count', fill_value=0)
    # % of weeks per year (52 or 53)
    sub_yr = sub.copy(); sub_yr['year'] = sub_yr['week_start'].dt.year
    weeks_per_year = sub_yr.groupby('year')['week_start'].nunique()
    for yr in matrix.columns:
        matrix[yr] = (matrix[yr] / weeks_per_year.get(yr, 52) * 100).clip(0, 100)

    matrix = matrix.reindex(columns=years, fill_value=0)
    matrix = matrix.sort_values(by=years, ascending=False)

    im = ax.imshow(matrix.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, fontsize=8, rotation=45)
    ax.set_yticks(range(len(matrix))); ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(f'{crop.capitalize()} — Top 30 markets\n(% weeks with data per year)',
                 fontsize=9, fontweight='bold', color=CROP_COLORS[crop])
    plt.colorbar(im, ax=ax, shrink=0.6, label='Coverage %')

fig.suptitle('Year-by-Year Data Coverage Heatmap — Top 30 Markets per Crop',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig04_coverage_heatmap.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Panel Timeline (weekly observation count 2017-2024)
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 5: Panel timeline ...')

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

for ax, crop in zip(axes, CROPS):
    # All markets
    all_wk = (df[df['crop']==crop]
               .groupby('week_start')['market'].count()
               .reset_index(name='all_markets'))
    # Selected markets
    sel_wk = (df_filtered[df_filtered['crop']==crop]
               .groupby('week_start')['market'].count()
               .reset_index(name='selected'))

    merged = all_wk.merge(sel_wk, on='week_start', how='left').fillna(0)

    ax.fill_between(merged['week_start'], merged['all_markets'],
                    alpha=0.25, color=CROP_COLORS[crop], label='All markets')
    ax.fill_between(merged['week_start'], merged['selected'],
                    alpha=0.7,  color=CROP_COLORS[crop], label='≥90% coverage')
    ax.set_ylabel('Market-weeks\nper week', fontsize=8)
    ax.set_title(crop.capitalize(), fontsize=10, fontweight='bold',
                 color=CROP_COLORS[crop], loc='left')
    ax.legend(fontsize=8, loc='upper left')

    # Shade COVID period
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-30'),
               alpha=0.12, color='grey', label='COVID lockdown')

axes[-1].set_xlabel('Week', fontsize=9)
fig.suptitle('Weekly Panel Density — All Markets vs ≥90% Coverage Selection (2017-2024)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig05_panel_timeline.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ── Final summary ─────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Script 11 complete. Outputs:')
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('fig0') or 'filtered' in f:
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f'  {f:<45} {sz:>7.1f} KB')
print()

print('Panel statistics (≥90% coverage):')
for crop in CROPS:
    sub = df_filtered[df_filtered['crop']==crop]
    print(f'  {crop:8s}: {sub["market"].nunique():>4} markets  '
          f'{sub["state"].nunique():>3} states  '
          f'{len(sub):>8,} rows')
print(f'  {"TOTAL":8s}: {df_filtered["market"].nunique():>4} markets  '
      f'{len(df_filtered):>8,} rows')
