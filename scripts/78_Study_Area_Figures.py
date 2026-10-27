# -*- coding: utf-8 -*-
"""
Script 78 -- Figures for the manuscript section "Study area and data".
Fig 1  map of markets (by crop) and production zones
Fig 2  weekly number of markets with an observed price, by crop
Fig 3  national median weekly price by crop, 2017-2026
Fig 4  markets per state, by crop
Fig 5  age of the latest data point of each layer, relative to the market panel
State outlines: Survey-of-India-based shapefile (path in TOP_INDIA_STATES_SHP, default in Downloads); the map is skipped if absent.
Outputs: Model_Output/fig_ms_study_area_{1..5}.png
"""
import os, re, io, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
SHP = os.environ.get('TOP_INDIA_STATES_SHP', r'C:\Users\masro\Downloads\India Shapefile With Kashmir\India Shape\india_st')
CROPS = ['tomato', 'onion', 'potato']
COL = {'tomato': '#DC2626', 'onion': '#7C3AED', 'potato': '#B45309'}
INK, MUTED, GRID = '#1F2933', '#52606D', '#E1E0D9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})


def panel():
    p = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv'), parse_dates=['week_start'])
    p['state'] = p['state'].replace({'Keralam': 'Kerala'})   # one label for the state (renamed in 2026)
    return p


def zones():
    src = io.open(os.path.join(BASE, 'scripts', '16_Zone_Assignment.py'), encoding='utf-8').read()
    out = []
    for m in re.finditer(r"'([TOP]\d+_\w+)':\s*\{'crop': '(\w+)',\s*'lat': ([\d.]+),\s*'lon': ([\d.]+)", src):
        out.append(dict(zone=m.group(1), crop=m.group(2), lat=float(m.group(3)), lon=float(m.group(4))))
    return pd.DataFrame(out)


def fig1(p):
    try:
        import shapefile
        r = shapefile.Reader(SHP)
    except Exception as e:
        print('[78] map skipped:', e); return
    za = pd.read_csv(os.path.join(BASE, 'data', 'zone_assignment.csv'))
    za['market_id'] = za['market_id'].astype(float).astype(int)
    panel_ids = {c: set(p[p.crop == c].market_id.dropna().astype(int)) for c in CROPS}
    za = za[za['geocode_level'].isin(['district', 'district_only']) & za['lat'].notna()]
    za = za[[mid in panel_ids[c] for c, mid in zip(za.crop, za.market_id)]]     # current panel markets only
    pts = za.groupby(['crop', 'lat', 'lon']).size().reset_index(name='n')
    Z = zones()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 5.2))
    for ax, crop in zip(axs, CROPS):
        for shp in r.shapes():
            parts = list(shp.parts) + [len(shp.points)]
            for a, b in zip(parts[:-1], parts[1:]):
                xy = np.array(shp.points[a:b]); ax.plot(xy[:, 0], xy[:, 1], color='#B8C0C8', lw=0.5, zorder=1)
        d = pts[pts.crop == crop]
        ax.scatter(d.lon, d.lat, s=8 + 9 * np.sqrt(d.n), color=COL[crop], alpha=0.55, linewidths=0, zorder=3, label='Markets (size = number per district)')
        zz = Z[Z.crop == crop]
        ax.scatter(zz.lon, zz.lat, marker='*', s=150, color='white', edgecolor=INK, linewidths=1.0, zorder=5, label='Production zone (30 km)')
        ax.set_title('%s: %d of %d markets mapped' % (crop.capitalize(), int(za[za.crop == crop].market_id.nunique()), len(panel_ids[crop])), fontsize=11, fontweight='bold', color=INK, loc='left')
        ax.set_xlim(67.5, 98); ax.set_ylim(6, 37.5); ax.set_aspect(1.05); ax.axis('off')
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig_ms_study_area_1.png'), dpi=200, facecolor='white'); plt.close(fig)


def fig2(p):
    o = p[p.imputed == 0].groupby(['crop', 'week_start']).market_id.nunique().unstack(0)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for c in CROPS:
        ax.plot(o.index, o[c], color=COL[c], lw=1.6, label=c.capitalize())
        ax.text(o.index[-1] + pd.Timedelta(days=20), o[c].dropna().iloc[-1] + {'tomato': 30, 'onion': -30, 'potato': 0}[c], c.capitalize(), color=COL[c], va='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Markets with an observed price'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(o.index.min(), o.index.max() + pd.Timedelta(days=110))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_ms_study_area_2.png'), dpi=200, facecolor='white'); plt.close(fig)


def fig3(p):
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True)
    for ax, c in zip(axs, CROPS):
        g = p[(p.crop == c) & (p.imputed == 0)].groupby('week_start').modal_price_weighted
        s = g.median()[g.count() >= (25 if c == 'potato' else 100)]
        ax.plot(s.index, s.values, color=COL[c], lw=1.4)
        ax.fill_between(s.index, g.quantile(.25).reindex(s.index), g.quantile(.75).reindex(s.index), color=COL[c], alpha=0.15, lw=0)
        ax.set_title(c.capitalize(), loc='left', fontsize=10, fontweight='bold', color=INK)
        ax.set_ylabel('Rs per quintal'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    axs[-1].xaxis.set_major_locator(mdates.YearLocator()); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.text(0.01, 0.005, 'Line: median across markets; band: interquartile range across markets. Observed weeks only.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(OUT, 'fig_ms_study_area_3.png'), dpi=200, facecolor='white'); plt.close(fig)


def fig4(p):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4), gridspec_kw={'width_ratios': [1, 1, 0.45]})
    for ax, c in zip(axs, CROPS):
        n = p[p.crop == c].groupby('state').market_id.nunique().sort_values(ascending=False).head(10)[::-1]
        ax.barh(n.index, n.values, color=COL[c], height=0.7)
        for y, v in enumerate(n.values): ax.text(v + n.max() * 0.015, y, str(v), va='center', fontsize=8, color=INK)
        ax.set_title('%s (%d states)' % (c.capitalize(), p[p.crop == c].state.nunique()), loc='left', fontsize=10, fontweight='bold', color=INK)
        ax.set_xlim(0, n.max() * 1.15); ax.xaxis.set_visible(False); ax.spines['bottom'].set_visible(False); ax.tick_params(length=0)
    fig.text(0.01, 0.01, 'Top 10 states by number of markets in the panel (potato: all states).', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, 'fig_ms_study_area_4.png'), dpi=200, facecolor='white'); plt.close(fig)


def fig5():
    ref = pd.Timestamp('2026-10-05')
    L = [('Market prices and arrivals (Agmarknet)', '2026-10-05'), ('Weather and satellite (zone features)', '2026-09-28'),
         ('Drought (VEDAS / IDM, panel)', '2026-08-31'), ('Fuel prices (PPAC)', '2026-08-31'), ('Oceanic Nino Index (NOAA)', '2026-08-31'),
         ('Rural price index (CPI-AL/RL)', '2026-08-31'), ('Fertiliser prices (fert.gov.in)', '2026-07-31'), ('Policy and trade log', '2026-07-30'),
         ('Macro (CMIE: credit, trade, wages, WPI)', '2026-07-01'), ('State agricultural wages (Labour Bureau)', '2026-03-31')]
    d = pd.DataFrame(L, columns=['layer', 'last']); d['last'] = pd.to_datetime(d['last']); d['lag'] = (ref - d['last']).dt.days / 7
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(d.layer, d.lag, color='#2a78d6', height=0.65)
    for y, v in enumerate(d.lag): ax.text(v + 0.5, y, '%.0f wk' % v, va='center', fontsize=8.5, color=INK)
    ax.set_xlabel('Weeks between the latest data point and the market panel (week of 5 Oct 2026)'); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlim(0, d.lag.max() * 1.15); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_ms_study_area_5.png'), dpi=200, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    p = panel()
    fig1(p); fig2(p); fig3(p); fig4(p); fig5()
    print('wrote figures to', OUT)
