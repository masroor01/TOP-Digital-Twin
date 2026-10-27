# -*- coding: utf-8 -*-
"""
Script 78 -- Figures for the manuscript section "Study area and data".
Fig 1  map of markets (by crop) and production zones
Fig 2  weekly number of markets with an observed price, by crop
Fig 3  national median weekly price by crop, 2017-2026
Fig 4  markets per state, by crop
Fig 5  age of the latest data point of each layer, relative to the market panel
Country outline: Survey of India based boundary (Datameet) (path in TOP_INDIA_STATES_SHP, default in Downloads); the map is skipped if absent.
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
OUTLINE = os.environ.get('TOP_INDIA_OUTLINE', r'C:\Users\masro\Downloads\maps-master\Country\india-soi.geojson')   # Datameet, Survey of India based country outline (CC-BY-SA 2.5 / ODbL)
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


GADM2 = os.environ.get('TOP_GADM2_SHP', r'C:\Users\masro\Downloads\gadm41_IND_2')
# Agmarknet district names the geocoder could not match (misspellings, market names used as districts) -> GADM district
FIX = {('Gujarat', 'Banaskanth'): 'Banas Kantha', ('Haryana', 'Jhajar'): 'Jhajjar', ('Himachal Pradesh', 'Sirmore'): 'Sirmaur', ('Kerala', 'Thirssur'): 'Thrissur',
       ('Maharashtra', 'Chattrapati Sambhajinagar'): 'Aurangabad', ('Odisha', 'Balasore'): 'Baleshwar', ('Odisha', 'Mayurbhanja'): 'Mayurbhanj', ('Punjab', 'Ferozpur'): 'Firozpur',
       ('Punjab', 'Tarntaran'): 'Tarn Taran', ('Rajasthan', 'Alwar FV'): 'Alwar', ('Rajasthan', 'Hanumangarh Town APMC'): 'Hanumangarh', ('Tamil Nadu', 'Nagercoil (Kannyiakumari)'): 'Kanniyakumari',
       ('Tamil Nadu', 'Thiruchirappalli'): 'Tiruchirappalli', ('Tamil Nadu', 'Thiruvellore'): 'Thiruvallur', ('Telangana', 'Kukatpally Rythu Bazar APMC'): 'Hyderabad',
       ('Uttar Pradesh', 'Bhadohi(Sant Ravi Nagar)'): 'Sant Ravi Das Nagar', ('Uttar Pradesh', 'Farukhabad'): 'Farrukhabad', ('Uttar Pradesh', 'Kannuj'): 'Kannauj',
       ('Uttar Pradesh', 'Mau(Maunathbhanjan)'): 'Mau', ('Uttar Pradesh', 'Pillibhit'): 'Pilibhit', ('Uttar Pradesh', 'Tamkuhi road APMC'): 'Kushinagar',
       ('Uttar Pradesh', 'Varanasi(F&V) APMC'): 'Varanasi', ('Uttarakhand', 'Dehradoon'): 'Dehradun', ('Uttarakhand', 'Nanital'): 'Nainital', ('Uttarakhand', 'Udhamsinghnagar'): 'Udham Singh Nagar',
       ('West Bengal', 'Purba Bardhaman'): 'Barddhaman', ('West Bengal', 'Sounth 24 Parganas'): 'South 24 Parganas'}


def gadm_centroids():
    import shapefile
    r = shapefile.Reader(GADM2)
    f = [x[0] for x in r.fields][1:]
    out = {}
    for sh, rec in zip(r.shapes(), r.records()):
        d = dict(zip(f, rec))
        parts = list(sh.parts) + [len(sh.points)]
        best = None
        for a, b in zip(parts[:-1], parts[1:]):
            xy = np.array(sh.points[a:b])
            x, y = xy[:, 0], xy[:, 1]
            cr = x[:-1] * y[1:] - x[1:] * y[:-1]
            ar = cr.sum() / 2
            if abs(ar) < 1e-12:
                continue
            cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * ar)
            cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * ar)
            if best is None or abs(ar) > best[0]:
                best = (abs(ar), cy, cx)
        if best:
            out[(d['NAME_1'], d['NAME_2'])] = (best[1], best[2])
    return out


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    a = np.radians
    h = np.sin((a(lat2) - a(lat1)) / 2) ** 2 + np.cos(a(lat1)) * np.cos(a(lat2)) * np.sin((a(lon2) - a(lon1)) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


def locations(p):
    """Coordinates for every market in the current panel (district centroid), nearest production zone and distance.
    Uses the geocoded district centroid in data/zone_assignment.csv where it is district-level, and the GADM district centroid otherwise."""
    pm = p.drop_duplicates(['crop', 'market_id'])[['crop', 'market_id', 'market', 'state', 'district']].copy()
    pm['market_id'] = pm['market_id'].astype(int)
    za = pd.read_csv(os.path.join(BASE, 'data', 'zone_assignment.csv'))
    za['market_id'] = za['market_id'].astype(float).astype(int)
    za = za[za['geocode_level'].isin(['district', 'district_only']) & za['lat'].notna()][['crop', 'market_id', 'lat', 'lon']]
    m = pm.merge(za, on=['crop', 'market_id'], how='left')
    m['source'] = np.where(m['lat'].notna(), 'geocoded district', '')
    G = gadm_centroids()
    st_alias = {'Keralam': 'Kerala'}
    for i in m.index[m['lat'].isna()]:
        s_ = st_alias.get(m.at[i, 'state'], m.at[i, 'state'])
        d_ = str(m.at[i, 'district'])
        key = (s_, FIX.get((s_, d_), d_))
        if key not in G:
            key = next((k for k in G if k[0] == s_ and k[1].lower() == d_.lower()), None)
        if key in G:
            m.at[i, 'lat'], m.at[i, 'lon'] = G[key]
            m.at[i, 'source'] = 'GADM district centroid'
    Z = zones()
    zl, dl = [], []
    for i, r_ in m.iterrows():
        zz = Z[Z.crop == r_.crop]
        if pd.isna(r_.lat):
            zl.append(None); dl.append(np.nan); continue
        d = hav(r_.lat, r_.lon, zz.lat.values, zz.lon.values)
        j = int(np.argmin(d))
        zl.append(zz.zone.values[j]); dl.append(d[j])
    m['zone'] = zl
    m['dist_km'] = dl
    m.loc[m['dist_km'] > 500, 'zone'] = None
    m.to_csv(os.path.join(OUT, 'table_market_locations.csv'), index=False)
    return m


def scalebar(ax, km=500, x0=68.2, y0=7.0):
    deg = km / (111.32 * np.cos(np.radians(y0)))
    ax.plot([x0, x0 + deg], [y0, y0], color=INK, lw=2.5, solid_capstyle='butt', zorder=6)
    ax.plot([x0, x0], [y0 - 0.25, y0 + 0.25], color=INK, lw=1.2, zorder=6)
    ax.plot([x0 + deg, x0 + deg], [y0 - 0.25, y0 + 0.25], color=INK, lw=1.2, zorder=6)
    ax.text(x0 + deg / 2, y0 + 0.55, '%d km' % km, ha='center', va='bottom', fontsize=8, color=INK)


def north(ax, x=96.0, y=11.0):
    ax.annotate('', xy=(x, y + 3.2), xytext=(x, y), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4), zorder=6)
    ax.text(x, y + 3.7, 'N', ha='center', va='bottom', fontsize=9, fontweight='bold', color=INK)


def fig1(p):
    try:
        import json
        gj = json.load(io.open(OUTLINE, encoding='utf-8'))
    except Exception as e:
        print('[78] map skipped:', e); return
    rings = []
    for ft in gj['features']:
        g = ft['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for poly in polys:
            for ring in poly:
                rings.append(np.array(ring))
    za = locations(p)
    za = za[za['lat'].notna()]
    pts = za.groupby(['crop', 'lat', 'lon']).size().reset_index(name='n')
    Z = zones()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 5.2))
    for ax, crop in zip(axs, CROPS):
        for xy in rings:
            ax.plot(xy[:, 0], xy[:, 1], color='#8A96A3', lw=0.7, zorder=1)
        d = pts[pts.crop == crop]
        ax.scatter(d.lon, d.lat, s=8 + 9 * np.sqrt(d.n), color=COL[crop], alpha=0.55, linewidths=0, zorder=3, label='Markets (size = number per district)')
        zz = Z[Z.crop == crop]
        ax.scatter(zz.lon, zz.lat, marker='*', s=150, color='white', edgecolor=INK, linewidths=1.0, zorder=5, label='Production zone (30 km)')
        ax.set_title('%s: %d markets' % (crop.capitalize(), int(za[za.crop == crop].market_id.nunique())), fontsize=11, fontweight='bold', color=INK, loc='left')
        ax.set_xlim(67.5, 98); ax.set_ylim(6, 37.5); ax.set_aspect(1.05); ax.axis('off'); scalebar(ax); north(ax)
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
