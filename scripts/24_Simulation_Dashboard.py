# -*- coding: utf-8 -*-
"""
Script 24 — TOP Digital Twin: Interactive Simulation Dashboard
====================================================================
A "what-if" scenario simulator built on the Script 23 production models
(M6: price + arrivals + macro + climate + satellite + infrastructure +
policy). Pick a crop, market, and forecast horizon; adjust policy,
climate, and macro inputs away from their current real-world values; see
how the model's predicted price responds.

This is a decision-support / illustration tool, not a new statistical
result — it visualizes what the already-validated M6 model has learned,
it doesn't add new evidence. Treat its outputs as "what the model implies
under this scenario," not as a new forecast to be taken at face value.

Requires: scripts/23_Train_Production_Models.py has been run first
(needs Model_Output/production_models/*.joblib + metadata files).

Run: streamlit run scripts/24_Simulation_Dashboard.py
"""

import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import anthropic
from scipy.interpolate import PchipInterpolator

# Portable path: resolved relative to this script's own location (scripts/..)
# rather than hardcoded to a specific machine — required for deployment to
# Streamlit Community Cloud, which clones the repo to its own path.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
DOW_PATTERN_FILE = os.path.join(BASE, 'Model_Output', 'table_dow_pattern.csv')

# Season definitions, identical to the month lists Script 23 uses to build
# the season_* model features (scripts/23_Train_Production_Models.py) --
# kept in sync so the chart's seasonal shading matches what the model
# actually learned from, not a separately-invented calendar.
SEASON_MONTHS = {
    'tomato': {'peak_arrival': [11, 12, 1, 2], 'lean': [5, 6, 7], 'kharif': [8, 9, 10]},
    'onion':  {'rabi_arrival': [2, 3, 4, 5],   'lean': [9, 10, 11], 'kharif': [8, 9]},
    'potato': {'harvest': [2, 3, 4], 'storage': [5, 6, 7, 8, 9], 'lean': [10, 11]},
}
SEASON_LABEL = {
    'peak_arrival': 'Peak arrival', 'lean': 'Lean season', 'kharif': 'Kharif',
    'rabi_arrival': 'Rabi arrival', 'harvest': 'Harvest', 'storage': 'Storage',
}
SEASON_COLOR = {
    'peak_arrival': 'rgba(30,92,55,0.10)', 'lean': 'rgba(168,50,50,0.10)',
    'kharif': 'rgba(150,102,11,0.10)', 'rabi_arrival': 'rgba(30,92,55,0.10)',
    'harvest': 'rgba(30,92,55,0.10)', 'storage': 'rgba(76,134,168,0.10)',
}

def season_for(crop, dt):
    """Season key for a given date, or None if it falls in an unlabeled
    transition month (not every month has a named season for every crop)."""
    for season, months in SEASON_MONTHS[crop].items():
        if dt.month in months:
            return season
    return None
CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]

# Plain-language explanation for every simulatable input: `help` drives the
# native Streamlit hover tooltip on each control; `mechanism` is the
# economic-reasoning sentence used in the scenario interpretation section
# below the chart. Grounded in the same market-structure reasoning used to
# interpret the ablation study (see Script 15/18b discussion): export
# controls matter most for onion, climate/logistics matter more at longer
# horizons, storage-buffered potato is largely insensitive to any of this.
FEATURE_INFO = {
    'export_banned': dict(
        label='Export ban in effect',
        help='When ON, the government prohibits exporting this crop abroad '
             '(e.g. India\'s Dec 2023-May 2024 onion export ban). The strongest '
             'policy lever — it forces all supply to stay in the domestic market.',
        mechanism='An export ban keeps supply that would have gone abroad inside '
                   'domestic markets, which tends to push domestic prices {dir}. '
                   'Historically this has mainly mattered for onion — tomato and '
                   'potato have had no significant export-ban history to learn from.'),
    'mep_usd_per_tonne': dict(
        label='Minimum Export Price (USD/tonne)',
        help='The floor price (USD/tonne) below which exporters may not sell '
             'abroad. A softer alternative to an outright ban — raising it prices '
             'exports out of the international market without banning them.',
        mechanism='A higher MEP discourages exports by making them less price-'
                   'competitive abroad, which — like a ban — tends to keep more '
                   'supply at home and push domestic prices {dir}.'),
    'export_duty_pct': dict(
        label='Export duty (%)',
        help='A tax (% of value) on exported goods, e.g. the 40% onion export '
             'duty imposed in Aug 2023. Raises the cost of exporting, discouraging '
             'outbound shipments similarly to a higher MEP.',
        mechanism='A higher export duty raises the cost of shipping abroad, '
                   'discouraging exports and tending to push domestic prices {dir}.'),
    'market_intervention_flag': dict(
        label='Government market intervention this week',
        help='Marks a reported NAFED/NCCF buffer-stock procurement or release, '
             'or a subsidised retail sale, in this exact week. These directly add '
             'or remove supply to manage price spikes or crashes.',
        mechanism='Interventions are usually a REACTION to price stress (they '
                   'happen because prices are already high or low), so this flag '
                   'can reflect "crisis conditions" as much as it drives price '
                   'itself — read its effect with that in mind.'),
    'era5_tmax': dict(
        label='Max temperature (°C)',
        help='Weekly maximum temperature in the growing region (ERA5 climate '
             'reanalysis). Extreme heat can stress crops and reduce yields, '
             'tightening supply in the weeks ahead.',
        mechanism='Higher extreme temperature is associated with crop stress and '
                   'reduced expected supply, which tends to push prices {dir}.'),
    'chirps_rain_mm': dict(
        label='Weekly rainfall (mm)',
        help='Satellite-estimated rainfall in the growing region (CHIRPS). '
             'Effect is two-sided: moderate rain supports growth, but excess '
             'rain can flood fields, damage crops, and disrupt harvest/transport.',
        mechanism='Rainfall\'s effect is non-monotonic — moderate increases can '
                   'support supply (pushing prices down), but large increases can '
                   'damage crops or disrupt logistics (pushing prices up). The '
                   'direction shown here is what the model learned for this '
                   'specific change, not a fixed rule.'),
    's2_ndvi': dict(
        label='Vegetation index (NDVI)',
        help='Crop health/density from Sentinel-2 satellite imagery (roughly '
             '0-1). Higher values generally mean healthier, denser vegetation — '
             'a proxy for expected yield.',
        mechanism='Higher NDVI (healthier growing conditions) generally signals '
                   'more supply ahead, which tends to push prices {dir}.'),
    'diesel_4city_rs_litre': dict(
        label='Diesel price (Rs/litre)',
        help='Average diesel price across 4 major Indian cities (PPAC data). '
             'Diesel is the dominant fuel for transporting produce from farms to '
             'markets, so it is a direct proxy for logistics cost.',
        mechanism='Higher diesel prices raise the cost of transporting produce '
                   'to market, which tends to push wholesale prices {dir}.'),
    'repo_rate_pct': dict(
        label='RBI repo rate (%)',
        help='The Reserve Bank of India\'s policy interest rate — the cost at '
             'which banks borrow. Affects the cost of credit for traders who '
             'borrow to finance stored inventory, particularly cold-stored potato.',
        mechanism='A higher repo rate raises the cost of holding inventory on '
                   'credit, which can discourage stockpiling and tends to push '
                   'prices {dir} — most relevant for storage-buffered crops.'),
    'usdinr_monthly_avg': dict(
        label='USD/INR exchange rate',
        help='The rupee-per-dollar exchange rate. A weaker rupee (higher number) '
             'makes Indian exports cheaper for foreign buyers in dollar terms.',
        mechanism='A weaker rupee makes exports more attractive, pulling supply '
                   'toward export markets and away from domestic ones, which '
                   'tends to push domestic prices {dir}.'),
}

st.set_page_config(page_title='TOP Digital Twin — Scenario Simulator', layout='wide')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (cached — these are static files, no need to reload per interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for crop in CROPS:
        for h in HORIZONS:
            path = os.path.join(MODEL_DIR, f'{crop}_{h}w.joblib')
            if os.path.exists(path):
                models[(crop, h)] = joblib.load(path)
    return models


@st.cache_data
def load_metadata():
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), encoding='utf-8') as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_ranges.json'), encoding='utf-8') as f:
        feature_ranges = json.load(f)
    uncertainty_path = os.path.join(MODEL_DIR, 'model_uncertainty.json')
    uncertainty = {}
    if os.path.exists(uncertainty_path):
        with open(uncertainty_path, encoding='utf-8') as f:
            uncertainty = json.load(f)
    reference = pd.read_csv(os.path.join(MODEL_DIR, 'reference_rows.csv'), parse_dates=['week_start', 'last_observed_date'])
    history = pd.read_csv(os.path.join(MODEL_DIR, 'price_history.csv'), parse_dates=['week_start'])
    staleness_path = os.path.join(MODEL_DIR, 'macro_climate_staleness.json')
    staleness = {}
    if os.path.exists(staleness_path):
        with open(staleness_path, encoding='utf-8') as f:
            staleness = json.load(f)
    # Daily residual std per crop, for the daily-view uncertainty band (Script
    # 26). Day-of-week factors themselves are NOT loaded/used here -- a
    # backtest found them negligible (<1%) and marginally worse than flat
    # interpolation, so only the overall historical day-to-day noise level
    # is kept.
    daily_noise = {}
    if os.path.exists(DOW_PATTERN_FILE):
        dow_df = pd.read_csv(DOW_PATTERN_FILE)
        daily_noise = dow_df.groupby('crop')['factor_std'].mean().to_dict()
    return feature_columns, feature_ranges, uncertainty, reference, history, staleness, daily_noise


if not os.path.exists(MODEL_DIR):
    st.error(f'No production models found at {MODEL_DIR}. '
             f'Run `python scripts/23_Train_Production_Models.py` first.')
    st.stop()

models = load_models()
feature_columns, feature_ranges, uncertainty, reference, history, staleness, daily_noise = load_metadata()


def predict(crop, h, feature_row):
    """feature_row: dict of {column: value}. Returns predicted price (Rs/quintal)."""
    cols = feature_columns[f'{crop}_{h}w']
    X = pd.DataFrame([{c: feature_row.get(c, 0) for c in cols}])
    log_pred = models[(crop, h)].predict(X)[0]
    return float(np.expm1(log_pred))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — crop / market / horizon selection
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title('TOP Digital Twin')
st.sidebar.caption('Price scenario simulator (M6 production models)')

with st.sidebar.expander('Data & coverage'):
    _counts = reference.groupby('crop')['market'].nunique()
    st.markdown(
        'Source: Agmarknet (price/arrivals), CMIE/RBI/PPAC (macro), '
        'GEE Sentinel-2/MODIS/ERA5/CHIRPS (satellite/climate), 2017-2026.\n\n'
        'Markets are filtered to those with **>=70% real (non-imputed) '
        'weekly coverage** over their own history (added 2026-07-27 — '
        'earlier versions of this dashboard had no such filter and '
        'included many thin-data markets).\n\n'
        f'Markets in current models: **{_counts.get("tomato", 0)} tomato, '
        f'{_counts.get("onion", 0)} onion, {_counts.get("potato", 0)} potato**.'
    )

crop = st.sidebar.selectbox('Crop', CROPS, format_func=str.capitalize)

crop_ref = reference[reference['crop'] == crop]
states = sorted(crop_ref['state'].dropna().unique())
state_sel = st.sidebar.selectbox('State / UT', states)

crop_markets = crop_ref[crop_ref['state'] == state_sel].sort_values('market')
market = st.sidebar.selectbox('Market', crop_markets['market'].unique())

base_row_df = crop_markets[crop_markets['market'] == market]
if base_row_df.empty:
    st.error('No baseline data for this crop/market combination.')
    st.stop()
base_row = base_row_df.iloc[0].to_dict()
as_of = pd.Timestamp(base_row['week_start'])
today = pd.Timestamp(datetime.date.today())
data_weeks_stale = int((today - as_of).days // 7)

st.sidebar.markdown(f"**Today:** {today.strftime('%d %b %Y')}")
st.sidebar.markdown(f"**Market data last updated:** {as_of.date()}"
                     + (f'  ({data_weeks_stale}w ago)' if data_weeks_stale > 0 else ''))

horizon = st.sidebar.select_slider('Forecast horizon (weeks ahead)', options=HORIZONS, value=4)
st.sidebar.caption(f'≈ {horizon * 7} days ahead of the market\'s last known data point.')


# ─────────────────────────────────────────────────────────────────────────────
# WHAT-IF CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown('---')
st.sidebar.subheader('Policy scenario')

scenario = dict(base_row)  # start from the real, current feature vector

def _num(col, default=0.0):
    """base_row.get(col) safely as a float, never NaN — `NaN or default`
    doesn't work because NaN is truthy in Python, so a missing/NaN policy
    value was silently passed straight into st.slider as NaN, rendering a
    degenerate slider with no visible track (found 2026-07-27)."""
    v = base_row.get(col)
    return default if v is None or pd.isna(v) else float(v)


def _policy_staleness_caption(col):
    stale = staleness.get(crop, {}).get(col)
    if stale:
        st.sidebar.caption(
            f'📌 Starting value carried forward from {stale["as_of"]} '
            f'({stale["weeks_stale"]}w stale) — this source hasn\'t published '
            f'more recent data yet.')


_ebi = FEATURE_INFO['export_banned']
export_banned = st.sidebar.checkbox(_ebi['label'], value=bool(_num('export_banned')),
                                     help=_ebi['help'])
scenario['export_banned'] = int(export_banned)
_policy_staleness_caption('export_banned')

if 'mep_usd_per_tonne' in feature_ranges:
    r = feature_ranges['mep_usd_per_tonne']
    _mi = FEATURE_INFO['mep_usd_per_tonne']
    scenario['mep_usd_per_tonne'] = st.sidebar.slider(
        _mi['label'], 0.0, max(r['max'], 900.0),
        _num('mep_usd_per_tonne'), step=10.0, help=_mi['help'])
    _policy_staleness_caption('mep_usd_per_tonne')

if 'export_duty_pct' in feature_ranges:
    _di = FEATURE_INFO['export_duty_pct']
    scenario['export_duty_pct'] = st.sidebar.slider(
        _di['label'], 0.0, 50.0, _num('export_duty_pct'),
        step=1.0, help=_di['help'])
    _policy_staleness_caption('export_duty_pct')

_mii = FEATURE_INFO['market_intervention_flag']
market_intervention = st.sidebar.checkbox(
    _mii['label'], value=bool(_num('market_intervention_flag')), help=_mii['help'])
scenario['market_intervention_flag'] = int(market_intervention)
_policy_staleness_caption('market_intervention_flag')

def safe_slider(col, extend_pct=0.0):
    """Slider with a fallback for degenerate (min==max) or missing ranges —
    a real issue found in testing: some features are near-constant for a
    given market's history, which crashes st.slider(min==max).

    extend_pct: widen the slider beyond the historically-observed min/max
    by this fraction. Macro variables like diesel price and USD/INR trend
    in one direction over 2017-2026, so "current value" often sits AT the
    historical max — leaving zero room to simulate a hike (flagged by a
    reviewer). Widening lets you explore beyond what's been observed, but
    LightGBM (tree-based) cannot truly extrapolate past its training
    range — it just repeats its most extreme leaf's prediction — so any
    value outside the original observed range is flagged with a warning
    rather than presented as equally reliable.
    """
    info = FEATURE_INFO[col]
    label = info['label']
    if col not in feature_ranges or col not in base_row or pd.isna(base_row[col]):
        return None
    r = feature_ranges[col]
    obs_lo, obs_hi, val = float(r['min']), float(r['max']), float(base_row[col])
    span = obs_hi - obs_lo
    lo, hi = obs_lo - span * extend_pct, obs_hi + span * extend_pct
    val = min(max(val, lo), hi)  # clamp in case of float edge cases
    if obs_hi <= obs_lo:
        st.sidebar.caption(f'{label}: {val:g} (fixed — no variation observed)')
        return val
    result = st.sidebar.slider(label, lo, hi, val, help=info['help'])
    if result < obs_lo or result > obs_hi:
        st.sidebar.caption(
            f'⚠️ {result:g} is outside the observed 2017-2026 range '
            f'({obs_lo:g}-{obs_hi:g}) — the model has never seen values here '
            f'and cannot reliably extrapolate; treat this result as speculative.')
    stale = staleness.get(crop, {}).get(col)
    if stale:
        st.sidebar.caption(
            f'📌 Starting value carried forward from {stale["as_of"]} '
            f'({stale["weeks_stale"]}w stale) — this source hasn\'t published '
            f'more recent data yet.')
    return result


st.sidebar.markdown('---')
st.sidebar.subheader('Climate scenario')

for col in ['era5_tmax', 'chirps_rain_mm', 's2_ndvi']:
    val = safe_slider(col)
    if val is not None:
        scenario[col] = val

st.sidebar.markdown('---')
st.sidebar.subheader('Macro / logistics scenario')

for col in ['diesel_4city_rs_litre', 'repo_rate_pct', 'usdinr_monthly_avg']:
    # Widened 20% beyond the observed range — these trend monotonically,
    # so "current" often sits at the historical max (e.g. USD/INR was
    # found to be exactly capped at its max with zero headroom).
    val = safe_slider(col, extend_pct=0.20)
    if val is not None:
        scenario[col] = val

reset = st.sidebar.button('Reset to current real-world values')
if reset:
    scenario = dict(base_row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.title(f'{crop.capitalize()} price scenario — {market}')
st.caption('Predictions are what the M6 model implies under the selected scenario, '
           'not a new statistically-validated forecast — see the ablation study '
           '(Script 15) and Diebold-Mariano tests (Scripts 18/18b) for validated results.')

# "Today" (live, real calendar date) vs. the market's actual last-data date
# are two different things — forecasts stay anchored to the latter (the
# model has no live feed), but that gap needs to be visible, not implied
# away, especially when it's large.
st.caption(f'📅 Today: **{today.strftime("%d %b %Y")}**  ·  '
           f'Market data last updated: **{as_of.date()}**')
if data_weeks_stale >= 8:
    st.warning(
        f'⚠️ This market\'s underlying data is {data_weeks_stale} weeks behind today. '
        f'Forecasts below are calculated {horizon} weeks ahead of that last known data '
        f'point ({(as_of + pd.Timedelta(weeks=horizon)).date()}), not from today\'s actual '
        f'conditions — there is no live data feed behind this dashboard.'
    )

# Data-sufficiency flag: warn when this market's recent history is mostly
# imputed (no real trades) rather than presenting every market's prediction
# with equal implied confidence — a reviewer correctly flagged this gap.
pct_imputed = base_row.get('pct_imputed_last_52w')
if pd.notna(pct_imputed) and pct_imputed >= 50:
    st.warning(
        f'⚠️ Data quality: {pct_imputed:.0f}% of this market\'s last 52 weeks have no '
        f'recorded trade (imputed/estimated). Predictions for thin-data markets like this '
        f'are less reliable than the validated error rates below suggest.'
    )
elif pd.notna(pct_imputed) and pct_imputed >= 20:
    st.info(f'ℹ️ Data quality: {pct_imputed:.0f}% of this market\'s last 52 weeks were imputed.')

# ─────────────────────────────────────────────────────────────────────────────
# PRICE FORECAST TICKER — the baseline model's real trained horizons, with
# their actual calendar dates. The model only has 4 trained horizons
# (1w/4w/13w/26w), not daily granularity, so this shows exactly those —
# no interpolated/invented daily points presented as real forecasts.
# Always shows the BASELINE (current real-world inputs), independent of
# the scenario sliders below, so it reads as "the model's live forecast."
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Price forecast ticker')
st.caption(
    'The baseline model\'s prediction at each of its trained horizons, dated to the '
    'real calendar week — not affected by the what-if sliders below. Forecasts start '
    'from the market\'s last known data point, not from today (see above). Season tags '
    'use the same calendar the model itself was trained on (Script 23).'
)
ticker_cols = st.columns(len(HORIZONS))
ticker_points = [(as_of, base_row.get('log_price'))]  # (date, log-price) incl. the starting point
for tcol, h in zip(ticker_cols, HORIZONS):
    fdate = as_of + pd.Timedelta(weeks=h)
    fprice = predict(crop, h, base_row)
    ticker_points.append((fdate, np.log1p(fprice)))
    herr = uncertainty.get(f'{crop}_{h}w', {})
    herr_note = (f' Typical error: ±Rs {herr["rmse"]:,.0f} ({herr["mape"]:.0f}% MAPE), '
                 f'from validated backtesting.') if herr.get('rmse') else ''
    season = season_for(crop, fdate)
    tcol.metric(
        f'h={h}w  ·  {fdate.strftime("%d %b %Y")}',
        f'Rs {fprice:,.0f}',
        help=f'Baseline forecast for {fdate.date()} ({h} weeks ahead of the market\'s '
             f'last known data point, {as_of.date()}).' + herr_note
    )
    if season:
        tcol.caption(f'🌾 {SEASON_LABEL[season]}')

# ─────────────────────────────────────────────────────────────────────────────
# DAILY PRICE VIEW (Script 26) — smooth interpolation through the ticker's own
# 4 validated points, shaded by the same season calendar as above, plus an
# uncertainty band from real historical daily volatility. Click-gated (not
# shown by default) since it's a secondary, derived view -- see Script 26's
# docstring for the full method and the day-of-week correction that was
# tested and dropped as net-negative.
# ─────────────────────────────────────────────────────────────────────────────
if pd.notna(ticker_points[0][1]) and crop in daily_noise:
    show_daily = st.button('📅 Show daily price forecast', key='show_daily_btn')
    if show_daily:
        st.session_state['daily_view_open'] = True
    if st.session_state.get('daily_view_open'):
        st.caption(
            'Smoothed daily curve built from the weekly forecast points above, widened by this '
            'crop\'s typical day-to-day price movement — a guide for planning, not an independent '
            'daily forecast. Shown from today onward only.'
        )

        pts_dates, pts_logp = zip(*ticker_points)
        pts_num = [(d - as_of).days for d in pts_dates]
        pchip = PchipInterpolator(pts_num, pts_logp)
        # Fit the curve over the FULL as_of-anchored range (needed for a
        # correctly-shaped interpolation through all validated points), but
        # only DISPLAY from today onward -- as_of (the market's last known
        # data point) can sit weeks behind today's real date (see the
        # "Market data last updated" staleness note above), and showing
        # that backfilled gap as part of a "daily forecast" would be
        # misleading; only genuinely forward-looking days are shown.
        daily_offsets_full = np.arange(0, HORIZONS[-1] * 7 + 1)
        daily_dates_full = [as_of + pd.Timedelta(days=int(o)) for o in daily_offsets_full]
        smooth_trend_full = np.expm1(pchip(daily_offsets_full))
        display_from = max(today, as_of)
        keep = [i for i, d in enumerate(daily_dates_full) if d >= display_from]
        daily_dates = [daily_dates_full[i] for i in keep]
        smooth_trend = smooth_trend_full[keep]
        band = smooth_trend * daily_noise[crop]

        if not daily_dates:
            st.info('Today is beyond this market\'s 26-week forecast horizon — no forward-looking '
                    'days to show. Try a market with more recent data.')
        else:
            # ── Running marquee of the daily values (slowed down for readability) ──
            marquee_items = ' &nbsp; • &nbsp; '.join(
                f'{d.strftime("%d %b")}: <b>Rs&nbsp;{p:,.0f}</b>'
                for d, p in zip(daily_dates, smooth_trend)
            )
            marquee_duration = max(90, len(daily_dates) * 1.5)  # scales with content length
            st.markdown(f"""
            <style>
            .daily-marquee-wrap {{
                overflow: hidden; white-space: nowrap; box-sizing: border-box;
                border: 1px solid rgba(128,128,128,0.3); border-radius: 8px;
                padding: 10px 0; margin-bottom: 12px; background: rgba(30,92,55,0.06);
            }}
            .daily-marquee-track {{
                display: inline-block; padding-left: 100%;
                animation: daily-marquee {marquee_duration}s linear infinite;
                font-size: 14px;
            }}
            .daily-marquee-wrap:hover .daily-marquee-track {{ animation-play-state: paused; }}
            @keyframes daily-marquee {{
                0%   {{ transform: translate(0, 0); }}
                100% {{ transform: translate(-100%, 0); }}
            }}
            </style>
            <div class="daily-marquee-wrap">
              <div class="daily-marquee-track">{marquee_items}</div>
            </div>
            """, unsafe_allow_html=True)

            fig_daily = go.Figure()

            # Season shading -- one vrect per contiguous same-season run of days
            seasons_by_day = [season_for(crop, d) for d in daily_dates]
            run_start = 0
            for i in range(1, len(daily_dates) + 1):
                if i == len(daily_dates) or seasons_by_day[i] != seasons_by_day[run_start]:
                    s = seasons_by_day[run_start]
                    if s:
                        fig_daily.add_vrect(
                            x0=daily_dates[run_start], x1=daily_dates[i - 1] + pd.Timedelta(days=1),
                            fillcolor=SEASON_COLOR[s], line_width=0, layer='below'
                        )
                        mid = daily_dates[run_start + (i - 1 - run_start) // 2]
                        fig_daily.add_annotation(x=mid, y=1.0, yref='paper', yanchor='bottom',
                                                  text=SEASON_LABEL[s], showarrow=False,
                                                  font=dict(size=9, color='#666'))
                    run_start = i

            fig_daily.add_trace(go.Scatter(
                x=daily_dates + daily_dates[::-1],
                y=list(smooth_trend + band) + list((smooth_trend - band)[::-1]),
                fill='toself', fillcolor='rgba(30,92,55,0.13)', line=dict(width=0),
                name='Typical day-to-day movement (±1 std)', hoverinfo='skip'
            ))
            fig_daily.add_trace(go.Scatter(
                x=daily_dates, y=smooth_trend, mode='lines',
                line=dict(color='#1E5C37', width=2), name='Smooth daily trend'
            ))
            visible_pts = [(d, p) for d, p in ticker_points if d >= display_from]
            if visible_pts:
                fig_daily.add_trace(go.Scatter(
                    x=[d for d, _ in visible_pts], y=[np.expm1(p) for _, p in visible_pts],
                    mode='markers', marker=dict(size=9, color='#1E5C37'), name='Weekly forecast point'
                ))
            fig_daily.update_layout(xaxis_title='Date', yaxis_title='Price (Rs/quintal)',
                                     height=420, hovermode='x unified',
                                     margin=dict(t=50),
                                     legend=dict(orientation='h', y=1.16))
            st.plotly_chart(fig_daily, use_container_width=True)

baseline_pred = predict(crop, horizon, base_row)
scenario_pred = predict(crop, horizon, scenario)
delta = scenario_pred - baseline_pred
delta_pct = 100 * delta / baseline_pred if baseline_pred else 0

# Validated uncertainty for this crop x horizon, from Script 15's actual
# out-of-sample CV results — not a statistical prediction interval, but a
# real, defensible sense of typical error magnitude (better than a bare
# point estimate with no uncertainty shown at all).
err = uncertainty.get(f'{crop}_{horizon}w', {})
rmse, mape = err.get('rmse'), err.get('mape')
uncertainty_note = (f' (typical error: ±Rs {rmse:,.0f}, ~{mape:.0f}% MAPE, from '
                     f'validated backtesting)') if rmse else ''

col1, col2, col3, col4 = st.columns(4)
col1.metric(f'Baseline prediction (h={horizon}w)', f'Rs {baseline_pred:,.0f}/quintal',
            help='What the model predicts under the CURRENT real-world feature values '
                 '(no sidebar changes applied) — this is the model\'s unmodified forecast.'
                 + uncertainty_note)
col2.metric(f'Scenario prediction (h={horizon}w)', f'Rs {scenario_pred:,.0f}/quintal',
            delta=f'{delta:+,.0f} ({delta_pct:+.1f}%)',
            help='What the model predicts after applying every change you made in the '
                 'sidebar. The delta (green/red) shows the net effect of ALL your changes '
                 'combined, not any single one — see "Scenario interpretation" below for '
                 'a feature-by-feature breakdown.' + uncertainty_note)

# "Accuracy" here is 100% - MAPE from Script 15's validated out-of-sample
# backtesting for this exact crop x horizon — a widely-used but approximate
# convention, not a formal confidence level. Shown alongside the raw MAPE/
# RMSE (in the tooltip) rather than replacing them.
if mape is not None:
    accuracy_pct = max(0.0, 100 - mape)
    col4.metric(f'Model accuracy (h={horizon}w)', f'~{accuracy_pct:.0f}%',
                help=f'100% − MAPE ({mape:.0f}%) from validated backtesting (Script 15). '
                     f'An approximate, commonly-used convention — not a formal statistical '
                     f'confidence level. See the ablation study for the full accuracy picture '
                     f'across crops/horizons.')
else:
    col4.metric(f'Model accuracy (h={horizon}w)', 'N/A',
                help='No validated backtesting result available for this crop/horizon.')

# "Last observed price" previously showed whatever the latest panel row
# was, even if that row was imputed (58.6% of markets' latest row is —
# flagged by a reviewer). Show the TRUE last genuinely-observed trade,
# and note staleness if it differs from the model's as-of week.
last_obs_price = base_row.get('last_observed_price')
last_obs_date = base_row.get('last_observed_date')
if pd.notna(last_obs_price):
    is_stale = pd.notna(last_obs_date) and pd.Timestamp(last_obs_date) < as_of
    label3 = 'Last observed (real trade) price'
    value3 = f'Rs {last_obs_price:,.0f}/quintal'
    help3 = ('The most recent week with an ACTUAL recorded trade for this market — '
             'not an imputed/estimated value.')
    if is_stale:
        weeks_stale = (as_of - pd.Timestamp(last_obs_date)).days // 7
        value3 += f'  (as of {pd.Timestamp(last_obs_date).date()}, {weeks_stale}w ago)'
        help3 += (f' This market has had no real trade recorded in {weeks_stale} weeks — '
                  f'the value shown as "current" elsewhere on this page is imputed/estimated.')
else:
    label3, value3, help3 = 'Last observed price', 'N/A', 'No non-imputed trade found for this market.'
col3.metric(label3, value3, help=help3)

st.markdown('---')

# Historical price + calendar-dated forecast chart
st.subheader('Price history and forecast')
st.caption(
    f'Forecasts only reach as far as the model was trained to predict — '
    f'{max(HORIZONS)} weeks past the last observed week. This is a genuine limit, '
    f'not a display cutoff: extending it requires training models at longer '
    f'horizons, not just changing this chart.'
)

mkt_history = (history[(history['crop'] == crop) & (history['market'] == market)]
               .sort_values('week_start'))

fig = go.Figure()

if not mkt_history.empty:
    fig.add_trace(go.Scatter(
        x=mkt_history['week_start'], y=mkt_history['modal_price_weighted'],
        mode='lines', name='Actual price (history)',
        line=dict(color='#495057', width=2)))

# Forecast traces: connect from the last actual point through each horizon,
# at the real calendar date it corresponds to (as_of + h weeks)
last_actual_date = mkt_history['week_start'].max() if not mkt_history.empty else as_of
last_actual_price = (mkt_history[mkt_history['week_start'] == last_actual_date]
                      ['modal_price_weighted'].iloc[0]) if not mkt_history.empty else baseline_pred

selected_date = as_of + pd.Timedelta(weeks=horizon)
selected_points_x, selected_points_y = [], []

# Validated uncertainty band around the baseline forecast (RMSE per
# horizon from Script 15's actual out-of-sample results) — a reviewer
# correctly flagged that point predictions were shown with no sense of
# reliability, especially for thin-data markets.
band_rmse = [uncertainty.get(f'{crop}_{h}w', {}).get('rmse', 0) for h in HORIZONS]
if any(band_rmse):
    band_dates = [last_actual_date] + [as_of + pd.Timedelta(weeks=h) for h in HORIZONS]
    band_upper = [last_actual_price] + [predict(crop, h, base_row) + r for h, r in zip(HORIZONS, band_rmse)]
    band_lower = [last_actual_price] + [predict(crop, h, base_row) - r for h, r in zip(HORIZONS, band_rmse)]
    fig.add_trace(go.Scatter(
        x=band_dates + band_dates[::-1], y=band_upper + band_lower[::-1],
        fill='toself', fillcolor='rgba(173,181,189,0.20)', line=dict(width=0),
        name='±RMSE (validated, baseline)', hoverinfo='skip'))

for label, feature_row, color, dash in [
        ('Baseline forecast', base_row, '#adb5bd', 'dot'),
        ('Scenario forecast', scenario, '#e64980', 'dash')]:
    fc_dates = [last_actual_date] + [as_of + pd.Timedelta(weeks=h) for h in HORIZONS]
    fc_prices = [last_actual_price] + [predict(crop, h, feature_row) for h in HORIZONS]
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_prices, mode='lines+markers', name=label,
        line=dict(color=color, width=2, dash=dash), marker=dict(size=7)))
    # Track the point matching the sidebar's selected horizon, so moving
    # that slider visibly changes the chart too (previously the chart always
    # plotted all 4 horizons regardless of the slider — a real UX bug a
    # reviewer flagged as "the horizon slider doesn't do anything").
    selected_points_x.append(selected_date)
    selected_points_y.append(fc_prices[HORIZONS.index(horizon) + 1])

fig.add_trace(go.Scatter(
    x=selected_points_x, y=selected_points_y, mode='markers', name=f'Selected horizon (h={horizon}w)',
    marker=dict(size=16, symbol='star', color='#ffd43b', line=dict(width=1.5, color='#495057')),
    showlegend=True))

fig.add_vline(x=as_of, line_dash='dot', line_color='#888',
              annotation_text='data as-of', annotation_position='top')
fig.add_vline(x=selected_date, line_dash='dash', line_color='#ffd43b',
              annotation_text=f'selected: h={horizon}w (~{horizon * 7}d)', annotation_position='top')
fig.update_layout(xaxis_title='Date', yaxis_title='Price (Rs/quintal)',
                   legend=dict(orientation='h', y=1.15), height=460,
                   hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET COMPARISON — top markets by price and arrivals, for this crop
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('---')
TOP_N = 15
st.subheader(f'Market comparison — top {TOP_N} {crop} markets')
st.caption(
    'Ranked by each market\'s most recent REAL observed values (not imputed, not a model '
    'prediction). Price and arrivals may come from different weeks per market since data '
    'recency varies market to market. The currently selected market is highlighted in pink '
    'if it appears in the top ranking.'
)

price_top = (crop_ref.dropna(subset=['last_observed_price'])
             .nlargest(TOP_N, 'last_observed_price')
             .sort_values('last_observed_price'))
arr_top = (crop_ref.dropna(subset=['log_arr'])
           .assign(arrivals_tonnes=lambda d: np.expm1(d['log_arr']))
           .nlargest(TOP_N, 'arrivals_tonnes')
           .sort_values('arrivals_tonnes'))

mkt_col1, mkt_col2 = st.columns(2)
with mkt_col1:
    if price_top.empty:
        st.info('No real (non-imputed) price data available to rank markets.')
    else:
        fig_price = go.Figure(go.Bar(
            x=price_top['last_observed_price'], y=price_top['market'], orientation='h',
            marker_color=['#e64980' if m == market else '#495057' for m in price_top['market']]))
        fig_price.update_layout(title='By last observed price (Rs/quintal)', height=440,
                                 margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_price, use_container_width=True)
with mkt_col2:
    if arr_top.empty:
        st.info('No arrivals data available to rank markets.')
    else:
        fig_arr = go.Figure(go.Bar(
            x=arr_top['arrivals_tonnes'], y=arr_top['market'], orientation='h',
            marker_color=['#e64980' if m == market else '#495057' for m in arr_top['market']]))
        fig_arr.update_layout(title='By arrivals, latest week (tonnes)', height=440,
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_arr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRICE-TREND COMPARISON — independent Crop / State-UT / Market picker,
# deliberately separate from the sidebar's main scenario-simulator
# selectors so you can compare markets in a different crop/state without
# disturbing the main scenario above. User-driven multi-select rather
# than an auto-populated top-N, per explicit user decision.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Price-trend comparison')
st.caption(
    'Pick a crop, one or more states, and up to 15 markets to compare their real observed '
    'weekly price history (up to the last 2 years) on the same chart — markets can be mixed '
    'across states (or all left within one), independent of the main scenario controls in '
    'the sidebar and the rankings above.'
)

trend_col1, trend_col2 = st.columns(2)
with trend_col1:
    trend_crop = st.selectbox('Crop', CROPS, format_func=str.capitalize,
                               index=CROPS.index(crop), key='trend_crop')
with trend_col2:
    trend_crop_ref = reference[reference['crop'] == trend_crop]
    trend_states = sorted(trend_crop_ref['state'].dropna().unique())
    trend_state_default = [state_sel] if state_sel in trend_states else trend_states[:1]
    trend_states_sel = st.multiselect(
        'State / UT', trend_states, default=trend_state_default, key='trend_states',
        help='Pick one state to compare markets within it, or several to compare across states.'
    )

trend_crop_history = history[history['crop'] == trend_crop]
market_state = trend_crop_ref.drop_duplicates('market').set_index('market')['state']
trend_selectable_markets = set(trend_crop_ref[trend_crop_ref['state'].isin(trend_states_sel)]['market'])
history_markets = sorted(m for m in trend_crop_history['market'].unique() if m in trend_selectable_markets)
default_trend_markets = [market] if (trend_crop == crop and market in history_markets) else []
selected_trend_markets = st.multiselect(
    'Markets to compare', options=history_markets, default=default_trend_markets,
    max_selections=15, format_func=lambda m: f'{m} — {market_state.get(m, "?")}',
    help='Select up to 15 markets from the state(s) chosen above — defaults to the sidebar\'s '
         'currently selected market when crop and state both match. Mixing states is fine; '
         'each market keeps its own line and legend label.'
)

if not selected_trend_markets:
    st.info('Select at least one market above to see its price trend.')
else:
    fig_trend = go.Figure()
    no_data_markets = []
    for m in selected_trend_markets:
        mdata = trend_crop_history[trend_crop_history['market'] == m].sort_values('week_start')
        if mdata['modal_price_weighted'].notna().sum() == 0:
            no_data_markets.append(m)
            continue
        legend_name = m if len(trend_states_sel) <= 1 else f'{m} ({market_state.get(m, "?")})'
        fig_trend.add_trace(go.Scatter(
            x=mdata['week_start'], y=mdata['modal_price_weighted'],
            mode='lines', name=legend_name,
            line=dict(width=3 if (trend_crop == crop and m == market) else 1.5)
        ))
    if no_data_markets:
        st.warning(
            f'⚠️ No real observed trades in this window for: {", ".join(no_data_markets)} '
            f'— entirely imputed/missing, so left off the chart rather than plotting a '
            f'flat/empty line.'
        )
    if len(fig_trend.data) == 0:
        st.info('None of the selected markets have real price history to plot in this window.')
    else:
        fig_trend.update_layout(xaxis_title='Date', yaxis_title='Price (Rs/quintal)',
                                 height=460, hovermode='x unified',
                                 legend=dict(orientation='h', y=1.15))
        st.plotly_chart(fig_trend, use_container_width=True)


def _differs(a, b):
    """NaN-safe inequality — plain != treats NaN as never equal to itself,
    which flagged every NaN-valued feature as 'changed' even when untouched."""
    if pd.isna(a) and pd.isna(b):
        return False
    return a != b


diff_cols = [c for c in scenario if _differs(scenario.get(c), base_row.get(c))]


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO INTERPRETATION — feature-by-feature breakdown of the price change
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Scenario interpretation')

if not diff_cols:
    st.info('No changes from baseline yet. Adjust a control in the sidebar to see '
            'how the model\'s prediction responds and why.')
else:
    st.caption(
        'For each input you changed, this isolates ITS OWN effect on the h='
        f'{horizon}w prediction — changing only that one input from the baseline, '
        'holding everything else fixed. Because the model is non-linear, these '
        'isolated effects don\'t always add up exactly to the combined scenario '
        'delta shown above (features can interact) — treat this as "what each '
        'change contributes on its own," not a precise accounting.'
    )

    isolated_effects = []
    for col in diff_cols:
        single = dict(base_row)
        single[col] = scenario[col]
        single_pred = predict(crop, horizon, single)
        eff = single_pred - baseline_pred
        isolated_effects.append((col, eff))

    sum_isolated = sum(e for _, e in isolated_effects)
    interaction_gap = delta - sum_isolated

    for col, eff in sorted(isolated_effects, key=lambda x: -abs(x[1])):
        info = FEATURE_INFO.get(col)
        label = info['label'] if info else col
        before, after = base_row.get(col), scenario.get(col)
        direction = 'higher' if eff > 0 else ('lower' if eff < 0 else 'about the same')
        arrow = '⬆️' if eff > 0 else ('⬇️' if eff < 0 else '➡️')

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f'**{label}**  ·  {before:g} → {after:g}')
                if info:
                    st.caption(info['mechanism'].format(dir=direction))
            with c2:
                st.markdown(f'### {arrow} {eff:+,.0f}')
                st.caption(f'Rs/quintal, isolated')

    if abs(interaction_gap) > 1:
        st.caption(
            f'Sum of isolated effects: {sum_isolated:+,.0f} Rs — actual combined '
            f'effect: {delta:+,.0f} Rs (difference of {interaction_gap:+,.0f} Rs is '
            f'due to interactions between your changes, not measurement error).'
        )

st.markdown('---')

# ─────────────────────────────────────────────────────────────────────────────
# AI POLICY RECOMMENDATION — button-triggered (not auto-run on every rerun,
# to keep API calls bounded on a public dashboard), and grounded strictly in
# the isolated effects / delta / uncertainty already computed above rather
# than letting the model free-associate about numbers it wasn't given.
# ─────────────────────────────────────────────────────────────────────────────
st.subheader('AI policy recommendation')


def _api_key():
    try:
        return st.secrets.get('ANTHROPIC_API_KEY')
    except Exception:
        return None


if not diff_cols:
    st.caption('Adjust a scenario control above, then generate an AI policy read on it.')
elif not _api_key():
    st.info(
        'AI recommendations need an `ANTHROPIC_API_KEY` configured for this deployment '
        '(Streamlit Cloud: App settings -> Secrets). See README.md for setup.'
    )
else:
    scenario_key = hashlib.md5(
        f"{crop}|{market}|{horizon}|{sorted((c, scenario.get(c)) for c in diff_cols)}".encode()
    ).hexdigest()
    reco_state_key = f'ai_reco_{scenario_key}'

    if st.button('Generate AI policy recommendation'):
        changes_text = '\n'.join(
            f"- {FEATURE_INFO.get(c, {}).get('label', c)}: "
            f"{base_row.get(c):g} -> {scenario.get(c):g} "
            f"(isolated effect: {eff:+,.0f} Rs/quintal)"
            for c, eff in sorted(isolated_effects, key=lambda x: -abs(x[1]))
        )
        prompt = (
            'You are a policy-analysis assistant embedded in an agricultural price '
            'forecasting dashboard for Indian APMC markets (Tomato/Onion/Potato, HADP-04, '
            'SKUAST-K). A user ran a what-if scenario. Ground your answer STRICTLY in the '
            'numbers given below — do not invent statistics, events, or data you were not given.\n\n'
            f'Crop: {crop.capitalize()}\nMarket: {market}\nForecast horizon: {horizon} weeks ahead\n'
            f'Baseline prediction: Rs {baseline_pred:,.0f}/quintal\n'
            f'Scenario prediction: Rs {scenario_pred:,.0f}/quintal '
            f'({delta_pct:+.1f}%, {delta:+,.0f} Rs/quintal)\n'
            f"Model's typical error at this horizon: "
            f"{f'±Rs {rmse:,.0f} ({mape:.0f}% MAPE)' if rmse else 'not available'}\n\n"
            f'Changes made in this scenario, with the isolated effect of each:\n{changes_text}\n\n'
            'Write ONE paragraph (120-160 words) of plain-language policy commentary for an '
            'agricultural-market analyst. Cover: (1) what this price move would mean for '
            'farmers vs consumers, (2) which lever is doing most of the work and whether that '
            'matches known market structure for this crop, (3) one caveat about relying on this '
            'scenario (it is a what-if from a single model, not a validated forecast; thin-data '
            'markets and feature interactions add uncertainty). Plain prose only — no bullet '
            'points, headers, or markdown.'
        )
        try:
            client = anthropic.Anthropic(api_key=_api_key())
            resp = client.messages.create(
                model='claude-haiku-4-5-20251001', max_tokens=400,
                messages=[{'role': 'user', 'content': prompt}])
            st.session_state[reco_state_key] = resp.content[0].text
        except Exception as e:
            st.session_state[reco_state_key] = None
            st.error(f'AI recommendation failed: {e}')

    cached = st.session_state.get(reco_state_key)
    if cached:
        st.markdown(f'> {cached}')
        st.caption(
            'AI-generated commentary grounded in this scenario\'s model outputs above — '
            'not an independently validated forecast or official policy advice.'
        )

st.markdown('---')

with st.expander('Full scenario feature vector (debug view)'):
    if diff_cols:
        st.write('Changed from baseline:', {c: (base_row.get(c), scenario.get(c)) for c in diff_cols})
    else:
        st.write('No changes from baseline yet — adjust the sidebar controls.')
