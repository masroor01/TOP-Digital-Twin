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

# Portable path: resolved relative to this script's own location (scripts/..)
# rather than hardcoded to a specific machine — required for deployment to
# Streamlit Community Cloud, which clones the repo to its own path.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
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
    return feature_columns, feature_ranges, uncertainty, reference, history


if not os.path.exists(MODEL_DIR):
    st.error(f'No production models found at {MODEL_DIR}. '
             f'Run `python scripts/23_Train_Production_Models.py` first.')
    st.stop()

models = load_models()
feature_columns, feature_ranges, uncertainty, reference, history = load_metadata()


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

_ebi = FEATURE_INFO['export_banned']
export_banned = st.sidebar.checkbox(_ebi['label'], value=bool(base_row.get('export_banned', 0)),
                                     help=_ebi['help'])
scenario['export_banned'] = int(export_banned)

if 'mep_usd_per_tonne' in feature_ranges:
    r = feature_ranges['mep_usd_per_tonne']
    _mi = FEATURE_INFO['mep_usd_per_tonne']
    scenario['mep_usd_per_tonne'] = st.sidebar.slider(
        _mi['label'], 0.0, max(r['max'], 900.0),
        float(base_row.get('mep_usd_per_tonne', 0) or 0), step=10.0, help=_mi['help'])

if 'export_duty_pct' in feature_ranges:
    _di = FEATURE_INFO['export_duty_pct']
    scenario['export_duty_pct'] = st.sidebar.slider(
        _di['label'], 0.0, 50.0, float(base_row.get('export_duty_pct', 0) or 0),
        step=1.0, help=_di['help'])

_mii = FEATURE_INFO['market_intervention_flag']
market_intervention = st.sidebar.checkbox(
    _mii['label'], value=bool(base_row.get('market_intervention_flag', 0)), help=_mii['help'])
scenario['market_intervention_flag'] = int(market_intervention)

def safe_slider(col, extend_pct=0.0):
    """Slider with a fallback for degenerate (min==max) or missing ranges —
    a real issue found in testing: some features are near-constant for a
    given market's history, which crashes st.slider(min==max).

    extend_pct: widen the slider beyond the historically-observed min/max
    by this fraction. Macro variables like diesel price and USD/INR trend
    in one direction over 2017-2025, so "current value" often sits AT the
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
            f'⚠️ {result:g} is outside the observed 2017-2025 range '
            f'({obs_lo:g}-{obs_hi:g}) — the model has never seen values here '
            f'and cannot reliably extrapolate; treat this result as speculative.')
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
