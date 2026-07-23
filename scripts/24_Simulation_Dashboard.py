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
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

BASE = r'C:\Users\masro\Documents\TOP_Digital_Twin'
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]

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
    reference = pd.read_csv(os.path.join(MODEL_DIR, 'reference_rows.csv'), parse_dates=['week_start'])
    return feature_columns, feature_ranges, reference


if not os.path.exists(MODEL_DIR):
    st.error(f'No production models found at {MODEL_DIR}. '
             f'Run `python scripts/23_Train_Production_Models.py` first.')
    st.stop()

models = load_models()
feature_columns, feature_ranges, reference = load_metadata()


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

crop_markets = reference[reference['crop'] == crop].sort_values('market')
market = st.sidebar.selectbox('Market', crop_markets['market'].unique())

base_row_df = crop_markets[crop_markets['market'] == market]
if base_row_df.empty:
    st.error('No baseline data for this crop/market combination.')
    st.stop()
base_row = base_row_df.iloc[0].to_dict()

st.sidebar.markdown(f"**As-of week:** {pd.Timestamp(base_row['week_start']).date()}")
st.sidebar.markdown(f"**State:** {base_row.get('state', 'N/A')}")

horizon = st.sidebar.select_slider('Forecast horizon (weeks ahead)', options=HORIZONS, value=4)


# ─────────────────────────────────────────────────────────────────────────────
# WHAT-IF CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown('---')
st.sidebar.subheader('Policy scenario')

scenario = dict(base_row)  # start from the real, current feature vector

export_banned = st.sidebar.checkbox('Export ban in effect',
                                     value=bool(base_row.get('export_banned', 0)))
scenario['export_banned'] = int(export_banned)

if 'mep_usd_per_tonne' in feature_ranges:
    r = feature_ranges['mep_usd_per_tonne']
    scenario['mep_usd_per_tonne'] = st.sidebar.slider(
        'Minimum Export Price (USD/tonne)', 0.0, max(r['max'], 900.0),
        float(base_row.get('mep_usd_per_tonne', 0) or 0), step=10.0)

if 'export_duty_pct' in feature_ranges:
    scenario['export_duty_pct'] = st.sidebar.slider(
        'Export duty (%)', 0.0, 50.0, float(base_row.get('export_duty_pct', 0) or 0), step=1.0)

market_intervention = st.sidebar.checkbox(
    'Government market intervention this week (buffer procurement/release, subsidised sale)',
    value=bool(base_row.get('market_intervention_flag', 0)))
scenario['market_intervention_flag'] = int(market_intervention)

def safe_slider(col, label):
    """Slider with a fallback for degenerate (min==max) or missing ranges —
    a real issue found in testing: some features are near-constant for a
    given market's history, which crashes st.slider(min==max)."""
    if col not in feature_ranges or col not in base_row or pd.isna(base_row[col]):
        return None
    r = feature_ranges[col]
    lo, hi, val = float(r['min']), float(r['max']), float(base_row[col])
    val = min(max(val, lo), hi)  # clamp in case of float edge cases
    if hi <= lo:
        st.sidebar.caption(f'{label}: {val:g} (fixed — no variation observed)')
        return val
    return st.sidebar.slider(label, lo, hi, val)


st.sidebar.markdown('---')
st.sidebar.subheader('Climate scenario')

for col, label in [('era5_tmax', 'Max temperature (°C)'),
                    ('chirps_rain_mm', 'Weekly rainfall (mm)'),
                    ('s2_ndvi', 'Vegetation index (NDVI)')]:
    val = safe_slider(col, label)
    if val is not None:
        scenario[col] = val

st.sidebar.markdown('---')
st.sidebar.subheader('Macro / logistics scenario')

for col, label in [('diesel_4city_rs_litre', 'Diesel price (Rs/litre)'),
                    ('repo_rate_pct', 'RBI repo rate (%)'),
                    ('usdinr_monthly_avg', 'USD/INR exchange rate')]:
    val = safe_slider(col, label)
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

baseline_pred = predict(crop, horizon, base_row)
scenario_pred = predict(crop, horizon, scenario)
delta = scenario_pred - baseline_pred
delta_pct = 100 * delta / baseline_pred if baseline_pred else 0

col1, col2, col3 = st.columns(3)
col1.metric(f'Baseline prediction (h={horizon}w)', f'Rs {baseline_pred:,.0f}/quintal')
col2.metric(f'Scenario prediction (h={horizon}w)', f'Rs {scenario_pred:,.0f}/quintal',
            delta=f'{delta:+,.0f} ({delta_pct:+.1f}%)')
col3.metric('Last observed price',
            f"Rs {np.expm1(base_row.get('log_price', 0)):,.0f}/quintal" if pd.notna(base_row.get('log_price')) else 'N/A')

st.markdown('---')

# All-horizon comparison chart
st.subheader('Scenario impact across all forecast horizons')
rows = []
for h in HORIZONS:
    rows.append({'horizon': f'{h}w', 'variant': 'Baseline (current conditions)',
                 'price': predict(crop, h, base_row)})
    rows.append({'horizon': f'{h}w', 'variant': 'Scenario',
                 'price': predict(crop, h, scenario)})
chart_df = pd.DataFrame(rows)

fig = go.Figure()
for variant, color in [('Baseline (current conditions)', '#adb5bd'), ('Scenario', '#e64980')]:
    sub = chart_df[chart_df['variant'] == variant]
    fig.add_trace(go.Bar(x=sub['horizon'], y=sub['price'], name=variant,
                          marker_color=color))
fig.update_layout(barmode='group', xaxis_title='Forecast horizon',
                   yaxis_title='Predicted price (Rs/quintal)',
                   legend=dict(orientation='h', y=1.1), height=420)
st.plotly_chart(fig, use_container_width=True)

def _differs(a, b):
    """NaN-safe inequality — plain != treats NaN as never equal to itself,
    which flagged every NaN-valued feature as 'changed' even when untouched."""
    if pd.isna(a) and pd.isna(b):
        return False
    return a != b


with st.expander('Full scenario feature vector (debug view)'):
    diff_cols = [c for c in scenario if _differs(scenario.get(c), base_row.get(c))]
    if diff_cols:
        st.write('Changed from baseline:', {c: (base_row.get(c), scenario.get(c)) for c in diff_cols})
    else:
        st.write('No changes from baseline yet — adjust the sidebar controls.')
