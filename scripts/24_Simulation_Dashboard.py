# -*- coding: utf-8 -*-
"""
Script 24 — TOP Digital Twin: Interactive Simulation Dashboard
====================================================================
React / Next.js & Tailwind/Shadcn SaaS Architecture for Streamlit
Decision-support scenario simulator powered by M6 Production Models
(Price + Arrivals + Macro + Climate + Satellite + Infrastructure + Policy).

Maintains 100% of underlying models, causal constraints, and data feeds.

Run:
  cd C:\\Users\\masro\\Documents\\TOP_Digital_Twin
  python -m streamlit run scripts/24_Simulation_Dashboard.py
"""

import os
import json
import hashlib
import datetime
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import anthropic
from scipy.interpolate import PchipInterpolator

# Portable base path
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
DOW_PATTERN_FILE = os.path.join(BASE, 'Model_Output', 'table_dow_pattern.csv')

# Crop Seasonal Calendar
SEASON_MONTHS = {
    'tomato': {'peak_arrival': [11, 12, 1, 2], 'lean': [5, 6, 7], 'kharif': [8, 9, 10]},
    'onion':  {'rabi_arrival': [2, 3, 4, 5],   'lean': [9, 10, 11], 'kharif': [8, 9]},
    'potato': {'harvest': [2, 3, 4], 'storage': [5, 6, 7, 8, 9], 'lean': [10, 11]},
}
SEASON_LABEL = {
    'peak_arrival': 'Peak Arrival', 'lean': 'Lean Season', 'kharif': 'Kharif Season',
    'rabi_arrival': 'Rabi Arrival', 'harvest': 'Harvest Season', 'storage': 'Storage Period',
}
SEASON_COLOR = {
    'peak_arrival': 'rgba(16, 185, 129, 0.14)', 'lean': 'rgba(239, 68, 68, 0.14)',
    'kharif': 'rgba(245, 158, 11, 0.14)', 'rabi_arrival': 'rgba(16, 185, 129, 0.14)',
    'harvest': 'rgba(16, 185, 129, 0.14)', 'storage': 'rgba(59, 130, 246, 0.14)',
}

def season_for(crop, dt):
    for season, months in SEASON_MONTHS[crop].items():
        if dt.month in months:
            return season
    return None

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]

# Feature Info & Economic Mechanism Meta
FEATURE_INFO = {
    'export_banned': dict(
        label='Export Ban in Effect',
        help='When ON, the government prohibits exporting this crop abroad '
             '(e.g. India\'s Dec 2023-May 2024 onion export ban). The strongest '
             'policy lever — it forces all supply to stay in the domestic market.',
        mechanism='An export ban keeps supply that would have gone abroad inside '
                   'domestic markets, which tends to push domestic prices {dir}. '
                   'Historically this has mainly mattered for onion — tomato and '
                   'potato have had no significant export-ban history to learn from.'),
    'mep_usd_per_tonne': dict(
        label='Minimum Export Price (USD/t)',
        help='The floor price (USD/tonne) below which exporters may not sell '
             'abroad. A softer alternative to an outright ban — raising it prices '
             'exports out of the international market without banning them.',
        mechanism='A higher MEP discourages exports by making them less price-'
                   'competitive abroad, which — like a ban — tends to keep more '
                   'supply at home and push domestic prices {dir}.'),
    'export_duty_pct': dict(
        label='Export Duty (%)',
        help='A tax (% of value) on exported goods, e.g. the 40% onion export '
             'duty imposed in Aug 2023. Raises the cost of exporting, discouraging '
             'outbound shipments similarly to a higher MEP.',
        mechanism='A higher export duty raises the cost of shipping abroad, '
                   'discouraging exports and tending to push domestic prices {dir}.'),
    'market_intervention_flag': dict(
        label='Market Intervention This Week',
        help='Marks a reported NAFED/NCCF buffer-stock procurement or release, '
             'or a subsidised retail sale, in this exact week. These directly add '
             'or remove supply to manage price spikes or crashes.',
        mechanism='Interventions are usually a REACTION to price stress (they '
                   'happen because prices are already high or low), so this flag '
                   'can reflect "crisis conditions" as much as it drives price '
                   'itself — read its effect with that in mind.'),
    'era5_tmax': dict(
        label='Max Temperature (°C)',
        help='Weekly maximum temperature in the growing region (ERA5 climate '
             'reanalysis). Extreme heat can stress crops and reduce yields, '
             'tightening supply in the weeks ahead.',
        mechanism='Higher extreme temperature is associated with crop stress and '
                   'reduced expected supply, which tends to push prices {dir}.'),
    'chirps_rain_mm': dict(
        label='Weekly Rainfall (mm)',
        help='Satellite-estimated rainfall in the growing region (CHIRPS). '
             'Effect is two-sided: moderate rain supports growth, but excess '
             'rain can flood fields, damage crops, and disrupt harvest/transport.',
        mechanism='Rainfall\'s effect is non-monotonic — moderate increases can '
                   'support supply (pushing prices down), but large increases can '
                   'damage crops or disrupt logistics (pushing prices up). The '
                   'direction shown here is what the model learned for this '
                   'specific change, not a fixed rule.'),
    's2_ndvi': dict(
        label='Vegetation Index (NDVI)',
        help='Crop health/density from Sentinel-2 satellite imagery (roughly '
             '0-1). Higher values generally mean healthier, denser vegetation — '
             'a proxy for expected yield.',
        mechanism='Higher NDVI (healthier growing conditions) generally signals '
                   'more supply ahead, which tends to push prices {dir}.'),
    'diesel_4city_rs_litre': dict(
        label='Diesel Price (Rs/Litre)',
        help='Average diesel price across 4 major Indian cities (PPAC data). '
             'Diesel is the dominant fuel for transporting produce from farms to '
             'markets, so it is a direct proxy for logistics cost.',
        mechanism='Higher diesel prices raise the cost of transporting produce '
                   'to market, which tends to push wholesale prices {dir}.'),
    'repo_rate_pct': dict(
        label='RBI Repo Rate (%)',
        help='The Reserve Bank of India\'s policy interest rate — the cost at '
             'which banks borrow. Affects the cost of credit for traders who '
             'borrow to finance stored inventory, particularly cold-stored potato.',
        mechanism='A higher repo rate raises the cost of holding inventory on '
                   'credit, which can discourage stockpiling and tends to push '
                   'prices {dir} — most relevant for storage-buffered crops.'),
    'usdinr_monthly_avg': dict(
        label='USD/INR Exchange Rate',
        help='The rupee-per-dollar exchange rate. A weaker rupee (higher number) '
             'makes Indian exports cheaper for foreign buyers in dollar terms.',
        mechanism='A weaker rupee makes exports more attractive, pulling supply '
                   'toward export markets and away from domestic ones, which '
                   'tends to push domestic prices {dir}.'),
    'operation_greens_active': dict(
        label='Operation Greens Active',
        help='Marks whether the Centre\'s Operation Greens (TOP — Tomato, Onion, '
             'Potato) price-stabilisation scheme, including its transport/storage '
             'subsidy, was active that week for this crop.',
        mechanism='Operation Greens subsidises storage and interstate transport '
                   'to smooth out price swings, so an active scheme tends to '
                   'push prices {dir} toward normal levels rather than in one '
                   'fixed direction.'),

    # ── Climate (M3) ──────────────────────────────────────────────────────
    'era5_tmin': dict(
        label='Min Temperature (°C)',
        help='Weekly minimum temperature in the growing region (ERA5). Cold '
             'snaps can damage tender crops (tomato especially) and delay growth.',
        mechanism='Lower minimum temperature is associated with cold-stress risk '
                   'to the crop, which tends to push prices {dir} via reduced '
                   'expected supply.'),
    'era5_tmean': dict(
        label='Mean Temperature (°C)',
        help='Weekly average temperature in the growing region (ERA5) — the '
             'general thermal backdrop for the growing season, distinct from '
             'the daily extremes (max/min).',
        mechanism='Average temperature away from the crop\'s optimal range '
                   'tends to push prices {dir} via reduced expected supply.'),
    'era5_dtr': dict(
        label='Diurnal Temp. Range (°C)',
        help='The gap between daily max and min temperature (ERA5). A wide '
             'swing can stress crops even when the average temperature looks fine.',
        mechanism='A wider day-night temperature swing is associated with '
                   'additional crop stress, which tends to push prices {dir}.'),
    'era5_heat_35': dict(
        label='Days ≥35°C (weekly count)',
        help='Number of days in the week the growing region hit 35°C or hotter '
             '(ERA5) — a standard heat-stress threshold for vegetable crops.',
        mechanism='More extreme-heat days signal crop stress and reduced '
                   'expected supply, which tends to push prices {dir}.'),
    'era5_heat_38': dict(
        label='Days ≥38°C (weekly count)',
        help='Number of days in the week the growing region hit 38°C or hotter '
             '(ERA5) — a more severe heat-stress threshold than the 35°C count.',
        mechanism='Severe-heat days are a stronger stress signal than the 35°C '
                   'count, tending to push prices {dir} via reduced supply.'),
    'chirps_rain_max': dict(
        label='Peak Daily Rainfall (mm)',
        help='The single wettest day\'s rainfall within the week (CHIRPS) — '
             'captures a flooding/waterlogging risk that a weekly total can hide.',
        mechanism='A very wet single day raises flood/waterlogging risk even if '
                   'the week\'s total rain looks moderate, which tends to push '
                   'prices {dir}.'),
    'chirps_excess': dict(
        label='Excess Rainfall Index',
        help='A derived measure (CHIRPS) of how far weekly rainfall exceeds a '
             'normal/expected level for the region and season.',
        mechanism='Rainfall well above the seasonal norm signals flood/logistics '
                   'disruption risk, which tends to push prices {dir}.'),

    # ── Satellite (M4) ────────────────────────────────────────────────────
    's2_valid_frac': dict(
        label='Sentinel-2 Cloud-Free Fraction',
        help='Fraction of the growing region with a usable (cloud-free) '
             'Sentinel-2 image that week (0-1). Low values mean the NDVI/EVI '
             'readings that week are based on thin coverage — a data-quality '
             'signal, not an agronomic one.',
        mechanism='This is a coverage/data-quality signal rather than a supply '
                   'driver; the model may still have learned indirect '
                   'associations with it, shown as {dir} here.'),
    's2_ndvi_anom': dict(
        label='NDVI Anomaly',
        help='How far this week\'s Sentinel-2 vegetation index is from the '
             'region\'s normal level for this time of year — isolates an '
             'unusual season from a normal one better than raw NDVI alone.',
        mechanism='A negative anomaly (worse than normal for the season) '
                   'signals below-normal expected supply, which tends to push '
                   'prices {dir}.'),
    'modis_ndvi': dict(
        label='MODIS Vegetation Index',
        help='Crop health/density from MODIS satellite imagery — a second, '
             'coarser-resolution vegetation signal alongside Sentinel-2\'s NDVI, '
             'with more frequent (near-daily) revisits.',
        mechanism='Higher MODIS NDVI (healthier growing conditions) generally '
                   'signals more supply ahead, which tends to push prices {dir}.'),
    'modis_evi': dict(
        label='MODIS Enhanced Veg. Index',
        help='A vegetation index (MODIS) that corrects for canopy background '
             'and atmospheric effects better than plain NDVI, especially in '
             'denser vegetation.',
        mechanism='Higher EVI (healthier, denser vegetation) generally signals '
                   'more supply ahead, which tends to push prices {dir}.'),
    'modis_lst_mean': dict(
        label='MODIS Land Surface Temp. (mean)',
        help='Average land surface temperature from MODIS thermal imagery — '
             'a satellite-based temperature signal distinct from ERA5\'s '
             'ground-station-based reanalysis.',
        mechanism='Higher land surface temperature is associated with crop '
                   'heat stress, tending to push prices {dir} via reduced '
                   'expected supply.'),
    'modis_lst_max': dict(
        label='MODIS Land Surface Temp. (max)',
        help='Peak land surface temperature from MODIS that week — captures '
             'the hottest surface reading, not just the average.',
        mechanism='A higher peak surface temperature signals acute heat-stress '
                   'risk, tending to push prices {dir}.'),
    'modis_lst_frac35': dict(
        label='MODIS Land ≥35°C Fraction',
        help='Fraction of the growing region (0-1) whose MODIS-measured land '
             'surface hit 35°C or hotter that week.',
        mechanism='A larger heat-affected area signals broader crop stress, '
                   'tending to push prices {dir}.'),

    # ── Macro (M2) ────────────────────────────────────────────────────────
    'agri_wages_rs_day': dict(
        label='National Agri. Daily Wage (Rs)',
        help='CMIE\'s all-India average agricultural daily wage — a national '
             'labour-cost benchmark distinct from the state-specific wage '
             'figures below.',
        mechanism='Higher farm labour costs raise the cost of producing and '
                   'harvesting the crop, which tends to push prices {dir}.'),
    'bank_credit_agri_cr': dict(
        label='Bank Credit to Agriculture (Rs Cr)',
        help='Outstanding scheduled-bank credit to the agriculture sector '
             '(CMIE, Rs crore) — a proxy for how much financing is available '
             'for farm inputs, storage, and trade.',
        mechanism='More credit available can support both planting/storage '
                   'decisions in ways that push prices either direction — read '
                   'the {dir} shown here as what the model learned for this '
                   'specific change, not a fixed rule.'),
    'crude_oil_usd_bbl': dict(
        label='Crude Oil (USD/barrel)',
        help='International crude oil price — the upstream driver behind '
             'diesel and LPG prices, which affect transport and cold-chain costs.',
        mechanism='Higher crude oil prices raise transport and cold-storage '
                   'energy costs, tending to push prices {dir}.'),
    'diesel_delhi_per_L': dict(
        label='Diesel Price — Delhi (Rs/L)',
        help='Diesel price in Delhi specifically (PPAC), alongside the '
             '4-city average — captures local pump-price variation.',
        mechanism='Higher diesel prices raise transport costs to market, '
                   'tending to push prices {dir}.'),
    'export_veg_usd_mn': dict(
        label='Vegetable Exports (USD mn)',
        help='India\'s total monthly vegetable export value (USD million) — '
             'a demand-pull signal: more exports mean less supply staying '
             'in the domestic market.',
        mechanism='Higher export volumes pull supply away from domestic '
                   'markets, tending to push domestic prices {dir}.'),
    'import_veg_usd_mn': dict(
        label='Vegetable Imports (USD mn)',
        help='India\'s total monthly vegetable import value (USD million) — '
             'imported supply competing with domestic produce.',
        mechanism='Higher imports add to domestic supply, tending to push '
                   'prices {dir}.'),
    'iip_food_proc': dict(
        label='Food Processing IIP',
        help='Index of Industrial Production for food processing (CMIE) — '
             'a proxy for processing-sector demand (e.g. tomato going into '
             'paste/ketchup) pulling on the same raw supply as fresh markets.',
        mechanism='Higher processing-sector activity competes with fresh '
                   'markets for the same raw crop, tending to push prices {dir}.'),
    'lpg_nonsub_4city_rs_cyl': dict(
        label='LPG Price, 4-City Avg (Rs/cyl)',
        help='Non-subsidised LPG cylinder price averaged across 4 major '
             'cities (PPAC) — a cold-chain and local-market energy cost proxy.',
        mechanism='Higher LPG prices raise cold-chain and local operating '
                   'costs, tending to push prices {dir}.'),
    'lpg_nonsub_delhi_per14kg': dict(
        label='LPG Price — Delhi (Rs/14kg)',
        help='Non-subsidised LPG price in Delhi specifically (PPAC), '
             'alongside the 4-city average.',
        mechanism='Higher LPG prices raise cold-chain and local operating '
                   'costs, tending to push prices {dir}.'),
    'reverse_repo_pct': dict(
        label='RBI Reverse Repo Rate (%)',
        help='The rate at which RBI absorbs liquidity from banks — moves '
             'alongside the repo rate as part of the same monetary-policy '
             'stance affecting credit costs.',
        mechanism='Same broad channel as the repo rate — a higher reverse '
                   'repo rate is part of a tighter monetary stance, tending '
                   'to push prices {dir} via the cost of holding inventory.'),
    'wpi_fruits_vegetables': dict(
        label='WPI: Fruits & Vegetables',
        help='Wholesale Price Index for the fruits-and-vegetables group '
             '(RBI/CMIE) — a broad wholesale-inflation signal for the whole '
             'category this crop sits in.',
        mechanism='This is itself a price index for the crop\'s broader '
                   'category, so it tends to move together with this crop\'s '
                   'own price ({dir} shown here is the isolated model effect).'),
    'wpi_vegetables_total': dict(
        label='WPI: Vegetables (Total)',
        help='Wholesale Price Index for vegetables overall (RBI/CMIE) — '
             'narrower than the fruits-and-vegetables group above.',
        mechanism='Another close relative of this crop\'s own price series; '
                   '{dir} shown here is the isolated model effect net of the '
                   'other WPI/price features.'),
    'wpi_potato': dict(
        label='WPI: Potato',
        help='Wholesale Price Index specifically for potato (RBI/CMIE) — '
             'relevant as a cross-crop signal even when simulating tomato or '
             'onion (substitution/basket effects).',
        mechanism='Potato\'s own wholesale index can pick up shared seasonal '
                   'or macro effects across the vegetable basket; {dir} shown '
                   'here is the isolated model effect.'),
    'wpi_onion': dict(
        label='WPI: Onion',
        help='Wholesale Price Index specifically for onion (RBI/CMIE) — '
             'relevant as a cross-crop signal even when simulating tomato or '
             'potato.',
        mechanism='Onion\'s own wholesale index can pick up shared seasonal '
                   'or macro effects across the vegetable basket; {dir} shown '
                   'here is the isolated model effect.'),
    'wpi_tomato': dict(
        label='WPI: Tomato',
        help='Wholesale Price Index specifically for tomato (RBI/CMIE) — '
             'relevant as a cross-crop signal even when simulating onion or '
             'potato.',
        mechanism='Tomato\'s own wholesale index can pick up shared seasonal '
                   'or macro effects across the vegetable basket; {dir} shown '
                   'here is the isolated model effect.'),

    # ── Infrastructure (M5) ───────────────────────────────────────────────
    'wage_agri_men': dict(
        label='Agri. Wage — Men (Rs/day)',
        help='State-level average daily wage for male agricultural labour '
             '(Labour Bureau/CMIE) — a direct farm-labour cost input.',
        mechanism='Higher labour costs raise the cost of producing and '
                   'harvesting the crop, which tends to push prices {dir}.'),
    'wage_agri_women': dict(
        label='Agri. Wage — Women (Rs/day)',
        help='State-level average daily wage for female agricultural labour '
             '(Labour Bureau/CMIE) — a direct farm-labour cost input.',
        mechanism='Higher labour costs raise the cost of producing and '
                   'harvesting the crop, which tends to push prices {dir}.'),
    'cold_storage_n_facilities': dict(
        label='Cold Storage Facilities (count)',
        help='Number of registered cold-storage facilities in this state — '
             'more facilities mean more capacity to hold produce off the '
             'market and smooth out price swings.',
        mechanism='More cold-storage facilities give traders more ability to '
                   'hold stock rather than dump it immediately, which tends '
                   'to push prices {dir} by reducing forced-sale pressure.'),
    'cold_storage_capacity_mt': dict(
        label='Cold Storage Capacity (MT)',
        help='Total registered cold-storage capacity (metric tonnes) in this '
             'state — this is a "what if this state had more/less storage '
             'capacity" policy-investment lever, not a week-to-week variable.',
        mechanism='More storage capacity lets traders buffer supply across '
                   'time instead of selling immediately, which tends to push '
                   'prices {dir} by smoothing out short-term gluts.'),
    'road_density_per_100_sqkm': dict(
        label='Road Density (km per 100 km²)',
        help='State road network density (CEIC/MORTH) — a proxy for how '
             'easily produce moves from farm to market. Like cold storage, '
             'this is a slow-moving infrastructure lever, not a weekly variable.',
        mechanism='Denser road networks lower the effective cost and delay of '
                   'getting produce to market, which tends to push prices {dir}.'),
}

st.set_page_config(
    page_title='TOP Digital Twin · APMC Scenario Intelligence',
    layout='wide',
    page_icon='⚡',
    initial_sidebar_state='expanded'
)

# Crop identity & design tokens
CROP_ICON = {'tomato': '🍅', 'onion': '🧅', 'potato': '🥔'}
CROP_COLOR = {'tomato': '#EF4444', 'onion': '#A855F7', 'potato': '#F59E0B'}
CROP_GRADIENT = {
    'tomato': 'linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)',
    'onion': 'linear-gradient(135deg, #A855F7 0%, #7E22CE 100%)',
    'potato': 'linear-gradient(135deg, #F59E0B 0%, #B45309 100%)'
}

# ─────────────────────────────────────────────────────────────────────────────
# REACT / NEXT.JS & TAILWIND CSS STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --bg-app: #F8FAFC;
    --card-bg: #FFFFFF;
    --border-color: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #64748B;
    --text-muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: var(--text-primary);
    background-color: var(--bg-app);
}

/* ── Top Navbar (Next.js / Vercel layout) ── */
.react-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
.react-nav-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.react-brand-badge {
    background: #0F172A;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.02em;
}
.react-breadcrumb {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}
.react-breadcrumb b {
    color: var(--text-primary);
}
.react-nav-right {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Live Pulse Badge ── */
.live-pulse {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-ring 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
}
@keyframes pulse-ring {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* ── React Metric Cards (Shadcn UI style) ── */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px 18px 14px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div[data-testid="stMetric"]:hover {
    border-color: #CBD5E1 !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.02) !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Modern Tabs (Shadcn segmented tabs) ── */
div[data-testid="stTabs"] {
    background: transparent;
}
div[data-testid="stTabs"] button[role="tab"] {
    background: #F1F5F9;
    border-radius: 8px;
    padding: 8px 18px;
    margin-right: 6px;
    font-size: 0.86rem;
    font-weight: 600;
    color: #64748B;
    border: 1px solid transparent;
    transition: all 0.18s ease;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border-color: #0F172A !important;
    box-shadow: 0 4px 10px -1px rgba(15, 23, 42, 0.2);
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #0F172A;
    background: #E2E8F0;
}

/* ── Sidebar (Tailwind / Next.js App Shell) ── */
section[data-testid="stSidebar"] {
    background: #FAFAFC;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1rem;
}
section[data-testid="stSidebar"] hr {
    border-color: #E2E8F0 !important;
    margin: 14px 0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label, 
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #334155;
    letter-spacing: 0.01em;
}

/* ── Buttons (Tailwind Button styling) ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
.stButton > button:hover {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border-color: #0F172A !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
}

/* ── Container Cards with subtle borders ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
}

/* ── Impact Chips ── */
.chip-up {
    background: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FCA5A5;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 0.85rem;
}
.chip-down {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #6EE7B7;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 0.85rem;
}
.chip-neutral {
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 0.85rem;
}

/* ── Custom Badges ── */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY MODERN RECHARTS/TREMOR THEME
# ─────────────────────────────────────────────────────────────────────────────
def apply_react_chart_theme(fig, height=440, title=None):
    """Applies a minimalist React Recharts / Tremor UI aesthetic."""
    fig.update_layout(
        template='plotly_white',
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=13, family="Inter, sans-serif", color="#0F172A"),
            x=0.01,
            y=0.96
        ) if title else None,
        font=dict(family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif', size=11, color='#475569'),
        margin=dict(l=15, r=15, t=40 if title else 20, b=20),
        hoverlabel=dict(
            bgcolor='#0F172A',
            font_size=11,
            font_family='Inter, sans-serif',
            font_color='#FFFFFF',
            bordercolor='#1E293B'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color='#64748B'),
            title=dict(font=dict(size=11, color='#475569', family='Inter, sans-serif')),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color='#64748B'),
            title=dict(font=dict(size=11, color='#475569', family='Inter, sans-serif')),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#E2E8F0',
            borderwidth=1,
            font=dict(size=10, color='#475569')
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
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
    daily_noise = {}
    if os.path.exists(DOW_PATTERN_FILE):
        dow_df = pd.read_csv(DOW_PATTERN_FILE)
        daily_noise = dow_df.groupby('crop')['factor_std'].mean().to_dict()
    return feature_columns, feature_ranges, uncertainty, reference, history, staleness, daily_noise


if not os.path.exists(MODEL_DIR):
    st.error(f'🛑 Production models not found at `{MODEL_DIR}`. Run `python scripts/23_Train_Production_Models.py` first.')
    st.stop()

models = load_models()
feature_columns, feature_ranges, uncertainty, reference, history, staleness, daily_noise = load_metadata()


def predict(crop, h, feature_row):
    """feature_row: dict of {column: value}. Returns predicted price (Rs/quintal).

    FIXED 2026-09-13 (train/serve skew, flagged in the 2026-09-02 audit):
    `feature_row.get(c, 0)` only substitutes 0 for a column MISSING from
    the dict -- if the key exists but its value is a real NaN (confirmed
    live in reference_rows.csv, e.g. price_lag_52/wage_agri_men for
    thinner-history markets), the raw NaN passed straight through to the
    model instead of the 0 Script 23's training-time `fillna(0)` used.
    The explicit `.fillna(0)` below catches both cases the same way
    training did, closing the gap between what the model was trained to
    see for "no value" and what it was actually being served.
    """
    cols = feature_columns[f'{crop}_{h}w']
    X = pd.DataFrame([{c: feature_row.get(c, 0) for c in cols}]).fillna(0)
    log_pred = models[(crop, h)].predict(X)[0]
    return float(np.expm1(log_pred))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — REACT CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.4rem;">⚡</span>
                <div>
                    <h3 style="font-size:1rem; font-weight:800; margin:0; color:#0F172A; letter-spacing:-0.02em;">TOP Digital Twin</h3>
                    <p style="font-size:0.7rem; color:#64748B; margin:0; font-weight:600;">HADP-04 · APMC Simulator</p>
                </div>
            </div>
            <span class="live-pulse"><span class="pulse-dot"></span> M6</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander('📦 Data & Market Specs', expanded=False):
        _counts = reference.groupby('crop')['market_id'].nunique()
        st.markdown(
            '**Feeds:** Agmarknet (prices/arrivals), CMIE/RBI/PPAC (macro), '
            'Sentinel-2/MODIS/ERA5/CHIRPS (remote sensing), Labour Bureau/CMIE '
            '(state wages) + CEIC/MORTH (cold storage, roads), PIB/DGFT '
            '(export policy events), 2017-2026.\n\n'
            f'- 🍅 Tomato: **{_counts.get("tomato", 0)} APMCs**\n'
            f'- 🧅 Onion: **{_counts.get("onion", 0)} APMCs**\n'
            f'- 🥔 Potato: **{_counts.get("potato", 0)} APMCs**'
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:6px;'>Market Selection</p>", unsafe_allow_html=True)

    crop = st.selectbox('Crop Commodity', CROPS, format_func=lambda x: f"{CROP_ICON.get(x, '')} {x.capitalize()}")

    crop_ref = reference[reference['crop'] == crop]
    states = sorted(crop_ref['state'].dropna().unique())
    state_sel = st.selectbox('State / UT', states)

    crop_markets = crop_ref[crop_ref['state'] == state_sel].sort_values('market')
    market = st.selectbox('APMC Terminal', crop_markets['market'].unique())

    base_row_df = crop_markets[crop_markets['market'] == market]
    if base_row_df.empty:
        st.error('No baseline data for this crop/market combination.')
        st.stop()
    base_row = base_row_df.iloc[0].to_dict()
    market_id = base_row['market_id']
    as_of = pd.Timestamp(base_row['week_start'])
    today = pd.Timestamp(datetime.date.today())
    data_weeks_stale = int((today - as_of).days // 7)

    horizon = st.select_slider('Forecast Horizon', options=HORIZONS, value=4, format_func=lambda h: f"{h}W (~{h*7}d)")
    st.caption(f'🎯 Target Date: **{(as_of + pd.Timedelta(weeks=horizon)).strftime("%d %b %Y")}**')

    # ── Scenario Modifiers ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:6px;'>Policy Scenarios</p>", unsafe_allow_html=True)

    scenario = dict(base_row)

    def _num(col, default=0.0):
        v = base_row.get(col)
        return default if v is None or pd.isna(v) else float(v)

    def _policy_staleness_caption(col):
        stale = staleness.get(crop, {}).get(col)
        if stale:
            st.caption(f'📌 Stale feed: {stale["as_of"]} ({stale["weeks_stale"]}w)')

    _ebi = FEATURE_INFO['export_banned']
    export_banned = st.checkbox(_ebi['label'], value=bool(_num('export_banned')), help=_ebi['help'])
    scenario['export_banned'] = int(export_banned)
    _policy_staleness_caption('export_banned')

    def safe_slider(col, extend_pct=0.0, slider_range=None, step=None):
        info = FEATURE_INFO[col]
        label = info['label']
        if col not in feature_ranges or col not in base_row or pd.isna(base_row[col]):
            return None
        r = feature_ranges[col]
        obs_lo, obs_hi, val = float(r['min']), float(r['max']), float(base_row[col])
        if slider_range is not None:
            # Explicit UI bounds (e.g. a domain cap wider than the observed
            # data span) -- the "speculative" check below still compares
            # against the real observed obs_lo/obs_hi, not these UI bounds.
            lo, hi = slider_range
        else:
            span = obs_hi - obs_lo
            lo, hi = obs_lo - span * extend_pct, obs_hi + span * extend_pct
        val = min(max(val, lo), hi)
        if obs_hi <= obs_lo:
            st.caption(f'{label}: {val:g} (fixed)')
            return val
        slider_kwargs = dict(help=info['help'])
        if step is not None:
            slider_kwargs['step'] = step
        result = st.slider(label, lo, hi, val, **slider_kwargs)
        if result < obs_lo or result > obs_hi:
            st.caption(f'⚠️ Speculative range ({obs_lo:g}–{obs_hi:g})')
        stale = staleness.get(crop, {}).get(col)
        if stale:
            st.caption(f'📌 Stale feed: {stale["as_of"]} ({stale["weeks_stale"]}w)')
        return result

    if 'mep_usd_per_tonne' in feature_ranges:
        r = feature_ranges['mep_usd_per_tonne']
        val = safe_slider('mep_usd_per_tonne', slider_range=(0.0, max(r['max'], 900.0)), step=10.0)
        if val is not None:
            scenario['mep_usd_per_tonne'] = val

    if 'export_duty_pct' in feature_ranges:
        val = safe_slider('export_duty_pct', slider_range=(0.0, 50.0), step=1.0)
        if val is not None:
            scenario['export_duty_pct'] = val

    _mii = FEATURE_INFO['market_intervention_flag']
    market_intervention = st.checkbox(
        _mii['label'], value=bool(_num('market_intervention_flag')), help=_mii['help']
    )
    scenario['market_intervention_flag'] = int(market_intervention)
    _policy_staleness_caption('market_intervention_flag')

    _ogi = FEATURE_INFO['operation_greens_active']
    op_greens = st.checkbox(_ogi['label'], value=bool(_num('operation_greens_active')), help=_ogi['help'])
    scenario['operation_greens_active'] = int(op_greens)
    _policy_staleness_caption('operation_greens_active')

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:6px;'>Climate & Satellite</p>", unsafe_allow_html=True)
    for col in ['era5_tmax', 'chirps_rain_mm', 's2_ndvi']:
        val = safe_slider(col)
        if val is not None:
            scenario[col] = val
    # Every other raw M3/M4 climate/satellite column the model actually
    # trains on (2026-09-25 dashboard audit) -- collapsed by default so the
    # sidebar's default view is unchanged, but genuinely reachable rather
    # than invisible. 's2_evi' deliberately excluded -- see Script 23's
    # SIMULATABLE comment (its source data is corrupted for ~50% of rows).
    with st.expander('More climate & satellite variables'):
        for col in ['era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38',
                    'chirps_rain_max', 'chirps_excess', 's2_valid_frac', 's2_ndvi_anom',
                    'modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']:
            val = safe_slider(col)
            if val is not None:
                scenario[col] = val

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:6px;'>Macro & Logistics</p>", unsafe_allow_html=True)
    for col in ['diesel_4city_rs_litre', 'repo_rate_pct', 'usdinr_monthly_avg']:
        val = safe_slider(col, extend_pct=0.20)
        if val is not None:
            scenario[col] = val
    # Every other raw M2 macro column (CMIE/RBI/PPAC) the model actually
    # trains on -- same collapsed-by-default treatment as climate/satellite.
    # No extend_pct here (unlike the 3 primary sliders above): found
    # 2026-09-25 that a 20% extension on a column whose observed minimum
    # sits near zero (WPI series, bank credit) pushes the slider's lower
    # bound negative, which is nonsensical for an index/count -- the 3
    # primary sliders' own observed minimums are comfortably above zero, so
    # this never showed up there.
    with st.expander('More macro & trade variables'):
        for col in ['agri_wages_rs_day', 'bank_credit_agri_cr', 'crude_oil_usd_bbl',
                    'diesel_delhi_per_L', 'export_veg_usd_mn', 'import_veg_usd_mn',
                    'iip_food_proc', 'lpg_nonsub_4city_rs_cyl', 'lpg_nonsub_delhi_per14kg',
                    'reverse_repo_pct', 'wpi_fruits_vegetables', 'wpi_vegetables_total',
                    'wpi_potato', 'wpi_onion', 'wpi_tomato']:
            val = safe_slider(col)
            if val is not None:
                scenario[col] = val

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:6px;'>Infrastructure & Labor (M5)</p>", unsafe_allow_html=True)
    # M5 had NO presence anywhere in the dashboard before 2026-09-25 despite
    # being real model inputs -- wage_agri_men/women vary weekly by state;
    # cold storage/road density are slow-moving state facts (see their
    # FEATURE_INFO 'help' text), so moving their sliders is a genuine "what
    # if this state invested in more capacity" policy question, not a
    # week-to-week what-if -- collapsed by default for that reason too.
    for col in ['wage_agri_men', 'wage_agri_women']:
        val = safe_slider(col)
        if val is not None:
            scenario[col] = val
    with st.expander('Cold storage & road density (state infrastructure)'):
        # No extend_pct -- cold_storage_n_facilities' observed minimum is
        # already near zero, so a 20% extension pushes the slider negative
        # (same issue as the macro WPI columns above).
        for col in ['cold_storage_n_facilities', 'cold_storage_capacity_mt', 'road_density_per_100_sqkm']:
            val = safe_slider(col)
            if val is not None:
                scenario[col] = val

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button('🔄 Reset to Baseline Vector', use_container_width=True):
        scenario = dict(base_row)


# Dynamic Crop Border for Metric Cards
st.markdown(f"""
<style>
div[data-testid="stMetric"] {{
    border-left: 4px solid {CROP_COLOR[crop]} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP APP HEADER BAR (React / Vercel style)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="react-navbar">
    <div class="react-nav-left">
        <span class="react-brand-badge">{CROP_ICON.get(crop, "")} {crop.upper()}</span>
        <div class="react-breadcrumb">
            <span>HADP-04</span> / <span>{base_row.get('state', '')}</span> / <b>{market} APMC</b>
        </div>
    </div>
    <div class="react-nav-right">
        <span class="badge-pill">📍 ID: {market_id}</span>
        <span class="badge-pill">🗓️ Feed: {as_of.strftime("%d %b %Y")}</span>
        <span class="live-pulse"><span class="pulse-dot"></span> Live Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Data Quality Notices
if data_weeks_stale >= 8:
    st.warning(f'⚠️ **Feed Staleness:** Market data is **{data_weeks_stale} weeks** behind current calendar date.')

sufficient_history = base_row.get('sufficient_history')
if pd.notna(sufficient_history) and not bool(sufficient_history):
    st.error('🛑 **Insufficient History:** Market lacks pre-baseline trade data; treated as placeholder.')

stale_reference = base_row.get('stale_reference')
if pd.notna(stale_reference) and bool(stale_reference) and (pd.isna(sufficient_history) or bool(sufficient_history)):
    last_real = base_row.get('last_observed_date')
    last_real_str = pd.to_datetime(last_real).date() if pd.notna(last_real) else 'unknown'
    st.warning(f'⚠️ **Stale Reference Price:** Most recent trade occurred on **{last_real_str}**.')

pct_imputed = base_row.get('pct_imputed_last_52w')
if pd.notna(pct_imputed) and pct_imputed >= 50:
    st.warning(f'⚠️ **Data Quality:** {pct_imputed:.0f}% of last 52 weeks are imputed.')


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE MODEL PREDICTIONS & UNCERTAINTIES
# ─────────────────────────────────────────────────────────────────────────────
_ticker_hist = (history[(history['crop'] == crop) & (history['market_id'] == market_id)
                         & (history['week_start'] <= as_of)]
                 .sort_values('week_start')
                 .tail(12))
recent_trend = _ticker_hist['modal_price_weighted'].dropna().tolist()

ticker_points = [(as_of, base_row.get('log_price'))]
ticker_results = {}
for h in HORIZONS:
    fdate = as_of + pd.Timedelta(weeks=h)
    fprice = predict(crop, h, base_row)
    ticker_points.append((fdate, np.log1p(fprice)))
    herr = uncertainty.get(f'{crop}_{h}w', {})
    season = season_for(crop, fdate)
    spark = (recent_trend + [fprice]) if recent_trend else None
    ticker_results[h] = {
        'date': fdate,
        'price': fprice,
        'rmse': herr.get('rmse'),
        'wape': herr.get('wape'),
        'season': season,
        'spark': spark
    }

baseline_pred = predict(crop, horizon, base_row)
scenario_pred = predict(crop, horizon, scenario)
delta = scenario_pred - baseline_pred
delta_pct = 100 * delta / baseline_pred if baseline_pred else 0

err = uncertainty.get(f'{crop}_{horizon}w', {})
rmse, wape = err.get('rmse'), err.get('wape')

def _differs(a, b):
    if pd.isna(a) and pd.isna(b):
        return False
    return a != b

diff_cols = [c for c in scenario if _differs(scenario.get(c), base_row.get(c))]


# ─────────────────────────────────────────────────────────────────────────────
# REACT SEGMENTED TAB ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
tab_sim, tab_attrib, tab_bench, tab_multi, tab_ai, tab_audit = st.tabs([
    "⚡ Simulation & Trajectory",
    "🔍 Feature Attribution",
    "🏆 Market Benchmarks",
    "📊 Cross-Market Analytics",
    "🤖 AI Policy Intelligence",
    "🛠️ Technical Audit"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: SIMULATION & TRAJECTORY
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    # 1. Multi-Horizon Ticker
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Multi-Horizon Baseline Forecasts</p>", unsafe_allow_html=True)
    t_cols = st.columns(len(HORIZONS))
    for tcol, h in zip(t_cols, HORIZONS):
        res = ticker_results[h]
        herr_note = f" ±₹{res['rmse']:,.0f} ({res['wape']:.0f}% WAPE)" if res['rmse'] is not None else ""
        with tcol:
            st.metric(
                f"Horizon {h}W · {res['date'].strftime('%d %b')}",
                f"₹ {res['price']:,.0f}",
                border=True,
                chart_data=res['spark'],
                chart_type='line',
                help=f"Baseline forecast for {res['date'].date()} ({h}w ahead).{herr_note}"
            )
            if res['season']:
                st.markdown(
                    f"<div style='text-align:center; margin-top:-6px; margin-bottom:6px;'>"
                    f"<span class='badge-pill'>🌾 {SEASON_LABEL[res['season']]}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # 2. Executive KPI Cards
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        f"Baseline ({horizon}W)",
        f"₹ {baseline_pred:,.0f} / q",
        border=True,
        chart_data=(recent_trend + [baseline_pred]) if recent_trend else None,
        chart_type='line',
        help="Unmodified baseline model projection."
    )

    c2.metric(
        f"Scenario ({horizon}W)",
        f"₹ {scenario_pred:,.0f} / q",
        delta=f"{delta:+,.0f} ({delta_pct:+.1f}%)",
        delta_color='inverse' if delta > 0 else 'normal',
        border=True,
        chart_data=(recent_trend + [scenario_pred]) if recent_trend else None,
        chart_type='line',
        help="Model projection with active scenario modifier inputs."
    )

    last_obs_price = base_row.get('last_observed_price')
    last_obs_date = base_row.get('last_observed_date')
    if pd.notna(last_obs_price):
        is_stale = pd.notna(last_obs_date) and pd.Timestamp(last_obs_date) < as_of
        v3 = f"₹ {last_obs_price:,.0f} / q"
        if is_stale:
            w_stale = (as_of - pd.Timestamp(last_obs_date)).days // 7
            v3 += f" ({w_stale}w ago)"
        c3.metric("Last Real Trade", v3, border=True, help="Most recent non-imputed trade price.")
    else:
        c3.metric("Last Real Trade", "N/A", border=True)

    if wape is not None:
        c4.metric("Model Accuracy", f"~{max(0.0, 100 - wape):.0f}%", border=True, help=f"100% - WAPE ({wape:.0f}%) from backtesting.")
    else:
        c4.metric("Model Accuracy", "N/A", border=True)

    # 3. Main Chart: Price History & Projections
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    mkt_history = (history[(history['crop'] == crop) & (history['market_id'] == market_id)]
                   .sort_values('week_start'))

    fig_main = go.Figure()
    if not mkt_history.empty:
        fig_main.add_trace(go.Scatter(
            x=mkt_history['week_start'], y=mkt_history['modal_price_weighted'],
            mode='lines', name='Historical Observed Price',
            line=dict(color='#334155', width=2)
        ))

    last_actual_date = mkt_history['week_start'].max() if not mkt_history.empty else as_of
    last_actual_price = (mkt_history[mkt_history['week_start'] == last_actual_date]
                          ['modal_price_weighted'].iloc[0]) if not mkt_history.empty else baseline_pred
    selected_date = as_of + pd.Timedelta(weeks=horizon)

    band_rmse = [uncertainty.get(f'{crop}_{h}w', {}).get('rmse', 0) for h in HORIZONS]
    if any(band_rmse):
        band_dates = [last_actual_date] + [as_of + pd.Timedelta(weeks=h) for h in HORIZONS]
        band_upper = [last_actual_price] + [predict(crop, h, base_row) + r for h, r in zip(HORIZONS, band_rmse)]
        band_lower = [last_actual_price] + [predict(crop, h, base_row) - r for h, r in zip(HORIZONS, band_rmse)]
        fig_main.add_trace(go.Scatter(
            x=band_dates + band_dates[::-1], y=band_upper + band_lower[::-1],
            fill='toself', fillcolor='rgba(148, 163, 184, 0.16)', line=dict(width=0),
            name='±RMSE Uncertainty Range', hoverinfo='skip'
        ))

    for label, f_row, color, dash in [
            ('Baseline Forecast', base_row, '#64748B', 'dot'),
            ('Scenario Forecast', scenario, '#EF4444', 'dash')]:
        fc_dates = [last_actual_date] + [as_of + pd.Timedelta(weeks=h) for h in HORIZONS]
        fc_prices = [last_actual_price] + [predict(crop, h, f_row) for h in HORIZONS]
        fig_main.add_trace(go.Scatter(
            x=fc_dates, y=fc_prices, mode='lines+markers', name=label,
            line=dict(color=color, width=2.4, dash=dash), marker=dict(size=6, color=color)
        ))

    fig_main.add_trace(go.Scatter(
        x=[selected_date], y=[predict(crop, horizon, scenario)], mode='markers',
        name=f'Target ({horizon}W)',
        marker=dict(size=13, symbol='star', color='#F59E0B', line=dict(width=1.5, color='#0F172A')),
        showlegend=True,
        hovertemplate=f'Target ({horizon}W)<br>%{{x|%d %b %Y}}<br>₹%{{y:,.0f}}<extra></extra>'
    ))

    fig_main.add_vline(x=as_of, line_dash='dot', line_color='#94A3B8', annotation_text='Data As-Of', annotation_position='top left', annotation_font=dict(size=9, color='#64748B'))
    fig_main.add_vline(x=selected_date, line_dash='dash', line_color='#F59E0B', annotation_text=f'Target: {horizon}W', annotation_position='top right', annotation_font=dict(size=9, color='#D97706'))

    fig_main.update_layout(xaxis_title='Date', yaxis_title='Price (Rs/quintal)', hovermode='x unified')
    apply_react_chart_theme(fig_main, height=450)
    st.plotly_chart(fig_main, use_container_width=True)

    # 4. Daily Disaggregation Collapsible
    if pd.notna(ticker_points[0][1]) and crop in daily_noise:
        st.session_state.setdefault('daily_view_open', False)
        b_col, _ = st.columns([1, 3])
        with b_col:
            d_btn_label = '🔼 Hide Daily Disaggregation' if st.session_state['daily_view_open'] else '📅 Expand Daily Disaggregation Curve'
            if st.button(d_btn_label, key='daily_toggle_btn', use_container_width=True):
                st.session_state['daily_view_open'] = not st.session_state['daily_view_open']
                st.rerun()

        if st.session_state['daily_view_open']:
            with st.container(border=True):
                pts_dates, pts_logp = zip(*ticker_points)
                pts_num = [(d - as_of).days for d in pts_dates]
                pchip = PchipInterpolator(pts_num, pts_logp)

                daily_offsets_full = np.arange(0, HORIZONS[-1] * 7 + 1)
                daily_dates_full = [as_of + pd.Timedelta(days=int(o)) for o in daily_offsets_full]
                smooth_trend_full = np.expm1(pchip(daily_offsets_full))
                display_from = max(today, as_of)
                keep = [i for i, d in enumerate(daily_dates_full) if d >= display_from]
                daily_dates = [daily_dates_full[i] for i in keep]
                smooth_trend = smooth_trend_full[keep]
                band = smooth_trend * daily_noise[crop]

                if not daily_dates:
                    st.info('Today is beyond this market\'s 26-week horizon.')
                else:
                    marquee_items = ' &nbsp;&nbsp;•&nbsp;&nbsp; '.join(
                        f'<span style="background:#FFFFFF; padding:2px 8px; border-radius:6px; border:1px solid #E2E8F0;">'
                        f'{d.strftime("%d %b")}: <b>₹&nbsp;{p:,.0f}</b></span>'
                        for d, p in zip(daily_dates, smooth_trend)
                    )
                    marquee_duration = max(80, len(daily_dates) * 1.4)
                    st.markdown(f"""
                    <style>
                    .react-marquee-wrap {{
                        overflow: hidden; white-space: nowrap; box-sizing: border-box;
                        border: 1px solid #E2E8F0; border-radius: 8px;
                        padding: 8px 0; margin-bottom: 12px; background: #F8FAFC;
                    }}
                    .react-marquee-track {{
                        display: inline-block; padding-left: 100%;
                        animation: r-marquee {marquee_duration}s linear infinite;
                        font-size: 12px; color: #1E293B;
                    }}
                    .react-marquee-wrap:hover .react-marquee-track {{ animation-play-state: paused; }}
                    @keyframes r-marquee {{
                        0%   {{ transform: translate(0, 0); }}
                        100% {{ transform: translate(-100%, 0); }}
                    }}
                    </style>
                    <div class="react-marquee-wrap">
                      <div class="react-marquee-track">{marquee_items}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    fig_d = go.Figure()
                    seasons_by_day = [season_for(crop, d) for d in daily_dates]
                    run_start = 0
                    for i in range(1, len(daily_dates) + 1):
                        if i == len(daily_dates) or seasons_by_day[i] != seasons_by_day[run_start]:
                            s = seasons_by_day[run_start]
                            if s:
                                fig_d.add_vrect(
                                    x0=daily_dates[run_start], x1=daily_dates[i - 1] + pd.Timedelta(days=1),
                                    fillcolor=SEASON_COLOR[s], line_width=0, layer='below'
                                )
                            run_start = i

                    fig_d.add_trace(go.Scatter(
                        x=daily_dates + daily_dates[::-1],
                        y=list(smooth_trend + band) + list((smooth_trend - band)[::-1]),
                        fill='toself', fillcolor='rgba(16, 185, 129, 0.10)', line=dict(width=0),
                        name='Daily Noise Band (±1 SD)', hoverinfo='skip'
                    ))
                    fig_d.add_trace(go.Scatter(
                        x=daily_dates, y=smooth_trend, mode='lines',
                        line=dict(color='#059669', width=2), name='PCHIP Daily Trend'
                    ))
                    apply_react_chart_theme(fig_d, height=360, title="Smooth Daily Trajectory & Volatility")
                    st.plotly_chart(fig_d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: FEATURE ATTRIBUTION & ISOLATED EFFECTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_attrib:
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Scenario Feature Decomposition (h=4W)</p>", unsafe_allow_html=True)
    if not diff_cols:
        st.info('💡 No changes from baseline. Adjust modifier controls in the sidebar to view isolated factor contributions.')
    else:
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
            chip_cls = "chip-up" if eff > 0 else ("chip-down" if eff < 0 else "chip-neutral")

            with st.container(border=True):
                ca, cb = st.columns([3, 1])
                with ca:
                    st.markdown(f"**{label}** &nbsp;·&nbsp; `{before:g}` → `<b>{after:g}</b>`", unsafe_allow_html=True)
                    if info:
                        st.caption(info['mechanism'].format(dir=direction))
                with cb:
                    st.markdown(
                        f"""
                        <div style="text-align:right;">
                            <span class="{chip_cls}">{eff:+,.0f} Rs/q</span>
                            <p style="font-size:0.7rem; color:#64748B; margin:4px 0 0 0; font-weight:600; text-transform:uppercase;">Isolated</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if abs(interaction_gap) > 1:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px; margin-top:8px;">
                    <span style="font-size:0.82rem; color:#475569;">
                        <b>Sum of isolated effects:</b> {sum_isolated:+,.0f} Rs/q &nbsp;·&nbsp; 
                        <b>Combined delta:</b> {delta:+,.0f} Rs/q &nbsp;·&nbsp; 
                        <b>Non-linear interaction term:</b> {interaction_gap:+,.0f} Rs/q
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: MARKET BENCHMARKS (TOP 15)
# ═════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Top 15 APMC Terminals by Observed Volume & Price</p>", unsafe_allow_html=True)
    TOP_N = 15
    price_top = (crop_ref.dropna(subset=['last_observed_price'])
                 .nlargest(TOP_N, 'last_observed_price')
                 .sort_values('last_observed_price'))
    arr_top = (crop_ref.dropna(subset=['log_arr'])
               .assign(arrivals_tonnes=lambda d: np.expm1(d['log_arr']))
               .nlargest(TOP_N, 'arrivals_tonnes')
               .sort_values('arrivals_tonnes'))

    def _disambiguated_labels(df):
        dup_names = set(df['market'][df['market'].duplicated(keep=False)])
        return [f"{m} ({s})" if m in dup_names else m for m, s in zip(df['market'], df['state'])]

    bc1, bc2 = st.columns(2)
    with bc1:
        if price_top.empty:
            st.info('No price data available.')
        else:
            fig_p = go.Figure(go.Bar(
                x=price_top['last_observed_price'], y=_disambiguated_labels(price_top), orientation='h',
                marker_color=['#EF4444' if mid == market_id else '#334155' for mid in price_top['market_id']],
                marker_line=dict(width=0)
            ))
            apply_react_chart_theme(fig_p, height=420, title='Top 15 Markets by Real Trade Price (Rs/quintal)')
            fig_p.update_layout(yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_p, use_container_width=True)

    with bc2:
        if arr_top.empty:
            st.info('No arrivals data available.')
        else:
            fig_a = go.Figure(go.Bar(
                x=arr_top['arrivals_tonnes'], y=_disambiguated_labels(arr_top), orientation='h',
                marker_color=['#EF4444' if mid == market_id else '#0F766E' for mid in arr_top['market_id']],
                marker_line=dict(width=0)
            ))
            apply_react_chart_theme(fig_a, height=420, title='Top 15 Markets by Weekly Arrivals (Tonnes)')
            fig_a.update_layout(yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_a, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: CROSS-MARKET ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Cross-Market Multi-Series Price Comparison</p>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        trend_crop = st.selectbox('Commodity', CROPS, format_func=lambda x: f"{CROP_ICON.get(x, '')} {x.capitalize()}",
                                   index=CROPS.index(crop), key='t_crop')
    with mc2:
        trend_crop_ref = reference[reference['crop'] == trend_crop]
        trend_states = sorted(trend_crop_ref['state'].dropna().unique())
        trend_state_default = [state_sel] if state_sel in trend_states else trend_states[:1]
        trend_states_sel = st.multiselect('States / UTs', trend_states, default=trend_state_default, key='t_states')

    trend_crop_history = history[history['crop'] == trend_crop]
    id_to_name = trend_crop_ref.drop_duplicates('market_id').set_index('market_id')['market']
    id_to_state = trend_crop_ref.drop_duplicates('market_id').set_index('market_id')['state']
    trend_selectable_ids = set(trend_crop_ref[trend_crop_ref['state'].isin(trend_states_sel)]['market_id'])
    history_market_ids = sorted(mid for mid in trend_crop_history['market_id'].unique() if mid in trend_selectable_ids)
    default_trend_ids = [market_id] if (trend_crop == crop and market_id in history_market_ids) else []
    selected_trend_ids = st.multiselect(
        'APMC Terminals (Max 15)', options=history_market_ids, default=default_trend_ids,
        max_selections=15, format_func=lambda mid: f'{id_to_name.get(mid, "?")} — {id_to_state.get(mid, "?")}',
        key='t_markets'
    )

    if not selected_trend_ids:
        st.info('👉 Select APMC markets above to compare trajectories.')
    else:
        fig_multi = go.Figure()
        no_data_labels = []
        for mid in selected_trend_ids:
            mdata = trend_crop_history[trend_crop_history['market_id'] == mid].sort_values('week_start')
            m_name, m_state = id_to_name.get(mid, '?'), id_to_state.get(mid, '?')
            if mdata['modal_price_weighted'].notna().sum() == 0:
                no_data_labels.append(f'{m_name} ({m_state})')
                continue
            legend_name = m_name if len(trend_states_sel) <= 1 else f'{m_name} ({m_state})'
            fig_multi.add_trace(go.Scatter(
                x=mdata['week_start'], y=mdata['modal_price_weighted'],
                mode='lines', name=legend_name,
                line=dict(width=2.5 if (trend_crop == crop and mid == market_id) else 1.5)
            ))
        if no_data_labels:
            st.warning(f'⚠️ No non-imputed trade data available in this window for: {", ".join(no_data_labels)}')
        if len(fig_multi.data) > 0:
            apply_react_chart_theme(fig_multi, height=440, title="Comparative Historical Timeline")
            st.plotly_chart(fig_multi, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: AI POLICY INTELLIGENCE (Claude Haiku)
# ═════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Executive Policy Synthesis</p>", unsafe_allow_html=True)
    def _api_key():
        try:
            return st.secrets.get('ANTHROPIC_API_KEY')
        except Exception:
            return None

    if not diff_cols:
        st.info('💡 Modify sidebar scenario inputs to generate an AI policy memo.')
    elif not _api_key():
        st.info('ℹ️ AI policy commentary requires an `ANTHROPIC_API_KEY` configured in Streamlit Secrets.')
    else:
        scenario_key = hashlib.md5(
            f"{crop}|{market_id}|{horizon}|{sorted((c, scenario.get(c)) for c in diff_cols)}".encode()
        ).hexdigest()
        reco_state_key = f'ai_reco_{scenario_key}'

        AI_BRIEF_COOLDOWN_SECONDS = 30

        if st.button('✨ Generate Policy Briefing', key='ai_brief_btn'):
            _last_brief_time = st.session_state.get('last_ai_brief_time')
            _now = time.time()
            if _last_brief_time is not None and (_now - _last_brief_time) < AI_BRIEF_COOLDOWN_SECONDS:
                st.warning(
                    f'⏳ Please wait a moment before requesting another AI briefing '
                    f'({int(AI_BRIEF_COOLDOWN_SECONDS - (_now - _last_brief_time))}s).'
                )
            else:
                st.session_state['last_ai_brief_time'] = _now
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
                    f'Crop: {crop.capitalize()}\nMarket: {market}, {base_row.get("state", "")}\n'
                    f'Forecast horizon: {horizon} weeks ahead\n'
                    f'Baseline prediction: Rs {baseline_pred:,.0f}/quintal\n'
                    f'Scenario prediction: Rs {scenario_pred:,.0f}/quintal '
                    f'({delta_pct:+.1f}%, {delta:+,.0f} Rs/quintal)\n'
                    f"Model's typical error at this horizon: "
                    f"{f'±Rs {rmse:,.0f} ({wape:.0f}% WAPE)' if rmse is not None else 'not available'}\n\n"
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
                    with st.spinner('Generating analysis...'):
                        client = anthropic.Anthropic(api_key=_api_key())
                        resp = client.messages.create(
                            model='claude-haiku-4-5-20251001', max_tokens=400,
                            messages=[{'role': 'user', 'content': prompt}])
                        st.session_state[reco_state_key] = resp.content[0].text
                except Exception as e:
                    st.session_state[reco_state_key] = None
                    st.error(f'AI generation failed: {e}')

        cached = st.session_state.get(reco_state_key)
        if cached:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #0F172A; padding-left:14px; margin:4px 0;">
                        <p style="font-size:0.92rem; line-height:1.6; color:#0F172A; margin:0;">
                            {cached}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.caption('📌 Grounded in empirical model outputs · decision-support commentary only.')


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: TECHNICAL AUDIT & FEATURE VECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-top:8px; margin-bottom:10px;'>Active Feature Vector Inspection</p>", unsafe_allow_html=True)
    if diff_cols:
        debug_df = pd.DataFrame([
            {
                'Feature': FEATURE_INFO.get(c, {}).get('label', c),
                'Baseline Value': base_row.get(c),
                'Scenario Value': scenario.get(c),
                'Field Key': c
            }
            for c in diff_cols
        ])
        st.dataframe(debug_df, use_container_width=True, hide_index=True)
    else:
        st.write('All scenario variables are currently aligned with the empirical baseline vector.')
