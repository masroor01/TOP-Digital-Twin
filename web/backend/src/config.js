import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Reference data lives inside web/data/ (a bundled copy), NOT the wider
// TOP_Digital_Twin repo's Model_Output/ -- deliberate. Hosting platforms
// that deploy from a "root directory" subtree (e.g. Hostinger's Deploy Web
// App pointed at `web`) never check out anything outside that subtree, so
// a path reaching up to ../../Model_Output would silently not exist there
// and crash the process on startup (confirmed: this caused a 503 on first
// deploy attempt). Keeping web/ fully self-contained avoids that whole
// class of bug regardless of a given platform's checkout behavior. See
// web/README.md "Updating the bundled data" for how to refresh this copy.
const WEB_ROOT = path.resolve(__dirname, '..', '..');
export const MODEL_DIR = path.join(WEB_ROOT, 'data', 'production_models');
export const DOW_PATTERN_FILE = path.join(WEB_ROOT, 'data', 'table_dow_pattern.csv');
export const DIRECTIONAL_ACCURACY_FILE = path.join(WEB_ROOT, 'data', 'table_directional_accuracy.csv');
export const MARKET_ACCURACY_FILE = path.join(WEB_ROOT, 'data', 'table_market_level_accuracy.csv');
export const SHAP_LAYER_FILE = path.join(WEB_ROOT, 'data', 'table_shap_by_layer.csv');
export const SHAP_FEATURES_FILE = path.join(WEB_ROOT, 'data', 'table_shap_top_features.csv');

export const CROPS = ['tomato', 'onion', 'potato'];
export const HORIZONS = [1, 4, 13, 26];

export const CROP_ICON = { tomato: '🍅', onion: '🧅', potato: '🥔' };
export const CROP_COLOR = { tomato: '#EF4444', onion: '#A855F7', potato: '#F59E0B' };

export const PORT = process.env.PORT || 4000;

// Ported verbatim from scripts/24_Simulation_Dashboard.py — keep in sync if
// the Python dashboard's feature metadata ever changes.
export const SEASON_MONTHS = {
  tomato: { peak_arrival: [11, 12, 1, 2], lean: [5, 6, 7], kharif: [8, 9, 10] },
  onion: { rabi_arrival: [2, 3, 4, 5], lean: [9, 10, 11], kharif: [8, 9] },
  potato: { harvest: [2, 3, 4], storage: [5, 6, 7, 8, 9], lean: [10, 11] },
};

export const SEASON_LABEL = {
  peak_arrival: 'Peak Arrival', lean: 'Lean Season', kharif: 'Kharif Season',
  rabi_arrival: 'Rabi Arrival', harvest: 'Harvest Season', storage: 'Storage Period',
};

export const SEASON_COLOR = {
  peak_arrival: 'rgba(16, 185, 129, 0.14)', lean: 'rgba(239, 68, 68, 0.14)',
  kharif: 'rgba(245, 158, 11, 0.14)', rabi_arrival: 'rgba(16, 185, 129, 0.14)',
  harvest: 'rgba(16, 185, 129, 0.14)', storage: 'rgba(59, 130, 246, 0.14)',
};

export function seasonFor(crop, date) {
  const month = date.getUTCMonth() + 1;
  const months = SEASON_MONTHS[crop] || {};
  for (const [season, list] of Object.entries(months)) {
    if (list.includes(month)) return season;
  }
  return null;
}

export const FEATURE_INFO = {
  export_banned: {
    label: 'Export Ban in Effect',
    help: "When ON, the government prohibits exporting this crop abroad (e.g. India's Dec 2023-May 2024 onion export ban). The strongest policy lever — it forces all supply to stay in the domestic market.",
    mechanism: 'An export ban keeps supply that would have gone abroad inside domestic markets, which tends to push domestic prices {dir}. Historically this has mainly mattered for onion — tomato and potato have had no significant export-ban history to learn from.',
  },
  mep_usd_per_tonne: {
    label: 'Minimum Export Price (USD/t)',
    help: 'The floor price (USD/tonne) below which exporters may not sell abroad. A softer alternative to an outright ban — raising it prices exports out of the international market without banning them.',
    mechanism: 'A higher MEP discourages exports by making them less price-competitive abroad, which — like a ban — tends to keep more supply at home and push domestic prices {dir}.',
  },
  export_duty_pct: {
    label: 'Export Duty (%)',
    help: 'A tax (% of value) on exported goods, e.g. the 40% onion export duty imposed in Aug 2023. Raises the cost of exporting, discouraging outbound shipments similarly to a higher MEP.',
    mechanism: 'A higher export duty raises the cost of shipping abroad, discouraging exports and tending to push domestic prices {dir}.',
  },
  market_intervention_flag: {
    label: 'Market Intervention This Week',
    help: 'Marks a reported NAFED/NCCF buffer-stock procurement or release, or a subsidised retail sale, in this exact week. These directly add or remove supply to manage price spikes or crashes.',
    mechanism: 'Interventions are usually a REACTION to price stress (they happen because prices are already high or low), so this flag can reflect "crisis conditions" as much as it drives price itself — read its effect with that in mind.',
  },
  era5_tmax: {
    label: 'Max Temperature (°C)',
    help: 'Weekly maximum temperature in the growing region (ERA5 climate reanalysis). Extreme heat can stress crops and reduce yields, tightening supply in the weeks ahead.',
    mechanism: 'Higher extreme temperature is associated with crop stress and reduced expected supply, which tends to push prices {dir}.',
  },
  chirps_rain_mm: {
    label: 'Weekly Rainfall (mm)',
    help: "Satellite-estimated rainfall in the growing region (CHIRPS). Effect is two-sided: moderate rain supports growth, but excess rain can flood fields, damage crops, and disrupt harvest/transport.",
    mechanism: "Rainfall's effect is non-monotonic — moderate increases can support supply (pushing prices down), but large increases can damage crops or disrupt logistics (pushing prices up). The direction shown here is what the model learned for this specific change, not a fixed rule.",
  },
  s2_ndvi: {
    label: 'Vegetation Index (NDVI)',
    help: 'Crop health/density from Sentinel-2 satellite imagery (roughly 0-1). Higher values generally mean healthier, denser vegetation — a proxy for expected yield.',
    mechanism: 'Higher NDVI (healthier growing conditions) generally signals more supply ahead, which tends to push prices {dir}.',
  },
  diesel_4city_rs_litre: {
    label: 'Diesel Price (Rs/Litre)',
    help: 'Average diesel price across 4 major Indian cities (PPAC data). Diesel is the dominant fuel for transporting produce from farms to markets, so it is a direct proxy for logistics cost.',
    mechanism: 'Higher diesel prices raise the cost of transporting produce to market, which tends to push wholesale prices {dir}.',
  },
  repo_rate_pct: {
    label: 'RBI Repo Rate (%)',
    help: "The Reserve Bank of India's policy interest rate — the cost at which banks borrow. Affects the cost of credit for traders who borrow to finance stored inventory, particularly cold-stored potato.",
    mechanism: 'A higher repo rate raises the cost of holding inventory on credit, which can discourage stockpiling and tends to push prices {dir} — most relevant for storage-buffered crops.',
  },
  usdinr_monthly_avg: {
    label: 'USD/INR Exchange Rate',
    help: 'The rupee-per-dollar exchange rate. A weaker rupee (higher number) makes Indian exports cheaper for foreign buyers in dollar terms.',
    mechanism: 'A weaker rupee makes exports more attractive, pulling supply toward export markets and away from domestic ones, which tends to push domestic prices {dir}.',
  },
  operation_greens_active: {
    label: 'Operation Greens Active',
    help: "Marks whether the Centre's Operation Greens (TOP — Tomato, Onion, Potato) price-stabilisation scheme, including its transport/storage subsidy, was active that week for this crop.",
    mechanism: 'Operation Greens subsidises storage and interstate transport to smooth out price swings, so an active scheme tends to push prices {dir} toward normal levels rather than in one fixed direction.',
  },

  // ── Climate (M3) ─────────────────────────────────────────────────────
  era5_tmin: {
    label: 'Min Temperature (°C)',
    help: 'Weekly minimum temperature in the growing region (ERA5). Cold snaps can damage tender crops (tomato especially) and delay growth.',
    mechanism: 'Lower minimum temperature is associated with cold-stress risk to the crop, which tends to push prices {dir} via reduced expected supply.',
  },
  era5_tmean: {
    label: 'Mean Temperature (°C)',
    help: "Weekly average temperature in the growing region (ERA5) — the general thermal backdrop for the growing season, distinct from the daily extremes (max/min).",
    mechanism: "Average temperature away from the crop's optimal range tends to push prices {dir} via reduced expected supply.",
  },
  era5_dtr: {
    label: 'Diurnal Temp. Range (°C)',
    help: 'The gap between daily max and min temperature (ERA5). A wide swing can stress crops even when the average temperature looks fine.',
    mechanism: 'A wider day-night temperature swing is associated with additional crop stress, which tends to push prices {dir}.',
  },
  era5_heat_35: {
    label: 'Days ≥35°C (weekly count)',
    help: "Number of days in the week the growing region hit 35°C or hotter (ERA5) — a standard heat-stress threshold for vegetable crops.",
    mechanism: 'More extreme-heat days signal crop stress and reduced expected supply, which tends to push prices {dir}.',
  },
  era5_heat_38: {
    label: 'Days ≥38°C (weekly count)',
    help: "Number of days in the week the growing region hit 38°C or hotter (ERA5) — a more severe heat-stress threshold than the 35°C count.",
    mechanism: 'Severe-heat days are a stronger stress signal than the 35°C count, tending to push prices {dir} via reduced supply.',
  },
  chirps_rain_max: {
    label: 'Peak Daily Rainfall (mm)',
    help: "The single wettest day's rainfall within the week (CHIRPS) — captures a flooding/waterlogging risk that a weekly total can hide.",
    mechanism: "A very wet single day raises flood/waterlogging risk even if the week's total rain looks moderate, which tends to push prices {dir}.",
  },
  chirps_excess: {
    label: 'Excess Rainfall Index',
    help: 'A derived measure (CHIRPS) of how far weekly rainfall exceeds a normal/expected level for the region and season.',
    mechanism: 'Rainfall well above the seasonal norm signals flood/logistics disruption risk, which tends to push prices {dir}.',
  },

  // ── Satellite (M4) ───────────────────────────────────────────────────
  s2_valid_frac: {
    label: 'Sentinel-2 Cloud-Free Fraction',
    help: "Fraction of the growing region with a usable (cloud-free) Sentinel-2 image that week (0-1). Low values mean the NDVI/EVI readings that week are based on thin coverage — a data-quality signal, not an agronomic one.",
    mechanism: 'This is a coverage/data-quality signal rather than a supply driver; the model may still have learned indirect associations with it, shown as {dir} here.',
  },
  s2_ndvi_anom: {
    label: 'NDVI Anomaly',
    help: "How far this week's Sentinel-2 vegetation index is from the region's normal level for this time of year — isolates an unusual season from a normal one better than raw NDVI alone.",
    mechanism: 'A negative anomaly (worse than normal for the season) signals below-normal expected supply, which tends to push prices {dir}.',
  },
  modis_ndvi: {
    label: 'MODIS Vegetation Index',
    help: "Crop health/density from MODIS satellite imagery — a second, coarser-resolution vegetation signal alongside Sentinel-2's NDVI, with more frequent (near-daily) revisits.",
    mechanism: 'Higher MODIS NDVI (healthier growing conditions) generally signals more supply ahead, which tends to push prices {dir}.',
  },
  modis_evi: {
    label: 'MODIS Enhanced Veg. Index',
    help: 'A vegetation index (MODIS) that corrects for canopy background and atmospheric effects better than plain NDVI, especially in denser vegetation.',
    mechanism: 'Higher EVI (healthier, denser vegetation) generally signals more supply ahead, which tends to push prices {dir}.',
  },
  modis_lst_mean: {
    label: 'MODIS Land Surface Temp. (mean)',
    help: "Average land surface temperature from MODIS thermal imagery — a satellite-based temperature signal distinct from ERA5's ground-station-based reanalysis.",
    mechanism: 'Higher land surface temperature is associated with crop heat stress, tending to push prices {dir} via reduced expected supply.',
  },
  modis_lst_max: {
    label: 'MODIS Land Surface Temp. (max)',
    help: 'Peak land surface temperature from MODIS that week — captures the hottest surface reading, not just the average.',
    mechanism: 'A higher peak surface temperature signals acute heat-stress risk, tending to push prices {dir}.',
  },
  modis_lst_frac35: {
    label: 'MODIS Land ≥35°C Fraction',
    help: 'Fraction of the growing region (0-1) whose MODIS-measured land surface hit 35°C or hotter that week.',
    mechanism: 'A larger heat-affected area signals broader crop stress, tending to push prices {dir}.',
  },

  // ── Macro (M2) ───────────────────────────────────────────────────────
  agri_wages_rs_day: {
    label: 'National Agri. Daily Wage (Rs)',
    help: "CMIE's all-India average agricultural daily wage — a national labour-cost benchmark distinct from the state-specific wage figures below.",
    mechanism: 'Higher farm labour costs raise the cost of producing and harvesting the crop, which tends to push prices {dir}.',
  },
  bank_credit_agri_cr: {
    label: 'Bank Credit to Agriculture (Rs Cr)',
    help: 'Outstanding scheduled-bank credit to the agriculture sector (CMIE, Rs crore) — a proxy for how much financing is available for farm inputs, storage, and trade.',
    mechanism: 'More credit available can support both planting/storage decisions in ways that push prices either direction — read the {dir} shown here as what the model learned for this specific change, not a fixed rule.',
  },
  crude_oil_usd_bbl: {
    label: 'Crude Oil (USD/barrel)',
    help: 'International crude oil price — the upstream driver behind diesel and LPG prices, which affect transport and cold-chain costs.',
    mechanism: 'Higher crude oil prices raise transport and cold-storage energy costs, tending to push prices {dir}.',
  },
  diesel_delhi_per_L: {
    label: 'Diesel Price — Delhi (Rs/L)',
    help: 'Diesel price in Delhi specifically (PPAC), alongside the 4-city average — captures local pump-price variation.',
    mechanism: 'Higher diesel prices raise transport costs to market, tending to push prices {dir}.',
  },
  export_veg_usd_mn: {
    label: 'Vegetable Exports (USD mn)',
    help: "India's total monthly vegetable export value (USD million) — a demand-pull signal: more exports mean less supply staying in the domestic market.",
    mechanism: 'Higher export volumes pull supply away from domestic markets, tending to push domestic prices {dir}.',
  },
  import_veg_usd_mn: {
    label: 'Vegetable Imports (USD mn)',
    help: "India's total monthly vegetable import value (USD million) — imported supply competing with domestic produce.",
    mechanism: 'Higher imports add to domestic supply, tending to push prices {dir}.',
  },
  iip_food_proc: {
    label: 'Food Processing IIP',
    help: 'Index of Industrial Production for food processing (CMIE) — a proxy for processing-sector demand (e.g. tomato going into paste/ketchup) pulling on the same raw supply as fresh markets.',
    mechanism: 'Higher processing-sector activity competes with fresh markets for the same raw crop, tending to push prices {dir}.',
  },
  lpg_nonsub_4city_rs_cyl: {
    label: 'LPG Price, 4-City Avg (Rs/cyl)',
    help: 'Non-subsidised LPG cylinder price averaged across 4 major cities (PPAC) — a cold-chain and local-market energy cost proxy.',
    mechanism: 'Higher LPG prices raise cold-chain and local operating costs, tending to push prices {dir}.',
  },
  lpg_nonsub_delhi_per14kg: {
    label: 'LPG Price — Delhi (Rs/14kg)',
    help: 'Non-subsidised LPG price in Delhi specifically (PPAC), alongside the 4-city average.',
    mechanism: 'Higher LPG prices raise cold-chain and local operating costs, tending to push prices {dir}.',
  },
  reverse_repo_pct: {
    label: 'RBI Reverse Repo Rate (%)',
    help: 'The rate at which RBI absorbs liquidity from banks — moves alongside the repo rate as part of the same monetary-policy stance affecting credit costs.',
    mechanism: 'Same broad channel as the repo rate — a higher reverse repo rate is part of a tighter monetary stance, tending to push prices {dir} via the cost of holding inventory.',
  },
  wpi_fruits_vegetables: {
    label: 'WPI: Fruits & Vegetables',
    help: 'Wholesale Price Index for the fruits-and-vegetables group (RBI/CMIE) — a broad wholesale-inflation signal for the whole category this crop sits in.',
    mechanism: "This is itself a price index for the crop's broader category, so it tends to move together with this crop's own price ({dir} shown here is the isolated model effect).",
  },
  wpi_vegetables_total: {
    label: 'WPI: Vegetables (Total)',
    help: 'Wholesale Price Index for vegetables overall (RBI/CMIE) — narrower than the fruits-and-vegetables group above.',
    mechanism: "Another close relative of this crop's own price series; {dir} shown here is the isolated model effect net of the other WPI/price features.",
  },
  wpi_potato: {
    label: 'WPI: Potato',
    help: 'Wholesale Price Index specifically for potato (RBI/CMIE) — relevant as a cross-crop signal even when simulating tomato or onion (substitution/basket effects).',
    mechanism: "Potato's own wholesale index can pick up shared seasonal or macro effects across the vegetable basket; {dir} shown here is the isolated model effect.",
  },
  wpi_onion: {
    label: 'WPI: Onion',
    help: 'Wholesale Price Index specifically for onion (RBI/CMIE) — relevant as a cross-crop signal even when simulating tomato or potato.',
    mechanism: "Onion's own wholesale index can pick up shared seasonal or macro effects across the vegetable basket; {dir} shown here is the isolated model effect.",
  },
  wpi_tomato: {
    label: 'WPI: Tomato',
    help: 'Wholesale Price Index specifically for tomato (RBI/CMIE) — relevant as a cross-crop signal even when simulating onion or potato.',
    mechanism: "Tomato's own wholesale index can pick up shared seasonal or macro effects across the vegetable basket; {dir} shown here is the isolated model effect.",
  },

  // ── Infrastructure (M5) ──────────────────────────────────────────────
  wage_agri_men: {
    label: 'Agri. Wage — Men (Rs/day)',
    help: 'State-level average daily wage for male agricultural labour (Labour Bureau/CMIE) — a direct farm-labour cost input.',
    mechanism: 'Higher labour costs raise the cost of producing and harvesting the crop, which tends to push prices {dir}.',
  },
  wage_agri_women: {
    label: 'Agri. Wage — Women (Rs/day)',
    help: 'State-level average daily wage for female agricultural labour (Labour Bureau/CMIE) — a direct farm-labour cost input.',
    mechanism: 'Higher labour costs raise the cost of producing and harvesting the crop, which tends to push prices {dir}.',
  },
  cold_storage_n_facilities: {
    label: 'Cold Storage Facilities (count)',
    help: 'Number of registered cold-storage facilities in this state — more facilities mean more capacity to hold produce off the market and smooth out price swings.',
    mechanism: 'More cold-storage facilities give traders more ability to hold stock rather than dump it immediately, which tends to push prices {dir} by reducing forced-sale pressure.',
  },
  cold_storage_capacity_mt: {
    label: 'Cold Storage Capacity (MT)',
    help: 'Total registered cold-storage capacity (metric tonnes) in this state — this is a "what if this state had more/less storage capacity" policy-investment lever, not a week-to-week variable.',
    mechanism: 'More storage capacity lets traders buffer supply across time instead of selling immediately, which tends to push prices {dir} by smoothing out short-term gluts.',
  },
  road_density_per_100_sqkm: {
    label: 'Road Density (km per 100 km²)',
    help: 'State road network density (CEIC/MORTH) — a proxy for how easily produce moves from farm to market. Like cold storage, this is a slow-moving infrastructure lever, not a weekly variable.',
    mechanism: 'Denser road networks lower the effective cost and delay of getting produce to market, which tends to push prices {dir}.',
  },
  s2_evi: {
    label: 'Sentinel-2 Enhanced Veg. Index',
    help: 'A vegetation index (Sentinel-2) that corrects for canopy background and atmospheric effects better than plain NDVI, especially in denser vegetation.',
    mechanism: 'Higher EVI (healthier, denser vegetation) generally signals more supply ahead, which tends to push prices {dir}.',
  },
};

export const POLICY_FIELDS = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct', 'market_intervention_flag', 'operation_greens_active'];
export const CLIMATE_FIELDS = [
  'era5_tmax', 'chirps_rain_mm', 's2_ndvi',
  'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38',
  'chirps_rain_max', 'chirps_excess', 's2_valid_frac', 's2_ndvi_anom',
  'modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35',
  's2_evi',
  // 's2_evi' was excluded here 2026-09-21 (~50% of its source values were
  // wildly implausible, up to ~1.2 billion, a divide-by-near-zero blowup in
  // the raw per-pixel EVI formula) and re-added the same day after the fix
  // landed at both layers -- see scripts/14_Satellite_Climate_Features.py's
  // sanity guard and scripts/23_Train_Production_Models.py's SIMULATABLE
  // comment for the full account. Verified range now roughly [-1.4, 1.5].
];
export const MACRO_FIELDS = [
  'diesel_4city_rs_litre', 'repo_rate_pct', 'usdinr_monthly_avg',
  'agri_wages_rs_day', 'bank_credit_agri_cr', 'crude_oil_usd_bbl',
  'diesel_delhi_per_L', 'export_veg_usd_mn', 'import_veg_usd_mn',
  'iip_food_proc', 'lpg_nonsub_4city_rs_cyl', 'lpg_nonsub_delhi_per14kg',
  'reverse_repo_pct', 'wpi_fruits_vegetables', 'wpi_vegetables_total',
  'wpi_potato', 'wpi_onion', 'wpi_tomato',
];
// M5 -- had NO presence anywhere in this app before 2026-09-21 despite being
// real model inputs (mirrors the same gap found and fixed in the Streamlit
// dashboard, scripts/24_Simulation_Dashboard.py, commit 0b759e5).
export const INFRA_FIELDS = [
  'wage_agri_men', 'wage_agri_women',
  'cold_storage_n_facilities', 'cold_storage_capacity_mt', 'road_density_per_100_sqkm',
];
