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
};

export const POLICY_FIELDS = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct', 'market_intervention_flag'];
export const CLIMATE_FIELDS = ['era5_tmax', 'chirps_rain_mm', 's2_ndvi'];
export const MACRO_FIELDS = ['diesel_4city_rs_litre', 'repo_rate_pct', 'usdinr_monthly_avg'];
