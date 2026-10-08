export const CROP_ICON = { tomato: '🍅', onion: '🧅', potato: '🥔' };
export const CROP_COLOR = { tomato: '#DC2626', onion: '#7C3AED', potato: '#B45309' };
export const HORIZONS = [1, 4, 13, 26];

export function fmtRs(v, opts = {}) {
  if (v == null || Number.isNaN(v)) return '—';
  return `₹ ${Math.round(v).toLocaleString('en-IN')}${opts.perQ ? ' / q' : ''}`;
}

// Price per kg from a price per quintal (1 quintal = 100 kg).
export function fmtRsKg(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return `₹ ${(v / 100).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} / kg`;
}

export function fmtDate(iso, opts = {}) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: opts.year ? 'numeric' : undefined });
}

export function fmtPct(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return '—';
  const s = v >= 0 ? '+' : '';
  return `${s}${v.toFixed(digits)}%`;
}
