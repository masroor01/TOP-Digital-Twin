import React, { useEffect, useMemo, useState } from 'react';
import { Card, SectionLabel, Spinner, Alert } from '../ui';
import PlotChart from '../PlotChart';
import { api } from '../../lib/api';

const PALETTE = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#A855F7', '#06B6D4', '#EC4899', '#84CC16'];

export default function MultiMarketTab({ crop, markets }) {
  const [selected, setSelected] = useState([]);
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { setSelected([]); setSeries(null); }, [crop]);

  const options = useMemo(() => markets.slice(0, 200), [markets]);

  function toggle(id) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 8) return prev;
      return [...prev, id];
    });
  }

  async function compare() {
    if (!selected.length) return;
    setLoading(true);
    try {
      const s = await api.multiHistory(crop, selected);
      setSeries(s);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <SectionLabel>Multi-Market Price Comparison</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-3">Select up to 8 markets to overlay their historical weekly price trajectories.</p>

      <Card className="p-3 mb-4">
        <div className="max-h-40 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1.5 mb-3">
          {options.map((m) => (
            <label key={m.marketId} className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] cursor-pointer">
              <input type="checkbox" checked={selected.includes(m.marketId)} onChange={() => toggle(m.marketId)} className="accent-[var(--brand)]" />
              <span className="truncate">{m.market} ({m.state})</span>
            </label>
          ))}
        </div>
        <button onClick={compare} disabled={!selected.length || loading}
          className="text-sm font-semibold rounded-lg border border-[var(--border-color-strong)] bg-white px-4 py-2 hover:bg-[var(--brand)] hover:text-white hover:border-[var(--brand)] transition disabled:opacity-40 disabled:pointer-events-none">
          {loading ? 'Loading…' : `Compare ${selected.length} Market${selected.length === 1 ? '' : 's'}`}
        </button>
      </Card>

      {loading && <Spinner label="Fetching multi-market history…" />}
      {!loading && series && series.length > 0 && (
        <Card className="p-3">
          <PlotChart height={460} layout={{ xaxis: { title: { text: 'Date' } }, yaxis: { title: { text: 'Price (Rs/quintal)' } }, hovermode: 'x unified' }}
            data={series.map((s, i) => ({
              x: s.points.map((p) => p.weekStart), y: s.points.map((p) => p.price),
              type: 'scatter', mode: 'lines', name: `${s.market} (${s.state})`,
              line: { color: PALETTE[i % PALETTE.length], width: 1.8 },
            }))} />
        </Card>
      )}
      {!loading && series && series.length === 0 && <Alert tone="info">No data for the selected markets.</Alert>}
    </div>
  );
}
