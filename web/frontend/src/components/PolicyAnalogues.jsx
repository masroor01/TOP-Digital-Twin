import React, { useEffect, useState } from 'react';
import { Card, Alert } from './ui';
import PlotChart from './PlotChart';
import { api } from '../lib/api';
import { fmtPct } from '../lib/theme';

const KIND_LABEL = { ban: 'Export ban', mep: 'Minimum export price', duty: 'Export duty' };
const COLORS = ['#B45309', '#7C3AED', '#0369A1', '#BE123C', '#15803D'];

// Which export controls differ from the baseline the user is looking at.
function activeKinds(overrides, baseRow) {
  const on = (f) => overrides[f] !== undefined && Number(overrides[f]) > Number(baseRow?.[f] ?? 0);
  return [on('export_banned') && 'ban', on('mep_usd_per_tonne') && 'mep', on('export_duty_pct') && 'duty'].filter(Boolean);
}

// The forecasting models do not respond to export-policy inputs (see scripts/83_Policy_Analogues.py), so a switched-on export
// control shows what onion prices actually did around the real dated restrictive actions instead of a model effect.
export default function PolicyAnalogues({ crop, overrides, baseRow }) {
  const [data, setData] = useState(null);
  useEffect(() => { api.policyAnalogues().then(setData).catch(() => setData(null)); }, []);

  if (crop !== 'onion') return null;
  const kinds = activeKinds(overrides, baseRow);
  if (!kinds.length) return null;
  if (!data) return null;

  const eps = data.episodes.filter((e) => kinds.includes(e.kind));
  const traces = eps.map((e, i) => ({
    x: e.weeks, y: e.median, type: 'scatter', mode: 'lines', name: `${e.date} ${KIND_LABEL[e.kind].toLowerCase()}`,
    line: { color: COLORS[i % COLORS.length], width: 2 },
    hovertemplate: `${e.date}<br>week %{x}<br>index %{y:.0f}<extra></extra>`,
  }));
  traces.push({ x: [0, 0], y: [0, 400], type: 'scatter', mode: 'lines', showlegend: false, hoverinfo: 'skip',
    line: { color: '#6B7280', width: 1, dash: 'dot' } });

  return (
    <Card className="p-4 mb-4" accent="#B45309">
      <p className="font-display text-sm font-bold text-[var(--text-primary)] mb-1">
        Historical analogues — {kinds.map((k) => KIND_LABEL[k].toLowerCase()).join(', ')}
      </p>
      <Alert tone="warning">
        <b>This is not a model forecast.</b> The forecasting model does not use export-policy inputs, so the scenario price above has not
        changed. The chart shows what onion prices actually did around real past actions. Policy was set after prices had already
        risen, the duty, minimum price and ban came in one escalating sequence in 2023, and the price effect of onion export policy
        is not identified in this project, so read these as history, not as what would happen now.
      </Alert>
      <PlotChart data={traces} height={320} layout={{
        xaxis: { title: { text: 'Weeks from the action (0 = week of the action)' } },
        yaxis: { title: { text: 'Price index (action week = 100)' }, range: [0, Math.min(400, Math.max(...eps.flatMap((e) => e.median)) * 1.15)] },
        hovermode: 'x unified',
      }} />
      <div className="overflow-x-auto mt-2">
        <table className="w-full text-xs text-[var(--text-secondary)]">
          <thead>
            <tr className="text-left text-[var(--text-primary)]">
              <th className="py-1 pr-3">Action</th><th className="pr-3">Rise in prior 13 wk</th><th className="pr-3">After 4 wk</th>
              <th className="pr-3">After 13 wk</th><th className="pr-3">After 26 wk</th><th>Markets</th>
            </tr>
          </thead>
          <tbody>
            {eps.map((e) => (
              <tr key={e.id} className="border-t border-[var(--border-color)]">
                <td className="py-1 pr-3">{e.date} · {KIND_LABEL[e.kind]}</td>
                <td className="pr-3">{fmtPct(e.runUpPct)}</td>
                <td className="pr-3">{fmtPct(e.afterPct['4'])}</td>
                <td className="pr-3">{fmtPct(e.afterPct['13'])}</td>
                <td className="pr-3">{fmtPct(e.afterPct['26'])}</td>
                <td>{e.nMarkets}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[0.7rem] text-[var(--text-secondary)] mt-2">
        Median across markets of each market's price indexed to its own price in the action week; markets with a price in that week and
        in at least 80% of the surrounding 39 weeks. Source: verified policy log and Agmarknet prices (Script 83).
      </p>
    </Card>
  );
}
