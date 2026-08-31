import React from 'react';
import { Card, SectionLabel, Alert, Spinner } from '../ui';
import PlotChart from '../PlotChart';
import { fmtRs } from '../../lib/theme';

export default function AttributionTab({ sim }) {
  if (!sim) return <Spinner label="Running simulation…" />;
  const { isolatedEffects, sumIsolated, interactionGap, kpis } = sim;

  if (!isolatedEffects.length) {
    return <Alert tone="info">No scenario modifiers are active. Adjust a policy, climate, or macro field in the sidebar to see its isolated price effect here.</Alert>;
  }

  const sorted = [...isolatedEffects].sort((a, b) => a.effect - b.effect);
  const colors = sorted.map((e) => (e.effect >= 0 ? '#EF4444' : '#10B981'));

  return (
    <div>
      <SectionLabel>Isolated Feature Effects</SectionLabel>
      <p className="text-sm text-slate-500 mb-3 max-w-3xl">
        Each bar holds every other field at its baseline value and changes only that one field — isolating its individual
        contribution to the price move. The sum of isolated effects vs. the actual combined delta reveals interaction effects
        between variables.
      </p>
      <Card className="p-3 mb-4">
        <PlotChart height={Math.max(240, sorted.length * 52)} layout={{
          xaxis: { title: { text: 'Isolated effect on price (Rs/quintal)' } },
          yaxis: { automargin: true },
        }} data={[{
          type: 'bar', orientation: 'h',
          x: sorted.map((e) => e.effect),
          y: sorted.map((e) => e.label),
          marker: { color: colors },
          text: sorted.map((e) => `${e.effect >= 0 ? '+' : ''}${Math.round(e.effect).toLocaleString()}`),
          textposition: 'outside',
          hovertemplate: '%{y}<br>%{x:,.0f} Rs/quintal<extra></extra>',
        }]} />
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <Card className="p-4">
          <p className="text-[0.72rem] font-semibold text-slate-500 uppercase tracking-wide mb-1">Sum of Isolated Effects</p>
          <p className="text-xl font-bold text-slate-900">{fmtRs(sumIsolated)}</p>
        </Card>
        <Card className="p-4">
          <p className="text-[0.72rem] font-semibold text-slate-500 uppercase tracking-wide mb-1">Actual Combined Delta</p>
          <p className="text-xl font-bold text-slate-900">{fmtRs(kpis.delta)}</p>
        </Card>
        <Card className="p-4">
          <p className="text-[0.72rem] font-semibold text-slate-500 uppercase tracking-wide mb-1">Interaction Gap</p>
          <p className={`text-xl font-bold ${Math.abs(interactionGap) > 1 ? 'text-amber-600' : 'text-slate-900'}`}>{fmtRs(interactionGap)}</p>
        </Card>
      </div>

      <Card className="p-4 overflow-x-auto">
        <table className="w-full text-sm min-w-[640px]">
          <thead>
            <tr className="text-left text-slate-500 text-xs uppercase tracking-wide border-b border-slate-200">
              <th className="py-2 pr-3">Field</th>
              <th className="py-2 pr-3">Before</th>
              <th className="py-2 pr-3">After</th>
              <th className="py-2 pr-3">Isolated Effect</th>
              <th className="py-2">Mechanism</th>
            </tr>
          </thead>
          <tbody>
            {isolatedEffects.map((e) => (
              <tr key={e.field} className="border-b border-slate-100 last:border-0">
                <td className="py-2 pr-3 font-semibold text-slate-800">{e.label}</td>
                <td className="py-2 pr-3 font-mono text-slate-600">{String(e.before)}</td>
                <td className="py-2 pr-3 font-mono text-slate-600">{String(e.after)}</td>
                <td className={`py-2 pr-3 font-mono font-semibold ${e.effect >= 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                  {e.effect >= 0 ? '+' : ''}{Math.round(e.effect).toLocaleString()}
                </td>
                <td className="py-2 text-slate-500 text-xs max-w-xs">{e.mechanism}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
