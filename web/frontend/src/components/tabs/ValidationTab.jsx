import React, { useEffect, useState } from 'react';
import { Card, SectionLabel, Spinner, Alert, Badge, FadeIn } from '../ui';
import PlotChart from '../PlotChart';
import { api } from '../../lib/api';
import { CROP_ICON, CROP_COLOR } from '../../lib/theme';

const CROPS = ['tomato', 'onion', 'potato'];
const HORIZONS = [1, 4, 13, 26];
const M0_COLOR = '#94A3B8';

export default function ValidationTab() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.directionalAccuracy().then((d) => setData(d.rows));
  }, []);

  if (!data) return <Spinner label="Loading validation results…" />;

  return (
    <div>
      <SectionLabel>Directional Accuracy — Did the Model Call the Right Direction?</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-4 max-w-3xl">
        Every other metric in this dashboard (RMSE, WAPE, Model Accuracy) measures how close the predicted price
        <em> level</em> was. This measures something different: for every per-market forecast, did the model correctly
        call whether the price would go <b>up</b> or <b>down</b> from where it stood when the forecast was made —
        regardless of by how much? Tested against a 50% coin-flip null with a binomial test (all cells below are
        statistically significant given large per-market sample sizes).
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {CROPS.map((crop, i) => (
          <FadeIn key={crop} delay={0.05 * i}>
            <Card className="p-3" accent={CROP_COLOR[crop]}>
              <p className="text-sm font-semibold text-[var(--text-primary)] mb-2 px-1 flex items-center gap-1.5">
                <span>{CROP_ICON[crop]}</span> {crop.charAt(0).toUpperCase() + crop.slice(1)}
              </p>
              <PlotChart
                height={320}
                layout={{
                  xaxis: { title: { text: 'Horizon' } },
                  yaxis: { title: { text: 'Accuracy (%)' }, range: [0, 100] },
                  shapes: [{ type: 'line', x0: -0.5, x1: 3.5, y0: 50, y1: 50, line: { dash: 'dash', color: '#B45309', width: 1 } }],
                  barmode: 'group',
                  legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
                }}
                data={['M0', 'M6'].map((variant) => ({
                  type: 'bar',
                  name: variant === 'M0' ? 'M0 (price-only)' : 'M6 (full model)',
                  x: HORIZONS.map((h) => `${h}W`),
                  y: HORIZONS.map((h) => {
                    const row = data.find((r) => r.crop === crop && r.variant === variant && Number(r.horizon_weeks) === h);
                    return row ? Number(row.directional_accuracy_pct) : null;
                  }),
                  marker: { color: variant === 'M0' ? M0_COLOR : CROP_COLOR[crop] },
                  text: HORIZONS.map((h) => {
                    const row = data.find((r) => r.crop === crop && r.variant === variant && Number(r.horizon_weeks) === h);
                    return row ? `${row.directional_accuracy_pct}%` : '';
                  }),
                  textposition: 'outside',
                }))}
              />
            </Card>
          </FadeIn>
        ))}
      </div>

      <Alert tone="info">
        📐 <b>Naive persistence scores 0.0% here for every crop and horizon</b> — by mathematical construction, not a
        weak result. It always predicts "no change from today," so it never actually calls a direction. Reported
        separately (crop-level, coarser than the per-market numbers above) — see the table below, not a fair
        apples-to-apples comparison with M0/M6.
      </Alert>

      <SectionLabel>Full Results</SectionLabel>
      <Card className="p-4 overflow-x-auto">
        <table className="w-full text-sm min-w-[640px]">
          <thead>
            <tr className="text-left text-[var(--text-secondary)] text-xs uppercase tracking-wide border-b border-[var(--border-color)]">
              <th className="py-2 pr-3">Crop</th>
              <th className="py-2 pr-3">Horizon</th>
              <th className="py-2 pr-3">Variant</th>
              <th className="py-2 pr-3">n (scored)</th>
              <th className="py-2 pr-3">Directional Accuracy</th>
              <th className="py-2">vs. 50% Null</th>
            </tr>
          </thead>
          <tbody>
            {data
              .slice()
              .sort((a, b) => a.crop.localeCompare(b.crop) || Number(a.horizon_weeks) - Number(b.horizon_weeks) || a.variant.localeCompare(b.variant))
              .map((r, i) => (
                <tr key={i} className="border-b border-[var(--border-color)] last:border-0">
                  <td className="py-2 pr-3 capitalize">{CROP_ICON[r.crop]} {r.crop}</td>
                  <td className="py-2 pr-3 font-mono">{r.horizon_weeks}W</td>
                  <td className="py-2 pr-3 font-mono">{r.variant}</td>
                  <td className="py-2 pr-3 font-mono text-[var(--text-secondary)]">{Number(r.n_scored).toLocaleString('en-IN')}</td>
                  <td className="py-2 pr-3 font-mono font-semibold">{r.directional_accuracy_pct}%</td>
                  <td className="py-2">
                    <Badge tone={r.significant_at_05 ? 'down' : 'neutral'}>
                      {r.significant_at_05 ? 'Significant (p<0.05)' : 'Not significant'}
                    </Badge>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
