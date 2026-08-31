import React, { useEffect, useState } from 'react';
import { Card, SectionLabel, Spinner } from '../ui';
import PlotChart from '../PlotChart';
import { api } from '../../lib/api';
import { CROP_COLOR } from '../../lib/theme';

export default function BenchmarksTab({ crop }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    setData(null);
    api.benchmarks(crop).then(setData);
  }, [crop]);

  if (!data) return <Spinner label="Loading benchmarks…" />;

  const color = CROP_COLOR[crop];

  return (
    <div>
      <SectionLabel>Cross-Market Benchmarks</SectionLabel>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="p-3">
          <p className="text-sm font-semibold text-slate-700 mb-2 px-1">Top 15 Markets by Last Observed Price</p>
          <PlotChart height={460} layout={{ xaxis: { title: { text: 'Rs/quintal' } }, yaxis: { automargin: true } }} data={[{
            type: 'bar', orientation: 'h',
            x: data.priceTop.map((r) => r.value),
            y: data.priceTop.map((r) => r.label),
            marker: { color },
            hovertemplate: '%{y}<br>₹%{x:,.0f}/quintal<extra></extra>',
          }]} />
        </Card>
        <Card className="p-3">
          <p className="text-sm font-semibold text-slate-700 mb-2 px-1">Top 15 Markets by Arrivals (Tonnes)</p>
          <PlotChart height={460} layout={{ xaxis: { title: { text: 'Tonnes' } }, yaxis: { automargin: true } }} data={[{
            type: 'bar', orientation: 'h',
            x: data.arrTop.map((r) => r.value),
            y: data.arrTop.map((r) => r.label),
            marker: { color: '#64748B' },
            hovertemplate: '%{y}<br>%{x:,.0f} t<extra></extra>',
          }]} />
        </Card>
      </div>
    </div>
  );
}
