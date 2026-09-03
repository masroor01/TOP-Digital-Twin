import React, { useEffect, useState } from 'react';
import { Card, SectionLabel, Spinner, Alert, InfoButton, FadeIn } from '../ui';
import PlotChart from '../PlotChart';
import { api } from '../../lib/api';
import { CROP_ICON, CROP_COLOR, HORIZONS } from '../../lib/theme';

const LAYER_COLOR = {
  Price: '#0F172A',
  Arrivals: '#0891B2',
  Macro: '#7C3AED',
  Climate: '#16A34A',
  Satellite: '#65A30D',
  Infrastructure: '#D97706',
  Policy: '#DC2626',
};
const LAYER_ORDER = ['Price', 'Arrivals', 'Macro', 'Climate', 'Satellite', 'Infrastructure', 'Policy'];

export default function FeatureImportanceTab({ crop, horizon }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.shap().then(setData);
  }, []);

  if (!data) return <Spinner label="Loading feature-importance results…" />;

  const { layers, features } = data;
  const cropLabel = crop.charAt(0).toUpperCase() + crop.slice(1);
  const layersForCrop = layers.filter((r) => r.crop === crop);
  const topFeatures = features
    .filter((r) => r.crop === crop && Number(r.horizon_weeks) === horizon)
    .slice()
    .sort((a, b) => Number(a.rank) - Number(b.rank))
    .slice(0, 12)
    .reverse(); // horizontal bar: rank 1 at top

  return (
    <div>
      <div className="flex items-center gap-2">
        <SectionLabel>Feature Importance (SHAP) — {CROP_ICON[crop]} {cropLabel}</SectionLabel>
        <InfoButton title="What is SHAP feature importance?">
          <p className="mb-2">
            SHAP (SHapley Additive exPlanations) measures how much each input feature actually pushed the model's
            prediction up or down, averaged across a sample of real training rows. It answers "what did the model
            actually lean on," as opposed to Attribution's "what happens if I change one thing right now."
          </p>
          <p className="mb-2">
            <b>Granularity — read this before drawing conclusions:</b> this analysis was computed once per
            <b> (crop, horizon)</b> pair, on a sample of that crop's training rows. There is one shared model per
            crop+horizon that serves every state and market, so there is no separate SHAP breakdown <i>per state or
            market</i> to show — the model itself doesn't vary by market, only by crop and horizon. A genuine
            per-market SHAP analysis would require retraining or re-explaining market-specific models, which hasn't
            been done. What's shown here is the real, available granularity: crop and horizon, both selectable from
            the sidebar.
          </p>
        </InfoButton>
      </div>

      <Alert tone="info">
        📊 <b>Granularity note:</b> feature importance below is computed per <b>crop + horizon</b>, matching the
        sidebar's Crop and Horizon selectors — it is not available broken down by state or market, because one
        shared model serves every market for a given crop and horizon.
      </Alert>

      <SectionLabel>How Importance Shifts Across Horizons</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-4 max-w-3xl">
        Each bar is 100% of the model's explained prediction for {cropLabel} at that horizon, split by data layer.
        Watch how <b>Price</b>'s share typically shrinks at longer horizons as recent price momentum becomes a
        weaker signal, while seasonal/macro/climate layers pick up more of the weight.
      </p>
      <Card className="p-4 mb-6">
        <PlotChart
          height={360}
          layout={{
            barmode: 'stack',
            xaxis: { title: { text: 'Horizon' } },
            yaxis: { title: { text: '% of model explanation' }, range: [0, 100] },
            legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
          }}
          data={LAYER_ORDER.map((layer) => ({
            type: 'bar',
            name: layer,
            x: HORIZONS.map((h) => `${h}W`),
            y: HORIZONS.map((h) => {
              const row = layersForCrop.find((r) => r.layer === layer && Number(r.horizon_weeks) === h);
              return row ? Number(row.pct_of_total) : 0;
            }),
            marker: { color: LAYER_COLOR[layer] },
          }))}
        />
      </Card>

      <SectionLabel>Top Individual Features — {horizon}W Horizon</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-4 max-w-3xl">
        The specific features the model relied on most at the currently-selected {horizon}-week horizon, colored by
        the data layer each belongs to.
      </p>
      <Card className="p-4">
        {topFeatures.length === 0 ? (
          <p className="text-sm text-[var(--text-secondary)] p-4">No SHAP data for this crop/horizon combination.</p>
        ) : (
          <PlotChart
            height={40 * topFeatures.length + 80}
            layout={{
              margin: { l: 180 },
              xaxis: { title: { text: '% of model explanation' } },
              yaxis: { automargin: true },
              showlegend: false,
            }}
            data={[{
              type: 'bar',
              orientation: 'h',
              x: topFeatures.map((f) => Number(f.pct_of_total)),
              y: topFeatures.map((f) => f.feature),
              marker: { color: topFeatures.map((f) => LAYER_COLOR[f.layer] || '#64748B') },
              text: topFeatures.map((f) => `${f.layer} · ${Number(f.pct_of_total).toFixed(1)}%`),
              textposition: 'outside',
            }]}
          />
        )}
      </Card>
    </div>
  );
}
