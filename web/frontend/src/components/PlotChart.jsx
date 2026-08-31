import React from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import PlotlyLib from 'plotly.js-dist-min';

const Plot = createPlotlyComponent(PlotlyLib);

export function themedLayout(overrides = {}) {
  return {
    template: 'plotly_white',
    font: { family: 'Inter, sans-serif', size: 11, color: '#475569' },
    margin: { l: 50, r: 15, t: overrides.title ? 40 : 20, b: 40 },
    hoverlabel: { bgcolor: '#0F172A', font: { size: 11, family: 'Inter, sans-serif', color: '#FFFFFF' }, bordercolor: '#1E293B' },
    xaxis: { showgrid: true, gridcolor: '#F1F5F9', zeroline: false, tickfont: { size: 10, color: '#64748B' } },
    yaxis: { showgrid: true, gridcolor: '#F1F5F9', zeroline: false, tickfont: { size: 10, color: '#64748B' } },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#E2E8F0', borderwidth: 1, font: { size: 10, color: '#475569' } },
    plot_bgcolor: '#FFFFFF',
    paper_bgcolor: '#FFFFFF',
    autosize: true,
    ...overrides,
  };
}

export default function PlotChart({ data, layout, height = 420, className = '' }) {
  return (
    <div className={className}>
      <Plot
        data={data}
        layout={{ ...themedLayout(layout), height }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
