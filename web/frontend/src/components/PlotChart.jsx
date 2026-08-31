import React from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import PlotlyLib from 'plotly.js-dist-min';

const Plot = createPlotlyComponent(PlotlyLib);

export function themedLayout(overrides = {}) {
  return {
    template: 'plotly_white',
    font: { family: 'Inter, sans-serif', size: 11, color: '#5B6B60' },
    margin: { l: 50, r: 15, t: overrides.title ? 40 : 20, b: 40 },
    hoverlabel: { bgcolor: '#0F2D20', font: { size: 11, family: 'Inter, sans-serif', color: '#FFFFFF' }, bordercolor: '#1B4332' },
    xaxis: { showgrid: true, gridcolor: '#EEF0EA', zeroline: false, tickfont: { size: 10, color: '#5B6B60' } },
    yaxis: { showgrid: true, gridcolor: '#EEF0EA', zeroline: false, tickfont: { size: 10, color: '#5B6B60' } },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#E3E6DE', borderwidth: 1, font: { size: 10, color: '#5B6B60' } },
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
