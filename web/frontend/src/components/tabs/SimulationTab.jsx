import React, { useState } from 'react';
import { Card, SectionLabel, Metric, Badge, Spinner } from '../ui';
import PlotChart from '../PlotChart';
import { fmtRs, fmtPct, fmtDate, HORIZONS } from '../../lib/theme';
import { api } from '../../lib/api';

const SEASON_LABEL = {
  peak_arrival: 'Peak Arrival', lean: 'Lean Season', kharif: 'Kharif Season',
  rabi_arrival: 'Rabi Arrival', harvest: 'Harvest Season', storage: 'Storage Period',
};

export default function SimulationTab({ sim, crop, market, marketId, overrides }) {
  const [dailyOpen, setDailyOpen] = useState(false);
  const [daily, setDaily] = useState(null);
  const [dailyLoading, setDailyLoading] = useState(false);

  if (!sim) return <Spinner label="Running simulation…" />;

  const { ticker, kpis, chart } = sim;

  async function toggleDaily() {
    if (!dailyOpen && !daily) {
      setDailyLoading(true);
      try {
        const d = await api.dailyCurve({ crop, marketId, overrides });
        setDaily(d);
      } finally {
        setDailyLoading(false);
      }
    }
    setDailyOpen((v) => !v);
  }

  const mainTraces = [
    {
      x: chart.history.map((h) => h.weekStart), y: chart.history.map((h) => h.price),
      type: 'scatter', mode: 'lines', name: 'Historical Observed Price', line: { color: '#334155', width: 2 },
    },
    {
      x: [...chart.dates, ...chart.dates.slice().reverse()],
      y: [...chart.bandUpper, ...chart.bandLower.slice().reverse()],
      fill: 'toself', fillcolor: 'rgba(148,163,184,0.16)', line: { width: 0 },
      name: '±RMSE Uncertainty Range', hoverinfo: 'skip', type: 'scatter',
    },
    {
      x: chart.dates, y: chart.baselineCurve, type: 'scatter', mode: 'lines+markers', name: 'Baseline Forecast',
      line: { color: '#64748B', width: 2.4, dash: 'dot' }, marker: { size: 6, color: '#64748B' },
    },
    {
      x: chart.dates, y: chart.scenarioCurve, type: 'scatter', mode: 'lines+markers', name: 'Scenario Forecast',
      line: { color: '#EF4444', width: 2.4, dash: 'dash' }, marker: { size: 6, color: '#EF4444' },
    },
    {
      x: [sim.targetDate], y: [kpis.scenario], type: 'scatter', mode: 'markers', name: `Target`,
      marker: { size: 13, symbol: 'star', color: '#F59E0B', line: { width: 1.5, color: '#0F172A' } },
      hovertemplate: `Target<br>%{x}<br>₹%{y:,.0f}<extra></extra>`,
    },
  ];

  const deltaTone = kpis.delta > 0 ? 'up' : kpis.delta < 0 ? 'down' : null;

  return (
    <div>
      <SectionLabel>Multi-Horizon Baseline Forecasts</SectionLabel>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {ticker.map((t) => (
          <div key={t.horizon}>
            <Metric label={`Horizon ${t.horizon}W · ${fmtDate(t.date)}`} value={fmtRs(t.price)}
              spark={t.spark} help={t.rmse ? `±₹${Math.round(t.rmse).toLocaleString()} (${t.mape.toFixed(0)}% MAPE)` : ''} />
            {t.season && (
              <div className="text-center -mt-1.5">
                <Badge>🌾 {SEASON_LABEL[t.season] || t.season}</Badge>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <Metric label={`Baseline`} value={fmtRs(kpis.baseline, { perQ: true })} help="Unmodified baseline model projection." />
        <Metric label={`Scenario`} value={fmtRs(kpis.scenario, { perQ: true })}
          delta={`${fmtRs(kpis.delta)} (${fmtPct(kpis.deltaPct)})`} deltaTone={deltaTone}
          help="Model projection with active scenario modifier inputs." />
        <Metric label="Last Real Trade"
          value={kpis.lastObservedPrice != null ? fmtRs(kpis.lastObservedPrice, { perQ: true }) : 'N/A'}
          help="Most recent non-imputed trade price." />
        <Metric label="Model Accuracy" value={kpis.mape != null ? `~${Math.max(0, 100 - kpis.mape).toFixed(0)}%` : 'N/A'}
          help={kpis.mape != null ? `100% - MAPE (${kpis.mape.toFixed(0)}%) from backtesting.` : ''} />
      </div>

      <Card className="p-3">
        <PlotChart data={mainTraces} height={440} layout={{
          xaxis: { title: { text: 'Date' } }, yaxis: { title: { text: 'Price (Rs/quintal)' } }, hovermode: 'x unified',
          shapes: [
            { type: 'line', x0: sim.asOf, x1: sim.asOf, yref: 'paper', y0: 0, y1: 1, line: { dash: 'dot', color: '#94A3B8' } },
            { type: 'line', x0: sim.targetDate, x1: sim.targetDate, yref: 'paper', y0: 0, y1: 1, line: { dash: 'dash', color: '#F59E0B' } },
          ],
        }} />
      </Card>

      <div className="mt-3">
        <button onClick={toggleDaily} className="text-sm font-semibold rounded-lg border border-slate-300 bg-white px-4 py-2 hover:bg-slate-900 hover:text-white hover:border-slate-900 transition">
          {dailyOpen ? '🔼 Hide Daily Disaggregation' : '📅 Expand Daily Disaggregation Curve'}
        </button>
      </div>

      {dailyOpen && (
        <Card className="p-4 mt-3">
          {dailyLoading && <Spinner label="Interpolating daily curve…" />}
          {!dailyLoading && daily && daily.points.length === 0 && (
            <p className="text-sm text-slate-500">No daily curve available (today is beyond this market's horizon, or the crop lacks a daily-noise factor).</p>
          )}
          {!dailyLoading && daily && daily.points.length > 0 && (
            <>
              <div className="marquee-wrap overflow-hidden whitespace-nowrap border border-slate-200 rounded-lg py-2 mb-3 bg-slate-50">
                <div className="marquee-track text-xs text-slate-700" style={{ animationDuration: `${Math.max(80, daily.points.length * 1.4)}s` }}>
                  {daily.points.map((p) => (
                    <span key={p.date} className="bg-white px-2 py-0.5 rounded-md border border-slate-200 mr-4">
                      {fmtDate(p.date)}: <b>{fmtRs(p.price)}</b>
                    </span>
                  ))}
                </div>
              </div>
              <PlotChart height={360} layout={{ title: { text: 'Smooth Daily Trajectory & Volatility' } }} data={[
                {
                  x: [...daily.points.map((p) => p.date), ...daily.points.map((p) => p.date).reverse()],
                  y: [...daily.points.map((p) => p.price + p.band), ...daily.points.map((p) => p.price - p.band).reverse()],
                  fill: 'toself', fillcolor: 'rgba(16,185,129,0.10)', line: { width: 0 }, name: 'Daily Noise Band (±1 SD)', hoverinfo: 'skip', type: 'scatter',
                },
                {
                  x: daily.points.map((p) => p.date), y: daily.points.map((p) => p.price),
                  type: 'scatter', mode: 'lines', line: { color: '#059669', width: 2 }, name: 'PCHIP Daily Trend',
                },
              ]} />
            </>
          )}
        </Card>
      )}
    </div>
  );
}
