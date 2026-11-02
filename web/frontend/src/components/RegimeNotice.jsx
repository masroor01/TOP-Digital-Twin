import React from 'react';
import { Alert } from './ui';

const pct = (v, d = 0) => `${v >= 0 ? '+' : '−'}${Math.abs(100 * v).toFixed(d)}%`;

// Shown when the market's price is far from its usual level (median of the last 52 weeks). The forecasting model has learned that such prices move back
// towards the usual level; this notice shows what actually followed similar situations in the backtest (scripts/88_Regime_Outcomes_and_Bands.py).
export default function RegimeNotice({ regime, horizon }) {
  if (!regime || !regime.notable) return null;
  const rec = regime.byHorizon?.[horizon];
  if (!rec) return null;
  const high = regime.level >= 1;
  const a = rec.actualChange;
  const years = rec.years?.length ? `${rec.years[0]} to ${rec.years[rec.years.length - 1]}` : '';
  return (
    <Alert tone="warning">
      <b>{high ? 'Unusually high price.' : 'Unusually low price.'}</b>{' '}
      This market's price (₹{Math.round(regime.price).toLocaleString()} per quintal) is <b>{regime.level.toFixed(1)} times</b> its usual level
      (the median of the last 52 weeks, ₹{Math.round(regime.med52).toLocaleString()}). The forecast {high ? 'expects the price to fall back' : 'expects the price to recover'} towards
      that level{rec.forecastChange != null ? <> (<b>{pct(rec.forecastChange)}</b> at {horizon} weeks)</> : null}.
      After {rec.n.toLocaleString()} similar past forecasts ({rec.originWeeks} weeks, {years}), the price {horizon} weeks later had changed by a median of{' '}
      <b>{pct(a.q50)}</b> (middle half {pct(a.q25)} to {pct(a.q75)}; 10th to 90th percentile {pct(a.q10)} to {pct(a.q90)}):
      {' '}{Math.round(100 * (high ? rec.shareFall25 : rec.shareRise25))}% moved by 25% or more in the expected direction and{' '}
      {Math.round(100 * (high ? rec.shareRise25 : rec.shareFall25))}% moved by 25% or more the other way.
      These cases come from only a few price cycles, and a price that is already far from its usual level can keep moving away from it, so read the forecast
      together with the range shown on the chart.
    </Alert>
  );
}
