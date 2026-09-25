import React, { useState } from 'react';
import { Card, Spinner, Alert } from '../ui';
import { api } from '../../lib/api';

export default function AITab({ sim, crop, market, horizon }) {
  const [brief, setBrief] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!sim) return <Spinner label="Running simulation…" />;

  async function generate() {
    setLoading(true);
    setError(null);
    setBrief(null);
    try {
      const res = await api.aiBrief({
        crop, marketId: market?.marketId, horizon,
        baselinePred: sim.kpis.baseline, scenarioPred: sim.kpis.scenario,
        deltaPct: sim.kpis.deltaPct, delta: sim.kpis.delta,
        rmse: sim.kpis.rmse, wape: sim.kpis.wape,
        isolatedEffects: sim.isolatedEffects,
      });
      setBrief(res.text);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-emerald-800 to-emerald-950 text-white shadow-sm mb-4 max-w-3xl">
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 rounded-lg bg-white/20 flex items-center justify-center text-xl">
            🤖
            <span className="absolute bottom-0.5 right-0.5 w-2 h-2 rounded-full bg-emerald-400 border border-emerald-900"></span>
          </div>
          <div>
            <h3 className="font-bold text-base text-white tracking-tight leading-none">MIC Assistant</h3>
          </div>
        </div>
        <span className="text-xs font-semibold bg-white/15 px-2.5 py-1 rounded-md text-emerald-100">HADP-04</span>
      </div>

      <p className="text-sm text-[var(--text-secondary)] mb-3 max-w-3xl">
        Grounded, plain-language agricultural policy commentary on the active scenario — strictly based on baseline/scenario
        numbers and isolated feature effects. Powered by Anthropic Claude for decision support.
      </p>
      <button onClick={generate} disabled={loading}
        className="text-sm font-semibold rounded-lg border border-[var(--border-color-strong)] bg-[var(--card-bg)] px-4 py-2 hover:bg-[var(--brand)] hover:text-white hover:border-[var(--brand)] transition disabled:opacity-40 mb-4">
        {loading ? 'Consulting MIC Assistant…' : '✨ Ask MIC Assistant'}
      </button>

      {loading && <Spinner label="Calling Claude…" />}
      {error && <Alert tone="error">{error}</Alert>}
      {brief && (
        <Card className="p-5 max-w-3xl">
          <p className="text-sm leading-relaxed text-[var(--text-primary)]">{brief}</p>
        </Card>
      )}
    </div>
  );
}
