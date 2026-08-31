import React, { useState } from 'react';
import { Card, SectionLabel, Spinner, Alert } from '../ui';
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
        crop, market: market?.market, state: market?.state, horizon,
        baselinePred: sim.kpis.baseline, scenarioPred: sim.kpis.scenario,
        deltaPct: sim.kpis.deltaPct, delta: sim.kpis.delta,
        rmse: sim.kpis.rmse, mape: sim.kpis.mape,
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
      <SectionLabel>AI Policy Briefing</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-3 max-w-3xl">
        Generates a grounded, plain-language commentary on the current scenario — strictly based on the baseline/scenario
        numbers and isolated feature effects computed above. Not a validated forecast; treat as decision-support only.
      </p>
      <button onClick={generate} disabled={loading}
        className="text-sm font-semibold rounded-lg border border-[var(--border-color-strong)] bg-white px-4 py-2 hover:bg-[var(--brand)] hover:text-white hover:border-[var(--brand)] transition disabled:opacity-40 mb-4">
        {loading ? 'Generating…' : '✨ Generate Briefing'}
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
