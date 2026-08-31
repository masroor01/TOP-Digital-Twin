import React from 'react';
import { Card, SectionLabel, Alert, Spinner } from '../ui';

export default function AuditTab({ sim, overrides }) {
  if (!sim) return <Spinner label="Running simulation…" />;
  const { baseRow, diffCols } = sim;
  const cols = Object.keys(baseRow).sort();

  return (
    <div>
      <SectionLabel>Feature Vector Audit</SectionLabel>
      <p className="text-sm text-[var(--text-secondary)] mb-3">
        Full baseline feature vector for this crop/market, with any active scenario overrides highlighted. Use this to
        verify exactly what the model sees.
      </p>
      {diffCols.length === 0 && <Alert tone="info">No overrides active — showing unmodified baseline vector.</Alert>}
      <Card className="p-4">
        <div className="overflow-auto max-h-[600px]">
          <table className="w-full text-xs min-w-[480px]">
            <thead className="sticky top-0 bg-[var(--card-bg)]">
              <tr className="text-left text-[var(--text-secondary)] uppercase tracking-wide border-b border-[var(--border-color)]">
                <th className="py-1.5 pr-3">Field</th>
                <th className="py-1.5 pr-3">Baseline Value</th>
                <th className="py-1.5">Scenario Override</th>
              </tr>
            </thead>
            <tbody>
              {cols.map((c) => {
                const changed = diffCols.includes(c);
                return (
                  <tr key={c} className={`border-b border-[var(--border-color)] last:border-0 ${changed ? 'bg-amber-50' : ''}`}>
                    <td className="py-1 pr-3 font-mono text-[var(--text-secondary)]">{c}</td>
                    <td className="py-1 pr-3 font-mono text-[var(--text-primary)]">{String(baseRow[c])}</td>
                    <td className={`py-1 font-mono ${changed ? 'text-amber-700 font-semibold' : 'text-[var(--text-muted)]'}`}>
                      {changed ? String(overrides[c]) : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
