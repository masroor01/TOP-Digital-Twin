import React, { useEffect, useMemo, useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import SimulationTab from './components/tabs/SimulationTab';
import AttributionTab from './components/tabs/AttributionTab';
import BenchmarksTab from './components/tabs/BenchmarksTab';
import MultiMarketTab from './components/tabs/MultiMarketTab';
import AITab from './components/tabs/AITab';
import AuditTab from './components/tabs/AuditTab';
import { Spinner, Alert } from './components/ui';
import { api } from './lib/api';
import { HORIZONS } from './lib/theme';

const TABS = [
  { key: 'simulation', label: '📈 Simulation' },
  { key: 'attribution', label: '🎯 Attribution' },
  { key: 'benchmarks', label: '🏆 Benchmarks' },
  { key: 'multimarket', label: '🗺️ Multi-Market' },
  { key: 'ai', label: '✨ AI Briefing' },
  { key: 'audit', label: '🔍 Audit' },
];

export default function App() {
  const [meta, setMeta] = useState(null);
  const [healthy, setHealthy] = useState(false);

  const [crop, setCrop] = useState('tomato');
  const [stateSel, setStateSel] = useState('');
  const [states, setStates] = useState([]);
  const [markets, setMarkets] = useState([]);
  const [market, setMarket] = useState(null);
  const [horizon, setHorizon] = useState(HORIZONS[1]);
  const [overrides, setOverrides] = useState({});

  const [baseRow, setBaseRow] = useState(null);
  const [sim, setSim] = useState(null);
  const [simError, setSimError] = useState(null);
  const [tab, setTab] = useState('simulation');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    api.meta().then(setMeta);
    api.health().catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.health().then((h) => { if (!cancelled) setHealthy(!!h.inference?.ok); }).catch(() => setHealthy(false));
    const id = setInterval(() => {
      api.health().then((h) => setHealthy(!!h.inference?.ok)).catch(() => setHealthy(false));
    }, 15000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // crop change -> reload states
  useEffect(() => {
    setOverrides({});
    api.states(crop).then((s) => {
      setStates(s);
      setStateSel(s[0] || '');
    });
  }, [crop]);

  // state change -> reload markets
  useEffect(() => {
    if (!stateSel) return;
    api.markets(crop, stateSel).then((m) => {
      setMarkets(m);
      setMarket(m[0] || null);
    });
  }, [crop, stateSel]);

  // market change -> reload baseline reference row
  useEffect(() => {
    if (!market) return;
    setOverrides({});
    api.reference(crop, market.marketId).then(setBaseRow).catch(() => setBaseRow(null));
  }, [crop, market?.marketId]);

  // scenario change -> re-run simulation
  useEffect(() => {
    if (!market) return;
    setSimError(null);
    api.simulate({ crop, marketId: market.marketId, horizon, overrides })
      .then(setSim)
      .catch((e) => setSimError(e?.response?.data?.error || e.message));
  }, [crop, market?.marketId, horizon, overrides]);

  const marketCounts = meta?.marketCounts;
  const targetDate = sim?.targetDate;

  return (
    <div className="h-screen flex flex-col">
      <Navbar crop={crop} market={market} healthy={healthy} onMenuClick={() => setSidebarOpen(true)} />
      <div className="flex flex-1 min-h-0 relative">
        {sidebarOpen && (
          <div className="fixed inset-0 bg-slate-900/40 z-30 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}
        <div className={`fixed inset-y-0 left-0 z-40 transition-transform duration-200 md:static md:translate-x-0 md:z-auto ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <Sidebar
            meta={meta} crop={crop} setCrop={setCrop}
            states={states} stateSel={stateSel} setStateSel={setStateSel}
            markets={markets} market={market} setMarket={setMarket}
            horizon={horizon} setHorizon={setHorizon}
            baseRow={baseRow} overrides={overrides} setOverrides={setOverrides}
            targetDate={targetDate} marketCounts={marketCounts}
            onClose={() => setSidebarOpen(false)}
          />
        </div>
        <main className="flex-1 min-w-0 overflow-y-auto px-3 sm:px-6 py-5">
          <div className="flex gap-1.5 mb-4 border-b border-slate-200 overflow-x-auto whitespace-nowrap">
            {TABS.map((t) => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`shrink-0 text-sm font-semibold px-3.5 py-2 border-b-2 -mb-px transition ${
                  tab === t.key ? 'border-slate-900 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}>
                {t.label}
              </button>
            ))}
          </div>

          {simError && <Alert tone="error">Simulation error: {simError}</Alert>}
          {!meta || !market ? (
            <Spinner label="Loading dashboard…" />
          ) : (
            <>
              {tab === 'simulation' && (
                <SimulationTab sim={sim} crop={crop} market={market} marketId={market.marketId} overrides={overrides} />
              )}
              {tab === 'attribution' && <AttributionTab sim={sim} />}
              {tab === 'benchmarks' && <BenchmarksTab crop={crop} />}
              {tab === 'multimarket' && <MultiMarketTab crop={crop} markets={markets} />}
              {tab === 'ai' && <AITab sim={sim} crop={crop} market={market} horizon={horizon} />}
              {tab === 'audit' && <AuditTab sim={sim} overrides={overrides} />}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
