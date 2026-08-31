import React from 'react';
import { CROP_ICON, HORIZONS } from '../lib/theme';
import { SectionLabel } from './ui';

function fnum(v, d = 0) {
  return v == null || Number.isNaN(v) ? d : Number(v);
}

function StaleCaption({ staleness, crop, field }) {
  const s = staleness?.[crop]?.[field];
  if (!s) return null;
  return <p className="text-[0.7rem] text-slate-400 mt-0.5">📌 Stale feed: {s.as_of} ({s.weeks_stale}w)</p>;
}

function SliderField({ label, help, min, max, step = 1, value, onChange, staleness, crop, field, obsMin, obsMax }) {
  const outOfObserved = obsMin != null && obsMax != null && (value < obsMin || value > obsMax);
  return (
    <div className="mb-3.5" title={help}>
      <div className="flex items-center justify-between mb-1">
        <label className="text-[0.8rem] font-semibold text-slate-700">{label}</label>
        <span className="font-mono text-xs text-slate-500">{Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full bg-slate-200 cursor-pointer"
      />
      {outOfObserved && (
        <p className="text-[0.7rem] text-amber-600 mt-0.5">⚠️ Speculative range ({obsMin.toLocaleString()}–{obsMax.toLocaleString()})</p>
      )}
      <StaleCaption staleness={staleness} crop={crop} field={field} />
    </div>
  );
}

function CheckField({ label, help, checked, onChange, staleness, crop, field }) {
  return (
    <div className="mb-3.5" title={help}>
      <label className="flex items-center gap-2 text-[0.8rem] font-semibold text-slate-700 cursor-pointer">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="w-4 h-4 accent-slate-900" />
        {label}
      </label>
      <StaleCaption staleness={staleness} crop={crop} field={field} />
    </div>
  );
}

export default function Sidebar({
  meta, crop, setCrop, states, stateSel, setStateSel, markets, market, setMarket,
  horizon, setHorizon, baseRow, overrides, setOverrides, targetDate, marketCounts, onClose,
}) {
  if (!meta) return null;
  const { featureInfo, featureRanges, staleness, policyFields, climateFields, macroFields } = meta;

  const setOverride = (field, value) => setOverrides((prev) => ({ ...prev, [field]: value }));

  const val = (field) => (overrides[field] !== undefined ? overrides[field] : fnum(baseRow?.[field]));

  return (
    <aside className="w-[300px] max-w-[85vw] shrink-0 bg-[#FAFAFC] border-r border-slate-200 h-full overflow-y-auto px-4 py-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">⚡</span>
          <div>
            <h3 className="text-sm font-extrabold text-slate-900 tracking-tight leading-none">TOP Digital Twin</h3>
            <p className="text-[0.68rem] text-slate-500 font-semibold mt-0.5">HADP-04 · APMC Simulator</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-[0.7rem] font-semibold">
            <span className="pulse-dot" /> M6
          </span>
          {onClose && (
            <button onClick={onClose} aria-label="Close menu" className="md:hidden p-1 rounded-md text-slate-500 hover:bg-slate-200">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          )}
        </div>
      </div>

      <details className="mb-3 text-xs">
        <summary className="cursor-pointer text-slate-600 font-medium">📦 Data & Market Specs</summary>
        <div className="mt-2 text-slate-500 leading-relaxed">
          <p className="mb-1">Feeds: Agmarknet (prices/arrivals), CMIE/RBI/PPAC (macro), Sentinel-2/MODIS/ERA5/CHIRPS (remote sensing), 2017-2026.</p>
          <ul className="list-none space-y-0.5">
            {Object.entries(marketCounts || {}).map(([c, n]) => (
              <li key={c}>{CROP_ICON[c] || ''} {c[0].toUpperCase() + c.slice(1)}: <b>{n} APMCs</b></li>
            ))}
          </ul>
        </div>
      </details>

      <hr className="border-slate-200 my-3" />
      <SectionLabel>Market Selection</SectionLabel>

      <select value={crop} onChange={(e) => setCrop(e.target.value)} className="w-full mb-2.5 text-sm rounded-lg border border-slate-300 px-2.5 py-1.5 bg-white">
        {['tomato', 'onion', 'potato'].map((c) => (
          <option key={c} value={c}>{CROP_ICON[c]} {c[0].toUpperCase() + c.slice(1)}</option>
        ))}
      </select>

      <select value={stateSel || ''} onChange={(e) => setStateSel(e.target.value)} className="w-full mb-2.5 text-sm rounded-lg border border-slate-300 px-2.5 py-1.5 bg-white">
        {states.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>

      <select value={market?.marketId || ''} onChange={(e) => setMarket(markets.find((m) => String(m.marketId) === e.target.value))}
        className="w-full mb-3 text-sm rounded-lg border border-slate-300 px-2.5 py-1.5 bg-white">
        {markets.map((m) => <option key={m.marketId} value={m.marketId}>{m.market}</option>)}
      </select>

      <label className="text-[0.8rem] font-semibold text-slate-700">Forecast Horizon</label>
      <div className="flex gap-1.5 mt-1.5 mb-1.5">
        {HORIZONS.map((h) => (
          <button key={h} onClick={() => setHorizon(h)}
            className={`flex-1 text-xs font-semibold rounded-md py-1.5 border transition ${
              horizon === h ? 'bg-slate-900 text-white border-slate-900' : 'bg-white text-slate-600 border-slate-300 hover:border-slate-400'
            }`}>
            {h}W
          </button>
        ))}
      </div>
      {targetDate && <p className="text-xs text-slate-500 mb-3">🎯 Target Date: <b>{targetDate}</b></p>}

      <hr className="border-slate-200 my-3" />
      <SectionLabel>Policy Scenarios</SectionLabel>
      {policyFields.includes('export_banned') && (
        <CheckField label={featureInfo.export_banned.label} help={featureInfo.export_banned.help}
          checked={!!val('export_banned')} onChange={(v) => setOverride('export_banned', v ? 1 : 0)}
          staleness={staleness} crop={crop} field="export_banned" />
      )}
      {featureRanges.mep_usd_per_tonne && (
        <SliderField label={featureInfo.mep_usd_per_tonne.label} help={featureInfo.mep_usd_per_tonne.help}
          min={0} max={Math.max(featureRanges.mep_usd_per_tonne.max, 900)} step={10}
          value={val('mep_usd_per_tonne')} onChange={(v) => setOverride('mep_usd_per_tonne', v)}
          staleness={staleness} crop={crop} field="mep_usd_per_tonne" />
      )}
      {featureRanges.export_duty_pct && (
        <SliderField label={featureInfo.export_duty_pct.label} help={featureInfo.export_duty_pct.help}
          min={0} max={50} step={1} value={val('export_duty_pct')} onChange={(v) => setOverride('export_duty_pct', v)}
          staleness={staleness} crop={crop} field="export_duty_pct" />
      )}
      <CheckField label={featureInfo.market_intervention_flag.label} help={featureInfo.market_intervention_flag.help}
        checked={!!val('market_intervention_flag')} onChange={(v) => setOverride('market_intervention_flag', v ? 1 : 0)}
        staleness={staleness} crop={crop} field="market_intervention_flag" />

      <hr className="border-slate-200 my-3" />
      <SectionLabel>Climate & Satellite</SectionLabel>
      {climateFields.map((field) => {
        const r = featureRanges[field];
        if (!r || baseRow?.[field] == null) return null;
        return (
          <SliderField key={field} label={featureInfo[field].label} help={featureInfo[field].help}
            min={r.min} max={r.max} step={(r.max - r.min) / 100 || 1}
            value={val(field)} onChange={(v) => setOverride(field, v)}
            staleness={staleness} crop={crop} field={field} obsMin={r.min} obsMax={r.max} />
        );
      })}

      <hr className="border-slate-200 my-3" />
      <SectionLabel>Macro & Logistics</SectionLabel>
      {macroFields.map((field) => {
        const r = featureRanges[field];
        if (!r || baseRow?.[field] == null) return null;
        const span = r.max - r.min;
        const lo = r.min - span * 0.2, hi = r.max + span * 0.2;
        return (
          <SliderField key={field} label={featureInfo[field].label} help={featureInfo[field].help}
            min={lo} max={hi} step={span / 100 || 1}
            value={val(field)} onChange={(v) => setOverride(field, v)}
            staleness={staleness} crop={crop} field={field} obsMin={r.min} obsMax={r.max} />
        );
      })}

      <button onClick={() => setOverrides({})}
        className="w-full mt-2 text-sm font-semibold rounded-lg border border-slate-300 bg-white py-2 hover:bg-slate-900 hover:text-white hover:border-slate-900 transition">
        🔄 Reset to Baseline Vector
      </button>
    </aside>
  );
}
