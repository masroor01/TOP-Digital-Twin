import React from 'react';

export function Card({ children, className = '' }) {
  return (
    <div className={`bg-white border border-slate-200 rounded-xl shadow-sm ${className}`}>
      {children}
    </div>
  );
}

export function SectionLabel({ children }) {
  return (
    <p className="text-xs font-bold uppercase tracking-wide text-slate-500 mt-2 mb-2.5">
      {children}
    </p>
  );
}

export function Badge({ children, tone = 'neutral' }) {
  const tones = {
    neutral: 'bg-slate-100 text-slate-600 border-slate-200',
    up: 'bg-red-50 text-red-600 border-red-200',
    down: 'bg-emerald-50 text-emerald-600 border-emerald-200',
    live: 'bg-emerald-50 text-emerald-800 border-emerald-200',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Chip({ value, unit = '' }) {
  const tone = value > 0 ? 'up' : value < 0 ? 'down' : 'neutral';
  const sign = value > 0 ? '+' : '';
  return (
    <Badge tone={tone}>
      <span className="font-mono">{sign}{value.toLocaleString('en-IN', { maximumFractionDigits: 0 })} {unit}</span>
    </Badge>
  );
}

export function Sparkline({ points, color = '#0F172A', width = 88, height = 26 }) {
  if (!points || points.length < 2) return null;
  const min = Math.min(...points), max = Math.max(...points);
  const span = max - min || 1;
  const stepX = width / (points.length - 1);
  const path = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i * stepX).toFixed(1)} ${(height - ((p - min) / span) * height).toFixed(1)}`)
    .join(' ');
  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={path} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />
    </svg>
  );
}

export function Metric({ label, value, delta, deltaTone, help, spark, sparkColor, border = true }) {
  return (
    <Card className={`p-4 relative group ${border ? '' : 'shadow-none border-0'}`} title={help}>
      <p className="text-[0.72rem] font-semibold text-slate-500 uppercase tracking-wide mb-1">{label}</p>
      <div className="flex items-end justify-between gap-2">
        <p className="text-[1.55rem] font-bold text-slate-900 tracking-tight">{value}</p>
        {spark && <Sparkline points={spark} color={sparkColor || '#0F172A'} />}
      </div>
      {delta && (
        <p className={`font-mono text-[0.82rem] font-semibold mt-1 ${deltaTone === 'up' ? 'text-red-600' : deltaTone === 'down' ? 'text-emerald-600' : 'text-slate-500'}`}>
          {delta}
        </p>
      )}
    </Card>
  );
}

export function Spinner({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500 py-6">
      <span className="w-3.5 h-3.5 border-2 border-slate-300 border-t-slate-700 rounded-full animate-spin" />
      {label}
    </div>
  );
}

export function Alert({ tone = 'warning', children }) {
  const tones = {
    warning: 'bg-amber-50 border-amber-200 text-amber-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    info: 'bg-sky-50 border-sky-200 text-sky-800',
  };
  return <div className={`border rounded-lg px-4 py-2.5 text-sm mb-2 ${tones[tone]}`}>{children}</div>;
}
