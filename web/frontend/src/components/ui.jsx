import React from 'react';
import ReactDOM from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

const cardTransition = { type: 'spring', stiffness: 300, damping: 26 };

export function Card({ children, className = '', accent, hover = true, ...rest }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={cardTransition}
      whileHover={hover ? { y: -3, boxShadow: '0 1px 2px rgba(28,38,32,0.06), 0 16px 32px -14px rgba(28,38,32,0.22)' } : undefined}
      className={`bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl shadow-[0_1px_2px_rgba(28,38,32,0.04),0_8px_24px_-16px_rgba(28,38,32,0.12)] relative overflow-hidden ${className}`}
      {...rest}
    >
      {accent && <span className="absolute top-0 left-0 right-0 h-[3px]" style={{ background: accent }} />}
      {children}
    </motion.div>
  );
}

export function SectionLabel({ children }) {
  return (
    <p className="font-display text-[0.7rem] font-bold uppercase tracking-[0.08em] text-[var(--brand)] mt-2 mb-3 flex items-center gap-2">
      <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
      {children}
    </p>
  );
}

export function Badge({ children, tone = 'neutral', style }) {
  const tones = {
    neutral: 'bg-[var(--bg-app-alt)] text-[var(--text-secondary)] border-[var(--border-color-strong)]',
    up: 'bg-red-50 text-red-700 border-red-200',
    down: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    live: 'bg-[#EAF3EC] text-[var(--brand)] border-[#BFDCC9]',
  };
  return (
    <span style={style} className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function CropBadge({ crop, icon, color, size = 'md' }) {
  const dims = size === 'sm' ? 'w-6 h-6 text-sm' : size === 'lg' ? 'w-11 h-11 text-2xl' : 'w-8 h-8 text-base';
  return (
    <span
      className={`inline-flex items-center justify-center rounded-lg shrink-0 ${dims}`}
      style={{ background: `${color}1A`, border: `1px solid ${color}40` }}
      title={crop}
    >
      {icon}
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
      <motion.path
        d={path} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" opacity="0.85"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.7, ease: 'easeOut' }}
      />
    </svg>
  );
}

export function Metric({ label, value, delta, deltaTone, help, spark, sparkColor, border = true, accent, icon }) {
  return (
    <Card className={`p-4 relative group ${border ? '' : 'shadow-none border-0'}`} accent={accent} title={help}>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-[0.72rem] font-semibold text-[var(--text-secondary)] uppercase tracking-wide">{label}</p>
        {icon}
      </div>
      <div className="flex items-end justify-between gap-2">
        <AnimatePresence mode="wait">
          <motion.p
            key={value}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="font-display text-[1.6rem] font-bold text-[var(--text-primary)] tracking-tight"
          >
            {value}
          </motion.p>
        </AnimatePresence>
        {spark && <Sparkline points={spark} color={sparkColor || 'var(--brand)'} />}
      </div>
      {delta && (
        <p className={`font-mono text-[0.82rem] font-semibold mt-1 ${deltaTone === 'up' ? 'text-red-600' : deltaTone === 'down' ? 'text-emerald-600' : 'text-[var(--text-secondary)]'}`}>
          {delta}
        </p>
      )}
    </Card>
  );
}

export function InfoButton({ title, children }) {
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState(null);
  const btnRef = React.useRef(null);
  const panelRef = React.useRef(null);
  const PANEL_WIDTH = 288; // matches w-72

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    const MARGIN = 8;
    const left = Math.min(Math.max(MARGIN, r.right - PANEL_WIDTH), window.innerWidth - PANEL_WIDTH - MARGIN);
    const spaceBelow = window.innerHeight - r.bottom - MARGIN - 6;
    const spaceAbove = r.top - MARGIN - 6;
    // Prefer opening below the button (natural reading direction); flip
    // above only when below is cramped AND above genuinely has more room --
    // either way the panel gets a real maxHeight + its own scrollbar so
    // long content (e.g. every crop/state/market tier explained at once)
    // is never silently cut off with no way to reach the rest.
    if (spaceBelow >= 160 || spaceBelow >= spaceAbove) {
      setPos({ left, top: r.bottom + 6, maxHeight: Math.max(120, Math.min(spaceBelow, 420)) });
    } else {
      setPos({ left, bottom: window.innerHeight - r.top + 6, maxHeight: Math.max(120, Math.min(spaceAbove, 420)) });
    }
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    function onClick(e) {
      if (btnRef.current?.contains(e.target) || panelRef.current?.contains(e.target)) return;
      setOpen(false);
    }
    function onKey(e) { if (e.key === 'Escape') setOpen(false); }
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', reposition);
      window.removeEventListener('scroll', reposition, true);
    };
  }, [open, reposition]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        aria-label={title ? `About: ${title}` : 'More info'}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        className="w-4 h-4 rounded-full inline-flex items-center justify-center text-[0.62rem] font-bold leading-none
                   border border-[var(--border-color-strong)] text-[var(--text-secondary)] bg-[var(--card-bg)]
                   hover:bg-[var(--brand)] hover:text-white hover:border-[var(--brand)] transition-colors"
      >
        i
      </button>
      {ReactDOM.createPortal(
        <AnimatePresence>
          {open && pos && (
            <motion.div
              ref={panelRef}
              initial={{ opacity: 0, y: -4, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -4, scale: 0.97 }}
              transition={{ duration: 0.15 }}
              onClick={(e) => e.stopPropagation()}
              style={{
                position: 'fixed',
                ...(pos.top != null ? { top: pos.top } : { bottom: pos.bottom }),
                left: pos.left,
                width: PANEL_WIDTH,
                maxHeight: pos.maxHeight,
              }}
              className="z-[100] max-w-[85vw] p-3 rounded-xl border border-[var(--border-color-strong)]
                         bg-[var(--card-bg)] shadow-[0_4px_16px_-4px_rgba(28,38,32,0.28)] text-left
                         overflow-y-auto overscroll-contain"
            >
              {title && <p className="text-[0.72rem] font-bold uppercase tracking-wide text-[var(--brand)] mb-1.5">{title}</p>}
              <div className="text-[0.78rem] leading-snug text-[var(--text-secondary)] space-y-1.5">{children}</div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  );
}

export function Spinner({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)] py-6">
      <span className="w-3.5 h-3.5 border-2 border-[var(--border-color-strong)] border-t-[var(--brand)] rounded-full animate-spin" />
      {label}
    </div>
  );
}

export function Alert({ tone = 'warning', children }) {
  const tones = {
    warning: 'bg-amber-50 border-amber-200 text-amber-900',
    error: 'bg-red-50 border-red-200 text-red-900',
    info: 'bg-[#EAF3EC] border-[#BFDCC9] text-[var(--brand-dark)]',
  };
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} transition={{ duration: 0.2 }}
      className={`border rounded-xl px-4 py-2.5 text-sm mb-2 ${tones[tone]}`}
    >
      {children}
    </motion.div>
  );
}

export function FadeIn({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
