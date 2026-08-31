import React from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { CROP_ICON, CROP_COLOR } from '../lib/theme';
import { Badge } from './ui';

export default function Navbar({ crop, market, healthy, onMenuClick }) {
  return (
    <header className="h-16 shrink-0 flex items-center justify-between px-3 sm:px-5 gap-2 bg-[var(--brand-dark)] shadow-[0_2px_12px_-4px_rgba(15,45,32,0.4)] relative z-10">
      <div className="flex items-center gap-3 text-sm min-w-0">
        <button onClick={onMenuClick} aria-label="Open menu"
          className="md:hidden shrink-0 -ml-1 p-1.5 rounded-md text-white/80 hover:bg-white/10">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <Link to="/" aria-label="Back to home" title="Back to home"
          className="hidden md:flex shrink-0 -ml-1 p-1.5 rounded-md text-white/70 hover:text-white hover:bg-white/10 transition">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12l9-9 9 9M5 10v10a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V10" />
          </svg>
        </Link>
        <AnimatePresence mode="wait">
          <motion.span
            key={crop}
            initial={{ opacity: 0, scale: 0.6, rotate: -10 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
            className="w-9 h-9 rounded-xl flex items-center justify-center text-lg shrink-0 border"
            style={{ background: `${CROP_COLOR[crop]}30`, borderColor: `${CROP_COLOR[crop]}55` }}
          >
            {CROP_ICON[crop]}
          </motion.span>
        </AnimatePresence>
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-display font-bold text-white capitalize shrink-0 tracking-tight">{crop}</span>
          <span className="text-white/25 hidden sm:inline">/</span>
          <span className="text-white/70 truncate hidden sm:inline">{market?.market || '—'}</span>
          <span className="text-white/25 hidden sm:inline">/</span>
          <span className="text-white/45 truncate hidden sm:inline">{market?.state || '—'}</span>
        </div>
      </div>
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold ${
        healthy ? 'bg-[var(--brand-light)]/20 border-[var(--brand-light)]/40 text-emerald-200' : 'bg-white/5 border-white/15 text-white/50'
      }`}>
        <span className="pulse-dot" /> <span className="hidden sm:inline">{healthy ? 'Inference Engine Live' : 'Connecting…'}</span>
        <span className="sm:hidden">{healthy ? 'Live' : '…'}</span>
      </span>
    </header>
  );
}
