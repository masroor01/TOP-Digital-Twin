import React from 'react';
import { CROP_ICON } from '../lib/theme';
import { Badge } from './ui';

export default function Navbar({ crop, market, healthy, onMenuClick }) {
  return (
    <header className="h-14 shrink-0 border-b border-slate-200 bg-white/90 backdrop-blur flex items-center justify-between px-3 sm:px-5 gap-2">
      <div className="flex items-center gap-2.5 text-sm min-w-0">
        <button onClick={onMenuClick} aria-label="Open menu"
          className="md:hidden shrink-0 -ml-1 p-1.5 rounded-md text-slate-600 hover:bg-slate-100">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <span className="text-xl shrink-0">{CROP_ICON[crop]}</span>
        <span className="font-bold text-slate-900 capitalize shrink-0">{crop}</span>
        <span className="text-slate-300 hidden sm:inline">/</span>
        <span className="text-slate-600 truncate hidden sm:inline">{market?.market || '—'}</span>
        <span className="text-slate-300 hidden sm:inline">/</span>
        <span className="text-slate-500 truncate hidden sm:inline">{market?.state || '—'}</span>
      </div>
      <Badge tone={healthy ? 'live' : 'neutral'}>
        <span className="pulse-dot" /> <span className="hidden sm:inline">{healthy ? 'Inference Engine Live' : 'Connecting…'}</span>
        <span className="sm:hidden">{healthy ? 'Live' : '…'}</span>
      </Badge>
    </header>
  );
}
