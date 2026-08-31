import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import ThemeToggle from './ThemeToggle';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
];

function NavLinks({ compact }) {
  const { pathname } = useLocation();
  return (
    <>
      {LINKS.map((l) => {
        const active = pathname === l.to;
        return (
          <Link key={l.to} to={l.to}
            className={`relative px-2.5 sm:px-3 py-2 text-sm font-semibold rounded-lg transition ${
              active ? 'text-[var(--brand)]' : 'text-[var(--text-secondary)] hover:text-[var(--brand)]'
            }`}>
            {l.label}
            {active && (
              <motion.span layoutId={compact ? 'site-nav-underline-m' : 'site-nav-underline'} className="absolute left-2.5 right-2.5 sm:left-3 sm:right-3 -bottom-[1px] h-[2px] rounded-full"
                style={{ background: 'var(--accent)' }} transition={{ type: 'spring', stiffness: 500, damping: 35 }} />
            )}
          </Link>
        );
      })}
    </>
  );
}

export default function SiteNav() {
  return (
    <header className="sticky top-0 z-20 backdrop-blur bg-[var(--bg-app)]/85 border-b border-[var(--border-color)]">
      <div className="max-w-6xl mx-auto px-3 sm:px-5">
        {/* Desktop: single row */}
        <div className="hidden sm:flex items-center justify-between gap-2 h-16">
          <Link to="/" className="flex items-center gap-2.5 min-w-0 shrink-0">
            <span className="w-9 h-9 rounded-xl bg-[var(--brand)] flex items-center justify-center text-lg shrink-0">⚡</span>
            <span className="font-display font-bold text-[var(--text-primary)] tracking-tight text-base whitespace-nowrap">TOP Digital Twin</span>
          </Link>
          <nav className="flex items-center gap-1 shrink-0">
            <NavLinks />
            <span className="ml-2"><ThemeToggle compact /></span>
            <Link to="/dashboard"
              className="ml-2 px-4 py-2 text-sm font-display font-semibold rounded-lg bg-[var(--brand)] text-white hover:bg-[var(--brand-light)] transition whitespace-nowrap">
              Open Dashboard
            </Link>
          </nav>
        </div>

        {/* Mobile: two rows, so nothing clips */}
        <div className="sm:hidden flex items-center justify-between gap-2 h-14">
          <Link to="/" className="flex items-center gap-2 min-w-0 shrink-0">
            <span className="w-8 h-8 rounded-xl bg-[var(--brand)] flex items-center justify-center text-base shrink-0">⚡</span>
            <span className="font-display font-bold text-[var(--text-primary)] tracking-tight text-sm whitespace-nowrap">TOP Digital Twin</span>
          </Link>
          <Link to="/dashboard"
            className="px-3 py-1.5 text-xs font-display font-semibold rounded-lg bg-[var(--brand)] text-white hover:bg-[var(--brand-light)] transition whitespace-nowrap shrink-0">
            Dashboard
          </Link>
        </div>
        <div className="sm:hidden flex items-center justify-between gap-2 pb-2.5">
          <nav className="flex items-center gap-0.5">
            <NavLinks compact />
          </nav>
          <ThemeToggle compact />
        </div>
      </div>
    </header>
  );
}
