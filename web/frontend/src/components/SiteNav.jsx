import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
];

export default function SiteNav() {
  const { pathname } = useLocation();
  return (
    <header className="sticky top-0 z-20 backdrop-blur bg-[var(--bg-app)]/85 border-b border-[var(--border-color)]">
      <div className="max-w-6xl mx-auto flex items-center justify-between gap-2 px-3 sm:px-5 h-16">
        <Link to="/" className="flex items-center gap-2 sm:gap-2.5 min-w-0 shrink-0">
          <span className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-[var(--brand)] flex items-center justify-center text-base sm:text-lg shrink-0">⚡</span>
          <span className="font-display font-bold text-[var(--text-primary)] tracking-tight text-sm sm:text-base whitespace-nowrap">TOP Digital Twin</span>
        </Link>
        <nav className="flex items-center gap-0.5 sm:gap-1 shrink-0">
          {LINKS.map((l) => {
            const active = pathname === l.to;
            return (
              <Link key={l.to} to={l.to}
                className={`relative px-2 sm:px-3 py-2 text-sm font-semibold rounded-lg transition ${
                  active ? 'text-[var(--brand)]' : 'text-[var(--text-secondary)] hover:text-[var(--brand)]'
                }`}>
                {l.label}
                {active && (
                  <motion.span layoutId="site-nav-underline" className="absolute left-2 right-2 sm:left-3 sm:right-3 -bottom-[1px] h-[2px] rounded-full"
                    style={{ background: 'var(--accent)' }} transition={{ type: 'spring', stiffness: 500, damping: 35 }} />
                )}
              </Link>
            );
          })}
          <Link to="/dashboard"
            className="ml-1 sm:ml-2 px-2.5 sm:px-4 py-2 text-xs sm:text-sm font-display font-semibold rounded-lg bg-[var(--brand)] text-white hover:bg-[var(--brand-light)] transition whitespace-nowrap">
            <span className="sm:hidden">Dashboard</span>
            <span className="hidden sm:inline">Open Dashboard</span>
          </Link>
        </nav>
      </div>
    </header>
  );
}
