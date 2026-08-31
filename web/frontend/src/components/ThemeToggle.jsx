import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../lib/ThemeContext';

const OPTIONS = [
  { key: 'light', icon: '☀️', label: 'Light' },
  { key: 'dark', icon: '🌙', label: 'Dark' },
  { key: 'warm', icon: '🍂', label: 'Warm' },
];

export default function ThemeToggle({ compact = false, onDark = false }) {
  const { theme, setTheme } = useTheme();
  const pillId = onDark ? 'theme-toggle-pill-dark' : 'theme-toggle-pill';
  return (
    <div className={`relative flex items-center gap-0.5 p-0.5 rounded-full border ${
      onDark ? 'border-white/15 bg-white/5' : 'border-[var(--border-color-strong)] bg-[var(--bg-app-alt)]'
    }`}>
      {OPTIONS.map((opt) => {
        const active = theme === opt.key;
        return (
          <button
            key={opt.key}
            onClick={() => setTheme(opt.key)}
            aria-label={`${opt.label} theme`}
            title={`${opt.label} theme`}
            className={`relative z-10 flex items-center justify-center gap-1 rounded-full text-xs font-semibold transition-colors ${
              compact ? 'w-7 h-7' : 'px-2.5 py-1.5'
            } ${active
              ? (onDark ? 'text-[var(--brand-dark)]' : 'text-white')
              : (onDark ? 'text-white/60 hover:text-white' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]')
            }`}
          >
            {active && (
              <motion.span
                layoutId={pillId}
                className="absolute inset-0 rounded-full -z-10"
                style={{ background: onDark ? 'var(--brand-light)' : 'var(--brand)' }}
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
            <span>{opt.icon}</span>
            {!compact && <span>{opt.label}</span>}
          </button>
        );
      })}
    </div>
  );
}
