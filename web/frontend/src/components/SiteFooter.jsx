import React from 'react';
import { Link } from 'react-router-dom';

export default function SiteFooter() {
  return (
    <footer className="border-t border-[var(--border-color)] mt-20">
      <div className="max-w-6xl mx-auto px-5 py-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <p className="font-display font-bold text-[var(--text-primary)] text-sm">TOP Digital Twin</p>
          <p className="text-xs text-[var(--text-secondary)] mt-1">SKUAST-K &middot; HADP-04 &middot; Sher-e-Kashmir University of Agricultural Sciences and Technology of Kashmir</p>
        </div>
        <div className="flex items-center gap-5 text-xs text-[var(--text-secondary)]">
          <Link to="/" className="hover:text-[var(--brand)] transition">Home</Link>
          <Link to="/about" className="hover:text-[var(--brand)] transition">About</Link>
          <Link to="/dashboard" className="hover:text-[var(--brand)] transition">Dashboard</Link>
        </div>
      </div>
    </footer>
  );
}
