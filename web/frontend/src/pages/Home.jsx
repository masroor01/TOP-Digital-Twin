import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import SiteNav from '../components/SiteNav';
import SiteFooter from '../components/SiteFooter';
import { Card, FadeIn } from '../components/ui';
import { api } from '../lib/api';
import { CROP_ICON, CROP_COLOR } from '../lib/theme';

const FEATURES = [
  {
    icon: '⚙️',
    gradient: 'linear-gradient(135deg, var(--brand) 0%, var(--brand-light) 100%)',
    title: 'What-If Scenario Simulator',
    body: 'Move a policy, climate, or macro lever — an export ban, a diesel price spike, a rainfall shock — and see the model\'s multi-horizon price response update live.',
  },
  {
    icon: '🎯',
    gradient: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%)',
    title: 'Isolated Feature Attribution',
    body: 'Every scenario change is decomposed into its individual price contribution, with the gap between the sum of isolated effects and the actual combined move surfaced explicitly.',
  },
  {
    icon: '🏆',
    gradient: 'linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%)',
    title: 'Cross-Market Benchmarks',
    body: 'Compare price levels and arrival volumes across every tracked APMC market for a crop, and overlay historical trajectories for up to 8 markets at once.',
  },
  {
    icon: '✨',
    gradient: 'linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%)',
    title: 'AI Policy Briefing',
    body: 'A grounded, plain-language commentary on the current scenario — generated strictly from the computed baseline, delta, and isolated effects, never free-form.',
  },
];

const LAYERS = [
  { icon: '🌾', label: 'Market' },
  { icon: '🛰️', label: 'Satellite' },
  { icon: '🌦️', label: 'Climate' },
  { icon: '💹', label: 'Macro' },
  { icon: '🏗️', label: 'Infrastructure' },
  { icon: '📜', label: 'Policy' },
];

export default function Home() {
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    api.meta().catch(() => null).then(setMeta);
  }, []);

  const counts = meta?.marketCounts;

  return (
    <div className="min-h-screen bg-[var(--bg-app)] overflow-x-hidden">
      <SiteNav />

      {/* Hero with gradient mesh background */}
      <section className="relative">
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[900px] h-[600px] rounded-full blur-3xl opacity-40"
            style={{ background: 'radial-gradient(circle, var(--brand) 0%, transparent 65%)' }} />
          <div className="absolute top-10 left-[8%] w-[420px] h-[420px] rounded-full blur-3xl opacity-30"
            style={{ background: 'radial-gradient(circle, var(--accent) 0%, transparent 70%)' }} />
          <div className="absolute top-24 right-[6%] w-[380px] h-[380px] rounded-full blur-3xl opacity-25"
            style={{ background: 'radial-gradient(circle, #7C3AED 0%, transparent 70%)' }} />
        </div>

        <div className="max-w-6xl mx-auto px-5 pt-20 pb-16 text-center">
          <FadeIn>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[var(--card-bg)]/80 backdrop-blur border border-[var(--border-color-strong)] text-[var(--brand-text)] text-xs font-semibold mb-6 shadow-sm">
              <span className="pulse-dot" /> SKUAST-K &middot; HADP-04
            </span>
          </FadeIn>
          <FadeIn delay={0.05}>
            <h1 className="font-display text-4xl sm:text-6xl font-bold tracking-tight leading-[1.08] max-w-4xl mx-auto">
              <span className="text-[var(--text-primary)]">Price forecasting for India's </span>
              <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'linear-gradient(120deg, var(--brand) 0%, var(--accent) 100%)' }}>
                Tomato, Onion &amp; Potato
              </span>
              <span className="text-[var(--text-primary)]"> markets</span>
            </h1>
          </FadeIn>
          <FadeIn delay={0.1}>
            <p className="text-[var(--text-secondary)] text-lg max-w-2xl mx-auto mt-6 leading-relaxed">
              A six-layer data-fusion model forecasting weekly APMC wholesale prices 1 to 26 weeks out —
              and a live simulator for testing how policy, climate, and macro shocks move them.
            </p>
          </FadeIn>
          <FadeIn delay={0.15}>
            <div className="flex flex-wrap items-center justify-center gap-3 mt-9">
              <Link to="/dashboard">
                <motion.span whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                  className="inline-block px-6 py-3 rounded-xl text-white font-display font-semibold shadow-[0_8px_28px_-8px_var(--brand)] hover:shadow-[0_12px_32px_-8px_var(--brand)] transition-shadow"
                  style={{ background: 'linear-gradient(135deg, var(--brand) 0%, var(--brand-light) 100%)' }}>
                  Open the Dashboard →
                </motion.span>
              </Link>
              <Link to="/about">
                <motion.span whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                  className="inline-block px-6 py-3 rounded-xl border border-[var(--border-color-strong)] bg-[var(--card-bg)] text-[var(--text-primary)] font-display font-semibold hover:border-[var(--brand)] hover:text-[var(--brand-text)] transition-colors">
                  Learn How It Works
                </motion.span>
              </Link>
            </div>
          </FadeIn>
        </div>

        {/* Crop + market-count strip */}
        <div className="max-w-6xl mx-auto px-5 pb-16">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {['tomato', 'onion', 'potato'].map((crop, i) => (
              <FadeIn key={crop} delay={0.05 * i}>
                <Card className="p-5 text-center relative overflow-hidden" accent={CROP_COLOR[crop]}>
                  <div className="absolute -top-8 -right-8 w-28 h-28 rounded-full blur-2xl opacity-20"
                    style={{ background: CROP_COLOR[crop] }} />
                  <div className="text-3xl mb-2 relative">{CROP_ICON[crop]}</div>
                  <p className="font-display font-bold text-[var(--text-primary)] capitalize relative">{crop}</p>
                  <p className="text-2xl font-display font-bold mt-1 relative" style={{ color: CROP_COLOR[crop] }}>
                    {counts?.[crop] != null ? counts[crop].toLocaleString('en-IN') : '—'}
                  </p>
                  <p className="text-xs text-[var(--text-secondary)] mt-0.5 relative">APMC markets tracked</p>
                </Card>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-5 pb-16">
        <FadeIn>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-primary)] text-center mb-2">What the dashboard does</h2>
          <p className="text-[var(--text-secondary)] text-center max-w-xl mx-auto mb-10">
            Built on 12 production LightGBM models, one per crop and forecast horizon.
          </p>
        </FadeIn>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {FEATURES.map((f, i) => (
            <FadeIn key={f.title} delay={0.05 * i}>
              <Card className="p-6 h-full group">
                <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-2xl mb-4 shadow-sm transition-transform group-hover:scale-110"
                  style={{ background: f.gradient }}>
                  {f.icon}
                </div>
                <p className="font-display font-bold text-[var(--text-primary)] mb-1.5">{f.title}</p>
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{f.body}</p>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* Data layers strip */}
      <section className="max-w-6xl mx-auto px-5 pb-24">
        <FadeIn>
          <div className="relative rounded-3xl p-8 overflow-hidden shadow-[0_20px_60px_-20px_rgba(15,45,32,0.5)]"
            style={{ background: 'linear-gradient(135deg, var(--brand-dark) 0%, var(--brand) 100%)' }}>
            <div className="pointer-events-none absolute -bottom-16 -left-16 w-64 h-64 rounded-full blur-3xl opacity-30"
              style={{ background: 'var(--accent-light)' }} />
            <div className="pointer-events-none absolute -top-16 -right-16 w-64 h-64 rounded-full blur-3xl opacity-20"
              style={{ background: '#7C3AED' }} />
            <p className="font-display font-bold text-white text-center mb-1 relative text-lg">Six data layers, fused into one forecast</p>
            <p className="text-white/60 text-sm text-center mb-6 relative">Market · Satellite vegetation · Climate stress · Macro-logistics · Infrastructure · Policy &amp; trade</p>
            <div className="flex flex-wrap items-center justify-center gap-3 relative">
              {LAYERS.map((l) => (
                <span key={l.label} className="flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/10 border border-white/15 text-white text-sm font-medium backdrop-blur-sm">
                  <span>{l.icon}</span>{l.label}
                </span>
              ))}
            </div>
          </div>
        </FadeIn>
      </section>

      <SiteFooter />
    </div>
  );
}
