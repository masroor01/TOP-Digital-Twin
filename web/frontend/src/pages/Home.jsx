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
    title: 'What-If Scenario Simulator',
    body: 'Move a policy, climate, or macro lever — an export ban, a diesel price spike, a rainfall shock — and see the model\'s multi-horizon price response update live.',
  },
  {
    icon: '🎯',
    title: 'Isolated Feature Attribution',
    body: 'Every scenario change is decomposed into its individual price contribution, with the gap between the sum of isolated effects and the actual combined move surfaced explicitly.',
  },
  {
    icon: '🏆',
    title: 'Cross-Market Benchmarks',
    body: 'Compare price levels and arrival volumes across every tracked APMC market for a crop, and overlay historical trajectories for up to 8 markets at once.',
  },
  {
    icon: '✨',
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
    <div className="min-h-screen bg-[var(--bg-app)]">
      <SiteNav />

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-5 pt-20 pb-16 text-center">
        <FadeIn>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#EAF3EC] border border-[#BFDCC9] text-[var(--brand)] text-xs font-semibold mb-6">
            <span className="pulse-dot" /> SKUAST-K &middot; HADP-04
          </span>
        </FadeIn>
        <FadeIn delay={0.05}>
          <h1 className="font-display text-4xl sm:text-5xl font-bold text-[var(--text-primary)] tracking-tight leading-tight max-w-3xl mx-auto">
            Price forecasting for India's Tomato, Onion &amp; Potato markets
          </h1>
        </FadeIn>
        <FadeIn delay={0.1}>
          <p className="text-[var(--text-secondary)] text-lg max-w-2xl mx-auto mt-5 leading-relaxed">
            A six-layer data-fusion model forecasting weekly APMC wholesale prices 1 to 26 weeks out —
            and a live simulator for testing how policy, climate, and macro shocks move them.
          </p>
        </FadeIn>
        <FadeIn delay={0.15}>
          <div className="flex items-center justify-center gap-3 mt-9">
            <Link to="/dashboard">
              <motion.span whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                className="inline-block px-6 py-3 rounded-xl bg-[var(--brand)] text-white font-display font-semibold shadow-[0_8px_24px_-8px_rgba(27,67,50,0.5)] hover:bg-[var(--brand-light)] transition-colors">
                Open the Dashboard →
              </motion.span>
            </Link>
            <Link to="/about">
              <motion.span whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                className="inline-block px-6 py-3 rounded-xl border border-[var(--border-color-strong)] bg-white text-[var(--text-primary)] font-display font-semibold hover:border-[var(--brand)] hover:text-[var(--brand)] transition-colors">
                Learn How It Works
              </motion.span>
            </Link>
          </div>
        </FadeIn>
      </section>

      {/* Crop + market-count strip */}
      <section className="max-w-6xl mx-auto px-5 pb-16">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {['tomato', 'onion', 'potato'].map((crop, i) => (
            <FadeIn key={crop} delay={0.05 * i}>
              <Card className="p-5 text-center" accent={CROP_COLOR[crop]}>
                <div className="text-3xl mb-2">{CROP_ICON[crop]}</div>
                <p className="font-display font-bold text-[var(--text-primary)] capitalize">{crop}</p>
                <p className="text-2xl font-display font-bold mt-1" style={{ color: CROP_COLOR[crop] }}>
                  {counts?.[crop] != null ? counts[crop].toLocaleString('en-IN') : '—'}
                </p>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">APMC markets tracked</p>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-5 pb-16">
        <FadeIn>
          <h2 className="font-display text-2xl font-bold text-[var(--text-primary)] text-center mb-2">What the dashboard does</h2>
          <p className="text-[var(--text-secondary)] text-center max-w-xl mx-auto mb-10">
            Built on 12 production LightGBM models, one per crop and forecast horizon.
          </p>
        </FadeIn>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {FEATURES.map((f, i) => (
            <FadeIn key={f.title} delay={0.05 * i}>
              <Card className="p-6 h-full">
                <div className="text-2xl mb-3">{f.icon}</div>
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
          <Card className="p-8 bg-[var(--brand-dark)] border-0" hover={false}>
            <p className="font-display font-bold text-white text-center mb-1">Six data layers, fused into one forecast</p>
            <p className="text-white/60 text-sm text-center mb-6">Market · Satellite vegetation · Climate stress · Macro-logistics · Infrastructure · Policy &amp; trade</p>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {LAYERS.map((l) => (
                <span key={l.label} className="flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/10 border border-white/15 text-white text-sm font-medium">
                  <span>{l.icon}</span>{l.label}
                </span>
              ))}
            </div>
          </Card>
        </FadeIn>
      </section>

      <SiteFooter />
    </div>
  );
}
