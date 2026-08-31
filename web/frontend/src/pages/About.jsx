import React from 'react';
import { Link } from 'react-router-dom';
import SiteNav from '../components/SiteNav';
import SiteFooter from '../components/SiteFooter';
import { Card, FadeIn, SectionLabel } from '../components/ui';

const LAYERS = [
  { icon: '🌾', title: 'Market', body: 'Weekly wholesale price and arrivals data from Agmarknet, covering APMC mandis across India since 2017.' },
  { icon: '🛰️', title: 'Satellite vegetation', body: 'Sentinel-2 and MODIS NDVI/EVI — a proxy for crop health and expected supply ahead of harvest.' },
  { icon: '🌦️', title: 'Climate stress', body: 'ERA5 temperature extremes and CHIRPS rainfall, tracking heat stress and flood/drought risk by growing zone.' },
  { icon: '💹', title: 'Macro-logistics', body: 'Diesel and LPG prices, the RBI repo rate, USD/INR, and wholesale price indices — the cost side of getting produce to market.' },
  { icon: '🏗️', title: 'Infrastructure', body: 'Cold storage capacity and road density by state, and agricultural wage data — how well a region can store and move produce.' },
  { icon: '📜', title: 'Policy & trade', body: 'Export bans, minimum export prices, export duties, and market interventions, drawn from a primary-source-verified event log.' },
];

const HORIZONS = [
  { h: '1 week', body: 'Dominated by price momentum — the model tracks what the market is already doing.' },
  { h: '4 weeks', body: 'Momentum still matters, but the other five layers begin contributing measurably.' },
  { h: '13 weeks', body: 'Slower-moving fundamentals — infrastructure, macro conditions — start to matter more than recent price alone.' },
  { h: '26 weeks', body: 'The longest horizon tested, where structural factors carry most of the signal for the crops they matter to.' },
];

export default function About() {
  return (
    <div className="min-h-screen bg-[var(--bg-app)]">
      <SiteNav />

      <section className="max-w-3xl mx-auto px-5 pt-16 pb-6">
        <FadeIn>
          <SectionLabel>About the project</SectionLabel>
          <h1 className="font-display text-3xl sm:text-4xl font-bold text-[var(--text-primary)] tracking-tight leading-tight">
            Forecasting isn't one-size-fits-all across crops
          </h1>
        </FadeIn>
        <FadeIn delay={0.05}>
          <p className="text-[var(--text-secondary)] leading-relaxed mt-5">
            TOP Digital Twin forecasts weekly wholesale prices for tomato, onion, and potato across India's APMC
            mandi network, at four horizons — 1, 4, 13, and 26 weeks — by fusing market data with satellite,
            climate, macro-economic, infrastructure, and policy signals. The project's central finding isn't a
            single "best" model: it's that <em>which</em> data layers help, and by how much, depends on the crop
            and the forecast horizon — and characterizing that heterogeneity, rather than papering over it, is the
            point.
          </p>
        </FadeIn>
        <FadeIn delay={0.1}>
          <p className="text-[var(--text-secondary)] leading-relaxed mt-4">
            Onion prices respond strongly to policy — export bans and duties have historically moved them in ways
            the model can pick up on. Tomato is highly perishable, so short-horizon forecasts are mostly about
            recent momentum, while longer horizons need slower-moving fundamentals. Potato's extensive cold-storage
            network already smooths out most of what the extra data layers could add, so simpler models hold their
            own there for longer. The dashboard's what-if simulator is built to make these differences visible,
            not hide them behind one number.
          </p>
        </FadeIn>
      </section>

      {/* Horizons */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>How the horizon changes what matters</SectionLabel>
        </FadeIn>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
          {HORIZONS.map((h, i) => (
            <FadeIn key={h.h} delay={0.04 * i}>
              <Card className="p-4">
                <p className="font-display font-bold text-[var(--brand)] text-sm">{h.h}</p>
                <p className="text-sm text-[var(--text-secondary)] mt-1 leading-relaxed">{h.body}</p>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* Data layers */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>The six data layers</SectionLabel>
          <p className="text-[var(--text-secondary)] mt-1 mb-5">
            Each layer is added incrementally in the underlying ablation study — from price alone up to the full
            fusion model — to measure exactly what each one contributes, crop by crop and horizon by horizon.
          </p>
        </FadeIn>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {LAYERS.map((l, i) => (
            <FadeIn key={l.title} delay={0.04 * i}>
              <Card className="p-4 flex gap-3 items-start">
                <span className="text-2xl shrink-0">{l.icon}</span>
                <div>
                  <p className="font-display font-bold text-[var(--text-primary)] text-sm">{l.title}</p>
                  <p className="text-sm text-[var(--text-secondary)] mt-0.5 leading-relaxed">{l.body}</p>
                </div>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* Model + honesty note */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>The model, and its limits</SectionLabel>
          <Card className="p-6">
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              Forecasts come from LightGBM (gradient-boosted trees) — one model per crop and horizon, 12 in total —
              chosen after being validated against tree-ensemble alternatives (XGBoost, CatBoost, Random Forest) and
              deep-learning baselines (LSTM, Transformer) on the same data. The what-if simulator's isolated feature
              effects and AI policy briefings are grounded strictly in what the model actually computed for a given
              scenario — not free-form commentary. Where the data or the model is genuinely uncertain (a stale
              market feed, a thin-data region, a scenario the model has little training signal for), the dashboard
              says so rather than presenting a confident number regardless.
            </p>
          </Card>
        </FadeIn>
      </section>

      <section className="max-w-3xl mx-auto px-5 py-14 text-center">
        <FadeIn>
          <p className="font-display text-xl font-bold text-[var(--text-primary)] mb-4">See it in action</p>
          <Link to="/dashboard"
            className="inline-block px-6 py-3 rounded-xl bg-[var(--brand)] text-white font-display font-semibold hover:bg-[var(--brand-light)] transition-colors">
            Open the Dashboard →
          </Link>
        </FadeIn>
      </section>

      <SiteFooter />
    </div>
  );
}
