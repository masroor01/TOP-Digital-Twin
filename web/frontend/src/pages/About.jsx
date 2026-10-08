import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import SiteNav from '../components/SiteNav';
import SiteFooter from '../components/SiteFooter';
import { Card, FadeIn, SectionLabel } from '../components/ui';

const LAYERS = [
  { icon: '🌾', title: 'Market', gradient: 'linear-gradient(135deg, var(--brand) 0%, var(--brand-light) 100%)', body: 'Weekly wholesale price and arrivals data from Agmarknet, covering APMC mandis across India since 2017.' },
  { icon: '🛰️', title: 'Satellite vegetation', gradient: 'linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%)', body: 'Sentinel-2 and MODIS NDVI/EVI — a proxy for crop health and expected supply ahead of harvest.' },
  { icon: '🌦️', title: 'Climate stress', gradient: 'linear-gradient(135deg, #0284C7 0%, #7DD3FC 100%)', body: 'ERA5 temperature extremes and CHIRPS rainfall, tracking heat stress and flood/drought risk by growing zone.' },
  { icon: '💹', title: 'Macro-logistics', gradient: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%)', body: 'Diesel and LPG prices, the RBI repo rate, USD/INR, and wholesale price indices — the cost side of getting produce to market.' },
  { icon: '🏗️', title: 'Infrastructure', gradient: 'linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%)', body: 'Cold storage capacity and road density by state, and agricultural wage data — how well a region can store and move produce.' },
  { icon: '📜', title: 'Policy & trade', gradient: 'linear-gradient(135deg, #B45309 0%, #F59E0B 100%)', body: 'Export bans, minimum export prices, export duties, and market interventions, drawn from a primary-source-verified event log.' },
];

const HORIZONS = [
  { h: '1 week', body: 'Dominated by price momentum — the model tracks what the market is already doing.' },
  { h: '4 weeks', body: 'Momentum still matters, but the other five layers begin contributing measurably.' },
  { h: '13 weeks', body: 'Slower-moving fundamentals — infrastructure, macro conditions — start to matter more than recent price alone.' },
  { h: '26 weeks', body: 'The longest horizon tested, where structural factors carry most of the signal for the crops they matter to.' },
];

export default function About() {
  return (
    <div className="min-h-screen bg-[var(--bg-app)] overflow-x-hidden">
      <SiteNav />

      <section className="relative">
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-[700px] h-[420px] rounded-full blur-3xl opacity-25"
            style={{ background: 'radial-gradient(circle, var(--brand) 0%, transparent 65%)' }} />
        </div>
        <div className="max-w-3xl mx-auto px-5 pt-16 pb-6">
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
              Onion prices have often moved around export bans and duties, and the model uses these policy events as
              inputs; the dashboard does not claim the policies caused those moves. Tomato is highly perishable, so short-horizon forecasts are mostly about
              recent momentum, while longer horizons need slower-moving fundamentals. Potato's extensive cold-storage
              network already smooths out most of what the extra data layers could add, so simpler models hold their
              own there for longer. The dashboard's what-if simulator is built to make these differences visible,
              not hide them behind one number.
            </p>
          </FadeIn>
        </div>
      </section>

      {/* Horizons */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>How the horizon changes what matters</SectionLabel>
        </FadeIn>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
          {HORIZONS.map((h, i) => (
            <FadeIn key={h.h} delay={0.04 * i}>
              <Card className="p-4" accent="var(--brand)">
                <p className="font-display font-bold text-[var(--brand-text)] text-sm">{h.h}</p>
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
                <span className="w-10 h-10 rounded-xl flex items-center justify-center text-lg shrink-0 shadow-sm" style={{ background: l.gradient }}>
                  {l.icon}
                </span>
                <div>
                  <p className="font-display font-bold text-[var(--text-primary)] text-sm">{l.title}</p>
                  <p className="text-sm text-[var(--text-secondary)] mt-0.5 leading-relaxed">{l.body}</p>
                </div>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* What the evidence shows */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>What the evidence shows (re-run 7 October 2026)</SectionLabel>
        </FadeIn>
        <div className="grid grid-cols-1 gap-3 mt-3">
          {[
            ['Forecast skill depends on the horizon', 'Against a no-change guess, tomato error is about 37 % and 38 % lower at 13 and 26 weeks; onion 6 % and 20 % lower. At one week no crop beats the guess, and potato does not beat it within 26 weeks.'],
            ['Beats classical methods at long range', 'At 13 and 26 weeks the model has lower error than the best of persistence, seasonal naive, a 4-week moving average and ARIMA in 5 of 6 crop-horizon cells (potato at 13 weeks is the exception).'],
            ['Price crises: medium-range warning, not week-ahead precision', 'In four documented episodes the model had far lower error than a no-change guess at 13 weeks in three (tomato 2023: 23 % vs 158 %; onion 2023–24: 21 % vs 51 %; potato crash 2025), and was worse in the 2024 potato spike. All four were missed at one week.'],
            ['Extra data layers: indicated, not certified', 'They lower error in many tomato and onion cells, and they carry most of the signal at long horizons, but no layer passes the strict yearly-fold test (five test years can certify only a 5-of-5 sweep). A 5-of-5 result for the El Niño layer appears in a different cell with each random seed, so it is not claimed.'],
            ['Early-warning decisions', 'Models trained directly on the event (a +30 % spike or −25 % crash within 13 weeks) with calibrated probabilities make alarms far more robust than alarms from a price forecast. A pre-registered test found no reliable decision benefit from the data layers, so we do not claim one. Onion is being confirmed on new weeks.'],
          ].map(([t, b], i) => (
            <FadeIn key={t} delay={0.04 * i}>
              <Card className="p-4" accent="var(--brand)">
                <p className="font-display font-bold text-[var(--text-primary)] text-sm">{t}</p>
                <p className="text-sm text-[var(--text-secondary)] mt-1 leading-relaxed">{b}</p>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* What we do not claim */}
      <section className="max-w-3xl mx-auto px-5 py-6">
        <FadeIn>
          <SectionLabel>What we do not claim</SectionLabel>
          <Card className="p-5" accent="var(--accent)">
            <ul className="text-sm text-[var(--text-secondary)] leading-relaxed list-disc pl-5 space-y-1.5">
              <li>Scenario results are <b>model sensitivities, not causal estimates</b>. Two quasi-experimental designs for the price effect of onion export policy failed their placebo and pre-trend checks, so no causal magnitude is claimed.</li>
              <li>The model is <b>not a week-ahead trading signal</b>.</li>
              <li>Forecast accuracy is reported as typical error (WAPE) by crop and horizon; the dashboard's "Model Accuracy" is 100 minus that error.</li>
              <li>All tests were run on unseen years with training and validation rows chosen by forecast-target date; results were repeated across random seeds to separate real effects from noise.</li>
            </ul>
          </Card>
        </FadeIn>
      </section>

      {/* Model + honesty note */}
      <section className="max-w-3xl mx-auto px-5 py-10">
        <FadeIn>
          <SectionLabel>The model, and its limits</SectionLabel>
          <div className="relative rounded-2xl p-6 overflow-hidden shadow-[0_20px_50px_-24px_rgba(15,45,32,0.4)]"
            style={{ background: 'linear-gradient(135deg, var(--brand-dark) 0%, var(--brand) 100%)' }}>
            <div className="pointer-events-none absolute -bottom-10 -right-10 w-48 h-48 rounded-full blur-3xl opacity-25"
              style={{ background: 'var(--accent-light)' }} />
            <p className="text-sm text-white/85 leading-relaxed relative">
              Forecasts come from LightGBM (gradient-boosted trees) — one model per crop and horizon, 12 in total —
              chosen after being validated against tree-ensemble alternatives (XGBoost, CatBoost, Random Forest) and
              deep-learning baselines (LSTM, Transformer) on the same data. The what-if simulator's isolated feature
              effects and AI policy briefings are grounded strictly in what the model actually computed for a given
              scenario — not free-form commentary. Where the data or the model is genuinely uncertain (a stale
              market feed, a thin-data region, a scenario the model has little training signal for), the dashboard
              says so rather than presenting a confident number regardless.
            </p>
          </div>
        </FadeIn>
      </section>

      <section className="max-w-3xl mx-auto px-5 py-14 text-center">
        <FadeIn>
          <p className="font-display text-xl font-bold text-[var(--text-primary)] mb-4">See it in action</p>
          <Link to="/dashboard">
            <motion.span whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
              className="inline-block px-6 py-3 rounded-xl text-white font-display font-semibold shadow-[0_8px_28px_-8px_var(--brand)]"
              style={{ background: 'linear-gradient(135deg, var(--brand) 0%, var(--brand-light) 100%)' }}>
              Open the Dashboard →
            </motion.span>
          </Link>
        </FadeIn>
      </section>

      <SiteFooter />
    </div>
  );
}
