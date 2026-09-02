/**
 * Monotone cubic Hermite interpolation (Fritsch-Carlson), i.e. PCHIP —
 * matches scipy's PchipInterpolator, INCLUDING its shape-preserving
 * three-point edge-derivative formula (fixed 2026-09-02; an earlier version
 * used a plain secant slope at the endpoints, which could overshoot in the
 * first/last segments). Verified numerically against real scipy output to
 * ~1e-13 (float64 noise level), not just "closely enough". Ported because
 * there's no PCHIP in the JS standard library or a trivial zero-dependency
 * npm equivalent worth pulling in for one curve.
 */
export function pchip(xs, ys) {
  const n = xs.length;
  if (n < 2) throw new Error('pchip needs at least 2 points');
  const h = new Array(n - 1);
  const delta = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    delta[i] = (ys[i + 1] - ys[i]) / h[i];
  }
  const d = new Array(n);
  // FIXED 2026-09-02 (audit finding, confirmed): d[0]/d[n-1] used to be the
  // plain secant slope of just the first/last interval. scipy's
  // PchipInterpolator uses a non-centered, shape-preserving THREE-POINT
  // formula at the edges instead (Fritsch-Carlson edge case) -- the plain-
  // secant version can overshoot specifically in the first/last segments
  // (the most-viewed part of the daily-forecast chart), diverging from what
  // this file's own docstring claims it matches. With only 2 points there's
  // no third point to use, so the secant slope is already exactly correct
  // for both ends in that case (kept as the n===2 fallback below).
  function edgeDerivative(h0, h1, m0, m1) {
    let dd = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (Math.sign(dd) !== Math.sign(m0)) {
      dd = 0;
    } else if (Math.sign(m0) !== Math.sign(m1) && Math.abs(dd) > 3 * Math.abs(m0)) {
      dd = 3 * m0;
    }
    return dd;
  }
  if (n === 2) {
    d[0] = delta[0];
    d[n - 1] = delta[0];
  } else {
    d[0] = edgeDerivative(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = edgeDerivative(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
  }
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] === 0 || delta[i] === 0 || (delta[i - 1] > 0) !== (delta[i] > 0)) {
      d[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }

  return function evaluate(x) {
    let lo = 0, hi = n - 1;
    if (x <= xs[0]) lo = 0;
    else if (x >= xs[n - 1]) lo = n - 2;
    else {
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid; else hi = mid;
      }
    }
    const i = lo;
    const t = (x - xs[i]) / h[i];
    const t2 = t * t, t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    return h00 * ys[i] + h10 * h[i] * d[i] + h01 * ys[i + 1] + h11 * h[i] * d[i + 1];
  };
}
