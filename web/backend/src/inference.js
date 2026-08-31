import { INFERENCE_URL } from './config.js';

/**
 * items: [{ id, crop, horizon, features }]
 * returns: { [id]: price }
 */
export async function predictBatch(items) {
  if (items.length === 0) return {};
  const res = await fetch(`${INFERENCE_URL}/predict/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Inference service error ${res.status}: ${text}`);
  }
  const data = await res.json();
  return data.predictions;
}

export async function inferenceHealth() {
  try {
    const res = await fetch(`${INFERENCE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return { ok: false };
    return { ok: true, ...(await res.json()) };
  } catch {
    return { ok: false };
  }
}
