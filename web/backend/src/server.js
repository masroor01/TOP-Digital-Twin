import 'dotenv/config';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import express from 'express';
import cors from 'cors';
import { rateLimit } from 'express-rate-limit';
import { loadAll } from './data.js';
import { buildRouter } from './routes.js';
import { PORT } from './config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const store = loadAll();

const app = express();
app.disable('x-powered-by');
app.set('trust proxy', 1); // behind Hostinger's edge/CDN -- needed for rate-limit to key on the real client IP, not the proxy's

// Minimal manual security headers (no helmet dependency for one line each).
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

app.use(cors());
app.use(express.json({ limit: '2mb' }));

// General API rate limit -- generous (this is a public, read-mostly demo
// tool with no accounts), but stops a single client from hammering it.
app.use('/api', rateLimit({ windowMs: 60 * 1000, limit: 120, standardHeaders: true, legacyHeaders: false }));

app.use('/api', buildRouter(store));

// Serve the built React frontend (web/frontend/dist) as static files, with
// an SPA catch-all so client-side routing works on a hard refresh. Lets the
// whole app run as ONE Express process -- deliberate, since Hostinger's
// "Deploy Web App" only supports single-process Node runtimes (no separate
// static host + API server pair, and no Python for a third process either).
const FRONTEND_DIST = path.resolve(__dirname, '..', '..', 'frontend', 'dist');
if (fs.existsSync(FRONTEND_DIST)) {
  app.use(express.static(FRONTEND_DIST));
  app.get(/^(?!\/api).*/, (req, res) => {
    res.sendFile(path.join(FRONTEND_DIST, 'index.html'));
  });
} else {
  console.warn(`[server] ${FRONTEND_DIST} not found -- run "npm run build" in web/frontend first. API-only mode.`);
}

app.listen(PORT, () => {
  console.log(`[server] TOP Digital Twin backend listening on http://localhost:${PORT}`);
});
