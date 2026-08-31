import 'dotenv/config';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import express from 'express';
import cors from 'cors';
import { loadAll } from './data.js';
import { buildRouter } from './routes.js';
import { PORT } from './config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const store = loadAll();

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
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
