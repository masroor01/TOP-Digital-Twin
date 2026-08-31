import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { loadAll } from './data.js';
import { buildRouter } from './routes.js';
import { PORT } from './config.js';

const store = loadAll();

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use('/api', buildRouter(store));

app.listen(PORT, () => {
  console.log(`[server] TOP Digital Twin backend listening on http://localhost:${PORT}`);
});
