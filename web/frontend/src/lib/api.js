import axios from 'axios';

const client = axios.create({ baseURL: '/api' });

export const api = {
  meta: () => client.get('/meta').then((r) => r.data),
  health: () => client.get('/health').then((r) => r.data),
  states: (crop) => client.get('/states', { params: { crop } }).then((r) => r.data),
  markets: (crop, state) => client.get('/markets', { params: { crop, state } }).then((r) => r.data),
  reference: (crop, marketId) => client.get('/reference', { params: { crop, marketId } }).then((r) => r.data),
  history: (crop, marketId) => client.get('/history', { params: { crop, marketId } }).then((r) => r.data),
  benchmarks: (crop) => client.get('/benchmarks', { params: { crop } }).then((r) => r.data),
  multiHistory: (crop, marketIds) =>
    client.get('/multi-history', { params: { crop, marketIds: marketIds.join(',') } }).then((r) => r.data),
  simulate: (payload) => client.post('/simulate', payload).then((r) => r.data),
  dailyCurve: (payload) => client.post('/daily-curve', payload).then((r) => r.data),
  aiBrief: (payload) => client.post('/ai-brief', payload).then((r) => r.data),
  directionalAccuracy: () => client.get('/directional-accuracy').then((r) => r.data),
};
