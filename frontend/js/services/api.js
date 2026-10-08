// ════════════════════════════════════════════════════════════
//  REST client for the FastAPI backend
// ════════════════════════════════════════════════════════════
import { API } from '../config.js';

async function request(path, { method = 'GET', body } = {}) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  // Backend always returns JSON — including {"error": "..."} on failures.
  return res.json();
}

const vecParam = v => v.map(x => x.toFixed(4)).join(',');

// ── Demo vectors ───────────────────────────────────────────
export const getItems = () => request('/items');

export const search = (emb, { k = 5, metric = 'cosine', algo = 'hnsw' } = {}) =>
  request(`/search?v=${vecParam(emb)}&k=${k}&metric=${metric}&algo=${algo}`);

export const benchmark = (emb, { k = 5, metric = 'cosine' } = {}) =>
  request(`/benchmark?v=${vecParam(emb)}&k=${k}&metric=${metric}`);

export const insertVector = (metadata, category, embedding) =>
  request('/insert', { method: 'POST', body: { metadata, category, embedding } });

export const deleteVector = id => request(`/delete/${id}`, { method: 'DELETE' });

export const getHnswInfo = () => request('/hnsw-info');

// ── Documents / RAG ────────────────────────────────────────
export const getStatus = () => request('/status');

export const listDocs = () => request('/doc/list');

export const insertDoc = (title, text) =>
  request('/doc/insert', { method: 'POST', body: { title, text } });

export const deleteDoc = id => request(`/doc/delete/${id}`, { method: 'DELETE' });

export const searchDocs = (question, k) =>
  request('/doc/search', { method: 'POST', body: { question, k } });

export const ask = (question, k) =>
  request('/doc/ask', { method: 'POST', body: { question, k } });
