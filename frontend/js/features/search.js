// ════════════════════════════════════════════════════════════
//  DEMO VECTORS — load, search, insert, delete
// ════════════════════════════════════════════════════════════
import { DIMS, colorFor } from '../config.js';
import { focusOnIds, setItems } from '../components/scatterPlot.js';
import { drawVectorChart } from '../components/vectorChart.js';
import { textToEmbedding } from '../lib/embedding.js';
import * as api from '../services/api.js';
import { emit, EVENTS, state } from '../state.js';
import { $, escapeHtml, formatLatency, note } from '../utils/dom.js';

export async function loadItems() {
  try {
    const items = await api.getItems();
    setItems(items);
    $('statsLabel').textContent = `${items.length} vectors · ${DIMS} dims`;
  } catch {
    $('statsLabel').textContent = 'backend offline';
  }
}

function renderResults(results) {
  const box = $('results');
  if (!results?.length) {
    box.innerHTML = note('No results');
    return;
  }
  box.innerHTML = results.map((r, i) => {
    const col = colorFor(r.category);
    return `<div class="rcard" data-id="${r.id}">
      <div class="rrank">#${i + 1} NEAREST</div>
      <div class="rmeta">${escapeHtml(r.metadata)}</div>
      <div class="rfoot">
        <span class="rcat" style="background:${col}18;color:${col};border:1px solid ${col}44">${escapeHtml(r.category.toUpperCase())}</span>
        <span class="rdist">dist: ${r.distance.toFixed(5)}</span>
        <button class="del" data-action="delete" data-id="${r.id}" title="Delete vector">✕</button>
      </div>
    </div>`;
  }).join('');
}

export async function runSearch() {
  const text = $('qInput').value.trim();
  if (!text) return;
  const emb = textToEmbedding(text);
  const k = parseInt($('kSlider').value, 10);
  const metric = $('metric').value;

  try {
    const data = await api.search(emb, { k, metric, algo: state.algo });
    if (data.error) throw new Error(data.error);
    state.searchResults = data.results || [];
    focusOnIds(state.searchResults.map(r => r.id));

    $('latBig').textContent = formatLatency(data.latencyUs || 0);
    $('latSub').textContent = `${state.algo.toUpperCase()}  ·  ${metric}  ·  k=${k}`;
    renderResults(state.searchResults);
    drawVectorChart(emb);
  } catch {
    alert('Cannot reach server — is it running on :8080?');
  }
}

async function addVector() {
  const meta = $('addMeta').value.trim();
  const cat = $('addCat').value;
  if (!meta) return;
  try {
    await api.insertVector(meta, cat, textToEmbedding(`${meta} ${cat}`));
    $('addMeta').value = '';
    emit(EVENTS.VECTORS_CHANGED);
  } catch { /* ignore */ }
}

async function deleteItem(id) {
  try {
    await api.deleteVector(id);
    state.searchResults = state.searchResults.filter(r => r.id !== id);
    state.hitIds.delete(id);
    renderResults(state.searchResults);
    emit(EVENTS.VECTORS_CHANGED);
  } catch { /* ignore */ }
}

function setAlgo(btn) {
  document.querySelectorAll('.algo-btn').forEach(b => b.classList.toggle('on', b === btn));
  state.algo = btn.dataset.algo;
}

export function initSearch() {
  $('searchBtn').addEventListener('click', runSearch);
  $('qInput').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
  $('addVectorBtn').addEventListener('click', addVector);
  $('addMeta').addEventListener('keydown', e => { if (e.key === 'Enter') addVector(); });
  $('kSlider').addEventListener('input', e => { $('kLabel').textContent = e.target.value; });
  document.querySelectorAll('.algo-btn').forEach(b => b.addEventListener('click', () => setAlgo(b)));

  // Event delegation for dynamically rendered result cards.
  const results = $('results');
  results.addEventListener('click', e => {
    const btn = e.target.closest('[data-action="delete"]');
    if (btn) deleteItem(Number(btn.dataset.id));
  });
  results.addEventListener('mouseover', e => {
    const card = e.target.closest('.rcard');
    state.hoverId = card ? Number(card.dataset.id) : null;
  });
  results.addEventListener('mouseleave', () => { state.hoverId = null; });
}
