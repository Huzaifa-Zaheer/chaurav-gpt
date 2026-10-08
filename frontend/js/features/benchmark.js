// ════════════════════════════════════════════════════════════
//  BENCHMARK + HNSW LAYER STATS
// ════════════════════════════════════════════════════════════
import { textToEmbedding } from '../lib/embedding.js';
import * as api from '../services/api.js';
import { $, formatLatency, note, withBusy } from '../utils/dom.js';

const BARS = [
  { key: 'bruteforceUs', label: 'Brute Force', color: '#f38ba8' },
  { key: 'kdtreeUs', label: 'KD-Tree', color: '#89dceb' },
  { key: 'hnswUs', label: 'HNSW', color: '#b388ff' },
];

export async function runBenchmark() {
  const text = $('qInput').value.trim() || 'binary tree algorithm';
  const emb = textToEmbedding(text);
  const metric = $('metric').value;
  try {
    const d = await api.benchmark(emb, { k: 5, metric });
    if (d.error) return;
    $('benchSec').classList.remove('hidden');
    const mx = Math.max(...BARS.map(b => d[b.key]), 1);
    $('benchBars').innerHTML = BARS.map(({ key, label, color }) => {
      const us = d[key];
      const pct = Math.max((us / mx) * 100, 2);
      return `<div class="brow">
        <div class="blabel"><span style="color:${color}">${label}</span><span class="note">${formatLatency(us)}</span></div>
        <div class="btrack"><div class="bfill" style="width:${pct}%;background:${color}"></div></div>
      </div>`;
    }).join('');
  } catch { /* ignore */ }
}

export async function loadHnswLayers() {
  try {
    const d = await api.getHnswInfo();
    const maxN = d.nodesPerLayer[0] || 1;
    $('layers').innerHTML = d.nodesPerLayer.map((cnt, lyr) => {
      const pct = Math.max((cnt / maxN) * 100, 2);
      const edges = d.edgesPerLayer[lyr] || 0;
      return `<div class="lrow">
        <div class="lnum">L${lyr}</div>
        <div class="ltrack"><div class="lfill" style="width:${pct}%"></div></div>
        <div class="lcount">${cnt}n · ${edges}e</div>
      </div>`;
    }).join('') || note('Empty');
  } catch { /* ignore */ }
}

export function initBenchmark() {
  const btn = $('benchmarkBtn');
  btn.addEventListener('click', () => withBusy(btn, 'Benchmarking…', runBenchmark));
}
