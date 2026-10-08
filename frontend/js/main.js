// ════════════════════════════════════════════════════════════
//  BOOT — wire modules together
// ════════════════════════════════════════════════════════════
import { initScatterPlot } from './components/scatterPlot.js';
import { initBenchmark, loadHnswLayers } from './features/benchmark.js';
import { checkOllamaStatus, initDocuments, loadDocList } from './features/documents.js';
import { initRag } from './features/rag.js';
import { initSearch, loadItems } from './features/search.js';
import { initTabs } from './features/tabs.js';
import { EVENTS, on } from './state.js';

async function refreshVectors() {
  await loadItems();
  await loadHnswLayers();
}

function boot() {
  initScatterPlot();
  initTabs();
  initSearch();
  initBenchmark();
  initDocuments();
  initRag();

  // Cross-module reactions
  on(EVENTS.VECTORS_CHANGED, refreshVectors);
  on(EVENTS.DOCS_CHANGED, () => { loadDocList(); checkOllamaStatus(); });
  on(EVENTS.TAB_CHANGED, tab => { if (tab === 'docs') loadDocList(); });

  refreshVectors();
  checkOllamaStatus();
}

boot();
