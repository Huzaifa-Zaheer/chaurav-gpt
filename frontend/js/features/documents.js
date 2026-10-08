// ════════════════════════════════════════════════════════════
//  DOCUMENTS — Ollama status, insert, list, delete
// ════════════════════════════════════════════════════════════
import { textToEmbedding } from '../lib/embedding.js';
import * as api from '../services/api.js';
import { emit, EVENTS } from '../state.js';
import { $, escapeHtml, note, withBusy } from '../utils/dom.js';

export async function checkOllamaStatus() {
  const badge = $('ollamaBadge');
  const box = $('ollamaStatus');
  try {
    const d = await api.getStatus();
    if (d.ollamaAvailable) {
      badge.className = 'badge ok';
      badge.textContent = 'OLLAMA ✓';
      box.className = 'ollama-status ok';
      box.innerHTML = `<span class="text-ok">● Online</span><br>
        Embed: <span class="text-acc">${escapeHtml(d.embedModel)}</span><br>
        Generate: <span class="text-acc">${escapeHtml(d.genModel)}</span><br>
        Dims: <span class="note">${d.docDims || '(first insert sets this)'}</span><br>
        Documents: <span>${d.docCount}</span>`;
    } else {
      badge.className = 'badge err';
      badge.textContent = 'OLLAMA ✗';
      box.className = 'ollama-status err';
      box.innerHTML = `<span class="text-err">● Offline</span><br><br>
        To enable RAG features:<br>
        <span class="note">1. Install from ollama.com<br>
        2. ollama pull nomic-embed-text<br>
        3. ollama pull llama3.2</span>`;
    }
  } catch {
    badge.className = 'badge err';
    badge.textContent = 'BACKEND ✗';
  }
}

export async function loadDocList() {
  try {
    const docs = await api.listDocs();
    $('docCountLabel').textContent = docs.length;
    $('docList').innerHTML = docs.length
      ? docs.map(d => `
        <div class="dcard">
          <div class="dcard-title">${escapeHtml(d.title)}</div>
          <div class="dcard-preview">${escapeHtml(d.preview)}</div>
          <div class="dcard-foot">
            <span class="dcard-words">${d.words} words</span>
            <button class="del" data-action="delete-doc" data-id="${d.id}" title="Delete chunk">✕</button>
          </div>
        </div>`).join('')
      : note('No documents yet.');
  } catch { /* ignore */ }
}

async function insertDocument() {
  const title = $('docTitle').value.trim();
  const text = $('docText').value.trim();
  const status = $('insertStatus');
  if (!title || !text) {
    status.textContent = '⚠ Need both a title and text.';
    return;
  }

  status.innerHTML = '<span class="note">Calling Ollama nomic-embed-text…</span>';
  try {
    const d = await api.insertDoc(title, text);
    if (d.error) {
      status.innerHTML = `<span class="text-err">✗ ${escapeHtml(d.error)}</span>`;
      return;
    }
    status.innerHTML = `<span class="text-ok">✓ Inserted ${d.chunks} chunk(s) · ${d.dims}D embeddings</span>`;
    $('docTitle').value = '';
    $('docText').value = '';

    // Mirror the doc as a 16-D demo vector so it shows up on the scatter plot.
    await api.insertVector(title, 'doc', textToEmbedding(`${title} ${text}`));
    emit(EVENTS.VECTORS_CHANGED);
    emit(EVENTS.DOCS_CHANGED);
  } catch {
    status.innerHTML = '<span class="text-err">✗ Server error</span>';
  }
}

async function deleteDoc(id) {
  try {
    await api.deleteDoc(id);
    emit(EVENTS.DOCS_CHANGED);
  } catch { /* ignore */ }
}

export function initDocuments() {
  const btn = $('insertDocBtn');
  btn.addEventListener('click', () => withBusy(btn, 'Embedding…', insertDocument));
  $('docList').addEventListener('click', e => {
    const del = e.target.closest('[data-action="delete-doc"]');
    if (del) deleteDoc(Number(del.dataset.id));
  });
}
