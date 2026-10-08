// ════════════════════════════════════════════════════════════
//  RAG — Ask AI
// ════════════════════════════════════════════════════════════
import { clearFocus, focusOn, focusOnIds } from '../components/scatterPlot.js';
import { textToEmbedding } from '../lib/embedding.js';
import * as api from '../services/api.js';
import { state } from '../state.js';
import { $, escapeHtml, withBusy } from '../utils/dom.js';

const scrollToBottom = el => { el.scrollTop = el.scrollHeight; };

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

/**
 * Move the query star on the scatter plot. Uses the real (768-D) document
 * retrieval when possible; otherwise falls back to the 16-D demo space.
 */
async function visualizeQuestion(question, k) {
  try {
    const data = await api.searchDocs(question, k);
    const pts = (data.contexts || [])
      .map(ctx => state.pcaPoints.find(p => p.item.category === 'doc' && ctx.title.startsWith(p.item.metadata)))
      .filter(Boolean);
    if (pts.length) {
      focusOn(pts);
      return;
    }
    clearFocus();
    const fallback = await api.search(textToEmbedding(question), { k: 3, metric: 'cosine', algo: 'hnsw' });
    if (fallback.results?.length) focusOnIds(fallback.results.map(r => r.id));
  } catch { /* visualization is best-effort */ }
}

function typewriter(target, text, container) {
  target.classList.add('typing');
  let i = 0;
  const timer = setInterval(() => {
    if (i >= text.length) {
      clearInterval(timer);
      target.classList.remove('typing');
      return;
    }
    target.textContent += text.slice(i, i + 3);
    i += 3;
    scrollToBottom(container);
  }, 18);
}

function renderAnswer(d) {
  const contexts = d.contexts || [];
  const chips = contexts.map((c, i) => `
    <button class="ctx-chip" data-ctx="${i}">#${i + 1} ${escapeHtml(c.title)} · ${c.distance.toFixed(3)}</button>
    <div class="ctx-expand" id="ctx-${i}">${escapeHtml(c.text)}</div>`).join('');

  const div = el('div', 'chat-a', `
    <div class="chat-a-label">🤖 ${escapeHtml(d.model || 'llm')}</div>
    <div class="chat-a-text"></div>
    <div class="chat-ctx">
      <div class="chat-ctx-label">RETRIEVED CONTEXT (${contexts.length} chunks)</div>
      ${chips}
    </div>`);
  return div;
}

const renderError = msg =>
  el('div', 'chat-a', `<div class="chat-a-label">ERROR</div><div class="chat-a-text text-err">${escapeHtml(msg)}</div>`);

async function askAI() {
  const question = $('ragQuestion').value.trim();
  if (!question) return;
  const k = parseInt($('ragK').value, 10);
  const history = $('chatHistory');

  history.innerHTML = '';
  history.appendChild(el('div', 'chat-q')).textContent = question;
  const thinking = history.appendChild(
    el('div', 'thinking', '<div class="spinner"></div>Retrieving context &amp; generating answer…'),
  );
  scrollToBottom(history);

  visualizeQuestion(question, k); // runs in parallel with generation

  try {
    const d = await api.ask(question, k);
    thinking.remove();
    if (d.error) {
      history.appendChild(renderError(d.error));
    } else {
      const answer = history.appendChild(renderAnswer(d));
      typewriter(answer.querySelector('.chat-a-text'), d.answer || '', history);
    }
  } catch {
    thinking.remove();
    history.appendChild(renderError('Server error — is the backend running?'));
  }

  $('ragQuestion').value = '';
  scrollToBottom(history);
}

export function initRag() {
  const btn = $('askBtn');
  const run = () => withBusy(btn, 'Thinking…', askAI);
  btn.addEventListener('click', run);
  $('ragQuestion').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey && !btn.disabled) run();
  });
  $('chatHistory').addEventListener('click', e => {
    const chip = e.target.closest('.ctx-chip');
    if (chip) $(`ctx-${chip.dataset.ctx}`)?.classList.toggle('open');
  });
}
