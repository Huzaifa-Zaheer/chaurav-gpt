// ════════════════════════════════════════════════════════════
//  TABS
// ════════════════════════════════════════════════════════════
import { emit, EVENTS } from '../state.js';

export function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === name));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('on', c.id === `tab-${name}`));
  emit(EVENTS.TAB_CHANGED, name);
}

export function initTabs() {
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));
}
