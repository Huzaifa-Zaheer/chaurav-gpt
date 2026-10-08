// ════════════════════════════════════════════════════════════
//  DOM helpers
// ════════════════════════════════════════════════════════════

export const $ = id => document.getElementById(id);

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape user-provided text before interpolating into innerHTML. */
export const escapeHtml = s => String(s ?? '').replace(/[&<>"']/g, c => ESC[c]);

export const formatLatency = us => (us < 1000 ? `${us} μs` : `${(us / 1000).toFixed(2)} ms`);

export const note = text => `<div class="note">${escapeHtml(text)}</div>`;

/** Temporarily disable a button and swap its label while `fn` runs. */
export async function withBusy(btn, busyLabel, fn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = busyLabel;
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}
