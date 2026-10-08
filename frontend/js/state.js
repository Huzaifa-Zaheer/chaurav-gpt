// ════════════════════════════════════════════════════════════
//  SHARED APP STATE + tiny event bus
// ════════════════════════════════════════════════════════════

export const state = {
  /** Demo vectors from /items */
  items: [],
  /** [{x, y, item}] — PCA-projected items */
  pcaPoints: [],
  /** IDs highlighted on the scatter plot */
  hitIds: new Set(),
  /** {x, y} position of the query star, or null */
  queryPt: null,
  /** ID of the item under the cursor / hovered result card */
  hoverId: null,
  /** Selected search algorithm */
  algo: 'hnsw',
  /** Last search results */
  searchResults: [],
};

const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, payload) {
  listeners.get(event)?.forEach(fn => fn(payload));
}

/** Event names used across modules. */
export const EVENTS = {
  VECTORS_CHANGED: 'vectors:changed',
  DOCS_CHANGED: 'docs:changed',
  TAB_CHANGED: 'tab:changed',
};
