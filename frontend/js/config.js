// ════════════════════════════════════════════════════════════
//  CONSTANTS
// ════════════════════════════════════════════════════════════

// When served by the FastAPI backend locally, use same-origin requests (no prefix).
// When deployed to Vercel, API routes live under /api.
// When index.html is opened directly from disk, fall back to localhost:8080.
function _resolveApi() {
  if (!location.protocol.startsWith('http')) return 'http://localhost:8080';
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') return '';
  return '/api';   // Vercel or any remote deployment
}
export const API = _resolveApi();

export const DIMS = 16;

export const COL = {
  cs: '#00d9ff',
  math: '#b388ff',
  food: '#ffb74d',
  sports: '#69f0ae',
  doc: '#a6e3a1',
  default: '#90a4ae',
};

export const DIM_GROUPS = [
  { label: 'CS', start: 0, color: COL.cs },
  { label: 'MATH', start: 4, color: COL.math },
  { label: 'FOOD', start: 8, color: COL.food },
  { label: 'SPORT', start: 12, color: COL.sports },
];

/** Colour of each of the 16 demo dimensions (4 per category). */
export const DIM_COL = DIM_GROUPS.flatMap(g => Array(4).fill(g.color));

export const colorFor = category => COL[category] || COL.default;
