// ════════════════════════════════════════════════════════════
//  TEXT → 16-D EMBEDDING (keyword-based, for demo vectors)
// ════════════════════════════════════════════════════════════
import { DIMS } from '../config.js';

const KW = {
  cs: ['algorithm', 'data', 'tree', 'graph', 'array', 'linked', 'hash', 'stack', 'queue', 'sort', 'binary', 'dynamic', 'programming', 'recursion', 'complexity', 'pointer', 'node', 'search', 'insert', 'bfs', 'dfs', 'heap', 'trie'],
  math: ['calculus', 'matrix', 'probability', 'theorem', 'integral', 'derivative', 'linear', 'algebra', 'equation', 'function', 'prime', 'modular', 'combinatorics', 'permutation', 'eigenvalue', 'statistics', 'proof'],
  food: ['food', 'pizza', 'sushi', 'ramen', 'pasta', 'recipe', 'cook', 'eat', 'restaurant', 'dish', 'ingredient', 'flavor', 'spice', 'noodle', 'bread', 'croissant', 'taco', 'fish', 'rice', 'soup'],
  sports: ['sport', 'basketball', 'football', 'tennis', 'chess', 'swim', 'game', 'play', 'score', 'team', 'athlete', 'competition', 'match', 'tournament', 'olympic', 'dribble', 'tackle', 'serve'],
};

const OFFSETS = { cs: 0, math: 4, food: 8, sports: 12 };

export function textToEmbedding(text) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const score = { cs: 0, math: 0, food: 0, sports: 0 };

  for (const w of words)
    for (const [cat, kws] of Object.entries(KW))
      if (kws.some(kw => w.includes(kw) || kw.startsWith(w))) score[cat] += 0.35;

  const mx = Math.max(...Object.values(score), 0.01);
  const norm = v => Math.min((v / mx) * 0.88, 0.94);
  const jitter = () => (Math.random() - 0.5) * 0.04;
  const emb = new Array(DIMS).fill(0.08);

  for (const [cat, i] of Object.entries(OFFSETS)) {
    if (score[cat] < 0.01) continue;
    const b = norm(score[cat]);
    emb[i] = Math.max(0.05, b + jitter());
    emb[i + 1] = Math.max(0.05, b + jitter());
    emb[i + 2] = Math.max(0.05, b * 0.92 + jitter());
    emb[i + 3] = Math.max(0.05, b * 0.87 + jitter());
  }
  return emb;
}
