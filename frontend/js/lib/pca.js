// ════════════════════════════════════════════════════════════
//  PCA → 2D via power iteration (top two principal components)
// ════════════════════════════════════════════════════════════

const dot = (a, b) => a.reduce((s, ai, i) => s + ai * b[i], 0);
const norm = v => Math.sqrt(dot(v, v));

function orthogonalize(v, against) {
  if (!against) return v;
  const d = dot(v, against);
  return v.map((vi, i) => vi - d * against[i]);
}

function powerIteration(X, d, exclude = null, iters = 200) {
  const n = X.length;
  let v = orthogonalize(Array.from({ length: d }, () => Math.random() - 0.5), exclude);
  let nrm = norm(v);
  v = v.map(x => x / nrm);

  for (let it = 0; it < iters; it++) {
    const Xv = X.map(row => dot(row, v));
    let nv = new Array(d).fill(0);
    for (let k = 0; k < n; k++) for (let j = 0; j < d; j++) nv[j] += X[k][j] * Xv[k];
    nv = orthogonalize(nv, exclude);
    nrm = norm(nv);
    if (nrm < 1e-10) break;
    const prev = v;
    v = nv.map(x => x / nrm);
    if (v.reduce((s, vi, i) => s + (vi - prev[i]) ** 2, 0) < 1e-12) break;
  }
  return v;
}

/** Project an array of equal-length vectors onto their top-2 principal components. */
export function pca2D(embs) {
  const n = embs.length;
  if (n < 2) return embs.map(() => [0, 0]);
  const d = embs[0].length;

  const mean = new Array(d).fill(0);
  for (const e of embs) for (let i = 0; i < d; i++) mean[i] += e[i] / n;
  const X = embs.map(e => e.map((v, i) => v - mean[i]));

  const pc1 = powerIteration(X, d);
  const pc2 = powerIteration(X, d, pc1);
  return X.map(x => [dot(x, pc1), dot(x, pc2)]);
}
