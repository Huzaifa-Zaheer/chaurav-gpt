// ════════════════════════════════════════════════════════════
//  SCATTER PLOT — 2D PCA projection of the semantic space
// ════════════════════════════════════════════════════════════
import { colorFor } from '../config.js';
import { pca2D } from '../lib/pca.js';
import { state } from '../state.js';
import { $, escapeHtml } from '../utils/dom.js';

const PAD = 70;
let canvas, ctx, tip;
let bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
let pulse = 0;

// ── coordinate transform ───────────────────────────────────
function worldToCanvas(wx, wy) {
  const W = canvas.width, H = canvas.height;
  const rx = bounds.maxX - bounds.minX || 1;
  const ry = bounds.maxY - bounds.minY || 1;
  return [
    PAD + ((wx - bounds.minX) / rx) * (W - 2 * PAD),
    H - PAD - ((wy - bounds.minY) / ry) * (H - 2 * PAD),
  ];
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  canvas.width = r.width;
  canvas.height = r.height;
}

// ── public API ─────────────────────────────────────────────

/** Recompute PCA projection + bounds for the current `state.items`. */
export function setItems(items) {
  state.items = items;
  if (items.length < 2) {
    state.pcaPoints = [];
    return;
  }
  const coords = pca2D(items.map(v => v.embedding));
  state.pcaPoints = items.map((item, i) => ({ x: coords[i][0], y: coords[i][1], item }));

  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of state.pcaPoints) {
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
    y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  const px = (x1 - x0) * 0.18 || 0.1, py = (y1 - y0) * 0.18 || 0.1;
  bounds = { minX: x0 - px, maxX: x1 + px, minY: y0 - py, maxY: y1 + py };
}

/**
 * Highlight the given points and place the query star at their
 * rank-weighted centroid (top 3, weights 1, 1/2, 1/3).
 * @param {Array<{x:number,y:number,item:object}>} points ordered by relevance
 */
export function focusOn(points) {
  state.hitIds = new Set(points.map(p => p.item.id));
  let sx = 0, sy = 0, sw = 0;
  points.slice(0, 3).forEach((pt, i) => {
    const w = 1 / (i + 1);
    sx += pt.x * w; sy += pt.y * w; sw += w;
  });
  if (sw > 0) {
    const j = () => (Math.random() - 0.5) * 0.015;
    state.queryPt = { x: sx / sw + j(), y: sy / sw + j() };
  }
}

/** Convenience: focus on demo items by id (in relevance order). */
export function focusOnIds(ids) {
  const pts = ids.map(id => state.pcaPoints.find(p => p.item.id === id)).filter(Boolean);
  focusOn(pts);
  state.hitIds = new Set(ids);
}

export function clearFocus() {
  state.hitIds = new Set();
}

// ── drawing ────────────────────────────────────────────────
function drawGrid() {
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#07070f';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#0e0e1e';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const tx = PAD + (i / 8) * (W - 2 * PAD), ty = PAD + (i / 8) * (H - 2 * PAD);
    ctx.beginPath(); ctx.moveTo(tx, PAD); ctx.lineTo(tx, H - PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, ty); ctx.lineTo(W - PAD, ty); ctx.stroke();
  }
  ctx.fillStyle = '#1a1a38';
  ctx.font = '11px Fira Code,monospace';
  ctx.fillText('PC₁ →', W / 2 - 40, H - 18);
  ctx.save(); ctx.translate(18, H / 2 + 50); ctx.rotate(-Math.PI / 2); ctx.fillText('PC₂ →', 0, 0); ctx.restore();
  ctx.fillStyle = '#151530';
  ctx.font = '12px Fira Code,monospace';
  ctx.fillText('2D PCA Projection  ·  Semantic Space', 80, 28);
}

function drawQueryLinks() {
  if (!state.queryPt || !state.hitIds.size) return;
  const [qx, qy] = worldToCanvas(state.queryPt.x, state.queryPt.y);
  ctx.strokeStyle = 'rgba(108,99,255,0.18)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  for (const pt of state.pcaPoints) {
    if (!state.hitIds.has(pt.item.id)) continue;
    const [px, py] = worldToCanvas(pt.x, pt.y);
    ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(px, py); ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawPoints() {
  for (const pt of state.pcaPoints) {
    const [cx, cy] = worldToCanvas(pt.x, pt.y);
    const col = colorFor(pt.item.category);
    const isHit = state.hitIds.has(pt.item.id);
    const r = isHit ? 10 : 7;

    if (isHit) {
      const pr = r + 7 + Math.sin(pulse) * 3.5;
      ctx.beginPath(); ctx.arc(cx, cy, pr, 0, 2 * Math.PI);
      ctx.strokeStyle = col + '55'; ctx.lineWidth = 1.5; ctx.stroke();
    }
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
    grd.addColorStop(0, col + (isHit ? 'bb' : '88'));
    grd.addColorStop(1, 'transparent');
    ctx.beginPath(); ctx.arc(cx, cy, r * 3, 0, 2 * Math.PI); ctx.fillStyle = grd; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();

    if (state.hoverId === pt.item.id) {
      ctx.beginPath(); ctx.arc(cx, cy, r + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
}

function drawQueryStar() {
  if (!state.queryPt) return;
  const [qx, qy] = worldToCanvas(state.queryPt.x, state.queryPt.y);
  ctx.save();
  ctx.translate(qx, qy);
  ctx.shadowColor = '#fff';
  ctx.shadowBlur = 18;
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = (i * Math.PI) / 5 - Math.PI / 2, rr = i % 2 === 0 ? 13 : 5;
    const fn = i === 0 ? 'moveTo' : 'lineTo';
    ctx[fn](Math.cos(a) * rr, Math.sin(a) * rr);
  }
  ctx.closePath();
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = '#aaaacc';
  ctx.font = '10px Fira Code,monospace';
  ctx.fillText('query', qx + 16, qy + 4);
}

function drawEmpty() {
  if (state.pcaPoints.length) return;
  ctx.fillStyle = '#1a1a38';
  ctx.font = '13px Fira Code,monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Connecting to VectorDB…', canvas.width / 2, canvas.height / 2);
  ctx.textAlign = 'left';
}

function frame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawQueryLinks();
  drawPoints();
  drawQueryStar();
  drawEmpty();
  pulse += 0.05;
  requestAnimationFrame(frame);
}

// ── hover tooltip ──────────────────────────────────────────
function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = 18, hover = null;
  for (const pt of state.pcaPoints) {
    const [cx, cy] = worldToCanvas(pt.x, pt.y);
    const d = Math.hypot(mx - cx, my - cy);
    if (d < best) { best = d; hover = pt.item; }
  }
  state.hoverId = hover?.id ?? null;
  if (hover) {
    tip.style.display = 'block';
    tip.style.left = `${e.clientX + 14}px`;
    tip.style.top = `${e.clientY - 8}px`;
    tip.innerHTML = `<span style="color:${colorFor(hover.category)}">[${escapeHtml(hover.category)}]</span><br>${escapeHtml(hover.metadata)}`;
  } else {
    tip.style.display = 'none';
  }
}

function onMouseLeave() {
  state.hoverId = null;
  tip.style.display = 'none';
}

export function initScatterPlot() {
  canvas = $('scatter');
  ctx = canvas.getContext('2d');
  tip = $('tip');
  resize();
  window.addEventListener('resize', resize);
  canvas.addEventListener('mousemove', onMouseMove);
  canvas.addEventListener('mouseleave', onMouseLeave);
  requestAnimationFrame(frame);
}
