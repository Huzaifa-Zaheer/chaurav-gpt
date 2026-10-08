// ════════════════════════════════════════════════════════════
//  QUERY EMBEDDING BAR CHART (16 dims)
// ════════════════════════════════════════════════════════════
import { DIMS, DIM_COL, DIM_GROUPS } from '../config.js';
import { $ } from '../utils/dom.js';

const H = 76;

export function drawVectorChart(emb) {
  const cvs = $('vecCvs');
  const W = cvs.parentElement.clientWidth;
  cvs.width = W;
  const g = cvs.getContext('2d');

  g.clearRect(0, 0, W, H);
  g.fillStyle = '#07070f';
  g.fillRect(0, 0, W, H);

  const bw = (W - 4) / DIMS;
  for (let i = 0; i < DIMS; i++) {
    const h = emb[i] * 58, x = 2 + i * bw, col = DIM_COL[i];
    g.shadowColor = col;
    g.shadowBlur = 5;
    g.fillStyle = col + 'aa';
    g.fillRect(x + 1, 63 - h, bw - 2, h);
  }

  g.shadowBlur = 0;
  g.font = '8px monospace';
  g.textAlign = 'center';
  for (const { label, start, color } of DIM_GROUPS) {
    g.fillStyle = color + '77';
    g.fillText(label, 2 + (start + 1.5) * bw, 74);
  }
  g.textAlign = 'left';
}
