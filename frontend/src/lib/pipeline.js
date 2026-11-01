// pipeline.js — pré/post-traitement PURS (aucune dépendance navigateur) :
// réplique fidèlement le pipeline Python (YOLO letterbox + NMS, CRNN gris 32×128,
// décodage CTC glouton, normalisation et validation du format de plaque).

// ── Constantes recopiées de model/CRNN/model_02.py ────────────────────────────
export const VOCAB = "0123456789أبھدو";
export const BLANK_IDX = VOCAB.length;            // blank = dernier index
export const CRNN_H = 32;
export const CRNN_W = 128;
const NORMALIZE = { "\u0647": "\u06BE", "\u0629": "\u06BE" };
const ARABIC = [...VOCAB].filter((c) => !/\d/.test(c)).join("");
const PLATE_RE = new RegExp(`(\\d{1,5})\\s*([${ARABIC}])\\s*(\\d{1,2})`, "u");

export const YOLO_SIZE = 640;

// ── YOLO ──────────────────────────────────────────────────────────────────────
/** Paramètres du letterbox (image w×h → carré 640, bandes grises 114). */
export function letterboxParams(w, h, size = YOLO_SIZE) {
  const scale = Math.min(size / w, size / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  return { scale, nw, nh, padX: Math.floor((size - nw) / 2), padY: Math.floor((size - nh) / 2) };
}

/** RGBA (taille size×size, déjà letterboxé) → Float32 CHW RGB normalisé 0..1. */
export function rgbaToChw(rgba, size = YOLO_SIZE) {
  const n = size * size, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[4 * i] / 255;
    out[n + i] = rgba[4 * i + 1] / 255;
    out[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

/**
 * Sortie YOLOv8 [1, 4+nc, N] (xywh en pixels du carré 640, puis scores de classes)
 * → détections [{box:[x1,y1,x2,y2] (pixels image d'origine), score}] après NMS.
 */
export function decodeYolo(output, dims, lb, origW, origH, conf = 0.25, iouThr = 0.45) {
  const [, rows, n] = dims;
  const nc = rows - 4;
  const cands = [];
  for (let i = 0; i < n; i++) {
    let score = 0;
    for (let c = 0; c < nc; c++) score = Math.max(score, output[(4 + c) * n + i]);
    if (score < conf) continue;
    const cx = output[i], cy = output[n + i], w = output[2 * n + i], h = output[3 * n + i];
    const x1 = (cx - w / 2 - lb.padX) / lb.scale, y1 = (cy - h / 2 - lb.padY) / lb.scale;
    const x2 = (cx + w / 2 - lb.padX) / lb.scale, y2 = (cy + h / 2 - lb.padY) / lb.scale;
    cands.push({
      box: [Math.max(0, x1), Math.max(0, y1), Math.min(origW, x2), Math.min(origH, y2)],
      score,
    });
  }
  cands.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cands) if (kept.every((k) => iou(k.box, c.box) < iouThr)) kept.push(c);
  return kept;
}

/** Boîte + marge de 4 px (comme main.py), bornée à l'image, en entiers. */
export function padBox(box, w, h, pad = 4) {
  return [
    Math.max(0, Math.floor(box[0]) - pad), Math.max(0, Math.floor(box[1]) - pad),
    Math.min(w, Math.ceil(box[2]) + pad), Math.min(h, Math.ceil(box[3]) + pad),
  ];
}

// ── CRNN ──────────────────────────────────────────────────────────────────────
/** RGBA → niveaux de gris entiers (cv2.COLOR_BGR2GRAY : 0.299 R + 0.587 G + 0.114 B). */
export function rgbaToGray(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    g[i] = Math.round(0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]);
  }
  return g;
}

function cubicCoeffs(t) {           // noyau bicubique d'OpenCV (a = -0.75)
  const A = -0.75;
  return [
    ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
    ((A + 2) * t - (A + 3)) * t * t + 1,
    ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1,
  ].concat([0]).map((v, i, arr) => (i === 3 ? 1 - arr[0] - arr[1] - arr[2] : v));
}

/** Équivalent de cv2.resize(gray, (dw, dh), INTER_CUBIC), sortie arrondie/saturée 0..255. */
export function resizeCubicGray(src, sw, sh, dw = CRNN_W, dh = CRNN_H) {
  const sx = sw / dw, sy = sh / dh;
  const clamp = (v, m) => (v < 0 ? 0 : v > m - 1 ? m - 1 : v);
  const xs = new Array(dw), ys = new Array(dh);
  for (let x = 0; x < dw; x++) {
    const fx = (x + 0.5) * sx - 0.5, ix = Math.floor(fx);
    xs[x] = { i: ix, c: cubicCoeffs(fx - ix) };
  }
  for (let y = 0; y < dh; y++) {
    const fy = (y + 0.5) * sy - 0.5, iy = Math.floor(fy);
    ys[y] = { i: iy, c: cubicCoeffs(fy - iy) };
  }
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      for (let j = 0; j < 4; j++) {
        const yy = clamp(ys[y].i - 1 + j, sh);
        let row = 0;
        for (let i = 0; i < 4; i++) row += src[yy * sw + clamp(xs[x].i - 1 + i, sw)] * xs[x].c[i];
        acc += row * ys[y].c[j];
      }
      out[y * dw + x] = Math.min(255, Math.max(0, Math.round(acc)));
    }
  }
  return out;
}

export function normalizeLabel(text) {
  return [...text].map((c) => NORMALIZE[c] ?? c).join("");
}

/** Décodage CTC glouton (identique à decode_ctc de 04_inference.py). probs: [T, C] aplati. */
export function ctcGreedy(probs, T, C) {
  const path = [];
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -1;
    for (let c = 0; c < C; c++) { const v = probs[t * C + c]; if (v > bv) { bv = v; best = c; } }
    path.push(best);
  }
  const collapsed = [path[0]];
  for (let t = 1; t < path.length; t++) if (path[t] !== collapsed[collapsed.length - 1]) collapsed.push(path[t]);
  const chars = [...VOCAB];
  return collapsed.filter((i) => i !== BLANK_IDX && i < chars.length).map((i) => chars[i]).join("");
}

/** Confiance = moyenne, sur les pas de temps, de la probabilité maximale. */
export function meanMaxProb(probs, T, C) {
  let s = 0;
  for (let t = 0; t < T; t++) {
    let m = 0;
    for (let c = 0; c < C; c++) m = Math.max(m, probs[t * C + c]);
    s += m;
  }
  return s / T;
}

/** Validation du format « 12345 أ 40 » (même logique que PlateReader.read). */
export function validatePlate(rawText, confidence) {
  const text = normalizeLabel(rawText || "");
  if (!text) return null;
  const m = PLATE_RE.exec(text);
  if (m) return { text: `${m[1]} ${m[2]} ${m[3]}`, confidence };
  const digits = text.replace(/[^0-9]/g, "");
  if (digits.length >= 3) return { text, confidence: confidence * 0.7 };
  return null;
}
