// localDetector.js — toute l'analyse tourne DANS le navigateur (onnxruntime-web).
// Aucune image n'est envoyée à un serveur.
import * as ort from "onnxruntime-web/wasm";
import {
  YOLO_SIZE, CRNN_H, CRNN_W, letterboxParams, rgbaToChw, decodeYolo, padBox,
  rgbaToGray, resizeCubicGray, ctcGreedy, meanMaxProb, validatePlate,
} from "./pipeline.js";

const BASE = import.meta.env.BASE_URL;
const abs = (p) => new URL(BASE + p, document.baseURI).href;

ort.env.wasm.wasmPaths = abs("ort/");
ort.env.wasm.numThreads = 1;                 // pas de SharedArrayBuffer requis (GitHub Pages)

let sessions = null;
let loading = null;
let vehicles = null;

export const isReady = () => sessions !== null;

async function fetchBuffer(path) {
  const r = await fetch(abs(path));
  if (!r.ok) throw new Error(`Modèle introuvable (${path}). Avez-vous lancé tools/convert_models.py ?`);
  return new Uint8Array(await r.arrayBuffer());
}

export function warmup() {
  if (!loading) {
    loading = (async () => {
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const [yoloBuf, crnnBuf, vjson] = await Promise.all([
        fetchBuffer("models/plate_yolo.onnx"), fetchBuffer("models/crnn.onnx"),
        fetch(abs("data/vehicules.json")).then((r) => r.json()),
      ]);
      const [yolo, crnn] = await Promise.all([
        ort.InferenceSession.create(yoloBuf, opts), ort.InferenceSession.create(crnnBuf, opts),
      ]);
      vehicles = new Map(vjson.map((v) => [v.immatriculation, v.statut]));
      sessions = { yolo, crnn };
    })().catch((e) => { loading = null; throw e; });
  }
  return loading;
}

function toCanvas(source, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  return [c, ctx];
}

/** file (File|Blob) → même structure de réponse que l'API /detect-image. */
export async function detectImageLocal(file) {
  const t0 = performance.now();
  await warmup();
  const bmp = await createImageBitmap(file);
  const W = bmp.width, H = bmp.height;
  const base = { image_size: [W, H] };

  // 1) YOLO : letterbox 640×640 gris 114
  const lb = letterboxParams(W, H);
  const [lc, lctx] = toCanvas(null, YOLO_SIZE, YOLO_SIZE);
  lctx.fillStyle = "rgb(114,114,114)";
  lctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  lctx.drawImage(bmp, lb.padX, lb.padY, lb.nw, lb.nh);
  const input = new ort.Tensor("float32", rgbaToChw(lctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE).data),
    [1, 3, YOLO_SIZE, YOLO_SIZE]);
  const yIn = sessions.yolo.inputNames[0];
  const yOut = (await sessions.yolo.run({ [yIn]: input }))[sessions.yolo.outputNames[0]];
  const dets = decodeYolo(yOut.data, yOut.dims, lb, W, H);
  if (!dets.length) return { ...base, success: false, message: "Aucune plaque detectee", statut: "inconnu" };

  // 2) Recadrage sur l'image d'ORIGINE (meilleure résolution que le carré 640)
  const best = dets[0];
  const bbox = padBox(best.box, W, H);
  const cw = bbox[2] - bbox[0], ch = bbox[3] - bbox[1];
  if (cw < 2 || ch < 2) return { ...base, success: false, message: "Aucune plaque detectee", statut: "inconnu" };
  const [cc, cctx] = toCanvas(null, cw, ch);
  cctx.drawImage(bmp, bbox[0], bbox[1], cw, ch, 0, 0, cw, ch);
  const plateImage = cc.toDataURL("image/jpeg", 0.85);

  // 3) CRNN : gris → resize bicubique 128×32 → softmax(T×C) → CTC glouton
  const gray = rgbaToGray(cctx.getImageData(0, 0, cw, ch).data, cw, ch);
  const resized = resizeCubicGray(gray, cw, ch, CRNN_W, CRNN_H);
  const cIn = new ort.Tensor("float32", resized, [1, CRNN_H, CRNN_W, 1]);
  const cOut = (await sessions.crnn.run({ [sessions.crnn.inputNames[0]]: cIn }))[sessions.crnn.outputNames[0]];
  const [, T, C] = cOut.dims;
  const raw = ctcGreedy(cOut.data, T, C);
  const valid = validatePlate(raw, meanMaxProb(cOut.data, T, C));
  if (!valid) {
    return { ...base, success: false, message: "Aucune plaque lisible", statut: "inconnu",
      bbox_plaque: bbox, plaque_image: plateImage };
  }

  // 4) Base de véhicules de démonstration
  const imm = valid.text.replace(/ /g, "");
  const statutDb = vehicles.get(imm);
  const statut = statutDb ?? "inconnu";
  const alerte = statut !== "safe";
  const now = new Date().toISOString().replace("Z", "");
  return {
    ...base, success: true, immatriculation: imm, texte_formate: valid.text,
    confiance: valid.confidence, confiance_detection_plaque: best.score,
    bbox_plaque: bbox, plaque_image: plateImage,
    statut, vehicule_trouve: statutDb !== undefined, alerte_creee: alerte,
    detection: { immatriculation: imm, confiance: valid.confidence, detected_at: now, source_type: "photo", camera_id: null },
    alert: alerte ? { immatriculation: imm, infraction: statut, temps: now } : null,
    temps_ms: Math.round(performance.now() - t0),
  };
}
