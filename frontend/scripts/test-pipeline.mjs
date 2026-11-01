// node frontend/scripts/test-pipeline.mjs /tmp/ref.json   (compare le JS au Python)
import { readFileSync } from "node:fs";
import * as P from "../src/lib/pipeline.js";
const ref = JSON.parse(readFileSync(process.argv[2], "utf8"));
let fail = 0; const ok = (c, m) => { console.log((c ? "✓ " : "✗ ") + m); if (!c) fail++; };

for (const r of ref.resize) {
  const out = P.resizeCubicGray(Float32Array.from(r.src), r.w, r.h);
  let md = 0; for (let i = 0; i < out.length; i++) md = Math.max(md, Math.abs(out[i] - r.out[i]));
  ok(md <= 2, `resize bicubique ${r.w}x${r.h} → écart max ${md} niveau(x) de gris vs cv2`);
}
for (const [i, r] of ref.ctc.entries()) {
  const t = P.ctcGreedy(Float32Array.from(r.p), 32, 16), c = P.meanMaxProb(Float32Array.from(r.p), 32, 16);
  ok(t === r.text && Math.abs(c - r.conf) < 1e-6, `CTC #${i} "${t}" = "${r.text}", conf ${c.toFixed(4)}`);
}
for (const r of ref.plates) {
  const o = P.validatePlate(r.raw, r.conf);
  const same = JSON.stringify(o && [o.text, +o.confidence.toFixed(9)]) === JSON.stringify(r.out && [r.out[0], +r.out[1].toFixed(9)]);
  ok(same, `plaque "${r.raw}" → ${JSON.stringify(o?.text ?? null)}`);
}
// letterbox + décodage YOLO : on fabrique une sortie dont la vérité terrain est connue
const W = 768, H = 1024, truth = [160, 490, 620, 610];
const lb = P.letterboxParams(W, H);
const N = 8400, out = new Float32Array(5 * N);
const put = (i, [x1, y1, x2, y2], s) => {
  const X1 = x1 * lb.scale + lb.padX, Y1 = y1 * lb.scale + lb.padY, X2 = x2 * lb.scale + lb.padX, Y2 = y2 * lb.scale + lb.padY;
  out[i] = (X1 + X2) / 2; out[N + i] = (Y1 + Y2) / 2; out[2 * N + i] = X2 - X1; out[3 * N + i] = Y2 - Y1; out[4 * N + i] = s;
};
put(10, truth, 0.91); put(11, [truth[0] + 4, truth[1] + 2, truth[2] + 3, truth[3] + 1], 0.85); // doublon → supprimé par NMS
put(500, [10, 10, 80, 50], 0.40); put(900, [300, 100, 340, 130], 0.10);                       // 2e plaque ; bruit < seuil
const dets = P.decodeYolo(out, [1, 5, N], lb, W, H);
const d0 = dets[0].box.map((v) => Math.round(v));
ok(dets.length === 2, `NMS : ${dets.length} détection(s) (attendu 2)`);
ok(d0.every((v, i) => Math.abs(v - truth[i]) <= 1), `retour au repère image d'origine : [${d0}] vs [${truth}]`);
ok(JSON.stringify(P.padBox([10.2, 5.5, 100.1, 50], 200, 60)) === JSON.stringify([6, 1, 105, 54]), "marge de 4 px bornée");
console.log(fail ? `\n${fail} échec(s)` : "\nTout est conforme au Python."); process.exit(fail ? 1 : 0);
