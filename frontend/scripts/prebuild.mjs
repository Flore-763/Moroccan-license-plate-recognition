// Prépare public/ avant `npm run dev|build` : runtime ONNX + liste des images d'exemple.
import { copyFileSync, mkdirSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "node_modules", "onnxruntime-web", "dist");
mkdirSync(join(root, "public", "ort"), { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(dist, f), join(root, "public", "ort", f));
}

const sdir = join(root, "public", "samples");
mkdirSync(sdir, { recursive: true });
const imgs = readdirSync(sdir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).sort();
const list = imgs.map((f) => ({
  name: f.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
  url: `samples/${f}`,
}));
writeFileSync(join(sdir, "samples.json"), JSON.stringify(list, null, 1));
console.log(`prebuild : runtime ONNX copié, ${list.length} image(s) d'exemple`);
for (const m of ["plate_yolo.onnx", "crnn.onnx"]) {
  if (!existsSync(join(root, "public", "models", m))) console.warn(`⚠ public/models/${m} absent → lancez tools/convert_models.py`);
}
