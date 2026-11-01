"""
plate_ocr_maroc.py — v4 (blob par blob)
----------------------------------------
Lecture OCR d'une plaque marocaine caractere par caractere.

Format plaque marocaine :
  [9][0][1][2][0] [|] [ا] [|] [7][2]
   0   1   2   3   4   5   6   7   8   9   <- index de blob

Pipeline :
  1. Trim (supprime bande pub en bas)
  2. Binarisation + recherche des composantes connexes (blobs)
  3. Filtrage des blobs (bruit, bordures) et tri gauche -> droite
  4. Reconciliation : on ajuste pour obtenir exactement 10 blobs
     - Trop de blobs  -> on fusionne les plus petits / adjacents
     - Pas assez      -> on coupe les blobs trop larges
  5. Attribution du role par index :
       0-4  -> chiffre  (allowlist 0-9)
       5,7  -> barre    (ignore)
       6    -> lettre arabe
       8-9  -> chiffre region
  6. OCR individuel par blob avec allowlist stricte
  7. Validation regex finale
  8. Fallback vers l'OCR global si < 7 blobs trouves
"""

from pathlib import Path
import argparse
import re

import cv2
import numpy as np


# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------

ARABIC_LETTERS = "أابتثجحخدذرزسشصضطظعغفقكلمنهويىءآإؤئ"

# Roles par index de blob (0-base)
# 'D' = digit, 'B' = barre (skip), 'A' = arabe
BLOB_ROLES = {0: "D", 1: "D", 2: "D", 3: "D", 4: "D",
              5: "B", 6: "A", 7: "B", 8: "D", 9: "D"}

DEFAULT_PLATE_MODEL = (
    "runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt"
)
DEFAULT_DEBUG_DIR = "outputs/ocr_debug"

_PLATE_REGEX = re.compile(
    r"(\d{1,5})\s*([" + ARABIC_LETTERS + r"])\s*(\d{1,2})"
)
_ARABIC_INDIC = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "0123456789"
)
_LETTER_FIXES = {
    "I": "\u0623", "1": "\u0623", "l": "\u0644",
    "J": "\u062c", "U": "\u0648", "|": "\u0623", "7": "\u0632",
}


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


def _trim(crop, bottom: float = 0.18, top: float = 0.08):
    """Coupe la bande publicitaire du bas et la bordure du haut."""
    if crop is None or crop.size == 0:
        return crop
    h = crop.shape[0]
    t = max(0, int(h * top))
    b = max(t + 1, int(h * (1.0 - bottom)))
    return crop[t:b, :]


def _binarize(gray):
    """OTSU sur image lissee. Retourne fond blanc, texte noir."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _preprocess_char(crop, scale: float = 3.0):
    """Agrandit et nettoie un crop de caractere unique."""
    if crop is None or crop.size == 0:
        return None
    gray = _to_gray(crop)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 17, 17)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# ------------------------------------------------------------------
# Detection et reconciliation des blobs
# ------------------------------------------------------------------

def _find_blobs(binary, min_area_ratio: float = 0.002, max_area_ratio: float = 0.40):
    """
    Trouve les composantes connexes, filtre le bruit, et retourne
    une liste de dicts {x, y, w, h, area} triee par x.
    """
    h_img, w_img = binary.shape
    total_area = h_img * w_img
    min_area = int(total_area * min_area_ratio)
    max_area = int(total_area * max_area_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs = []
    for i in range(1, num_labels):  # 0 = fond
        x, y, w, h, area = stats[i, :5]

        # Filtrer les blobs trop petits ou trop grands
        if area < min_area or area > max_area:
            continue

        # Filtrer les blobs qui touchent le bord (bordure de la plaque)
        margin = 3
        if x <= margin or y <= margin or (x + w) >= (w_img - margin) or (y + h) >= (h_img - margin):
            if w > w_img * 0.5:  # sauf si c'est tres large (impossible pour un char)
                continue

        blobs.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    # Tri par position X
    blobs.sort(key=lambda b: b["x"])
    return blobs


def _merge_two_blobs(a, b):
    """Fusionne deux blobs en un seul englobant."""
    x = min(a["x"], b["x"])
    y = min(a["y"], b["y"])
    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y, "area": a["area"] + b["area"]}


def _split_blob(blob):
    """Coupe un blob trop large en deux moitiés égales."""
    half = blob["w"] // 2
    left  = {"x": blob["x"],          "y": blob["y"], "w": half,              "h": blob["h"], "area": blob["area"] // 2}
    right = {"x": blob["x"] + half,   "y": blob["y"], "w": blob["w"] - half,  "h": blob["h"], "area": blob["area"] // 2}
    return left, right


def _reconcile_blobs(blobs, target: int = 10):
    """
    Ajuste la liste de blobs pour obtenir exactement `target` elements.

    Strategie :
      - Trop de blobs : fusionner les deux blobs les plus proches et les plus petits
      - Pas assez     : couper le blob le plus large
    Plafonne a 20 iterations pour eviter les boucles infinies.
    """
    MAX_ITER = 20
    for _ in range(MAX_ITER):
        n = len(blobs)
        if n == target:
            break

        if n > target:
            # Trouver la paire de blobs adjacents dont la somme d'aires est minimale
            best_pair = None
            best_score = float("inf")
            for i in range(len(blobs) - 1):
                gap   = blobs[i + 1]["x"] - (blobs[i]["x"] + blobs[i]["w"])
                score = blobs[i]["area"] + blobs[i + 1]["area"] + gap * 5
                if score < best_score:
                    best_score = score
                    best_pair  = i
            if best_pair is None:
                break
            merged = _merge_two_blobs(blobs[best_pair], blobs[best_pair + 1])
            blobs  = blobs[:best_pair] + [merged] + blobs[best_pair + 2:]

        else:  # n < target
            # Couper le blob le plus large
            widest = max(range(len(blobs)), key=lambda i: blobs[i]["w"])
            left, right = _split_blob(blobs[widest])
            blobs = blobs[:widest] + [left, right] + blobs[widest + 1:]

    return blobs


# ------------------------------------------------------------------
# OCR par caractere
# ------------------------------------------------------------------

def _normalize_digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text.translate(_ARABIC_INDIC))


def _ocr_digit(reader, crop):
    """OCR d'un chiffre unique avec allowlist stricte."""
    arabic_indic = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    proc = _preprocess_char(crop, scale=3.0)
    if proc is None:
        return "?", 0.0

    results = reader.readtext(
        proc, detail=1,
        allowlist="0123456789" + arabic_indic,
        paragraph=False
    )
    if not results:
        return "?", 0.0

    best  = max(results, key=lambda r: r[2])
    digit = _normalize_digits(best[1])
    return (digit[-1] if digit else "?"), float(best[2])


def _ocr_arabic(reader, crop):
    """OCR de la lettre arabe avec zoom fort et correction post-OCR."""
    proc = _preprocess_char(crop, scale=4.5)
    if proc is None:
        return "?", 0.0

    results = []
    try:
        results = reader.readtext(proc, detail=1,
                                  allowlist=ARABIC_LETTERS, paragraph=False)
    except Exception:
        pass
    if not results:
        try:
            results = reader.readtext(proc, detail=1, paragraph=False)
        except Exception:
            pass
    if not results:
        return "?", 0.0

    best       = max(results, key=lambda r: r[2])
    raw        = best[1].strip()
    score      = float(best[2])

    arabic = re.sub(r"[^\u0600-\u06FF]", "", raw)
    if arabic:
        return arabic[0], score

    for ch in raw:
        if ch in _LETTER_FIXES:
            return _LETTER_FIXES[ch], score * 0.7

    return "?", score * 0.4


# ------------------------------------------------------------------
# Fallback global (herite des versions precedentes)
# ------------------------------------------------------------------

def _ocr_global_fallback(reader, crop):
    if reader is None:
        return None, None

    trimmed = _trim(crop)
    gray    = _to_gray(trimmed)
    gray    = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray    = cv2.bilateralFilter(gray, 9, 17, 17)
    _, th   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dets = reader.readtext(th, detail=1, paragraph=False)
    if not dets:
        return None, None

    parts, scores = [], []
    for _, text, score in dets:
        cleaned = re.sub(r"\s+", " ", text.strip().translate(_ARABIC_INDIC))
        cleaned = re.sub(r"[^0-9\u0600-\u06FF ]", "", cleaned).strip()
        if cleaned:
            parts.append(cleaned)
            scores.append(float(score))

    if not parts:
        return None, None
    return " ".join(parts), float(np.mean(scores))


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def _validate(text: str):
    if not text:
        return None
    m = _PLATE_REGEX.search(text)
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else None


# ------------------------------------------------------------------
# Fonction principale
# ------------------------------------------------------------------

def read_plate_moroccan(reader, plate_crop, debug: bool = False):
    """
    Lecture d'une plaque marocaine caractere par caractere.

    Retourne (texte, score) ou (None, None).
    """
    if reader is None or plate_crop is None or plate_crop.size == 0:
        return None, None

    trimmed = _trim(plate_crop)
    gray    = _to_gray(trimmed)
    binary  = _binarize(gray)

    blobs   = _find_blobs(binary)

    if debug:
        print(f"[DBG] {len(blobs)} blobs bruts trouves")

    # Minimum viable : au moins 7 blobs (les 2 barres peuvent etre absentes)
    if len(blobs) < 5:
        print("[OCR] Trop peu de blobs, fallback global")
        text, score = _ocr_global_fallback(reader, plate_crop)
        if text:
            return (_validate(text) or text), score
        return None, None

    # Reconciliation vers 10 blobs
    blobs = _reconcile_blobs(blobs, target=10)

    if debug:
        for i, b in enumerate(blobs):
            role = BLOB_ROLES.get(i, "?")
            print(f"[DBG] blob {i:2d} role={role} x={b['x']:3d} w={b['w']:3d} h={b['h']:3d}")

    # OCR blob par blob
    chars  = []
    scores = []

    for idx, blob in enumerate(blobs):
        role = BLOB_ROLES.get(idx, "D")

        if role == "B":
            # Barre separatrice : on skip
            continue

        x1 = blob["x"]
        y1 = blob["y"]
        x2 = x1 + blob["w"]
        y2 = y1 + blob["h"]

        # Ajouter un petit padding
        h_img, w_img = trimmed.shape[:2]
        pad = 4
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w_img, x2 + pad)
        y2c = min(h_img, y2 + pad)

        char_crop = trimmed[y1c:y2c, x1c:x2c]

        if role == "D":
            ch, sc = _ocr_digit(reader, char_crop)
        elif role == "A":
            ch, sc = _ocr_arabic(reader, char_crop)
        else:
            ch, sc = "?", 0.0

        chars.append((idx, role, ch, sc))
        scores.append(sc)

        if debug:
            print(f"[DBG] blob {idx:2d} role={role} -> '{ch}' (conf {sc:.2f})")

    if not chars:
        return None, None

    # Reconstruction du texte
    num_chars = [ch for _, role, ch, _ in chars if role == "D" and _ < 8]
    # On reconstruit dans l'ordre : indices 0-4 = chiffres, 6 = arabe, 8-9 = region
    number_part = "".join(ch for idx, role, ch, _ in chars if role == "D" and idx <= 4)
    arabic_part = next((ch for idx, role, ch, _ in chars if role == "A"), "?")
    region_part = "".join(ch for idx, role, ch, _ in chars if role == "D" and idx >= 8)

    avg_score = float(np.mean(scores)) if scores else 0.0
    text      = f"{number_part} {arabic_part} {region_part}".strip()
    validated = _validate(text)

    if validated:
        print(f"[OCR] blob-par-blob | valide: {validated} (conf {avg_score:.2f})")
        return validated, avg_score

    # Résultat partiel acceptable si on a au moins les chiffres
    if number_part and region_part:
        print(f"[OCR] blob-par-blob | partiel: {text} (conf {avg_score:.2f})")
        return text, avg_score

    # Dernier recours : fallback global
    print("[OCR] Resultat insuffisant, fallback global")
    text, score = _ocr_global_fallback(reader, plate_crop)
    if text:
        return (_validate(text) or text), score

    return None, None


# ------------------------------------------------------------------
# Utilitaires debug
# ------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _crop_bbox(image, bbox, padding: int = 4):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def detect_plate_crop(image, model_path: str, conf: float = 0.25):
    from ultralytics import YOLO
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_file}")
    dets = YOLO(str(model_file))(image, conf=conf, verbose=False)[0]
    if not dets.boxes:
        return None, None, None
    best  = max(dets.boxes, key=lambda b: float(b.conf[0]))
    bbox  = [float(v) for v in best.xyxy[0]]
    score = float(best.conf[0])
    return _crop_bbox(image, bbox, padding=4), bbox, score


def _save_debug(output_dir, stem, source, plate_crop, blobs_reconciled=None, bbox=None):
    """Sauvegarde image source annotee + crop + chaque blob."""
    saved = {}

    src = source.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
    p = output_dir / f"{stem}_source.jpg"
    cv2.imwrite(str(p), src)
    saved["source"] = p

    if plate_crop is not None:
        trimmed = _trim(plate_crop)
        p = output_dir / f"{stem}_crop.jpg"
        cv2.imwrite(str(p), plate_crop)
        saved["crop"] = p
        p2 = output_dir / f"{stem}_trimmed.jpg"
        cv2.imwrite(str(p2), trimmed)
        saved["trimmed"] = p2

        # Sauvegarder les blobs individuels si dispo
        if blobs_reconciled is not None:
            h_img, w_img = trimmed.shape[:2]
            annotated = cv2.cvtColor(trimmed, cv2.COLOR_GRAY2BGR) if trimmed.ndim == 2 else trimmed.copy()
            colors = {"D": (0, 180, 0), "B": (120, 120, 120), "A": (0, 120, 255)}
            for i, blob in enumerate(blobs_reconciled):
                role  = BLOB_ROLES.get(i, "D")
                color = colors.get(role, (200, 200, 200))
                x1b, y1b = blob["x"], blob["y"]
                x2b, y2b = x1b + blob["w"], y1b + blob["h"]
                cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), color, 1)
                cv2.putText(annotated, str(i), (x1b, max(10, y1b - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            p3 = output_dir / f"{stem}_blobs.jpg"
            cv2.imwrite(str(p3), annotated)
            saved["blobs"] = p3

    return saved


# ------------------------------------------------------------------
# CLI de test
# ------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(description="OCR plaque marocaine v4 — blob par blob")
    parser.add_argument("image", help="Image source")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-detect", action="store_true",
                        help="L'image est deja un crop de plaque")
    parser.add_argument("--debug", action="store_true",
                        help="Affiche le detail de chaque blob")
    return parser


if __name__ == "__main__":
    import easyocr

    parser = _build_parser()
    args   = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de lire: {image_path}")

    print("[INFO] Chargement EasyOCR (ar, en)...")
    reader = easyocr.Reader(["ar", "en"], gpu=False)

    plate_crop, bbox, det_score = None, None, None
    if args.no_detect:
        plate_crop = img
    else:
        plate_crop, bbox, det_score = detect_plate_crop(
            img, args.plate_model, conf=args.conf
        )
        if plate_crop is None:
            print("[WARN] Aucune plaque detectee.")
            raise SystemExit(1)
        print(f"[INFO] Plaque detectee (score: {det_score:.2f})")

    text, score = read_plate_moroccan(reader, plate_crop, debug=args.debug)
    ocr_score   = 0.0 if score is None else score
    print(f"\nResultat : {text!r}  (confiance: {ocr_score:.2f})")

    # Pour le debug, recuperer les blobs apres reconciliation
    trimmed = _trim(plate_crop)
    gray    = _to_gray(trimmed)
    binary  = _binarize(gray)
    blobs   = _find_blobs(binary)
    blobs   = _reconcile_blobs(blobs, target=10) if len(blobs) >= 5 else blobs

    output_dir = _ensure_dir(Path(args.output_dir))
    saved      = _save_debug(output_dir, image_path.stem, img, plate_crop, blobs, bbox)
    print("\n[INFO] Fichiers debug:")
    for label, path in saved.items():
        print(f"       {label}: {path}")