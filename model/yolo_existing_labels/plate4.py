# -*- coding: utf-8 -*-
"""
plate_ocr_maroc.py — v5 (blob par blob avec dilatation horizontale)
--------------------------------------------------------------------
Correction majeure v5 :
  - Dilatation horizontale avant connectedComponents pour fusionner
    les traits d'un meme caractere en un seul blob
  - Suppression du bug num_chars / double underscore
  - Reconciliation plus souple (cible adaptative selon blobs trouves)
  - Chaque blob OCRise apres zoom fort sur le crop original (pas binaire)

Format plaque marocaine standard :
  [9][0][1][2][0] [|] [a] [|] [7][2]
   0   1   2   3   4   5   6   7   8   9  <- index blob

Roles :
  0-4  -> chiffre  (allowlist 0-9)
  5,7  -> barre    (skip)
  6    -> lettre arabe
  8-9  -> chiffre region
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

FIXED_BLOB_ROLES = {
    0: "D", 1: "D", 2: "D", 3: "D", 4: "D",
    5: "B",
    6: "A",
    7: "B",
    8: "D", 9: "D",
}

DEFAULT_PLATE_MODEL = (
    "runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt"
)
DEFAULT_DEBUG_DIR = "outputs/ocr_debug"

_PLATE_REGEX = re.compile(
    r"(\d{1,5})\s*([" + re.escape(ARABIC_LETTERS) + r"])\s*(\d{1,2})"
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
    """Coupe bande pub du bas + bordure du haut."""
    if crop is None or crop.size == 0:
        return crop
    h = crop.shape[0]
    t = max(0, int(h * top))
    b = max(t + 1, int(h * (1.0 - bottom)))
    return crop[t:b, :]


def _binarize(gray):
    """Binarisation OTSU inverse : texte blanc sur fond noir."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _preprocess_char(crop_bgr, scale: float = 4.0):
    """
    Zoom fort + debruitage sur le crop couleur d'un caractere.
    On travaille sur l'image couleur originale (pas le binaire)
    pour donner le maximum d'information a EasyOCR.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    # Si le crop est en niveaux de gris on le convertit
    img = crop_bgr if crop_bgr.ndim == 3 else cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.bilateralFilter(img, 9, 17, 17)
    return img


# ------------------------------------------------------------------
# Detection des blobs avec dilatation horizontale
# ------------------------------------------------------------------

def _remove_border_components(binary):
    """Remove the plate frame before looking for character components."""
    h_img, w_img = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = binary.copy()

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, :5]
        touches_border = (
            x <= 2 or y <= 2 or (x + w) >= (w_img - 2) or (y + h) >= (h_img - 2)
        )
        border_like = touches_border and (
            w > w_img * 0.45 or h > h_img * 0.75 or area > h_img * w_img * 0.08
        )
        if border_like:
            cleaned[labels == i] = 0

    return cleaned


def _merge_x_overlapping_blobs(blobs, min_overlap_ratio: float = 0.55):
    """Merge parts of the same character split vertically, like Arabic marks."""
    if not blobs:
        return []

    merged = []
    for blob in sorted(blobs, key=lambda b: (b["x"], b["y"])):
        if not merged:
            merged.append(blob)
            continue

        prev = merged[-1]
        left = max(prev["x"], blob["x"])
        right = min(prev["x"] + prev["w"], blob["x"] + blob["w"])
        overlap = max(0, right - left)
        min_width = max(1, min(prev["w"], blob["w"]))

        if overlap / min_width >= min_overlap_ratio:
            merged[-1] = _merge_two(prev, blob)
        else:
            merged.append(blob)

    merged.sort(key=lambda b: b["x"])
    return merged


def _find_char_blobs(binary, dilation_width_ratio: float = 0.01,
                     min_area_ratio: float = 0.003,
                     max_area_ratio: float = 0.45):
    """
    Trouve les blobs de caracteres en dilatant horizontalement d'abord.

    PROBLEME SANS DILATATION :
      Un "9" binarise peut donner 3 composantes connexes separees
      (la boucle du haut, la queue, un trait intermediate).
      connectedComponents les voit comme 3 blobs distincts.

    SOLUTION :
      On dilate horizontalement avec un noyau de largeur =
      ~2-3 % de la largeur de l'image. Cela fusionne les traits
      d'un meme caractere sans fusionner deux caracteres voisins
      (car les inter-caracteres font ~5-8 % de la largeur).

    Retourne liste de dicts {x,y,w,h,area} triee par x.
    """
    binary = _remove_border_components(binary)
    h_img, w_img = binary.shape

    # Noyau de dilatation horizontal uniquement
    dil_w = max(1, int(w_img * dilation_width_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Composantes connexes sur l'image dilatee
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    total_area = h_img * w_img
    min_area   = int(total_area * min_area_ratio)
    max_area   = int(total_area * max_area_ratio)
    margin     = 3

    blobs = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, :5]

        if area < min_area or area > max_area:
            continue

        # Exclure les blobs qui sont des bordures de plaque
        is_border = (
            x <= margin and w > w_img * 0.8  # ligne horizontale en haut/bas
        )
        if is_border:
            continue

        blobs.append({
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "area": int(area)
        })

    blobs.sort(key=lambda b: b["x"])
    return _merge_x_overlapping_blobs(blobs)


# ------------------------------------------------------------------
# Reconciliation vers le bon nombre de blobs
# ------------------------------------------------------------------

def _merge_two(a, b):
    x  = min(a["x"], b["x"])
    y  = min(a["y"], b["y"])
    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x, "y": y, "w": x2-x, "h": y2-y, "area": a["area"]+b["area"]}


def _split_one(blob):
    half  = blob["w"] // 2
    left  = {**blob, "w": half,              "area": blob["area"]//2}
    right = {**blob, "x": blob["x"]+half, "w": blob["w"]-half, "area": blob["area"]//2}
    return left, right


def _reconcile_blobs(blobs, target: int = 10, max_iter: int = 25):
    """
    Ajuste le nombre de blobs vers `target` par fusion/coupure iteratives.
    """
    for _ in range(max_iter):
        n = len(blobs)
        if n == target:
            break

        if n > target:
            # Fusionner la paire adjacente la plus proche et la plus petite
            best_i, best_score = 0, float("inf")
            for i in range(len(blobs) - 1):
                gap   = blobs[i+1]["x"] - (blobs[i]["x"] + blobs[i]["w"])
                score = blobs[i]["area"] + blobs[i+1]["area"] + gap * 8
                if score < best_score:
                    best_i, best_score = i, score
            merged = _merge_two(blobs[best_i], blobs[best_i+1])
            blobs  = blobs[:best_i] + [merged] + blobs[best_i+2:]
        else:
            # Couper le blob le plus large
            widest = max(range(len(blobs)), key=lambda i: blobs[i]["w"])
            left, right = _split_one(blobs[widest])
            blobs = blobs[:widest] + [left, right] + blobs[widest+1:]

    return blobs


def _target_blob_count(blobs):
    """
    Avoid inventing characters for shorter valid plates.

    Moroccan plates can have 1-5 leading digits and 1-2 region digits, so
    the full visual pattern can contain 5-10 blobs including separators.
    """
    return 10 if len(blobs) > 10 else len(blobs)


def _infer_blob_roles(blobs):
    """
    Deduce blob roles from visible separators when possible.

    Returns a list of "D" digit, "A" Arabic letter, or "B" separator roles.
    If the two separators are not clear, falls back to the standard 10-blob
    layout, then to a compact layout based on the rightmost region digits.
    """
    if not blobs:
        return []

    if len(blobs) == 10:
        return [FIXED_BLOB_ROLES.get(i, "D") for i in range(len(blobs))]

    widths = np.array([max(1, b["w"]) for b in blobs], dtype=float)
    heights = np.array([max(1, b["h"]) for b in blobs], dtype=float)
    median_w = float(np.median(widths))
    median_h = float(np.median(heights))
    center = (len(blobs) - 1) / 2.0

    separator_candidates = []
    for i, blob in enumerate(blobs):
        is_tall = blob["h"] >= median_h * 0.75
        is_narrow = blob["w"] <= median_w * 0.50
        aspect_is_bar = blob["h"] / max(1, blob["w"]) >= 3.50
        if is_tall and is_narrow and aspect_is_bar:
            separator_candidates.append(i)

    if len(separator_candidates) >= 2:
        sep1, sep2 = sorted(
            separator_candidates,
            key=lambda i: abs(i - center),
        )[:2]
        sep1, sep2 = sorted((sep1, sep2))

        roles = ["D"] * len(blobs)
        roles[sep1] = "B"
        roles[sep2] = "B"
        for i in range(sep1 + 1, sep2):
            roles[i] = "A"
        return roles

    if len(separator_candidates) == 1:
        sep = separator_candidates[0]
        roles = ["D"] * len(blobs)
        roles[sep] = "B"
        if sep >= center and sep > 0:
            roles[sep - 1] = "A"
        elif sep + 1 < len(blobs):
            roles[sep + 1] = "A"
        else:
            roles[max(0, len(blobs) - 3)] = "A"
        return roles

    arabic_idx = max(0, len(blobs) - 3)
    roles = ["D"] * len(blobs)
    roles[arabic_idx] = "A"
    return roles


# ------------------------------------------------------------------
# OCR par role
# ------------------------------------------------------------------

def _normalize_digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text.translate(_ARABIC_INDIC))


def _heuristic_digit(char_crop):
    """Recognize very clear plate-font digits before asking EasyOCR."""
    if char_crop is None or char_crop.size == 0:
        return None, 0.0

    gray = _to_gray(char_crop)
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    if h <= 0 or w <= 0:
        return None, 0.0

    roi = th[y:y + h, x:x + w]
    aspect = w / h

    holes = 0
    if hierarchy is not None:
        for item in hierarchy[0]:
            if item[3] != -1:
                holes += 1

    col_left = np.count_nonzero(roi[:, :max(1, w // 3)])
    col_mid = np.count_nonzero(roi[:, max(1, w // 3):max(2, 2 * w // 3)])
    col_right = np.count_nonzero(roi[:, max(1, 2 * w // 3):])
    upper_mass = np.count_nonzero(roi[:h // 2, :])
    lower_mass = np.count_nonzero(roi[h // 2:, :])
    left_bottom = np.count_nonzero(roi[h // 2:, :max(1, w // 2)])
    right_bottom = np.count_nonzero(roi[h // 2:, max(1, w // 2):])

    if aspect < 0.42 or (
        holes == 0 and aspect < 0.65
        and col_mid > col_left * 1.45 and col_mid > col_right * 1.45
    ):
        return "1", 0.92

    if holes >= 2:
        return "8", 0.90

    if holes >= 1:
        if lower_mass > 0 and upper_mass / lower_mass > 1.12:
            return "9", 0.90
        if right_bottom > 0 and left_bottom / right_bottom < 0.80:
            return "9", 0.90
        if lower_mass > 0 and upper_mass / lower_mass < 0.85:
            return "6", 0.88
        if (
            lower_mass > 0
            and upper_mass / lower_mass < 0.90
            and col_mid > col_left * 1.15
            and col_mid > col_right * 1.15
        ):
            return "6", 0.88
        if (
            aspect < 0.70
            and lower_mass > 0
            and upper_mass / lower_mass < 0.90
            and right_bottom > 0
            and left_bottom / right_bottom > 1.10
            and col_right > 0
            and col_left / col_right > 1.50
        ):
            return "6", 0.88
        if aspect < 0.50 and col_right > 0 and col_left / col_right > 1.25:
            return "6", 0.88
        return "0", 0.90

    if (
        0.45 <= aspect <= 0.70
        and lower_mass > 0
        and upper_mass / lower_mass > 1.40
        and right_bottom > 0
        and left_bottom / right_bottom < 1.10
    ):
        return "9", 0.88

    top_mass = np.count_nonzero(roi[:max(1, h // 3), :])
    bottom_mass = np.count_nonzero(roi[max(1, 2 * h // 3):, :])
    if (
        aspect < 0.72
        and col_left > 0
        and col_mid / col_left > 1.20
        and col_right / col_left > 0.90
        and bottom_mass > top_mass * 1.20
    ):
        return "4", 0.88

    return None, 0.0


def _ocr_digit(reader, char_crop):
    """
    OCR d'un chiffre unique.
    On utilise une allowlist stricte et on donne le crop couleur zoom a EasyOCR.
    """
    heuristic, heuristic_score = _heuristic_digit(char_crop)
    if heuristic is not None:
        return heuristic, heuristic_score

    arabic_indic = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    proc = _preprocess_char(char_crop, scale=4.0)
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
    return (digit[0] if digit else "?"), float(best[2])


def _ocr_arabic(reader, char_crop):
    """OCR de la lettre arabe avec zoom tres fort et correction."""
    if char_crop is not None and char_crop.size > 0:
        h, w = char_crop.shape[:2]
        if h > 0 and w / h < 0.65:
            return "ا", 0.85

    proc = _preprocess_char(char_crop, scale=5.0)
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

    best  = max(results, key=lambda r: r[2])
    raw   = best[1].strip()
    score = float(best[2])

    arabic = re.sub(r"[^\u0600-\u06FF]", "", raw)
    if arabic:
        return arabic[0], score

    for ch in raw:
        if ch in _LETTER_FIXES:
            return _LETTER_FIXES[ch], score * 0.7

    return "?", score * 0.4


# ------------------------------------------------------------------
# Fallback global
# ------------------------------------------------------------------

def _ocr_global_fallback(reader, crop):
    if reader is None:
        return None, None

    trimmed = _trim(crop)
    gray    = _to_gray(trimmed)
    gray    = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
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
    if "?" in text:
        return None
    m = _PLATE_REGEX.fullmatch(text.strip())
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

    # --- Detection des blobs avec dilatation horizontale ---
    blobs = _find_char_blobs(binary)

    if debug:
        print(f"[DBG] {len(blobs)} blobs apres dilatation horizontale")
        for i, b in enumerate(blobs):
            print(f"[DBG]   blob {i:2d} x={b['x']:3d} w={b['w']:3d} h={b['h']:3d} area={b['area']}")

    if len(blobs) < 5:
        print("[OCR] Trop peu de blobs, fallback global")
        text, score = _ocr_global_fallback(reader, plate_crop)
        return (_validate(text) or text, score) if text else (None, None)

    # --- Reconciliation souple : reduire les exces sans inventer des caracteres ---
    blobs = _reconcile_blobs(blobs, target=_target_blob_count(blobs))
    roles = _infer_blob_roles(blobs)

    if debug:
        print(f"[DBG] {len(blobs)} blobs apres reconciliation")
        for i, b in enumerate(blobs):
            role = roles[i] if i < len(roles) else "D"
            print(f"[DBG]   blob {i:2d} role={role} x={b['x']:3d} w={b['w']:3d} h={b['h']:3d}")

    # --- OCR blob par blob ---
    chars  = []
    scores = []

    for idx, blob in enumerate(blobs):
        role = roles[idx] if idx < len(roles) else "D"
        if role == "B":
            continue

        h_img, w_img = trimmed.shape[:2]
        pad = 4
        x1 = max(0, blob["x"] - pad)
        y1 = max(0, blob["y"] - pad)
        x2 = min(w_img, blob["x"] + blob["w"] + pad)
        y2 = min(h_img, blob["y"] + blob["h"] + pad)

        # Crop sur l'image couleur originale (pas le binaire)
        char_crop = trimmed[y1:y2, x1:x2]

        if role == "D":
            ch, sc = _ocr_digit(reader, char_crop)
        elif role == "A":
            ch, sc = _ocr_arabic(reader, char_crop)
        else:
            ch, sc = "?", 0.0

        chars.append((idx, role, ch, sc))
        scores.append(sc)

        if debug:
            print(f"[DBG] blob {idx:2d} role={role} size={char_crop.shape[1]}x{char_crop.shape[0]} -> '{ch}' (conf {sc:.2f})")

    if not chars:
        return None, None

    # --- Assemblage ---
    arabic_indices = [idx for idx, role, ch, sc in chars if role == "A"]
    arabic_idx = arabic_indices[0] if arabic_indices else len(blobs)
    number_part = "".join(ch for idx, role, ch, sc in chars if role == "D" and idx < arabic_idx)
    arabic_part = next((ch for idx, role, ch, sc in chars if role == "A"), "?")
    region_part = "".join(ch for idx, role, ch, sc in chars if role == "D" and idx > arabic_idx)

    avg_score = float(np.mean(scores)) if scores else 0.0
    text      = f"{number_part} {arabic_part} {region_part}".strip()
    validated = _validate(text)

    if validated:
        print(f"[OCR] blob-par-blob OK : {validated} (conf {avg_score:.2f})")
        return validated, avg_score

    if number_part and region_part:
        print(f"[OCR] blob-par-blob partiel : {text} (conf {avg_score:.2f})")
        return text, avg_score

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


def _save_debug(output_dir, stem, source, plate_crop, blobs=None, bbox=None):
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
        cv2.imwrite(str(output_dir / f"{stem}_crop.jpg"), plate_crop)
        cv2.imwrite(str(output_dir / f"{stem}_trimmed.jpg"), trimmed)
        saved["crop"]    = output_dir / f"{stem}_crop.jpg"
        saved["trimmed"] = output_dir / f"{stem}_trimmed.jpg"

        # Image annotee avec les blobs
        if blobs is not None:
            roles = _infer_blob_roles(blobs)
            ann = trimmed.copy() if trimmed.ndim == 3 else cv2.cvtColor(trimmed, cv2.COLOR_GRAY2BGR)
            colors = {"D": (0, 200, 0), "B": (160, 160, 160), "A": (0, 100, 255)}
            for i, blob in enumerate(blobs):
                role  = roles[i] if i < len(roles) else "D"
                color = colors.get(role, (200, 200, 200))
                x1b, y1b = blob["x"], blob["y"]
                x2b, y2b = x1b + blob["w"], y1b + blob["h"]
                cv2.rectangle(ann, (x1b, y1b), (x2b, y2b), color, 1)
                cv2.putText(ann, f"{i}:{role}", (x1b, max(10, y1b - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Sauvegarder aussi chaque caractere individuellement
            h_img, w_img = trimmed.shape[:2]
            for i, blob in enumerate(blobs):
                role = roles[i] if i < len(roles) else "D"
                if role == "B":
                    continue
                pad = 4
                x1c = max(0, blob["x"] - pad)
                y1c = max(0, blob["y"] - pad)
                x2c = min(w_img, blob["x"] + blob["w"] + pad)
                y2c = min(h_img, blob["y"] + blob["h"] + pad)
                char_img = trimmed[y1c:y2c, x1c:x2c]
                if char_img.size > 0:
                    char_zoomed = _preprocess_char(char_img, scale=4.0)
                    if char_zoomed is not None:
                        cv2.imwrite(
                            str(output_dir / f"{stem}_char_{i:02d}_{role}.jpg"),
                            char_zoomed
                        )

            p_blobs = output_dir / f"{stem}_blobs.jpg"
            cv2.imwrite(str(p_blobs), ann)
            saved["blobs"] = p_blobs

    return saved


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description="OCR plaque marocaine v5")
    p.add_argument("image")
    p.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL)
    p.add_argument("--output-dir",  default=DEFAULT_DEBUG_DIR)
    p.add_argument("--conf",        type=float, default=0.25)
    p.add_argument("--no-detect",   action="store_true")
    p.add_argument("--debug",       action="store_true")
    return p


if __name__ == "__main__":
    import easyocr

    args       = _build_parser().parse_args()
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
    print(f"\nResultat : {text!r}  (confiance: {0.0 if score is None else score:.2f})")

    # Blobs pour le debug visuel
    trimmed = _trim(plate_crop)
    gray    = _to_gray(trimmed)
    binary  = _binarize(gray)
    blobs   = _find_char_blobs(binary)
    if len(blobs) >= 5:
        blobs = _reconcile_blobs(blobs, target=_target_blob_count(blobs))

    output_dir = _ensure_dir(Path(args.output_dir))
    saved      = _save_debug(output_dir, image_path.stem, img, plate_crop, blobs, bbox)
    print("\n[INFO] Fichiers debug:")
    for label, path in saved.items():
        print(f"       {label}: {path}")
    print(f"\n[INFO] Chars individuels zoomes : {output_dir}/{image_path.stem}_char_*.jpg")
