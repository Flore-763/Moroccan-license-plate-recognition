"""
plate_ocr_maroc.py — v3
-----------------------
Lecture OCR robuste d'une plaque marocaine.

Ameliorations v3 :
  - Suppression de la bande publicitaire en bas (Auto Haus, etc.)
  - Detection des separateurs par projection de colonnes + peak-finding
    (plus robuste que la morphologie seule)
  - Segmentation proportionnelle fixe en fallback
    (les plaques marocaines ont des zones a positions relatives stables)
  - Normalisation des chiffres arabes-indiens (0123456789 arabe -> latin)
  - OCR lettre arabe en langue 'ar' uniquement pour eviter les confusions
  - Correction post-OCR des confusions frequentes (I -> a, 1 -> a, etc.)
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

DEFAULT_PLATE_MODEL = (
    "runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt"
)
DEFAULT_DEBUG_DIR = "outputs/ocr_debug"

_PLATE_REGEX = re.compile(
    r"(\d{1,5})\s*([" + ARABIC_LETTERS + r"])\s*(\d{1,2})"
)

# Chiffres arabes-indiens vers latins
_ARABIC_INDIC = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
                               "0123456789")

# Confusions frequentes EasyOCR en zone lettre
_LETTER_CORRECTIONS = {
    "I": "\u0623",  # alef
    "1": "\u0623",
    "l": "\u0644",  # lam
    "J": "\u062c",  # jim
    "U": "\u0648",  # waw
    "|": "\u0623",
    "7": "\u0632",  # zain
}

# Proportions fixes (mesure empirique sur plaques marocaines standard)
_SEP1_RATIO    = 0.53   # fin zone chiffres
_SEP2_RATIO    = 0.78   # debut zone region


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


def _preprocess(crop, scale: float = 2.0):
    if crop is None or crop.size == 0:
        return None
    gray = _to_gray(crop)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _trim_plate_crop(crop, bottom_ratio: float = 0.18, top_ratio: float = 0.08):
    """
    Supprime la bande du bas (publicite 'Auto Haus', etc.)
    et la fine bordure du haut.
    Ces zones parasitent le fallback OCR global.
    """
    if crop is None or crop.size == 0:
        return crop
    h = crop.shape[0]
    top    = max(0, int(h * top_ratio))
    bottom = max(top + 1, int(h * (1.0 - bottom_ratio)))
    return crop[top:bottom, :]


# ------------------------------------------------------------------
# Detection des separateurs par projection de colonnes
# ------------------------------------------------------------------

def _column_dark_ratio(gray):
    """Ratio de pixels sombres par colonne apres binarisation OTSU inverse."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.sum(axis=0) / (255.0 * gray.shape[0])


def _find_separators_by_projection(gray, search_start: float = 0.30,
                                   search_end: float = 0.85):
    """
    Cherche 2 barres verticales separatrices par projection de colonnes.

    Les separateurs sont de fins pics hauts et etroits dans le profil vertical.
    Retourne [x_sep1, x_sep2] ou [] si echec.
    """
    h, w = gray.shape
    x0 = int(w * search_start)
    x1 = int(w * search_end)
    if x1 <= x0:
        return []

    ratios = _column_dark_ratio(gray[:, x0:x1])

    # Seuil adaptatif : moyenne + 1.5 ecart-type, minimum 0.35
    threshold = max(0.35, ratios.mean() + 1.5 * ratios.std())
    hot = np.where(ratios >= threshold)[0]
    if len(hot) == 0:
        return []

    # Regrouper les colonnes adjacentes (gap <= 4 px)
    groups, current = [], [hot[0]]
    for col in hot[1:]:
        if col - current[-1] <= 4:
            current.append(col)
        else:
            groups.append(current)
            current = [col]
    groups.append(current)

    # Garder les groupes etroits (separateurs fins < 8% de la fenetre)
    max_group_width = max(3, int((x1 - x0) * 0.08))
    narrow = [g for g in groups if len(g) <= max_group_width]

    if len(narrow) < 2:
        return []

    sep1 = int(np.mean(narrow[0]))  + x0
    sep2 = int(np.mean(narrow[-1])) + x0

    if sep2 - sep1 < int(w * 0.05):
        return []

    return [sep1, sep2]


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------

def _segment_by_separators(crop):
    """Segmente en utilisant les separateurs detectes par projection."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    gray = _to_gray(crop)
    seps = _find_separators_by_projection(gray)
    if len(seps) < 2:
        return None

    sep1, sep2 = seps[0], seps[1]
    pad = max(3, int(w * 0.01))

    z_num = crop[:, :max(1, sep1 - pad)]
    z_let = crop[:, min(w - 1, sep1 + pad):max(1, sep2 - pad)]
    z_reg = crop[:, min(w - 1, sep2 + pad):]

    if z_num.shape[1] < int(w * 0.04) or z_reg.shape[1] < int(w * 0.04):
        return None

    return z_num, z_let, z_reg


def _segment_by_proportions(crop):
    """
    Segmentation de repli basee sur les proportions fixes.
    Essaie d'abord de raffiner chaque separateur dans une sous-fenetre etroite.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    gray = _to_gray(crop)

    # Recherche fine du separateur gauche autour de la zone attendue
    seps_left = _find_separators_by_projection(
        gray, search_start=max(0.0, _SEP1_RATIO - 0.12),
        search_end=min(1.0, _SEP1_RATIO + 0.08)
    )
    sep1 = seps_left[0] if seps_left else int(w * _SEP1_RATIO)

    # Recherche fine du separateur droit
    seps_right = _find_separators_by_projection(
        gray, search_start=max(0.0, _SEP2_RATIO - 0.10),
        search_end=min(1.0, _SEP2_RATIO + 0.10)
    )
    sep2 = seps_right[-1] if seps_right else int(w * _SEP2_RATIO)

    pad   = max(3, int(w * 0.01))
    z_num = crop[:, :max(1, sep1 - pad)]
    z_let = crop[:, min(w - 1, sep1 + pad):max(1, sep2 - pad)]
    z_reg = crop[:, min(w - 1, sep2 + pad):]

    return z_num, z_let, z_reg


# ------------------------------------------------------------------
# OCR par zone
# ------------------------------------------------------------------

def _normalize_digits(text: str) -> str:
    """Convertit chiffres arabes-indiens en latins et filtre tout le reste."""
    return re.sub(r"[^0-9]", "", text.translate(_ARABIC_INDIC))


def _ocr_digits(reader, crop):
    """OCR chiffres uniquement."""
    proc = _preprocess(crop, scale=2.5)
    if proc is None:
        return "", 0.0
    # On accepte aussi les chiffres arabes-indiens dans l'allowlist
    arabic_indic_chars = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    results = reader.readtext(
        proc, detail=1,
        allowlist="0123456789" + arabic_indic_chars,
        paragraph=False
    )
    if not results:
        return "", 0.0
    raw   = "".join(r[1] for r in results)
    score = float(np.mean([r[2] for r in results]))
    return _normalize_digits(raw), score


def _ocr_arabic_letter(reader, crop):
    """
    OCR lettre arabe avec zoom 4x.
    - Langue 'ar' uniquement pour eviter confusions avec caracteres latins
    - Correction post-OCR des erreurs frequentes
    """
    if crop is None or crop.size == 0:
        return "?", 0.0

    proc = _preprocess(crop, scale=4.0)
    if proc is None:
        return "?", 0.0

    # Tentative 1 : ar uniquement avec allowlist arabe
    results = []
    try:
        results = reader.readtext(
            proc, detail=1,
            allowlist=ARABIC_LETTERS,
            paragraph=False
        )
    except Exception:
        pass

    # Tentative 2 : pas d'allowlist
    if not results:
        try:
            results = reader.readtext(proc, detail=1, paragraph=False)
        except Exception:
            pass

    if not results:
        return "?", 0.0

    best       = max(results, key=lambda r: r[2])
    raw_letter = best[1].strip()
    score      = float(best[2])

    # Priorite aux caracteres arabes
    arabic_chars = re.sub(r"[^\u0600-\u06FF]", "", raw_letter)
    if arabic_chars:
        return arabic_chars[0], score

    # Correction des confusions latines
    for char in raw_letter:
        if char in _LETTER_CORRECTIONS:
            return _LETTER_CORRECTIONS[char], score * 0.75

    return "?", score * 0.5


# ------------------------------------------------------------------
# Fallback OCR global
# ------------------------------------------------------------------

def _ocr_global_fallback(reader, crop):
    """OCR global sur la plaque trimmee (sans la bande publicitaire)."""
    if reader is None:
        return None, None

    trimmed = _trim_plate_crop(crop)
    thresh  = _preprocess(trimmed, scale=2.0)
    if thresh is None:
        return None, None

    detections = reader.readtext(thresh, detail=1, paragraph=False)
    if not detections:
        return None, None

    parts, scores = [], []
    for _, text, score in detections:
        cleaned = text.strip().translate(_ARABIC_INDIC)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
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

def _validate_moroccan_format(text: str):
    if not text:
        return None
    m = _PLATE_REGEX.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    return None


# ------------------------------------------------------------------
# Fonction principale
# ------------------------------------------------------------------

def read_plate_moroccan(reader, plate_crop):
    """
    Lecture robuste d'une plaque marocaine.

    Pipeline :
      1. Trim (supprime la bande publicitaire du bas)
      2. Segmentation par projection de colonnes (separateurs detectes)
      3. Fallback : segmentation par proportions fixes
      4. OCR specialise par zone (chiffres / lettre / region)
      5. Fallback final : OCR global sur plaque trimmee

    Retourne (texte, score) ou (None, None).
    """
    if reader is None or plate_crop is None or plate_crop.size == 0:
        return None, None

    trimmed = _trim_plate_crop(plate_crop)

    # Methode 1 : separateurs detectes
    segments = _segment_by_separators(trimmed)
    method   = "separateurs"

    # Methode 2 : proportions fixes
    if segments is None:
        segments = _segment_by_proportions(trimmed)
        method   = "proportions"

    if segments is not None:
        z_num, z_let, z_reg = segments
        numbers, s1 = _ocr_digits(reader, z_num)
        letter,  s2 = _ocr_arabic_letter(reader, z_let)
        region,  s3 = _ocr_digits(reader, z_reg)

        scores    = [s for s in [s1, s2, s3] if s > 0]
        avg_score = float(np.mean(scores)) if scores else 0.0

        if numbers or region:
            text      = f"{numbers} {letter} {region}".strip()
            validated = _validate_moroccan_format(text)
            if validated:
                print(f"[OCR] {method} | valide: {validated}")
                return validated, avg_score
            if numbers and region:
                print(f"[OCR] {method} | partiel: {text}")
                return text, avg_score

    # Fallback global
    print("[OCR] Fallback OCR global")
    text, score = _ocr_global_fallback(reader, plate_crop)
    if text:
        validated = _validate_moroccan_format(text)
        return (validated or text), score

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
    detections = YOLO(str(model_file))(image, conf=conf, verbose=False)[0]
    if not detections.boxes:
        return None, None, None
    best_box = max(detections.boxes, key=lambda b: float(b.conf[0]))
    bbox  = [float(v) for v in best_box.xyxy[0]]
    score = float(best_box.conf[0])
    return _crop_bbox(image, bbox, padding=4), bbox, score


def _save_debug_outputs(output_dir, stem, source_img, plate_crop,
                        segments=None, bbox=None):
    saved = {}
    src = source_img.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
    p = output_dir / f"{stem}_source.jpg"
    cv2.imwrite(str(p), src)
    saved["source"] = p

    if plate_crop is not None:
        p = output_dir / f"{stem}_plate_crop.jpg"
        cv2.imwrite(str(p), plate_crop)
        saved["crop"] = p
        trimmed = _trim_plate_crop(plate_crop)
        p2 = output_dir / f"{stem}_plate_trimmed.jpg"
        cv2.imwrite(str(p2), trimmed)
        saved["trimmed"] = p2

    if segments is not None:
        for name, zone in zip(["numbers", "letter", "region"], segments):
            p = output_dir / f"{stem}_zone_{name}.jpg"
            cv2.imwrite(str(p), zone)
            saved[name] = p

    return saved


def _build_parser():
    parser = argparse.ArgumentParser(description="Test OCR plaque marocaine v3")
    parser.add_argument("image", help="Image source")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-detect", action="store_true",
                        help="Traiter l'image comme un crop de plaque")
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
        print(f"[INFO] Plaque detectee (score detection: {det_score:.2f})")

    text, score = read_plate_moroccan(reader, plate_crop)
    ocr_score   = 0.0 if score is None else score
    print(f"\nResultat : {text!r}  (confiance OCR: {ocr_score:.2f})")

    trimmed  = _trim_plate_crop(plate_crop)
    segments = _segment_by_separators(trimmed)
    label    = "separateurs"
    if segments is None:
        segments = _segment_by_proportions(trimmed)
        label    = "proportions (fallback)"
    print(f"[INFO] Segmentation utilisee: {label}")

    output_dir = _ensure_dir(Path(args.output_dir))
    saved      = _save_debug_outputs(
        output_dir, image_path.stem, img, plate_crop, segments, bbox
    )
    print("\n[INFO] Fichiers debug:")
    for label, path in saved.items():
        print(f"       {label}: {path}")