"""
plate_ocr_maroc.py
------------------
Lecture OCR d'une plaque marocaine a partir d'un crop de plaque.

Strategie:
  1. Detecter les barres separatrices verticales sur la plaque
  2. Decouper en 3 zones: [chiffres | lettre arabe | code region]
  3. Faire un OCR specialise par zone avec allowlist stricte
  4. Valider et assembler le resultat via regex
  5. Faire un fallback vers l'OCR global si la segmentation echoue

Le mode test de ce fichier peut aussi recevoir une image complete:
il detecte d'abord la plaque avec YOLO, puis applique l'OCR sur le crop.
"""

from pathlib import Path
import argparse
import re

import cv2
import numpy as np


ARABIC_LETTERS = "أابتثجحخدذرزسشصضطظعغفقكلمنهويىءآإؤئ"
DEFAULT_PLATE_MODEL = "runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt"
DEFAULT_DEBUG_DIR = "outputs/ocr_debug"

_PLATE_REGEX = re.compile(r"(\d{1,5})\s*([" + ARABIC_LETTERS + r"])\s*(\d{1,2})")


def _preprocess(crop, scale: float = 2.0):
    """Agrandit et nettoie un crop pour l'OCR."""
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _find_vertical_separators(gray, min_height_ratio: float = 0.45):
    """
    Retourne les positions X des barres separatrices verticales.
    """
    h, _ = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kh = max(5, int(h * min_height_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    col_sums = vertical.sum(axis=0).astype(float)
    if col_sums.max() == 0:
        return []

    threshold = col_sums.max() * 0.40
    hot_cols = np.where(col_sums > threshold)[0]
    if len(hot_cols) == 0:
        return []

    groups = []
    current = [hot_cols[0]]
    for col in hot_cols[1:]:
        if col - current[-1] <= 5:
            current.append(col)
        else:
            groups.append(current)
            current = [col]
    groups.append(current)

    return [int(np.mean(group)) for group in groups]


def _segment_moroccan_plate(crop):
    """
    Decoupe le crop de plaque en (zone_chiffres, zone_lettre, zone_region).
    Retourne None si les separateurs ne sont pas trouves.
    """
    if crop is None or crop.size == 0:
        return None

    _, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

    margin_left = int(w * 0.20)
    margin_right = int(w * 0.80)
    gray_center = gray[:, margin_left:margin_right]

    seps = _find_vertical_separators(gray_center)
    seps = [sep + margin_left for sep in seps]

    if len(seps) < 2:
        return None

    sep1, sep2 = seps[0], seps[-1]
    pad = max(3, int(w * 0.01))

    zone_numbers = crop[:, : max(1, sep1 - pad)]
    zone_letter = crop[:, min(w - 1, sep1 + pad): max(1, sep2 - pad)]
    zone_region = crop[:, min(w - 1, sep2 + pad):]

    min_width = int(w * 0.05)
    if zone_numbers.shape[1] < min_width or zone_region.shape[1] < min_width:
        return None

    return zone_numbers, zone_letter, zone_region


def _ocr_digits(reader, crop):
    """OCR sur une zone ne contenant que des chiffres."""
    proc = _preprocess(crop, scale=2.5)
    if proc is None:
        return "", 0.0

    results = reader.readtext(proc, detail=1, allowlist="0123456789", paragraph=False)
    if not results:
        return "", 0.0

    text = "".join(result[1] for result in results)
    score = float(np.mean([result[2] for result in results]))
    return re.sub(r"[^0-9]", "", text), score


def _ocr_arabic_letter(reader, crop):
    """
    OCR sur la zone centrale (lettre arabe).
    """
    proc = _preprocess(crop, scale=3.5)
    if proc is None:
        return "?", 0.0

    results = reader.readtext(proc, detail=1, allowlist=ARABIC_LETTERS, paragraph=False)
    if not results:
        results = reader.readtext(proc, detail=1, paragraph=False)
        if not results:
            return "?", 0.0

    best = max(results, key=lambda result: result[2])
    letter = re.sub(r"[^\u0600-\u06FF]", "", best[1].strip())
    return (letter[0] if letter else "?"), float(best[2])


def _ocr_global_fallback(reader, crop):
    """
    OCR global utilise si la segmentation a echoue.
    """
    if reader is None:
        return None, None

    thresh = _preprocess(crop, scale=2.0)
    if thresh is None:
        return None, None

    detections = reader.readtext(thresh, detail=1, paragraph=False)
    if not detections:
        return None, None

    parts = []
    scores = []
    for _, text, score in detections:
        cleaned = text.strip().upper()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[^0-9A-Z\u0600-\u06FF ]", "", cleaned).strip()
        if cleaned:
            parts.append(cleaned)
            scores.append(float(score))

    if not parts:
        return None, None

    return " ".join(parts), float(np.mean(scores))


def _validate_moroccan_format(text: str):
    """
    Verifie que le texte correspond au format plaque marocaine.
    Retourne le texte nettoye ou None.
    """
    if not text:
        return None

    match = _PLATE_REGEX.search(text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return None


def read_plate_moroccan(reader, plate_crop):
    """
    Lecture robuste d'une plaque marocaine.
    Retourne (texte, score) ou (None, None).
    """
    if reader is None or plate_crop is None or plate_crop.size == 0:
        return None, None

    segments = _segment_moroccan_plate(plate_crop)
    if segments is not None:
        zone_numbers, zone_letter, zone_region = segments

        numbers, score_numbers = _ocr_digits(reader, zone_numbers)
        letter, score_letter = _ocr_arabic_letter(reader, zone_letter)
        region, score_region = _ocr_digits(reader, zone_region)

        scores = [score for score in [score_numbers, score_letter, score_region] if score > 0]
        avg_score = float(np.mean(scores)) if scores else 0.0

        if numbers or region:
            text = f"{numbers} {letter} {region}".strip()
            validated = _validate_moroccan_format(text)
            if validated:
                return validated, avg_score
            if numbers and region:
                return text, avg_score

    text, score = _ocr_global_fallback(reader, plate_crop)
    if text:
        validated = _validate_moroccan_format(text)
        return (validated or text), score

    return None, None


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
    """
    Detecte la meilleure plaque dans une image complete et retourne (crop, bbox, score).
    """
    from ultralytics import YOLO

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modele de plaques introuvable: {model_file}")

    detections = YOLO(str(model_file))(image, conf=conf, verbose=False)[0]
    if not detections.boxes:
        return None, None, None

    best_box = max(detections.boxes, key=lambda box: float(box.conf[0]))
    bbox = [float(value) for value in best_box.xyxy[0]]
    score = float(best_box.conf[0])
    crop = _crop_bbox(image, bbox, padding=4)
    return crop, bbox, score


def _save_debug_outputs(output_dir: Path, stem: str, source_img, plate_crop, segments, bbox=None):
    """
    Sauvegarde les images utiles au debug sans utiliser cv2.imshow.
    """
    saved = {}

    source_path = output_dir / f"{stem}_source.jpg"
    source_to_save = source_img.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(source_to_save, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(source_path), source_to_save)
    saved["source"] = source_path

    if plate_crop is not None:
        crop_path = output_dir / f"{stem}_plate_crop.jpg"
        cv2.imwrite(str(crop_path), plate_crop)
        saved["crop"] = crop_path

    if segments is not None:
        zone_numbers, zone_letter, zone_region = segments
        numbers_path = output_dir / f"{stem}_zone_numbers.jpg"
        letter_path = output_dir / f"{stem}_zone_letter.jpg"
        region_path = output_dir / f"{stem}_zone_region.jpg"
        cv2.imwrite(str(numbers_path), zone_numbers)
        cv2.imwrite(str(letter_path), zone_letter)
        cv2.imwrite(str(region_path), zone_region)
        saved["numbers"] = numbers_path
        saved["letter"] = letter_path
        saved["region"] = region_path

    return saved


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Teste l'OCR plaque marocaine sur une image complete ou un crop."
    )
    parser.add_argument("image", help="Image source a tester")
    parser.add_argument(
        "--plate-model",
        default=DEFAULT_PLATE_MODEL,
        help="Modele YOLO de detection de plaque pour les images completes",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DEBUG_DIR,
        help="Dossier de sortie pour les fichiers de debug",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance de la detection de plaque",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Traite l'image comme un crop de plaque deja extrait",
    )
    return parser


if __name__ == "__main__":
    import easyocr

    parser = _build_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de lire: {image_path}")

    print("[INFO] Chargement EasyOCR (ar, en)...")
    reader = easyocr.Reader(["ar", "en"], gpu=False)

    plate_crop = img
    bbox = None
    det_score = None
    if not args.no_detect:
        plate_crop, bbox, det_score = detect_plate_crop(img, args.plate_model, conf=args.conf)
        if plate_crop is None:
            print("[WARN] Aucune plaque detectee. Test stoppe avant OCR.")
            raise SystemExit(1)
        print(f"[INFO] Plaque detectee avant OCR (score detection: {det_score:.2f})")

    text, score = read_plate_moroccan(reader, plate_crop)
    print(f"Resultat : {text!r}  (confiance OCR: {0.0 if score is None else score:.2f})")

    segments = _segment_moroccan_plate(plate_crop)
    if segments is None:
        print("[WARN] Segmentation echouee, OCR global utilise.")
    else:
        print("[INFO] Segmentation en 3 zones reussie.")

    output_dir = _ensure_dir(Path(args.output_dir))
    saved = _save_debug_outputs(output_dir, image_path.stem, img, plate_crop, segments, bbox=bbox)

    print("[INFO] Fichiers sauvegardes:")
    for label, path in saved.items():
        print(f"       {label}: {path}")
