from pathlib import Path
import importlib.util
import re

import cv2
import numpy as np
from plate_ocr_maroc import read_plate_moroccan

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def iter_image_files(folder: Path):
    for path in folder.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield path


class SimpleSortWrapper:
    def __init__(self, sort_cls):
        self.tracker = sort_cls()

    def update(self, vehicles):
        if not vehicles:
            detections = np.empty((0, 5), dtype=float)
        else:
            detections = np.array(
                [[*vehicle["bbox"], vehicle["score"]] for vehicle in vehicles],
                dtype=float,
            )

        tracks = self.tracker.update(detections)
        tracked_vehicles = []

        for x1, y1, x2, y2, track_id in tracks:
            tracked_vehicles.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "track_id": int(track_id),
                }
            )

        return tracked_vehicles


def load_sort_tracker():
    sort_path = Path(__file__).resolve().parents[1] / "sort" / "sort.py"
    if not sort_path.exists():
        print("[WARN] SORT introuvable, tracking desactive.")
        return None

    try:
        spec = importlib.util.spec_from_file_location("sort_module", sort_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return SimpleSortWrapper(module.Sort)
    except Exception as exc:
        print(f"[WARN] Impossible de charger SORT: {exc}")
        print("       Le pipeline continue sans tracking.")
        return None


def bbox_contains(inner_bbox, outer_bbox):
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def crop_bbox(frame, bbox, padding: int = 4):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def preprocess_plate_for_ocr(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def load_ocr_reader():
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "EasyOCR n'est pas installe. Lance: pip install easyocr"
        ) from exc

    print("[INFO] Chargement EasyOCR (ar, en)")
    return easyocr.Reader(["ar", "en"], gpu=False)


def normalize_plate_text(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"\s+", " ", text)
    allowed = re.sub(r"[^0-9A-Z\u0600-\u06FF ]", "", text)
    return allowed.strip()


def read_plate_text(ocr_reader, plate_crop):
    if ocr_reader is None:
        return None, None

    processed = preprocess_plate_for_ocr(plate_crop)
    if processed is None:
        return None, None

    detections = ocr_reader.readtext(processed, detail=1, paragraph=False)
    if not detections:
        return None, None

    best_text = None
    best_score = -1.0
    for _, text, score in detections:
        cleaned = normalize_plate_text(text)
        if not cleaned:
            continue
        if score > best_score:
            best_text = cleaned
            best_score = float(score)

    if best_text is None:
        return None, None

    return best_text, best_score


def get_plate_car_pairings(plates, vehicles):
    pairings = []

    for plate in plates:
        plate_bbox = plate["bbox"]
        best_vehicle = None
        best_vehicle_area = None

        for vehicle in vehicles:
            vehicle_bbox = vehicle["bbox"]
            if not bbox_contains(plate_bbox, vehicle_bbox):
                continue

            area = bbox_area(vehicle_bbox)
            if best_vehicle is None or area < best_vehicle_area:
                best_vehicle = vehicle
                best_vehicle_area = area

        if best_vehicle is None:
            continue

        pairings.append(
            {
                "vehicle": best_vehicle,
                "plate": plate,
                "ocr_text": None,
                "ocr_score": None,
            }
        )

    return pairings


def get_best_plate_text(frame, pairings, ocr_reader):
    for pairing in pairings:
        plate_crop = crop_bbox(frame, pairing["plate"]["bbox"])
        # text, score = read_plate_text(ocr_reader, plate_crop)
        text, score = read_plate_moroccan(ocr_reader,plate_crop)
        pairing["ocr_text"] = text
        pairing["ocr_score"] = score

    return pairings


def annotate_frame(frame, vehicles, pairings):
    annotated = frame.copy()

    for vehicle in vehicles:
        x1, y1, x2, y2 = map(int, vehicle["bbox"])
        track_suffix = f" ID {vehicle['track_id']}" if "track_id" in vehicle else ""
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            annotated,
            f"vehicle{track_suffix}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )

    for pairing in pairings:
        x1, y1, x2, y2 = map(int, pairing["plate"]["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = pairing["ocr_text"] or "plate"
        if pairing["ocr_score"] is not None:
            text = f"{text} ({pairing['ocr_score']:.2f})"

        cv2.putText(
            annotated,
            text,
            (x1, min(annotated.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return annotated
