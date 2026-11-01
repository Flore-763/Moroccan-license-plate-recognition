"""
Pipeline simple pour:
- detecter les voitures avec un modele COCO
- detecter les plaques avec le modele entraine sur mes labels
- lire le texte avec EasyOCR
- visualiser le resultat sur image ou video

Exemples:
    python training_me.py --source ../yolo/video.mp4 --save
    python training_me.py --source ../maroc_car/dataset/IMG_0001.jpg --save
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from util import (
    VEHICLE_CLASS_IDS,
    annotate_frame,
    ensure_output_dir,
    get_best_plate_text,
    get_plate_car_pairings,
    iter_image_files,
    load_ocr_reader,
    load_sort_tracker,
)


DEFAULT_COCO_MODEL = "yolov8n.pt"
DEFAULT_PLATE_MODEL = "runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt"
DEFAULT_OUTPUT_DIR = "outputs"


def load_models(coco_model_path: str, plate_model_path: str):
    plate_path = Path(plate_model_path)
    if not plate_path.exists():
        raise FileNotFoundError(
            f"Modele de plaques introuvable: {plate_path}\n"
            "Entraine d'abord le modele avec 02_train_yolov8.py."
        )

    coco_model = YOLO(coco_model_path)
    plate_model = YOLO(str(plate_path))
    return coco_model, plate_model


def detect_vehicles(frame, coco_model, conf: float):
    detections = coco_model(frame, conf=conf, verbose=False)[0]
    vehicles = []

    for box in detections.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASS_IDS:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        score = float(box.conf[0])
        vehicles.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class_id": class_id,
            }
        )

    return vehicles


def detect_plates(frame, plate_model, conf: float):
    detections = plate_model(frame, conf=conf, verbose=False)[0]
    plates = []

    for box in detections.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        score = float(box.conf[0])
        plates.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": score,
            }
        )

    return plates


def process_frame(frame, coco_model, plate_model, ocr_reader, vehicle_conf: float, plate_conf: float, tracker=None):
    vehicles = detect_vehicles(frame, coco_model, vehicle_conf)
    tracked_vehicles = vehicles

    if tracker is not None:
        tracked_vehicles = tracker.update(vehicles)

    plates = detect_plates(frame, plate_model, plate_conf)
    pairings = get_plate_car_pairings(plates, tracked_vehicles)
    pairings = get_best_plate_text(frame, pairings, ocr_reader)
    annotated = annotate_frame(frame, tracked_vehicles, pairings)
    return annotated, tracked_vehicles, pairings


def process_image(source: Path, coco_model, plate_model, ocr_reader, output_dir: Path, vehicle_conf: float, plate_conf: float):
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Impossible de lire l'image: {source}")

    annotated, vehicles, pairings = process_frame(
        frame, coco_model, plate_model, ocr_reader, vehicle_conf, plate_conf
    )

    output_path = output_dir / source.name
    cv2.imwrite(str(output_path), annotated)
    print(f"[INFO] Image -> {source.name}")
    print(f"       voitures detectees: {len(vehicles)}")
    print(f"       plaques associees : {len(pairings)}")
    print(f"       sortie            : {output_path}")


def process_folder(source: Path, coco_model, plate_model, ocr_reader, output_dir: Path, vehicle_conf: float, plate_conf: float):
    image_files = list(iter_image_files(source))
    print(f"[INFO] {len(image_files)} images trouvees dans {source}")

    for image_path in image_files:
        process_image(image_path, coco_model, plate_model, ocr_reader, output_dir, vehicle_conf, plate_conf)


def process_video(source: Path, coco_model, plate_model, ocr_reader, output_dir: Path, vehicle_conf: float, plate_conf: float, use_sort: bool):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{source.stem}_annotated.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = load_sort_tracker() if use_sort else None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, _ = process_frame(
            frame, coco_model, plate_model, ocr_reader, vehicle_conf, plate_conf, tracker=tracker
        )
        writer.write(annotated)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"[INFO] Video traitee: {frame_idx} frames")
    print(f"       sortie: {output_path}")


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description="Detection voiture + plaque + OCR")
    parser.add_argument("--source", required=True, help="Image, dossier d'images ou video")
    parser.add_argument("--coco-model", default=DEFAULT_COCO_MODEL, help="Modele COCO pour voitures")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL, help="Modele de detection des plaques")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Dossier de sortie")
    parser.add_argument("--vehicle-conf", type=float, default=0.25, help="Seuil de confiance pour les voitures")
    parser.add_argument("--plate-conf", type=float, default=0.25, help="Seuil de confiance pour les plaques")
    parser.add_argument("--no-sort", action="store_true", help="Desactive SORT pour la video")
    parser.add_argument("--no-ocr", action="store_true", help="Desactive EasyOCR")
    parser.add_argument("--save", action="store_true", help="Conserve pour compatibilite, les sorties sont sauvegardees")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source introuvable: {source}")

    output_dir = ensure_output_dir(args.output_dir)
    coco_model, plate_model = load_models(args.coco_model, args.plate_model)
    ocr_reader = None if args.no_ocr else load_ocr_reader()

    if source.is_dir():
        process_folder(source, coco_model, plate_model, ocr_reader, output_dir, args.vehicle_conf, args.plate_conf)
    elif is_video_file(source):
        process_video(source, coco_model, plate_model, ocr_reader, output_dir, args.vehicle_conf, args.plate_conf, use_sort=not args.no_sort)
    else:
        process_image(source, coco_model, plate_model, ocr_reader, output_dir, args.vehicle_conf, args.plate_conf)


if __name__ == "__main__":
    main()
