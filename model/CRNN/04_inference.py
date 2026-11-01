# -*- coding: utf-8 -*-
"""
04_inference.py  —  CRNN inference + envoi vers le backend FastAPI

NOUVEAUTES :
  - send_detection(immatriculation, confiance, source_type, camera_id)
    POST /detections → le backend gère l'alerte automatiquement
  - check_vehicle(immatriculation)  GET /vehicules/immatriculation/{imm}
    → retourne le statut du véhicule ou "inconnu"
  - PlateReader.read() envoie la détection si API_URL est défini

Usage autonome :
  python 04_inference.py image.jpg --debug --show
python 04_inference.py car_volee.jpg --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --char-model  models/best_model.keras --api-url http://127.0.0.1:8000 --camera-id 1 --debug --show
python 04_inference.py car_volee.jpg --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --char-model  models/best_model.keras --debug --show

      
"""

"""
CORRECTIONS APPORTEES :
=======================
  1. decode_ctc() : decodage greedy MANUEL coherent avec blank=0
  2. normalize_label() : meme normalisation que l'entrainement
  3. PlateReader.read() : retourne le texte normalise + confiance

Integration dans util.py (ton pipeline YOLO existant) :
  from inference_04 import PlateReader

  # Une seule fois au demarrage :
  _reader = PlateReader("models/best_model.keras")

  def get_best_plate_text(frame, pairings, ocr_reader):
      for p in pairings:
          crop = crop_bbox(frame, p["plate"]["bbox"])
          p["ocr_text"], p["ocr_score"] = _reader.read(crop)
      return pairings

Usage autonome :
  python 04_inference.py image.jpg
    --plate-model ../yolo.../best.pt
    --char-model  models/best_model.keras
    --debug
    --show
python 04_inference.py car5.jpg --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --char-model  models/best_model.keras --debug --show
"""




import argparse
import re
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
import importlib.util

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Import depuis model_02.py
# ------------------------------------------------------------------

def _load_module(filename):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location("_mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod            = _load_module("model_02.py")
VOCAB           = _mod.VOCAB
CHAR_TO_IDX     = _mod.CHAR_TO_IDX
IDX_TO_CHAR     = _mod.IDX_TO_CHAR
IMG_HEIGHT      = _mod.IMG_HEIGHT
IMG_WIDTH       = _mod.IMG_WIDTH
CTCLoss         = _mod.CTCLoss
BLANK_IDX       = _mod.BLANK_IDX
normalize_label = _mod.normalize_label

_ARABIC_CHARS = "".join(c for c in VOCAB if not c.isdigit())
_PLATE_REGEX  = re.compile( # on utilise recompile pour avoir un modele reutilisable
    r"(\d{1,5})\s*([" + re.escape(_ARABIC_CHARS) + r"])\s*(\d{1,2})"
)


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def check_vehicle(immatriculation: str, api_url: str) -> dict:
    """
    Cherche un véhicule par son immatriculation dans le backend.

    Retourne :
      {"found": True,  "statut": "safe"|"vole"|"suspicieux", ...}
      {"found": False, "statut": "inconnu"}
    """
    try:
        url = f"{api_url.rstrip('/')}/vehicules/immatriculation/{immatriculation}"
        r   = requests.get(url, timeout=3)
        if r.status_code == 200:
            data = r.json()
            return {"found": True, **data}
        elif r.status_code == 404:
            return {"found": False, "statut": "inconnu"}
        else:
            logger.warning(f"[API] check_vehicle HTTP {r.status_code}")
            return {"found": False, "statut": "inconnu"}
    except requests.RequestException as e:
        logger.warning(f"[API] check_vehicle error : {e}")
        return {"found": False, "statut": "inconnu"}


def send_detection(
    immatriculation: str,
    confiance: float,
    api_url: str,
    source_type: str = "photo",
    camera_id: int | None = None,
) -> dict | None:
    """
    Envoie une détection au backend FastAPI.
    POST /detections  →  le backend crée l'alerte si nécessaire.

    Retourne le JSON de réponse ou None en cas d'erreur.
    """
    payload = {
        "immatriculation": immatriculation,
        "confiance":       round(float(confiance), 4),
        "detected_at":     datetime.now(timezone.utc).isoformat(),
        "source_type":     source_type,
        "camera_id":       camera_id,
    }
    try:
        url = f"{api_url.rstrip('/')}/detections"
        r   = requests.post(url, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning(f"[API] send_detection error : {e}")
        return None


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def preprocess_image(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    return resized.astype(np.float32)[..., np.newaxis]



# ------------------------------------------------------------------
# Décodage CTC greedy (blank=0)
# ------------------------------------------------------------------

def decode_ctc(y_pred):
    """
    Décodage CTC greedy Manuel — blank=0 explicite.
    Évite le bug keras.backend.ctc_decode qui suppose blank = dernier index.
    """
    best_path = np.argmax(y_pred, axis=-1)
    results   = []
    for path in best_path:
        collapsed = [int(path[0])]
        for idx in path[1:]:
            if int(idx) != collapsed[-1]:
                collapsed.append(int(idx))
        text = "".join(
            IDX_TO_CHAR.get(idx, "")
            for idx in collapsed
            if idx != BLANK_IDX
        )
        results.append(text)
    return results


# ------------------------------------------------------------------
# PlateReader
# ------------------------------------------------------------------

class PlateReader:
    """
    Lecteur CRNN + intégration backend.

    Paramètres optionnels :
      api_url   — URL de base du backend FastAPI (ex: "http://127.0.0.1:8000")
      camera_id — identifiant de la caméra (int)
    """

    def __init__(self, model_path: str, api_url: str | None = None,
                 camera_id: int | None = None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle CRNN introuvable : {model_path}\n"
                f"Lance d'abord : python 03_train.py"
            )

        self.api_url   = api_url
        self.camera_id = camera_id

        print(f"[PlateReader] Chargement : {model_path}")
        self.model = keras.models.load_model(
            str(model_path),
            custom_objects={"CTCLoss": CTCLoss}
        )
        print(f"[PlateReader] OK — entrée={self.model.input_shape} "
              f"sortie={self.model.output_shape}")

    # ----------------------------------------------------------
    def read(self, plate_crop_bgr, debug=False, source_type="photo"):
        """
        Lit le texte d'un crop plaque, envoie la détection au backend.

        Retourne :
          (texte, confiance, vehicle_info)
          vehicle_info : {"found": bool, "statut": str, ...} ou {}
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return None, None, {}

        img = preprocess_image(plate_crop_bgr)
        if img is None:
            return None, None, {}

        batch  = img[np.newaxis, ...]
        y_pred = self.model.predict(batch, verbose=0)

        texts = decode_ctc(y_pred)
        text  = normalize_label(texts[0] if texts else "")

        if debug:
            avg_conf = float(y_pred[0].max(axis=-1).mean())
            print(f"  [CRNN] brut='{text}' conf_moy={avg_conf:.3f}")

        if not text:
            return None, None, {}

        confidence = float(y_pred[0].max(axis=-1).mean())

        # Validation format
        m = _PLATE_REGEX.search(text)
        if m:
            validated = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        else:
            digits = re.sub(r"[^0-9]", "", text)
            if len(digits) >= 3:
                validated, confidence = text, confidence * 0.7
            else:
                return None, None, {}

        if debug:
            print(f"  [CRNN] validé='{validated}' conf={confidence:.3f}")

        # ── Interrogation + envoi vers le backend ──────────────────────
        vehicle_info = {}
        if self.api_url:
            vehicle_info = check_vehicle(validated.replace(" ", ""),
                                         self.api_url)
            if debug:
                print(f"  [API]  véhicule={vehicle_info}")

            send_detection(
                immatriculation=validated.replace(" ", ""),
                confiance=confidence,
                api_url=self.api_url,
                source_type=source_type,
                camera_id=self.camera_id,
            )

        return validated, confidence, vehicle_info

    # ----------------------------------------------------------
    def read_batch(self, plate_crops, source_type="photo"):
        """
        Lit plusieurs plaques en un seul batch (efficace GPU).
        Retourne : list[(texte, confiance, vehicle_info)]
        """
        imgs, valid_idx = [], []
        for i, crop in enumerate(plate_crops):
            img = preprocess_image(crop)
            if img is not None:
                imgs.append(img)
                valid_idx.append(i)

        if not imgs:
            return [(None, None, {})] * len(plate_crops)

        batch  = np.array(imgs, dtype=np.float32)
        y_pred = self.model.predict(batch, verbose=0)
        texts  = decode_ctc(y_pred)

        results = [(None, None, {})] * len(plate_crops)
        for j, orig_idx in enumerate(valid_idx):
            text = normalize_label(texts[j])
            conf = float(y_pred[j].max(axis=-1).mean())
            m    = _PLATE_REGEX.search(text)
            if m:
                validated = f"{m.group(1)} {m.group(2)} {m.group(3)}"
            elif text and len(re.sub(r"[^0-9]", "", text)) >= 3:
                validated, conf = text, conf * 0.7
            else:
                continue

            imm          = validated.replace(" ", "")
            vehicle_info = {}
            if self.api_url:
                vehicle_info = check_vehicle(imm, self.api_url)
                send_detection(imm, conf, self.api_url, source_type,
                               self.camera_id)

            results[orig_idx] = (validated, conf, vehicle_info)

        return results


# ------------------------------------------------------------------
# Utilitaires CLI
# ------------------------------------------------------------------

def _detect_plate(image, model_path, conf=0.25):
    try:
        from ultralytics import YOLO
        m = Path(model_path)
        if not m.exists():
            return image, None
        dets = YOLO(str(m))(image, conf=conf, verbose=False)[0]
        if not dets.boxes:
            return None, None
        best = max(dets.boxes, key=lambda b: float(b.conf[0]))
        v    = [float(x) for x in best.xyxy[0]]
        h, w = image.shape[:2]
        crop = image[max(0,int(v[1])-4):min(h,int(v[3])+4),
                     max(0,int(v[0])-4):min(w,int(v[2])+4)]
        return crop, v
    except Exception as e:
        print(f"[WARN] YOLO : {e}")
        return image, None


_STATUS_COLOR = {
    "safe":        (0, 200, 0),    # vert
    "vole":        (0, 0, 220),    # rouge
    "suspicieux":  (0, 140, 255),  # orange
    "inconnu":     (180, 180, 0),  # jaune
}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inference CRNN plaque marocaine")
    p.add_argument("image")
    p.add_argument("--plate-model",
                   default="../yolo_existing_labels/runs/detect/runs/detect/"
                           "plate_maroc_existing_labels/weights/best.pt")
    p.add_argument("--char-model",   default="models/best_model.keras")
    p.add_argument("--api-url",      default="http://127.0.0.1:8000",
                   help="URL du backend FastAPI (laisser vide pour désactiver)")
    p.add_argument("--camera-id",    type=int, default=None)
    p.add_argument("--no-detect",    action="store_true")
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--show",         action="store_true")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"[ERREUR] Impossible de lire : {args.image}")

    if args.no_detect:
        plate_crop, bbox = img, None
    else:
        plate_crop, bbox = _detect_plate(img, args.plate_model)
        if plate_crop is None:
            raise SystemExit("[ERREUR] Aucune plaque détectée")
        print("[INFO] Plaque détectée")

    reader               = PlateReader(args.char_model,
                                       api_url=args.api_url or None,
                                       camera_id=args.camera_id)
    text, score, vinfo   = reader.read(plate_crop, debug=args.debug,
                                       source_type="photo")

    print(f"\nRésultat   : {text!r}")
    print(f"Confiance  : {0.0 if score is None else score:.3f}")
    print(f"Véhicule   : {vinfo}")

    if args.show:
        display = img.copy()
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            statut  = vinfo.get("statut", "inconnu")
            color   = _STATUS_COLOR.get(statut, (180, 180, 180))
            label   = f"{text or '?'}  ({score or 0:.2f})  [{statut}]"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Résultat OCR CRNN", display)
        print("[INFO] Appuie sur une touche pour fermer…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()