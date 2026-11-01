"""
routers/detect_video.py
========================

Pipeline corrigé — tracking sur les PLAQUES (pas les voitures) :

  Passe 1 :
    YOLOv8 (entraîné) + BoT-SORT  → plate_track_id STABLE par plaque
    YOLOv8n COCO (optionnel)       → car_track_id  (contexte / annotation)
    Filtre crops : taille min + netteté Laplacien
    Min-heap MAX_CROPS meilleurs crops par plate_track_id (CORRIGÉ)
    Vote OCR sur les N meilleurs crops → texte final unique par plaque

  Passe 2 :
    Re-lecture de la vidéo → annotation + écriture MP4 téléchargeable

Endpoints :
  POST /detect-video        → fichier vidéo complet
  GET  /video/{video_id}    → téléchargement vidéo annotée
  POST /detect-frame        → webcam live
  POST /reset-camera        → reset tracker webcam
"""

import re
import sys
import uuid
import heapq
import importlib.util
import tempfile
import os
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Tuple
import cv2
import time
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

try:
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_OK = True
except ImportError:
    _ARABIC_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CHEMINS
# ══════════════════════════════════════════════════════════════════════════════

import config
import registry
CRNN_DIR  = config.CRNN_DIR
CRNN_PATH = config.CRNN_INFERENCE_PY
PLATE_MODEL_PATH = config.PLATE_MODEL
BOTSORT_CFG   = Path(__file__).parent / "botsort.yaml"
# BYTETRACK_CFG = Path(__file__).parent / "bytetrack.yaml"

VIDEOS_DIR = Path(tempfile.gettempdir()) / "platewatch_videos"
VIDEOS_DIR.mkdir(exist_ok=True)
VIDEO_MAX_AGE_H = 2

if str(CRNN_DIR) not in sys.path:
    sys.path.insert(0, str(CRNN_DIR))


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_reader = _plate_yolo = _car_yolo = None

def _get_reader():
    return registry.get_reader()          # même instance que /detect-image

def _get_plate_yolo():
    return registry.get_plate_yolo()

def _get_car_yolo():
    global _car_yolo
    if _car_yolo is None:
        try:
            from ultralytics import YOLO
            _car_yolo = YOLO("yolov8n.pt")
        except Exception:
            _car_yolo = False
    return _car_yolo if _car_yolo else None


# ══════════════════════════════════════════════════════════════════════════════
#  ASSOCIATION PERSISTANTE PLAQUE ↔ VOITURE (CORRECTION 1)
# ══════════════════════════════════════════════════════════════════════════════

class CarPlateAssociation:
    """
    Association persistante entre car_track_id et plate_track_id.
    Résout le problème de multiples plate_track_id pour une même voiture.
    """
    
    def __init__(self):
        self.car_to_plate: Dict[int, int] = {}      # car_tid → plate_tid
        self.plate_to_car: Dict[int, int] = {}      # plate_tid → car_tid
        self.history: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)  # car_tid → [(plate_tid, frame, iou)]
        self.association_confidence: Dict[Tuple[int, int], float] = {}  # (car_tid, plate_tid) → max confidence
    
    def associate(self, plate_tid: int, car_tid: Optional[int], frame_idx: int, iou_score: float):
        """
        Associe une plaque à une voiture avec persistance.
        """
        if car_tid is None:
            return
        
        # Stocker l'historique
        self.history[car_tid].append((plate_tid, frame_idx, iou_score))
        
        # Mettre à jour la confiance d'association
        key = (car_tid, plate_tid)
        if key not in self.association_confidence:
            self.association_confidence[key] = iou_score
        else:
            self.association_confidence[key] = max(self.association_confidence[key], iou_score)
        
        # Si cette voiture a déjà une plaque assignée différente
        if car_tid in self.car_to_plate:
            current_plate = self.car_to_plate[car_tid]
            if current_plate != plate_tid:
                # Vérifier si la nouvelle association est meilleure
                current_conf = self.association_confidence.get((car_tid, current_plate), 0)
                new_conf = self.association_confidence.get((car_tid, plate_tid), 0)
                
                if new_conf > current_conf and new_conf > 0.7:
                    # Remplacer l'association
                    del self.plate_to_car[current_plate]
                    self.car_to_plate[car_tid] = plate_tid
                    self.plate_to_car[plate_tid] = car_tid
        else:
            # Nouvelle association
            self.car_to_plate[car_tid] = plate_tid
            self.plate_to_car[plate_tid] = car_tid
    
    def get_car_for_plate(self, plate_tid: int) -> Optional[int]:
        """Retourne le car_tid associé à une plaque."""
        return self.plate_to_car.get(plate_tid)
    
    def get_plate_for_car(self, car_tid: int) -> Optional[int]:
        """Retourne le plate_tid associé à une voiture."""
        return self.car_to_plate.get(car_tid)
    
    def get_merged_plates(self, threshold: float = 0.7) -> Dict[int, List[int]]:
        """
        Retourne un mapping car_tid → liste des plate_tid qui lui sont associés.
        Utilisé pour la fusion post-traitement.
        """
        merged = defaultdict(list)
        for car_tid, plate_tid in self.car_to_plate.items():
            merged[car_tid].append(plate_tid)
        return merged
    
    def get_best_plate_for_car(self, car_tid: int) -> Optional[int]:
        """
        Retourne le meilleur plate_tid pour une voiture (celui avec la plus haute confiance).
        """
        plates = self.get_merged_plates().get(car_tid, [])
        if not plates:
            return None
        
        if len(plates) == 1:
            return plates[0]
        
        # Choisir la plaque avec le plus de crops ou la meilleure confiance
        return max(plates, key=lambda p: len([h for h in self.history[car_tid] if h[0] == p]))


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def plate_similarity(p1, p2):
    """
    Similarité entre deux plaques OCR.
    Retourne une valeur entre 0 et 1.
    """
    return SequenceMatcher(None, p1, p2).ratio()


def merge_plate_candidates(detections, threshold=0.80):
    """
    Fusionne les lectures OCR proches appartenant
    au même véhicule.

    On conserve :
      - la plaque la plus votée
      - puis la plus confiante
    """

    by_car = defaultdict(list)

    for det in detections:
        by_car[det["car_track_id"]].append(det)

    merged = []

    for car_id, items in by_car.items():

        groups = []

        for det in items:

            assigned = False

            for group in groups:

                ref_plate = group[0]["immatriculation"]

                if plate_similarity(
                    det["immatriculation"],
                    ref_plate
                ) >= threshold:

                    group.append(det)
                    assigned = True
                    break

            if not assigned:
                groups.append([det])

        for group in groups:

            best = max(
                group,
                key=lambda x: (
                    x.get("votes", 0),
                    x.get("confiance", 0),
                    x.get("sharpness_max", 0)
                )
            )

            merged.append(best)

    return merged


def merge_plate_candidates_with_association(detections: List[dict], 
                                             association: CarPlateAssociation,
                                             threshold: float = 0.85) -> List[dict]:
    """
    Fusion améliorée utilisant les associations persistantes.
    """
    # 1. Regrouper par car_track_id via l'association
    by_car = defaultdict(list)
    for det in detections:
        car_id = det.get("car_track_id")
        if car_id is not None:
            # Utiliser le meilleur plate_tid associé si nécessaire
            best_plate = association.get_plate_for_car(car_id)
            if best_plate and best_plate != det.get("plate_track_id"):
                # Marquer comme potentiellement à fusionner
                det["should_merge_with"] = best_plate
        by_car[car_id].append(det)
    
    # 2. Regrouper par similarité de texte au sein du même car
    merged = []
    for car_id, items in by_car.items():
        if not items:
            continue
        
        groups = []
        for det in items:
            assigned = False
            for group in groups:
                ref_plate = group[0]["immatriculation"]
                if plate_similarity(det["immatriculation"], ref_plate) >= threshold:
                    group.append(det)
                    assigned = True
                    break
            if not assigned:
                groups.append([det])
        
        for group in groups:
            best = max(group, key=lambda x: (x.get("votes", 0), x.get("confiance", 0)))
            merged.append(best)
    
    return merged


# ── DB ────────────────────────────────────────────────────────────────────────
from db.database import SessionLocal
from crud import detection as detection_crud
from crud import vehicule as vehicule_crud
from schemas.classes import Detection, SourceType

router = APIRouter(tags=["video-detection"])

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CONF_PLATE    = 0.35
CONF_CAR      = 0.40

SHARPNESS_MIN = 50
MAX_CROPS     = 7

MIN_CROP_W    = 40
MIN_CROP_H    = 20
MAX_ASPECT    = 6.0

TRACK_TIMEOUT = 60
MAX_FRAMES    = 3000
PAD           = 6
CAR_CLASSES   = [2]

rejected_size = 0
rejected_ratio = 0

COLORS = {
    "safe":       (34,  197, 34),
    "vole":       (34,  34,  220),
    "suspicieux": (34,  140, 255),
    "inconnu":    (180, 180, 34),
    "default":    (140, 140, 140),
    "plate":      (34,  200, 34),
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _sharpness(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _get_crop(frame: np.ndarray, box) -> np.ndarray | None:
    """
    Extrait le crop avec PAD.
    Retourne None si le crop est trop petit ou a un ratio aberrant.
    """
    global rejected_size
    global rejected_ratio
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]) - PAD)
    y1 = max(0, int(box[1]) - PAD)
    x2 = min(w, int(box[2]) + PAD)
    y2 = min(h, int(box[3]) + PAD)

    cw, ch = x2 - x1, y2 - y1

    if cw < MIN_CROP_W or ch < MIN_CROP_H:
        rejected_size += 1
        return None

    ratio = cw / max(ch, 1)
    if ratio > MAX_ASPECT or ratio < 1.0:
        rejected_ratio += 1
        return None
    
    return frame[y1:y2, x1:x2].copy()


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _plate_in_car(plate, car) -> bool:
    ox = max(0, min(plate[2], car[2]) - max(plate[0], car[0]))
    oy = max(0, min(plate[3], car[3]) - max(plate[1], car[1]))
    pa = max(1, (plate[2]-plate[0]) * (plate[3]-plate[1]))
    return (ox * oy) / pa >= 0.50


def _normalize(text: str) -> str:
    t = text.replace(" ", "").strip()
    return t


def _fmt_ts(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def _enrich(imm: str, db: Session) -> dict:
    v = vehicule_crud.get_vehicule_by_immatriculation(db, imm)
    return {"found": v is not None, "statut": v.statut if v else "inconnu"}


def _save_detection(imm: str, conf: float, source: str, camera_id, db: Session):
    if not config.PERSIST_DETECTIONS:
        return
    try:
        detection_crud.create_detection_and_alert_if_needed(db, Detection(
            immatriculation=imm, confiance=round(float(conf), 4),
            detected_at=datetime.now(timezone.utc),
            source_type=source, camera_id=camera_id,
        ))
    except Exception:
        pass


def _tracker_cfg() -> str:
    return str(BOTSORT_CFG)


# ══════════════════════════════════════════════════════════════════════════════
#  VOTE OCR AMÉLIORÉ (CORRECTION 2)
# ══════════════════════════════════════════════════════════════════════════════

def _vote_ocr(crops: list, reader, source: str, 
              frame_indices: Optional[List[int]] = None,
              sharpness_values: Optional[List[float]] = None) -> Tuple[Optional[str], float]:
    """
    Vote OCR amélioré avec pondération par netteté.
    
    Lance le CRNN sur chaque crop.
    Retourne (texte_final, confiance) par vote majoritaire sur le texte normalisé,
    avec pondération par la netteté des crops.
    En cas d'égalité, retourne la lecture avec la meilleure confiance pondérée.
    """
    if not crops:
        return None, 0.0

    readings = []
    
    for idx, crop in enumerate(crops):
        # Calculer la netteté si non fournie
        sharp = sharpness_values[idx] if sharpness_values and idx < len(sharpness_values) else _sharpness(crop)
        
        text, conf, _ = reader.read(crop, source_type=source)
        if not text:
            continue
        
        # Pondération : netteté (30%) + conf OCR (70%)
        sharp_norm = min(sharp / 100.0, 1.0)  # Normaliser sharpness (max 100)
        weighted_conf = conf * 0.7 + sharp_norm * 0.3
        
        readings.append({
            "text": text,
            "norm": _normalize(text),
            "conf": float(conf or 0),
            "sharp": sharp,
            "weighted_conf": weighted_conf,
            "frame_idx": frame_indices[idx] if frame_indices and idx < len(frame_indices) else idx
        })

    if not readings:
        return None, 0.0

    # Vote pondéré par la confiance pondérée
    # 1. Regrouper par texte normalisé
    groups = defaultdict(list)
    for r in readings:
        groups[r["norm"]].append(r)
    
    # 2. Score par groupe = somme des confiances pondérées
    group_scores = {}
    for norm_text, group_readings in groups.items():
        group_scores[norm_text] = sum(r["weighted_conf"] for r in group_readings)
    
    # 3. Meilleur groupe
    best_norm = max(group_scores, key=group_scores.get)
    winners = groups[best_norm]
    
    # 4. Dans le groupe, prendre le meilleur (confiance * netteté)
    best = max(winners, key=lambda r: r["weighted_conf"])
    
    return best["text"], best["conf"]


# ══════════════════════════════════════════════════════════════════════════════
#  PLATE TRACK CORRIGÉ (CORRECTION 3 - Max-Heap)
# ══════════════════════════════════════════════════════════════════════════════

class PlateTrack:
    """
    Gère un track plaque BoT-SORT.
    CORRECTION: Utilise un max-heap via négation pour garder les MEILLEURS crops.
    """
    
    def __init__(self, plate_tid: int):
        self.plate_tid     = plate_tid
        self._heap: list   = []      # (-sharpness, seq, crop, frame_idx, bbox, car_tid) pour max-heap
        self._seq          = 0
        self.last_frame    = 0
        self.last_plt_bbox = None
        self.last_car_tid  = None
        self.done          = False
        self.result: dict  = {}
        self._sharpness_history: List[float] = []  # Historique des nettetés
        self._frame_history: List[int] = []       # Historique des frames

    def add(self, crop: np.ndarray, sharp: float,
            frame_idx: int, plt_bbox, car_tid=None):
        self.last_frame    = frame_idx
        self.last_plt_bbox = plt_bbox
        if car_tid is not None:
            self.last_car_tid = car_tid

        if crop is None or sharp < SHARPNESS_MIN:
            return

        # CORRECTION: Utiliser -sharp pour max-heap (les plus nets en tête)
        heapq.heappush(self._heap, (-sharp, self._seq, crop, frame_idx, plt_bbox, car_tid))
        self._seq += 1
        self._sharpness_history.append(sharp)
        self._frame_history.append(frame_idx)
        
        # CORRECTION: Retirer le moins bon (sharpness la plus faible)
        if len(self._heap) > MAX_CROPS:
            heapq.heappop(self._heap)  # Retire la plus petite sharpness (moins négative)

    def best_crops(self) -> list:
        """Retourne les crops triés par netteté décroissante."""
        # Trier par -sharp (du plus net au moins net)
        sorted_items = sorted(self._heap, key=lambda x: x[0])  # x[0] = -sharp
        return [item[2] for item in sorted_items]
    
    def best_metadata(self) -> List[dict]:
        """Retourne les métadonnées des meilleurs crops."""
        sorted_items = sorted(self._heap, key=lambda x: x[0])
        return [{
            "sharpness": -item[0],
            "seq": item[1],
            "frame_idx": item[3],
            "bbox": item[4],
            "car_tid": item[5]
        } for item in sorted_items]
    
    def best_sharpness(self) -> float:
        """Retourne la meilleure netteté."""
        if not self._heap:
            return 0.0
        return -min(self._heap, key=lambda x: x[0])[0]  # Le plus petit négatif = plus grande sharpness
    
    def get_sharpness_history(self) -> List[float]:
        """Retourne l'historique des nettetés."""
        return self._sharpness_history
    
    def get_frame_history(self) -> List[int]:
        """Retourne l'historique des frames."""
        return self._frame_history

    def finalize(self, reader, db: Session,
                 fps: float, source: str, camera_id) -> dict | None:
        if self.done:
            return self.result or None
        self.done = True

        crops = self.best_crops()
        if not crops:
            return None

        # Récupérer les frames indices pour le vote OCR
        frame_indices = [item[3] for item in sorted(self._heap, key=lambda x: x[0])]
        sharpness_values = [-item[0] for item in sorted(self._heap, key=lambda x: x[0])]
        
        # Vote OCR amélioré avec pondération
        text, conf = _vote_ocr(crops, reader, source, frame_indices, sharpness_values)
        
        if not text:
            return None

        imm  = text.replace(" ", "")
        info = _enrich(imm, db)
        _save_detection(imm, conf, source, camera_id, db)

        ts = self.last_frame / fps
        self.result = {
            "plate_track_id":  self.plate_tid,
            "car_track_id":    self.last_car_tid,
            "immatriculation": text,
            "confiance":       round(conf, 4),
            "votes":           len(crops),
            "sharpness_max":   round(self.best_sharpness(), 1),
            "statut":          info["statut"],
            "vehicule_trouve": info["found"],
            "alerte_creee":    info["statut"] not in ("safe", "inconnu"),
            "frame":           self.last_frame,
            "timestamp":       round(ts, 2),
            "timestamp_fmt":   _fmt_ts(ts),
            "bbox": self.last_plt_bbox
        }
        return self.result


# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

_font_cache: dict = {}

def _find_font(size=20):
    for c in ["C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/tahoma.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        if Path(c).exists():
            try: return ImageFont.truetype(c, size)
            except Exception: pass
    return ImageFont.load_default()

def _put_text(frame, text, pos, color, size=20):
    if _ARABIC_OK:
        try:
            bidi  = get_display(arabic_reshaper.reshape(text))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            draw  = ImageDraw.Draw(pil)
            if size not in _font_cache:
                _font_cache[size] = _find_font(size)
            draw.text(pos, bidi, font=_font_cache[size],
                      fill=(color[2], color[1], color[0]))
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame

def _annotate(frame: np.ndarray,
              meta: list[dict],
              results: dict[int, dict]) -> np.ndarray:
    """
    meta   : [{plate_tid, plt_bbox, car_tid, car_bbox}]
    results: plate_tid → résultat OCR final
    """
    out = frame.copy()
    for m in meta:
        ptid   = m["plate_tid"]
        pbbox  = m["plt_bbox"]
        cbbox  = m.get("car_bbox")
        res    = results.get(ptid)
        statut = res["statut"] if res else "inconnu"
        color  = COLORS.get(statut, COLORS["default"])

        # Boîte plaque
        cv2.rectangle(out, (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]),
                      COLORS["plate"], 2)

        # Boîte voiture (si disponible)
        if cbbox:
            cv2.rectangle(out, (cbbox[0], cbbox[1]), (cbbox[2], cbbox[3]),
                          color, 2)

        # Texte au-dessus de la plaque
        if res:
            label    = res["immatriculation"]
            font_sz  = max(14, int((pbbox[2]-pbbox[0]) * 0.22))
            text_y   = max(font_sz + 4, pbbox[1] - 6)
            out = _put_text(out, label, (pbbox[0], text_y - font_sz),
                            color, font_sz)
            # Badge statut non-safe
            if statut not in ("safe", "inconnu"):
                bw = len(statut) * 9 + 8
                cv2.rectangle(out, (pbbox[0], pbbox[3]+2),
                              (pbbox[0]+bw, pbbox[3]+16), color, -1)
                cv2.putText(out, statut.upper(), (pbbox[0]+3, pbbox[3]+13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  POST /detect-video
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/detect-video")
def detect_video(
    file: UploadFile = File(...),
    camera_id: int | None = Form(None),
    sharpness_threshold: int | None = Form(None),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    allowed_ext = {".mp4",".avi",".mov",".mkv",".webm"}
    allowed_ct  = {"video/mp4","video/avi","video/x-msvideo",
                   "video/quicktime","video/x-matroska","video/webm",
                   "application/octet-stream"}
    if file.content_type not in allowed_ct and ext not in allowed_ext:
        raise HTTPException(400, "Format vidéo non supporté")

    raw = file.file.read()
    with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
        tmp.write(raw); input_path = tmp.name

    video_id    = str(uuid.uuid4())
    output_path = str(VIDEOS_DIR / f"{video_id}.mp4")

    try:
        result = _run_pipeline(input_path, output_path, camera_id, db)
        result["video_id"]       = video_id
        result["video_download"] = f"/video/{video_id}"
        return result
    finally:
        try:
            os.unlink(input_path)
        except PermissionError:
            pass


def _run_pipeline(input_path: str, output_path: str,
                  camera_id, db: Session) -> dict:
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DÉBUT DU CHRONOMÉTRAGE GLOBAL
    # ═══════════════════════════════════════════════════════════════════════════
    # start_total = time.time()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(422, "Impossible d'ouvrir la vidéo")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config.DEMO_MODE and total / max(fps, 1) > config.MAX_VIDEO_SECONDS:
        cap.release()
        raise HTTPException(413, f"Vidéo trop longue (max {config.MAX_VIDEO_SECONDS} s sur la démo).")

    plate_yolo = _get_plate_yolo()
    car_yolo   = _get_car_yolo()
    reader     = _get_reader()
    tracker    = _tracker_cfg()

    # plate_track_id → PlateTrack
    plate_tracks: dict[int, PlateTrack] = {}
    
    # Association persistante plaque ↔ voiture
    association = CarPlateAssociation()

    # frame_idx → [{plate_tid, plt_bbox, car_tid, car_bbox}]
    frame_meta: dict[int, list] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # PASSE 1 : TRACKING + COLLECTE CROPS
    # ═══════════════════════════════════════════════════════════════════════════
    # start_pass1 = time.time()
    
    frame_idx = 0
    n_frames_yolo = 0
    n_plates_detected = 0
    
    while frame_idx < min(MAX_FRAMES, total):
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        frame_meta[frame_idx] = []

        # Voitures (BoT-SORT, optionnel)
        # start_car = time.time()
        car_boxes: list[tuple[int, list]] = []
        if car_yolo:
            for box in car_yolo.track(frame, persist=True, tracker=tracker,
                                      conf=CONF_CAR, verbose=False)[0].boxes:
                if int(box.cls[0]) not in CAR_CLASSES: continue
                if box.id is None: continue
                car_tid  = int(box.id[0])
                car_bbox = [round(float(v)) for v in box.xyxy[0]]
                car_boxes.append((car_tid, car_bbox))
        # time_car = time.time() - start_car

        # Plaques avec BoT-SORT → plate_track_id stable
        # start_plate = time.time()
        for box in plate_yolo.track(frame, persist=True, tracker=tracker,
                                    conf=CONF_PLATE, verbose=False)[0].boxes:
            if box.id is None: continue
            plate_tid = int(box.id[0])
            plt_bbox  = [round(float(v)) for v in box.xyxy[0]]
            n_plates_detected += 1
            
            # Filtre : crop valide ?
            crop  = _get_crop(frame, plt_bbox)
            if crop is None: continue
            sharp = _sharpness(crop)

            # Associer à une voiture
            best_car_tid  = None
            best_car_bbox = None
            best_score    = 0.0
            for car_tid, car_bbox in car_boxes:
                score = _iou(plt_bbox, car_bbox) + (0.5 if _plate_in_car(plt_bbox, car_bbox) else 0)
                if score > best_score:
                    best_score, best_car_tid, best_car_bbox = score, car_tid, car_bbox
            
            # Association persistante
            if best_car_tid is not None and best_score > 0.3:
                association.associate(plate_tid, best_car_tid, frame_idx, best_score)
                associated_car = association.get_car_for_plate(plate_tid)
                if associated_car is not None and associated_car != best_car_tid:
                    best_car_tid = associated_car

            # Initialiser le PlateTrack si nouveau
            if plate_tid not in plate_tracks:
                plate_tracks[plate_tid] = PlateTrack(plate_tid)

            plate_tracks[plate_tid].add(crop, sharp, frame_idx,
                                        plt_bbox, best_car_tid)

            frame_meta[frame_idx].append({
                "plate_tid": plate_tid,
                "plt_bbox":  plt_bbox,
                "car_tid":   best_car_tid,
                "car_bbox":  best_car_bbox,
            })
        # time_plate = time.time() - start_plate
        
        n_frames_yolo += 1

    cap.release()
    
    # end_pass1 = time.time()
    # time_pass1 = end_pass1 - start_pass1

    # ═══════════════════════════════════════════════════════════════════════════
    # PASSE 2 : VOTE OCR + FINALISATION
    # ═══════════════════════════════════════════════════════════════════════════
    # start_pass2 = time.time()
    
    results: dict[int, dict] = {}
    detections_out: list[dict] = []

    for ptid, track in plate_tracks.items():
        res = track.finalize(reader, db, fps, "video", camera_id)
        if res:
            best_car = association.get_car_for_plate(ptid)
            if best_car is not None:
                res["car_track_id"] = best_car
            results[ptid] = res
            detections_out.append(res)

    # Fusion améliorée avec association persistante
    detections_out = merge_plate_candidates_with_association(
        detections_out,
        association,
        threshold=0.80
    )

    detections_out.sort(key=lambda x: x["timestamp"])
    
    # end_pass2 = time.time()
    # time_pass2 = end_pass2 - start_pass2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASSE 3 : ANNOTATION VIDÉO
    # ═══════════════════════════════════════════════════════════════════════════
    # start_pass3 = time.time()
    
    cap2   = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    fi2 = 0
    while fi2 < min(MAX_FRAMES, total):
        ret, frame = cap2.read()
        if not ret: break
        fi2 += 1
        annotated = _annotate(frame, frame_meta.get(fi2, []), results)
        writer.write(annotated)

    cap2.release()
    writer.release()
    
    # end_pass3 = time.time()
    # time_pass3 = end_pass3 - start_pass3
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIN DU CHRONOMÉTRAGE GLOBAL
    # ═══════════════════════════════════════════════════════════════════════════
    # end_total = time.time()
    # time_total = end_total - start_total
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALCUL DES MÉTRIQUES DE PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # video_duration = total / fps if fps > 0 else 0
    
    # performance_metrics = {
    #     # Temps globaux
    #     "total_time_seconds": round(time_total, 2),
    #     "total_time_fmt": f"{int(time_total//60)}:{int(time_total%60):02d}",
        
    #     # Temps par passe
    #     "pass1_tracking_seconds": round(time_pass1, 2),
    #     "pass2_ocr_seconds": round(time_pass2, 2),
    #     "pass3_annotation_seconds": round(time_pass3, 2),
        
    #     # Pourcentages du temps total
    #     "pass1_percentage": round((time_pass1 / time_total) * 100, 1) if time_total > 0 else 0,
    #     "pass2_percentage": round((time_pass2 / time_total) * 100, 1) if time_total > 0 else 0,
    #     "pass3_percentage": round((time_pass3 / time_total) * 100, 1) if time_total > 0 else 0,
        
    #     # Performances de traitement
    #     "fps_processing": round(frame_idx / time_total, 2) if time_total > 0 else 0,
    #     "time_per_frame_ms": round((time_total / frame_idx) * 1000, 2) if frame_idx > 0 else 0,
    #     "real_time_ratio": round(time_total / video_duration, 2) if video_duration > 0 else 0,
        
    #     # Infos vidéo
    #     "video_duration_seconds": round(video_duration, 2),
    #     "video_fps_source": round(fps, 1),
    #     "total_frames": total,
    #     "frames_analysed": frame_idx,
        
    #     # Détections
    #     "tracks_total": len(plate_tracks),
    #     "plates_count": len(detections_out),
    #     "total_plate_detections_raw": n_plates_detected,
        
    #     # Filtres
    #     "rejected_by_size": rejected_size,
    #     "rejected_by_aspect_ratio": rejected_ratio,
    # }
    
    # print("\n" + "="*60)
    # print(" PERFORMANCES DE TRAITEMENT")
    # print("="*60)
    # print(f"⏱ Temps total: {performance_metrics['total_time_seconds']}s ({performance_metrics['total_time_fmt']})")
    # print(f" Ratio temps réel: {performance_metrics['real_time_ratio']}x")
    # print(f" FPS traitement: {performance_metrics['fps_processing']} f/s")
    # print(f" Temps par frame: {performance_metrics['time_per_frame_ms']} ms")
    # print(f"\n Répartition du temps:")
    # print(f"   Passe 1 (Tracking): {performance_metrics['pass1_percentage']}% ({performance_metrics['pass1_seconds']}s)")
    # print(f"   Passe 2 (OCR):      {performance_metrics['pass2_percentage']}% ({performance_metrics['pass2_seconds']}s)")
    # print(f"   Passe 3 (Annotation): {performance_metrics['pass3_percentage']}% ({performance_metrics['pass3_seconds']}s)")
    
    # print(f"\n Détections rejetées:")
    # print(f"   Taille min: {rejected_size}")
    # print(f"   Ratio aberrant: {rejected_ratio}")
    
    return {
        "success":         True,
        "frames_analysed": frame_idx,
        "frames_total":    total,
        "fps":             round(fps, 1),
        "tracks_total":    len(plate_tracks),
        "plates_count":    len(detections_out),
        "detections":      detections_out,
        # "performance":     performance_metrics  # ← AJOUT DES MÉTRIQUES
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /video/{video_id}  (CORRECTION 4 - Sécurité)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/video/{video_id}")
async def download_video(video_id: str):
    # CORRECTION: Validation stricte du format UUID pour éviter l'injection
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', video_id.lower()):
        raise HTTPException(400, "Format d'identifiant invalide")
    
    path = VIDEOS_DIR / f"{video_id}.mp4"
    
    # CORRECTION: Vérifier que le chemin résolu est bien dans VIDEOS_DIR
    try:
        resolved = path.resolve()
        if not str(resolved).startswith(str(VIDEOS_DIR.resolve())):
            raise HTTPException(403, "Accès interdit")
    except Exception:
        raise HTTPException(403, "Accès interdit")
    
    if not path.exists():
        raise HTTPException(404, "Vidéo expirée ou inexistante")
    
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"platewatch_{video_id[:8]}.mp4")


# ══════════════════════════════════════════════════════════════════════════════
#  POST /detect-frame  —  webcam live (avec corrections)
# ══════════════════════════════════════════════════════════════════════════════

_live_tracks:   dict[int, dict[int, PlateTrack]] = {}
_live_associations: dict[int, CarPlateAssociation] = {}  # CORRECTION: association par caméra
_live_counters: dict[int, int]                   = {}


@router.post("/detect-frame")
async def detect_frame(
    file: UploadFile = File(...),
    camera_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Frame illisible")

    cam = camera_id or 0
    if cam not in _live_tracks:
        _live_tracks[cam]   = {}
        _live_associations[cam] = CarPlateAssociation()
        _live_counters[cam] = 0

    _live_counters[cam] += 1
    fi       = _live_counters[cam]
    tracks   = _live_tracks[cam]
    association = _live_associations[cam]
    h, w     = frame.shape[:2]
    tracker  = _tracker_cfg()
    car_yolo = _get_car_yolo()

    # Voitures
    car_boxes = []
    if car_yolo:
        for box in car_yolo.track(frame, persist=True, tracker=tracker,
                                  conf=CONF_CAR, verbose=False)[0].boxes:
            if int(box.cls[0]) not in CAR_CLASSES: continue
            if box.id is None: continue
            car_boxes.append((int(box.id[0]),
                              [round(float(v)) for v in box.xyxy[0]]))

    # Plaques avec BoT-SORT
    visible  = []
    new_dets = []

    for box in _get_plate_yolo().track(frame, persist=True, tracker=tracker,
                                       conf=CONF_PLATE, verbose=False)[0].boxes:
        if box.id is None: continue
        plate_tid = int(box.id[0])
        plt_bbox  = [round(float(v)) for v in box.xyxy[0]]

        crop  = _get_crop(frame, plt_bbox)
        if crop is None: continue
        sharp = _sharpness(crop)

        best_car_tid = None
        best_score   = 0.0
        for car_tid, car_bbox in car_boxes:
            score = _iou(plt_bbox, car_bbox) + (0.5 if _plate_in_car(plt_bbox, car_bbox) else 0)
            if score > best_score:
                best_score, best_car_tid = score, car_tid
        
        # Association persistante
        if best_car_tid is not None and best_score > 0.3:
            association.associate(plate_tid, best_car_tid, fi, best_score)
            associated_car = association.get_car_for_plate(plate_tid)
            if associated_car is not None and associated_car != best_car_tid:
                best_car_tid = associated_car

        if plate_tid not in tracks:
            tracks[plate_tid] = PlateTrack(plate_tid)

        tracks[plate_tid].add(crop, sharp, fi, plt_bbox, best_car_tid)

        visible.append({
            "plate_track_id": plate_tid,
            "car_track_id":   best_car_tid,
            "bbox":     plt_bbox,
            "sharpness": round(sharp, 1),
            "frame_w": w, "frame_h": h,
        })

    # Finaliser les tracks perdus (timeout)
    for ptid, track in list(tracks.items()):
        if not track.done and fi - track.last_frame >= TRACK_TIMEOUT:
            res = track.finalize(_get_reader(), db, 30.0, "webcam", camera_id)
            if res:
                # Mettre à jour avec la meilleure association
                best_car = association.get_car_for_plate(ptid)
                if best_car is not None:
                    res["car_track_id"] = best_car
                new_dets.append(res)

    return {
        "frame_idx":      fi,
        "visible_plates": visible,
        "new_detections": new_dets,
        "frame_w": w, "frame_h": h,
    }


@router.post("/reset-camera")
async def reset_camera(camera_id: int | None = Form(None)):
    cam = camera_id or 0
    _live_tracks.pop(cam, None)
    _live_associations.pop(cam, None)
    _live_counters.pop(cam, None)
    return {"reset": True, "camera_id": cam}



























# # -*- coding: utf-8 -*-
# """
# routers/detect_video.py
# ========================
# Détection de plaques sur vidéo avec ByteTrack.

# Principe (simple) :
#   1. Pour chaque frame : YOLO.track() → chaque plaque reçoit un track_id stable
#   2. Par track_id : on garde uniquement le crop le plus net (variance Laplacien)
#   3. Fin de vidéo : on lit la plaque UNE SEULE FOIS sur le meilleur crop
#   → Une détection par véhicule, toujours sur la meilleure image disponible

# Endpoints :
#   POST /detect-video   — fichier vidéo complet
#   POST /detect-frame   — frame webcam (live)
#   POST /reset-camera   — remet à zéro le tracker webcam d'une caméra

# Ajouter dans main.py :
#   from routers.detect_video import router as detect_video_router
#   plate_detection_app.include_router(detect_video_router)

# Copier bytetrack.yaml dans le même dossier que ce fichier.
# """

# import sys
# import importlib.util
# import tempfile
# import os
# from pathlib import Path
# from datetime import datetime, timezone

# import cv2
# import numpy as np
# from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
# from sqlalchemy.orm import Session

# # ── Chemins ──────────────────────────────────────────────────────────────────
# CRNN_DIR  = Path(__file__).parent.parent.parent / "model" / "CRNN"
# CRNN_PATH = CRNN_DIR / "04_inference.py"
# PLATE_MODEL_PATH = (
#     CRNN_DIR.parent
#     / "yolo_existing_labels" / "runs" / "detect"
#     / "runs" / "detect"
#     / "plate_maroc_existing_labels" / "weights" / "best.pt"
# )
# BYTETRACK_CFG = Path(__file__).parent / "bytetrack.yaml"

# if str(CRNN_DIR) not in sys.path:
#     sys.path.insert(0, str(CRNN_DIR))

# # ── Singletons ────────────────────────────────────────────────────────────────
# _reader     = None
# _plate_yolo = None


# def _get_reader():
#     global _reader
#     if _reader is None:
#         spec = importlib.util.spec_from_file_location("crnn_inference", CRNN_PATH)
#         crnn = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(crnn)
#         _reader = crnn.PlateReader(str(CRNN_DIR / "models" / "best_model.keras"))
#     return _reader


# def _get_yolo():
#     global _plate_yolo
#     if _plate_yolo is None:
#         from ultralytics import YOLO
#         _plate_yolo = YOLO(str(PLATE_MODEL_PATH))
#     return _plate_yolo


# # ── DB ────────────────────────────────────────────────────────────────────────
# from db.database import SessionLocal
# from crud import detection as detection_crud
# from crud import vehicule as vehicule_crud
# from schemas.classes import Detection, SourceType

# router = APIRouter(tags=["video-detection"])


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# # ── Paramètres ────────────────────────────────────────────────────────────────
# CONF_YOLO      = 0.30   # seuil de détection YOLO
# PAD            = 6      # pixels de marge autour du crop plaque
# SHARPNESS_MIN  = 60     # variance Laplacien minimale pour stocker un crop
# DEDUP_FRAME_WINDOW = 90 # fusionne les tracks OCR proches vus a ~3s d'intervalle
# # Nombre de frames sans détection avant de finaliser un track (webcam live)
# TRACK_TIMEOUT  = 45     # ~1.5s à 30fps
# MAX_FRAMES     = 3000   # limite frames vidéo fichier


# # ── Utilitaires ───────────────────────────────────────────────────────────────

# def _sharpness(crop: np.ndarray) -> float:
#     """Variance du Laplacien — mesure de netteté d'un crop BGR."""
#     if crop is None or crop.size == 0:
#         return 0.0
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# def _get_crop(frame: np.ndarray, box) -> np.ndarray:
#     h, w = frame.shape[:2]
#     x1 = max(0, int(box[0]) - PAD)
#     y1 = max(0, int(box[1]) - PAD)
#     x2 = min(w, int(box[2]) + PAD)
#     y2 = min(h, int(box[3]) + PAD)
#     return frame[y1:y2, x1:x2].copy()


# def _fmt_ts(seconds: float) -> str:
#     return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# def _enrich(imm: str, db: Session) -> dict:
#     v = vehicule_crud.get_vehicule_by_immatriculation(db, imm)
#     return {"found": v is not None, "statut": v.statut if v else "inconnu"}


# def _save_detection(imm: str, conf: float, source: str, camera_id, db: Session):
#     try:
#         detection_crud.create_detection_and_alert_if_needed(db, Detection(
#             immatriculation=imm,
#             confiance=round(float(conf), 4),
#             detected_at=datetime.now(timezone.utc),
#             source_type=source,
#             camera_id=camera_id,
#         ))
#     except Exception:
#         pass


# # ══════════════════════════════════════════════════════════════════════════════
# #  TRACK STORE
# #  Stocke, par track_id, le meilleur crop (le plus net) vu jusqu'ici.
# # ══════════════════════════════════════════════════════════════════════════════

# def _plate_key(text: str) -> str:
#     """Normalise une lecture plaque pour comparer deux sorties OCR."""
#     return "".join(ch for ch in (text or "") if ch.isalnum())


# def _levenshtein(a: str, b: str) -> int:
#     """Distance d'edition simple, suffisante pour des plaques courtes."""
#     if a == b:
#         return 0
#     if not a:
#         return len(b)
#     if not b:
#         return len(a)
#     prev = list(range(len(b) + 1))
#     for i, ca in enumerate(a, start=1):
#         curr = [i]
#         for j, cb in enumerate(b, start=1):
#             curr.append(min(
#                 curr[j - 1] + 1,
#                 prev[j] + 1,
#                 prev[j - 1] + (ca != cb),
#             ))
#         prev = curr
#     return prev[-1]


# def _same_plate_reading(a: str, b: str) -> bool:
#     """Vrai si deux lectures OCR semblent representer la meme plaque."""
#     ka, kb = _plate_key(a), _plate_key(b)
#     if not ka or not kb:
#         return False
#     if ka == kb:
#         return True
#     if abs(len(ka) - len(kb)) > 1:
#         return False
#     return _levenshtein(ka, kb) <= 1


# def _result_score(result: dict) -> tuple[float, float]:
#     """Score utilise pour garder la meilleure lecture entre deux doublons."""
#     return (float(result.get("confiance") or 0), float(result.get("sharpness") or 0))


# def _dedupe_results(results: list[dict]) -> list[dict]:
#     """
#     ByteTrack peut casser une meme voiture en plusieurs IDs.
#     On fusionne les lectures OCR quasi identiques et proches dans le temps.
#     """
#     kept: list[dict] = []

#     for result in sorted(results, key=lambda d: d["frame"]):
#         duplicate_index = None
#         for i, existing in enumerate(kept):
#             close_in_time = abs(result["frame"] - existing["frame"]) <= DEDUP_FRAME_WINDOW
#             same_plate = _same_plate_reading(
#                 result["immatriculation"],
#                 existing["immatriculation"],
#             )
#             if close_in_time and same_plate:
#                 duplicate_index = i
#                 break

#         if duplicate_index is None:
#             kept.append(result)
#             continue

#         existing = kept[duplicate_index]
#         merged_track_ids = sorted({
#             *existing.get("merged_track_ids", [existing["track_id"]]),
#             result["track_id"],
#         })
#         if _result_score(result) > _result_score(existing):
#             result["merged_track_ids"] = merged_track_ids
#             kept[duplicate_index] = result
#         else:
#             existing["merged_track_ids"] = merged_track_ids

#     return kept


# class TrackStore:
#     """
#     Pour chaque track_id ByteTrack, mémorise :
#       - best_crop  : le crop BGR le plus net
#       - best_sharp : sa netteté (variance Laplacien)
#       - best_frame : index de frame correspondant
#       - last_frame : dernière frame où ce track a été vu (pour timeout webcam)
#       - done       : True une fois la lecture CRNN effectuée
#     """

#     def __init__(self):
#         self.tracks: dict[int, dict] = {}

#     def update(self, track_id: int, crop: np.ndarray,
#                sharp: float, frame_idx: int):
#         """Enregistre ou met à jour un track avec un nouveau crop."""
#         if track_id not in self.tracks:
#             self.tracks[track_id] = {
#                 "best_crop":  None,
#                 "best_sharp": 0.0,
#                 "best_frame": frame_idx,
#                 "last_frame": frame_idx,
#                 "done":       False,
#             }
#         t = self.tracks[track_id]
#         t["last_frame"] = frame_idx

#         # On ne garde un crop que s'il est net ET meilleur que le précédent
#         if sharp >= SHARPNESS_MIN and sharp > t["best_sharp"]:
#             t["best_crop"]  = crop
#             t["best_sharp"] = sharp
#             t["best_frame"] = frame_idx

#     def finalize(self, track_id: int, fps: float,
#                  reader, db: Session, source: str, camera_id,
#                  persist: bool = True) -> dict | None:
#         """
#         Lit la plaque sur le meilleur crop du track.
#         Retourne un dict résultat ou None si lecture impossible.
#         Marque le track comme 'done'.
#         """
#         t = self.tracks.get(track_id)
#         if t is None or t["done"]:
#             return None
#         t["done"] = True

#         crop = t["best_crop"]
#         if crop is None or crop.size == 0:
#             return None   # aucun crop assez net trouvé pour ce track

#         text, conf, _ = reader.read(crop, source_type=source)
#         if not text:
#             return None

#         imm  = text.replace(" ", "")
#         info = _enrich(imm, db)
#         if persist:
#             _save_detection(imm, conf or 0.0, source, camera_id, db)

#         timestamp = t["best_frame"] / fps
#         return {
#             "track_id":        track_id,
#             "immatriculation": text,
#             "confiance":       round(float(conf or 0.0), 4),
#             "sharpness":       round(t["best_sharp"], 1),
#             "statut":          info["statut"],
#             "vehicule_trouve": info["found"],
#             "alerte_creee":    info["statut"] not in ("safe", "inconnu"),
#             "frame":           t["best_frame"],
#             "timestamp":       round(timestamp, 2),
#             "timestamp_fmt":   _fmt_ts(timestamp),
#         }

#     def finalize_stale(self, current_frame: int, fps: float,
#                        reader, db: Session, source: str, camera_id,
#                        persist: bool = True) -> list[dict]:
#         """Finalise les tracks absents depuis TRACK_TIMEOUT frames (mode live)."""
#         results = []
#         for tid, t in self.tracks.items():
#             if not t["done"] and current_frame - t["last_frame"] >= TRACK_TIMEOUT:
#                 r = self.finalize(tid, fps, reader, db, source, camera_id, persist)
#                 if r:
#                     results.append(r)
#         return results

#     def finalize_all(self, fps: float,
#                      reader, db: Session, source: str, camera_id,
#                      persist: bool = True) -> list[dict]:
#         """Finalise tous les tracks restants (fin de vidéo)."""
#         results = []
#         for tid in list(self.tracks.keys()):
#             r = self.finalize(tid, fps, reader, db, source, camera_id, persist)
#             if r:
#                 results.append(r)
#         return results

#     def reset(self):
#         self.tracks.clear()


# # ══════════════════════════════════════════════════════════════════════════════
# #  POST /detect-video  —  fichier vidéo complet
# # ══════════════════════════════════════════════════════════════════════════════

# @router.post("/detect-video")
# async def detect_video(
#     file: UploadFile = File(...),
#     camera_id: int | None = Form(None),
#     db: Session = Depends(get_db),
# ):
#     ext = Path(file.filename or "").suffix.lower()
#     allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
#     allowed_ct  = {"video/mp4", "video/avi", "video/x-msvideo",
#                    "video/quicktime", "video/x-matroska", "video/webm",
#                    "application/octet-stream"}
#     if file.content_type not in allowed_ct and ext not in allowed_ext:
#         raise HTTPException(400, "Format vidéo non supporté (mp4/avi/mov/mkv/webm)")

#     raw = await file.read()
#     with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
#         tmp.write(raw)
#         tmp_path = tmp.name

#     try:
#         return _run_video(tmp_path, camera_id, db)
#     finally:
#         os.unlink(tmp_path)


# def _run_video(path: str, camera_id, db: Session) -> dict:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise HTTPException(422, "Impossible d'ouvrir la vidéo")

#     fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     yolo   = _get_yolo()
#     reader = _get_reader()
#     store  = TrackStore()

#     frame_idx = 0
#     while frame_idx < min(MAX_FRAMES, total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_idx += 1

#         # ByteTrack : persist=True maintient les IDs entre les frames
#         res = yolo.track(
#             frame,
#             persist=True,
#             tracker=str(BYTETRACK_CFG),
#             conf=CONF_YOLO,
#             verbose=False,
#         )

#         for box in res[0].boxes:
#             if box.id is None:
#                 continue
#             track_id = int(box.id[0])
#             x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
#             crop  = _get_crop(frame, (x1, y1, x2, y2))
#             sharp = _sharpness(crop)
#             store.update(track_id, crop, sharp, frame_idx)

#     cap.release()

#     # Lecture CRNN : une seule fois par track, sur le meilleur crop
#     results = store.finalize_all(fps, reader, db, "video", camera_id)
#     results.sort(key=lambda d: d["frame"])

#     return {
#         "success":         True,
#         "frames_analysed": frame_idx,
#         "frames_total":    total_frames,
#         "fps":             round(fps, 1),
#         "tracks_total":    len(store.tracks),
#         "plates_count":    len(results),
#         "detections":      results,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  POST /detect-frame  —  webcam live (une frame à la fois)
# # ══════════════════════════════════════════════════════════════════════════════

# # Un TrackStore par caméra, persistant entre les appels HTTP
# _live_stores:   dict[int, TrackStore] = {}
# _live_counters: dict[int, int]        = {}


# @router.post("/detect-frame")
# async def detect_frame(
#     file: UploadFile = File(...),
#     camera_id: int | None = Form(None),
#     db: Session = Depends(get_db),
# ):
#     """
#     Reçoit une frame JPEG de webcam.
#     Retourne :
#       - visible_plates : boîtes actuellement suivies (pour l'overlay)
#       - new_detections : plaques finalisées depuis le dernier appel
#     """
#     raw = await file.read()
#     arr = np.frombuffer(raw, np.uint8)
#     frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if frame is None:
#         raise HTTPException(422, "Frame illisible")

#     cam = camera_id or 0
#     if cam not in _live_stores:
#         _live_stores[cam]   = TrackStore()
#         _live_counters[cam] = 0

#     _live_counters[cam] += 1
#     frame_idx = _live_counters[cam]
#     store     = _live_stores[cam]
#     h, w      = frame.shape[:2]

#     yolo   = _get_yolo()
#     reader = _get_reader()

#     res = yolo.track(
#         frame,
#         persist=True,
#         tracker=str(BYTETRACK_CFG),
#         conf=CONF_YOLO,
#         verbose=False,
#     )

#     visible = []
#     for box in res[0].boxes:
#         if box.id is None:
#             continue
#         tid = int(box.id[0])
#         x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
#         crop  = _get_crop(frame, (x1, y1, x2, y2))
#         sharp = _sharpness(crop)
#         store.update(tid, crop, sharp, frame_idx)
#         visible.append({
#             "track_id": tid,
#             "bbox":     [round(x1), round(y1), round(x2), round(y2)],
#             "sharpness": round(sharp, 1),
#             "frame_w":   w,
#             "frame_h":   h,
#         })

#     # Finaliser les tracks qui ont disparu depuis TRACK_TIMEOUT frames
#     new_dets = store.finalize_stale(frame_idx, 30.0, reader, db, "video", camera_id)

#     return {
#         "frame_idx":      frame_idx,
#         "visible_plates": visible,
#         "new_detections": new_dets,
#         "frame_w": w,
#         "frame_h": h,
#     }


# @router.post("/reset-camera")
# async def reset_camera(camera_id: int | None = Form(None)):
#     """Réinitialise le TrackStore d'une caméra (fin de session webcam)."""
#     cam = camera_id or 0
#     if cam in _live_stores:
#         _live_stores[cam].reset()
#         _live_counters[cam] = 0
#     return {"reset": True, "camera_id": cam}
