"""
registry.py — chargement PARESSEUX et partagé des modèles (YOLO plaque + CRNN).

Pourquoi : l'ancien main.py chargeait le CRNN à l'import. Si le fichier .keras
manquait, tout le serveur plantait au démarrage (impossible de diagnostiquer
sur un hébergeur). Ici, l'API démarre toujours, /health indique ce qui manque,
et le CRNN n'est chargé qu'une seule fois pour l'image ET la vidéo.
"""
import importlib.util
import threading

import config

_lock = threading.Lock()
infer_lock = threading.Lock()   # TensorFlow/YOLO : une inférence à la fois

_reader = None
_plate_yolo = None
last_error: str | None = None


def get_reader():
    global _reader, last_error
    if _reader is None:
        with _lock:
            if _reader is None:
                if not config.CHAR_MODEL.exists():
                    raise FileNotFoundError(f"Modèle CRNN introuvable : {config.CHAR_MODEL}")
                spec = importlib.util.spec_from_file_location(
                    "crnn_inference", config.CRNN_INFERENCE_PY
                )
                crnn = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(crnn)
                _reader = crnn.PlateReader(str(config.CHAR_MODEL), api_url=None, camera_id=None)
    return _reader


def get_plate_yolo():
    global _plate_yolo
    if _plate_yolo is None:
        with _lock:
            if _plate_yolo is None:
                if not config.PLATE_MODEL.exists():
                    raise FileNotFoundError(f"Modèle YOLO introuvable : {config.PLATE_MODEL}")
                from ultralytics import YOLO

                _plate_yolo = YOLO(str(config.PLATE_MODEL))
    return _plate_yolo


def preload() -> None:
    """Appelé au démarrage : charge tout et mémorise l'erreur éventuelle."""
    global last_error
    try:
        get_plate_yolo()
        get_reader()
        last_error = None
    except Exception as exc:  # noqa: BLE001 — on veut démarrer quoi qu'il arrive
        last_error = f"{type(exc).__name__}: {exc}"
        print(f"[registry] Modèles indisponibles → {last_error}")


def status() -> dict:
    return {
        "yolo_loaded": _plate_yolo is not None,
        "crnn_loaded": _reader is not None,
        "yolo_file_present": config.PLATE_MODEL.exists(),
        "crnn_file_present": config.CHAR_MODEL.exists(),
        "error": last_error,
    }


def ready() -> bool:
    return _plate_yolo is not None and _reader is not None
