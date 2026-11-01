"""
config.py — configuration centralisée, pilotée par variables d'environnement.

En local      : rien à définir (SQLite + chemins par défaut).
En production : voir README (section « Déploiement gratuit »).
"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ── Modèles ────────────────────────────────────────────────────────────────
CRNN_DIR = ROOT / "model" / "CRNN"                 # contient 04_inference.py + model_02.py
CRNN_INFERENCE_PY = CRNN_DIR / "04_inference.py"
CHAR_MODEL = Path(os.getenv("CRNN_MODEL", ROOT / "models" / "crnn_best.keras"))
PLATE_MODEL = Path(os.getenv("YOLO_PLATE_MODEL", ROOT / "models" / "plate_yolo.pt"))

# ── Base de données ────────────────────────────────────────────────────────
# SQLite par défaut (zéro configuration). Pour PostgreSQL :
#   DATABASE_URL=postgresql://user:pass@host:5432/db
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'platewatch.db'}")
SEED_CSV = BACKEND_DIR / "annotations_original.csv"
SEED_ON_STARTUP = _bool("SEED_ON_STARTUP", True)

# ── Mode démo public ───────────────────────────────────────────────────────
# DEMO_MODE=1 : API en lecture seule (sauf analyse), rien n'est enregistré
# côté serveur (les photos des visiteurs ne laissent aucune trace).
DEMO_MODE = _bool("DEMO_MODE", False)
PERSIST_DETECTIONS = _bool("PERSIST_DETECTIONS", not DEMO_MODE)
READ_ONLY = _bool("READ_ONLY", DEMO_MODE)

# ── Fonctionnalités ────────────────────────────────────────────────────────
ENABLE_VIDEO = _bool("ENABLE_VIDEO", not DEMO_MODE)     # phase 2
ENABLE_WEBCAM = _bool("ENABLE_WEBCAM", False)           # trop lourd pour un CPU gratuit

# ── Limites (protègent le serveur gratuit) ─────────────────────────────────
MAX_IMAGE_MB = _int("MAX_IMAGE_MB", 8)
MAX_IMAGE_SIDE = _int("MAX_IMAGE_SIDE", 1600)          # px, redimensionnement avant YOLO
MAX_VIDEO_MB = _int("MAX_VIDEO_MB", 15)
MAX_VIDEO_SECONDS = _int("MAX_VIDEO_SECONDS", 10)
RATE_LIMIT_PER_MIN = _int("RATE_LIMIT_PER_MIN", 15)    # analyses / minute / IP (0 = off)

# ── CORS (utile seulement si le front est hébergé ailleurs, ex. Vercel) ────
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# ── Front compilé servi par FastAPI (Docker) ───────────────────────────────
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", ROOT / "frontend" / "dist"))
SAMPLES_DIR = BACKEND_DIR / "samples"
