"""
main.py — API FastAPI (détection de fraude d'immatriculation, plaques marocaines)

Lancement local :   uvicorn main:app --reload
Production (Docker) : voir Dockerfile à la racine du dépôt.
"""
import base64
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import config
import registry
from crud import alert as alert_crud
from crud import detection as detection_crud
from crud import vehicule as vehicule_crud
from db.database import SessionLocal
from routers.detect_video import router as detect_video_router
from schemas.classes import Alert, Detection, Vehicule
from seed import init_db

log = logging.getLogger("platewatch")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    created = init_db(seed=config.SEED_ON_STARTUP)
    if created:
        print(f"[seed] {created} véhicules insérés")
    # Chargement en tâche de fond : le port s'ouvre tout de suite (healthcheck
    # de l'hébergeur), /health indique « models_ready » quand c'est terminé.
    threading.Thread(target=registry.preload, daemon=True).start()
    yield


plate_detection_app = FastAPI(
    title="API Backend pour la Detection de Fraude",
    description=(
        "Backend pour inserer les vehicules, enregistrer les detections "
        "et creer des alertes lorsqu'une plaque est inconnue ou suspecte."
    ),
    version="v1",
    lifespan=lifespan,
)
app = plate_detection_app  # alias : `uvicorn main:app` ET `uvicorn main:plate_detection_app`

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Garde-fous du mode démo public ───────────────────────────────────────────
_WRITE_ALLOWED = {"/detect-image", "/detect-video", "/detect-frame", "/reset-camera"}
_VIDEO_PATHS = {"/detect-video"}
_WEBCAM_PATHS = {"/detect-frame", "/reset-camera"}
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Max RATE_LIMIT_PER_MIN analyses par minute et par IP (0 = désactivé)."""
    limit = config.RATE_LIMIT_PER_MIN
    if limit <= 0:
        return
    now = time.monotonic()
    q = _hits[_client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Trop de requêtes : réessayez dans une minute.")
    q.append(now)


@app.middleware("http")
async def guard(request: Request, call_next):
    path, method = request.url.path, request.method
    if path in _VIDEO_PATHS and not config.ENABLE_VIDEO:
        return JSONResponse({"detail": "L'analyse vidéo est désactivée sur cette démo."}, 403)
    if path in _WEBCAM_PATHS and not config.ENABLE_WEBCAM:
        return JSONResponse({"detail": "La webcam est désactivée sur cette démo."}, 403)
    if path in _VIDEO_PATHS:
        size = int(request.headers.get("content-length") or 0)
        if size > config.MAX_VIDEO_MB * 1024 * 1024:
            return JSONResponse({"detail": f"Vidéo trop lourde (max {config.MAX_VIDEO_MB} Mo)."}, 413)
        try:  # une HTTPException levée dans un middleware n'est pas convertie par FastAPI
            rate_limit(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, exc.status_code)
    if config.READ_ONLY and method not in {"GET", "HEAD", "OPTIONS"} and path not in _WRITE_ALLOWED:
        return JSONResponse({"detail": "Démo en lecture seule."}, 403)
    return await call_next(request)


app.include_router(detect_video_router)


# ── Santé / configuration (lues par le frontend) ─────────────────────────────
@app.get("/health")
def health():
    st = registry.status()
    return {"status": "ok", "models_ready": registry.ready(), **st}


@app.get("/config")
def public_config():
    return {
        "demo_mode": config.DEMO_MODE,
        "persist_detections": config.PERSIST_DETECTIONS,
        "enable_video": config.ENABLE_VIDEO,
        "enable_webcam": config.ENABLE_WEBCAM,
        "max_image_mb": config.MAX_IMAGE_MB,
        "max_video_mb": config.MAX_VIDEO_MB,
        "max_video_seconds": config.MAX_VIDEO_SECONDS,
    }


@app.get("/samples")
def list_samples():
    """Images d'exemple proposées dans l'interface (dossier backend_pfa/samples)."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in config.SAMPLES_DIR.glob("*") if p.suffix.lower() in exts)
    return [{"name": p.stem.replace("_", " ").title(), "url": f"/sample-images/{p.name}"} for p in files]


if config.SAMPLES_DIR.exists():
    app.mount("/sample-images", StaticFiles(directory=config.SAMPLES_DIR), name="samples")


def detect_plate_crop(image, conf: float = 0.25, padding: int = 4):
    """YOLO → (crop, bbox [x1,y1,x2,y2], confiance) de la meilleure plaque, ou (None,)*3."""
    yolo = registry.get_plate_yolo()
    with registry.infer_lock:
        result = yolo(image, conf=conf, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None

    best = max(boxes, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = [int(v) for v in best.xyxy[0]]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None
    return crop, [x1, y1, x2, y2], float(best.conf[0])


# Operations sur la table vehicule


@app.get("/vehicules")
def list_vehicules(db: Session = Depends(get_db)):
    return vehicule_crud.list_vehicules(db)


@app.get("/vehicules/immatriculation/{immatriculation}")
def get_vehicule_by_immatriculation(
    immatriculation: str, db: Session = Depends(get_db)
):
    db_vehicule = vehicule_crud.get_vehicule_by_immatriculation(db, immatriculation)
    if db_vehicule is None:
        raise HTTPException(status_code=404, detail="Vehicule introuvable")
    return db_vehicule


@app.get("/vehicules/{vehicule_id}")
def get_vehicule(vehicule_id: int, db: Session = Depends(get_db)):
    db_vehicule = vehicule_crud.get_vehicule(db, vehicule_id)
    if db_vehicule is None:
        raise HTTPException(status_code=404, detail="Vehicule introuvable")
    return db_vehicule


@app.post("/vehicules", status_code=status.HTTP_201_CREATED)
def create_vehicule(vehicule: Vehicule, db: Session = Depends(get_db)):
    existing = vehicule_crud.get_vehicule_by_immatriculation(
        db, vehicule.immatriculation
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Immatriculation deja existante")
    return vehicule_crud.create_vehicule(db, vehicule)


@app.put("/vehicules/{vehicule_id}")
def update_vehicule(
    vehicule_id: int, vehicule: Vehicule, db: Session = Depends(get_db)
):
    db_vehicule = vehicule_crud.update_vehicule(db, vehicule_id, vehicule)
    if db_vehicule is None:
        raise HTTPException(status_code=404, detail="Vehicule introuvable")
    return db_vehicule


@app.delete("/vehicules/{vehicule_id}")
def delete_vehicule(vehicule_id: int, db: Session = Depends(get_db)):
    deleted = vehicule_crud.delete_vehicule(db, vehicule_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Vehicule introuvable")
    return {"message": "Vehicule supprime"}


@app.delete("/vehicules")
def delete_all_vehicule(db: Session = Depends(get_db)):
    deleted_count = vehicule_crud.delete_all_vehicule(db)
    return {
        "message": "Tous les vehicules ont ete supprimes",
        "deleted_count": deleted_count,
    }




# Operations sur la table detection


@app.get("/detections")
def list_detections(db: Session = Depends(get_db)):
    return detection_crud.list_detections(db)


@app.get("/detections/interval")
def get_detections_interval(
    debut_date: datetime, fin_date: datetime, db: Session = Depends(get_db)
):
    return detection_crud.get_detections_interval(db, debut_date, fin_date)


@app.get("/detections/immatriculation/{immatriculation}/interval")
def get_detections_by_immatriculation_interval(
    immatriculation: str,
    debut_date: datetime,
    fin_date: datetime,
    db: Session = Depends(get_db),
):
    return detection_crud.get_detections_by_immatriculation_interval(
        db, immatriculation, debut_date, fin_date
    )


@app.get("/detections/{detection_id}")
def get_detection(detection_id: int, db: Session = Depends(get_db)):
    db_detection = detection_crud.get_detection(db, detection_id)
    if db_detection is None:
        raise HTTPException(status_code=404, detail="Detection introuvable")
    return db_detection


@app.post("/detections", status_code=status.HTTP_201_CREATED)
def create_detection(detection: Detection, db: Session = Depends(get_db)):
    return detection_crud.create_detection_and_alert_if_needed(db, detection)


@app.put("/detections/{detection_id}")
def update_detection(
    detection_id: int, detection: Detection, db: Session = Depends(get_db)
):
    db_detection = detection_crud.update_detection(db, detection_id, detection)
    if db_detection is None:
        raise HTTPException(status_code=404, detail="Detection introuvable")
    return db_detection


@app.delete("/detections/{detection_id}")
def delete_detection(detection_id: int, db: Session = Depends(get_db)):
    deleted = detection_crud.delete_detection(db, detection_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Detection introuvable")
    return {"message": "Detection supprimee"}


@app.delete("/detections")
def delete_all_detections(db: Session = Depends(get_db)):
    deleted_count = detection_crud.delete_all_detections(db)
    return {
        "message": "Toutes les detections ont ete supprimees",
        "deleted_count": deleted_count,
    }

# Operations sur la table alert


@app.get("/alerts")
def list_alerts(db: Session = Depends(get_db)):
    return alert_crud.list_alerts(db)


@app.get("/alerts/interval")
def get_alerts_interval(
    debut_date: datetime, fin_date: datetime, db: Session = Depends(get_db)
):
    return alert_crud.get_alerts_interval(db, debut_date, fin_date)


@app.get("/alerts/immatriculation/{immatriculation}/interval")
def get_alerts_by_immatriculation_interval(
    immatriculation: str,
    debut_date: datetime,
    fin_date: datetime,
    db: Session = Depends(get_db),
):
    return alert_crud.get_alerts_by_immatriculation_interval(
        db, immatriculation, debut_date, fin_date
    )


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    db_alert = alert_crud.get_alert(db, alert_id)
    if db_alert is None:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    return db_alert


@app.post("/alerts", status_code=status.HTTP_201_CREATED)
def create_alert(alert: Alert, db: Session = Depends(get_db)):
    return alert_crud.create_alert(db, alert)


@app.put("/alerts/{alert_id}")
def update_alert(alert_id: int, alert: Alert, db: Session = Depends(get_db)):
    db_alert = alert_crud.update_alert(db, alert_id, alert)
    if db_alert is None:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    return db_alert


@app.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    deleted = alert_crud.delete_alert(db, alert_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    return {"message": "Alerte supprimee"}


@app.delete("/alerts")
def delete_all_alerts(db: Session = Depends(get_db)):
    deleted_count = alert_crud.delete_all_alerts(db)
    return {
        "message": "Toutes les alertes ont ete supprimees",
        "deleted_count": deleted_count,
    }

# ── Analyse d'une image ──────────────────────────────────────────────────────
def _to_data_url(img_bgr, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


@app.post("/detect-image", dependencies=[Depends(rate_limit)])
def detect_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    `def` (et non `async def`) : l'inférence TensorFlow/YOLO est bloquante ;
    FastAPI l'exécute alors dans un thread et le serveur reste réactif.
    """
    t0 = time.perf_counter()
    if not registry.ready():
        detail = "Modèles en cours de chargement, réessayez dans quelques secondes."
        if registry.last_error:
            detail = f"Modèles indisponibles : {registry.last_error}"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)

    max_bytes = config.MAX_IMAGE_MB * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"Image trop lourde (max {config.MAX_IMAGE_MB} Mo).")

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image invalide")

    # Borne le temps de calcul sur CPU gratuit (les bbox restent dans le repère renvoyé).
    h0, w0 = image.shape[:2]
    scale = min(1.0, config.MAX_IMAGE_SIDE / max(h0, w0))
    if scale < 1.0:
        image = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]

    plate_crop, bbox, plate_confidence = detect_plate_crop(image)
    base = {"image_size": [w, h]}
    if plate_crop is None:
        return {**base, "success": False, "message": "Aucune plaque detectee", "statut": "inconnu"}

    with registry.infer_lock:
        text, confidence, _info = registry.get_reader().read(
            plate_crop, debug=False, source_type="photo"
        )
    if not text:
        return {
            **base, "success": False, "message": "Aucune plaque lisible",
            "statut": "inconnu", "bbox_plaque": bbox,
            "plaque_image": _to_data_url(plate_crop),
        }

    immatriculation = text.replace(" ", "")
    vehicule = vehicule_crud.get_vehicule_by_immatriculation(db, immatriculation)
    statut = vehicule.statut if vehicule is not None else "inconnu"

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # UTC naïf, comme les autres routes
    detection = Detection(
        immatriculation=immatriculation, confiance=confidence,
        detected_at=now, source_type="photo", camera_id=None,
    )
    if config.PERSIST_DETECTIONS:
        alert_created = detection_crud.create_detection_and_alert_if_needed(db, detection)["alert"] is not None
    else:  # démo publique : rien n'est stocké côté serveur
        alert_created = statut != "safe"

    return {
        **base,
        "success": True,
        "immatriculation": immatriculation,
        "texte_formate": text,
        "confiance": confidence,
        "confiance_detection_plaque": plate_confidence,
        "bbox_plaque": bbox,
        "plaque_image": _to_data_url(plate_crop),
        "statut": statut,
        "vehicule_trouve": vehicule is not None,
        "alerte_creee": alert_created,
        "detection": {
            "immatriculation": immatriculation, "confiance": confidence,
            "detected_at": now.isoformat(), "source_type": "photo", "camera_id": None,
        },
        "alert": ({"immatriculation": immatriculation, "infraction": statut, "temps": now.isoformat()}
                  if alert_created else None),
        "temps_ms": round((time.perf_counter() - t0) * 1000),
    }


# ── Frontend compilé (image Docker : un seul service, pas de CORS) ───────────
if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def home():
        return {"message": "API fonctionnelle", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
