from datetime import datetime

from sqlalchemy.orm import Session

from crud.alert import create_alert_from_detection
from models.tables import Detection as DetectionTable
from schemas.classes import Detection as DetectionSchema
from schemas.classes import Statut


def list_detections(db: Session):
    return db.query(DetectionTable).all()


def get_detection(db: Session, detection_id: int):
    return db.query(DetectionTable).filter(DetectionTable.id == detection_id).first()


def create_detection(db: Session, detection: DetectionSchema):
    db_detection = DetectionTable(**detection.model_dump())
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection


def create_detection_and_alert_if_needed(db: Session, detection: DetectionSchema):
    from crud.vehicule import get_vehicule_by_immatriculation

    db_detection = DetectionTable(**detection.model_dump())
    db.add(db_detection)

    vehicule = get_vehicule_by_immatriculation(db, detection.immatriculation)
    alert = None

    if vehicule is None:
        alert = create_alert_from_detection(
            db,
            detection.immatriculation,
            Statut.INCONNU.value,
        )
    elif vehicule.statut != Statut.SAFE.value:
        alert = create_alert_from_detection(
            db,
            detection.immatriculation,
            vehicule.statut,
        )

    db.commit()
    db.refresh(db_detection)
    if alert is not None:
        db.refresh(alert)

    return {"detection": db_detection, "alert": alert}


def update_detection(db: Session, detection_id: int, detection: DetectionSchema):
    db_detection = get_detection(db, detection_id)
    if db_detection is None:
        return None

    for field, value in detection.model_dump().items():
        setattr(db_detection, field, value)

    db.commit()
    db.refresh(db_detection)
    return db_detection


def delete_detection(db: Session, detection_id: int):
    db_detection = get_detection(db, detection_id)
    if db_detection is None:
        return None

    db.delete(db_detection)
    db.commit()
    return db_detection


def delete_all_detections(db: Session):
    deleted_count = db.query(DetectionTable).delete(synchronize_session=False)
    db.commit()
    return deleted_count  


def delete_all_detection(db: Session):
    return delete_all_detections(db)

def get_detections_interval(db: Session, debut_date: datetime, fin_date: datetime):
    return (
        db.query(DetectionTable)
        .filter(
            DetectionTable.detected_at >= debut_date,
            DetectionTable.detected_at <= fin_date,
        )
        .all()
    )


def get_detections_by_immatriculation_interval(
    db: Session, immatriculation: str, debut_date: datetime, fin_date: datetime
):
    return (
        db.query(DetectionTable)
        .filter(
            DetectionTable.immatriculation == immatriculation,
            DetectionTable.detected_at >= debut_date,
            DetectionTable.detected_at <= fin_date,
        )
        .all()
    )
