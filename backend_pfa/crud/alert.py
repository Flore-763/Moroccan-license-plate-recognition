from datetime import datetime

from sqlalchemy.orm import Session

from models.tables import Alert as AlertTable
from schemas.classes import Alert as AlertSchema


def list_alerts(db: Session):
    return db.query(AlertTable).all()


def get_alert(db: Session, alert_id: int):
    return db.query(AlertTable).filter(AlertTable.id == alert_id).first()


def create_alert(db: Session, alert: AlertSchema):
    db_alert = AlertTable(**alert.model_dump())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert


def create_alert_from_detection(db: Session, immatriculation: str, infraction: str):
    db_alert = AlertTable(
        immatriculation=immatriculation,
        infraction=infraction,
        temps=datetime.utcnow(),
    )
    db.add(db_alert)
    return db_alert


def update_alert(db: Session, alert_id: int, alert: AlertSchema):
    db_alert = get_alert(db, alert_id)
    if db_alert is None:
        return None

    for field, value in alert.model_dump().items():
        setattr(db_alert, field, value)

    db.commit()
    db.refresh(db_alert)
    return db_alert


def delete_alert(db: Session, alert_id: int):
    db_alert = get_alert(db, alert_id)
    if db_alert is None:
        return None

    db.delete(db_alert)
    db.commit()
    return db_alert


def delete_all_alerts(db: Session):
    deleted_count = db.query(AlertTable).delete(synchronize_session=False)
    db.commit()
    return deleted_count  


def delete_all_alert(db: Session):
    return delete_all_alerts(db)

def get_alerts_interval(db: Session, debut_date: datetime, fin_date: datetime):
    return (
        db.query(AlertTable)
        .filter(AlertTable.temps >= debut_date, AlertTable.temps <= fin_date)
        .all()
    )


def get_alerts_by_immatriculation_interval(
    db: Session, immatriculation: str, debut_date: datetime, fin_date: datetime
):
    return (
        db.query(AlertTable)
        .filter(
            AlertTable.immatriculation == immatriculation,
            AlertTable.temps >= debut_date,
            AlertTable.temps <= fin_date,
        )
        .all()
    )
