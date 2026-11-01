from sqlalchemy.orm import Session

from models.tables import Vehicule as VehiculeTable
from schemas.classes import Vehicule as VehiculeSchema


def list_vehicules(db: Session):
    return db.query(VehiculeTable).all()


def get_vehicule(db: Session, vehicule_id: int):
    return db.query(VehiculeTable).filter(VehiculeTable.id == vehicule_id).first()


def get_vehicule_by_immatriculation(db: Session, immatriculation: str):
    return (
        db.query(VehiculeTable)
        .filter(VehiculeTable.immatriculation == immatriculation)
        .first()
    )


def create_vehicule(db: Session, vehicule: VehiculeSchema):
    db_vehicule = VehiculeTable(**vehicule.model_dump())
    db.add(db_vehicule)
    db.commit()
    db.refresh(db_vehicule)
    return db_vehicule


def update_vehicule(db: Session, vehicule_id: int, vehicule: VehiculeSchema):
    db_vehicule = get_vehicule(db, vehicule_id)
    if db_vehicule is None:
        return None

    for field, value in vehicule.model_dump().items():
        setattr(db_vehicule, field, value)

    db.commit()
    db.refresh(db_vehicule)
    return db_vehicule


def delete_vehicule(db: Session, vehicule_id: int):
    db_vehicule = get_vehicule(db, vehicule_id)
    if db_vehicule is None:
        return None

    db.delete(db_vehicule)
    db.commit()
    return db_vehicule


def delete_all_vehicule(db: Session):
    deleted_count = db.query(VehiculeTable).delete(synchronize_session=False)
    db.commit()
    return deleted_count    