"""
seed.py — remplit la table `vehicule` à partir des annotations.

Remplace plate_db.py (psycopg2 + localhost en dur + statuts aléatoires
différents à chaque exécution). Ici : SQLAlchemy (SQLite ou PostgreSQL) et
graine fixe → la démo est reproductible.

Usage manuel :  python seed.py
"""
import csv
import random

from sqlalchemy.orm import Session

from config import SEED_CSV
from db.database import Base, SessionLocal, engine
from models import tables

STATUTS = ["safe", "vole", "suspicieux"]
POIDS = [0.80, 0.10, 0.10]
_NORMALIZE = {"\u0647": "\u06BE", "\u0629": "\u06BE"}  # même table que model_02.py


def normalize_plate(value: str) -> str:
    plate = str(value).strip()
    try:  # corrige un éventuel CSV mal décodé (latin1 → utf-8)
        plate = plate.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    plate = "".join(_NORMALIZE.get(c, c) for c in plate)
    return plate.replace(" ", "")


def read_unique_plates(csv_path=SEED_CSV) -> list[str]:
    plates = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            if label and (p := normalize_plate(label)):
                plates.add(p)
    return sorted(plates)


def seed_vehicules(db: Session, seed: int = 42) -> int:
    """Insère les plaques absentes. Retourne le nombre d'insertions."""
    if db.query(tables.Vehicule).count() > 0:
        return 0
    rng = random.Random(seed)
    rows = [
        tables.Vehicule(immatriculation=p, statut=rng.choices(STATUTS, POIDS)[0])
        for p in read_unique_plates()
    ]
    db.add_all(rows)
    db.commit()
    return len(rows)


def init_db(seed: bool = True) -> int:
    Base.metadata.create_all(bind=engine)
    if not seed:
        return 0
    with SessionLocal() as db:
        return seed_vehicules(db)


if __name__ == "__main__":
    print(f"{init_db()} véhicules insérés.")
