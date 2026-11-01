"""Exporte la base de véhicules de démonstration en JSON statique (mêmes statuts que seed.py)."""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend_pfa"))
import importlib.util
spec = importlib.util.spec_from_file_location("seed_light", ROOT / "backend_pfa" / "seed.py")
# seed.py importe SQLAlchemy ; on réutilise uniquement ses fonctions pures
src = (ROOT / "backend_pfa" / "seed.py").read_text(encoding="utf-8")
ns = {}
exec(compile(src.split("from sqlalchemy.orm import Session")[0], "seed_head", "exec"), ns)  # imports de tête
exec(compile("import csv, random\nSEED_CSV = %r\n" % str(ROOT / "backend_pfa" / "annotations_original.csv")
             + src[src.index("STATUTS"):src.index("def seed_vehicules")], "seed_fn", "exec"), ns)

plates = ns["read_unique_plates"]()
rng = random.Random(42)
rows = [{"id": i + 1, "immatriculation": p, "statut": rng.choices(ns["STATUTS"], ns["POIDS"])[0]}
        for i, p in enumerate(plates)]
out = ROOT / "frontend" / "public" / "data" / "vehicules.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
print(f"{len(rows)} véhicules → {out}")
