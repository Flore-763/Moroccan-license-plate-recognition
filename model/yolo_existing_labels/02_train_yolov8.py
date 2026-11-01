"""
Entrainement YOLOv8 avec labels existants.
"""
""" on importe argparse pour pouvoir lancer le script avec des options comme :
model epochs imgsz batch  autres que ceux déjà affectés a nos variables """


import argparse
from pathlib import Path
from ultralytics import YOLO

# ma configuration par defaut  peut etre modifier au lancement de l'entrainement
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "data": "dataset.yaml",
    "epochs": 50,
    "imgsz": 640, # taille par defaut mais peut etre modifie la seule regle est que ca doit etre un multiple de 32 et pas obligatoirement carre
    "batch": 16,
    "patience": 20, # l'entrainement s'arrete s'il n y a pas de changements au bout de 20 epochs
    "device": "",  #cpu  par defaut
    "project": "runs/detect", # le dernier et le meilleur modèle seront stockés dans ce dossier
    "name": "plate_maroc_existing_labels",
    "save": True,
}


def check_dataset():
    required = [
        Path("data/images/train"),
        Path("data/images/val"),
        Path("data/labels/train"),
        Path("data/labels/val"),
        Path("dataset.yaml"), # où on definit le nombre de classes, les noms de classe ainsi que les dossiers
    ]

    ok = True
    for path in required:
        if not path.exists():
            print(f"[ERREUR] Manquant: {path}")
            ok = False

    if not ok:
        print("\n Structure de  fichiers minimal inexistante , preparer d'abord les donnees")
        return False

    n_imgs = len(list(Path("data/images/train").glob("*")))
    n_labels = len(list(Path("data/labels/train").glob("*.txt")))
    print(f"[INFO] Train: {n_imgs} images | {n_labels} labels")

    return True


def train(config: dict, resume: bool = False):
    if not check_dataset():
        return

    if resume:
        last_ckpt = Path("runs/detect/plate_maroc_existing_labels/weights/last.pt") # dernier modele sauvegargé
        if not last_ckpt.exists():
            print(f"[ERREUR] Checkpoint introuvable: {last_ckpt}") 
            return
        model = YOLO(str(last_ckpt))
    else:
        model_name = config.pop("model")
        model = YOLO(model_name)

    results = model.train(resume=resume, **config)

    print("\n[OK] Entrainement termine")
    print("Best weights dans: runs/detect/plate_maroc_existing_labels/weights/best.pt")
    print(f"mAP50 -> {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"mAP50-95 -> {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement YOLOv8 - avec les labels existants")
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["model"] = f"{args.model}.pt"
    cfg["epochs"] = args.epochs
    cfg["imgsz"] = args.imgsz
    cfg["batch"] = args.batch

    train(cfg, resume=args.resume)
