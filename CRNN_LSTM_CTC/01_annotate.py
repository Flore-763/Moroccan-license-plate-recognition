# -*- coding: utf-8 -*-
"""
01_annotate.py
==============
ETAPE 1 : Creer le fichier d'annotations (annotations.csv)

Ce script parcourt tes images de plaques et te demande de taper
le texte present sur chaque plaque. Il detecte d'abord la plaque
avec YOLO, l'affiche, et tu tapes le texte.

Format du CSV produit :
  image_path,label
  ../maroc_car/dataset/IMG_0001.jpg,90120ا72
  ../maroc_car/dataset/IMG_0002.jpg,56651ا14
  ...

NOTE IMPORTANTE :
  Le label ne contient QUE les caracteres utiles, PAS les barres |
  Exemple : pour "90120 | ا | 72" tu tapes "90120ا72"
  Le modele CRNN apprend a les lire sans qu'on lui dise ou ils sont.

Usage :
  python 01_annotate.py --plates-dir ../maroc_car/dataset
                         --plate-model ../yolo.../best.pt
                         --output annotations.csv

Tu peux relancer le script : il reprend la ou il s'est arrete
(les images deja annotees sont sautees).
"""
import argparse
import csv
import cv2
from pathlib import Path


# ------------------------------------------------------------------
# Detection de la plaque avec YOLO (optionnel)
# ------------------------------------------------------------------

def detect_plate_crop(image, model_path, conf=0.25):
    """
    Detecte la plaque dans l'image et retourne le crop.
    Si le modele n'existe pas, retourne l'image entiere.
    """
    try:
        from ultralytics import YOLO
        m = Path(model_path)
        if not m.exists():
            print(f"  [WARN] Modele YOLO introuvable, image entiere utilisee")
            return image

        dets = YOLO(str(m))(image, conf=conf, verbose=False)[0]
        if not dets.boxes:
            return None  # Pas de plaque trouvee

        # Garder la detection avec la meilleure confiance
        best = max(dets.boxes, key=lambda b: float(b.conf[0]))
        v = [float(x) for x in best.xyxy[0]]
        h, w = image.shape[:2]
        x1, y1 = max(0, int(v[0])-4), max(0, int(v[1])-4)
        x2, y2 = min(w, int(v[2])+4), min(h, int(v[3])+4)
        return image[y1:y2, x1:x2]

    except Exception as e:
        print(f"  [WARN] Erreur YOLO : {e}")
        return image


# ------------------------------------------------------------------
# Interface d'annotation
# ------------------------------------------------------------------

def annotate(plates_dir, plate_model, output_csv, val_ratio=0.20):
    """
    Boucle principale d'annotation.
    """
    import random

    plates_dir = Path(plates_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([p for p in plates_dir.iterdir()
                         if p.suffix.lower() in extensions])

    # Charger les annotations deja existantes pour ne pas recommencer
    already_done = set()
    existing_rows = []
    if Path(output_csv).exists():
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(row["image_path"])
                existing_rows.append(row)
        print(f"[INFO] {len(already_done)} images deja annotees, on continue...")

    # Ouvrir le CSV en mode append
    csv_file = open(output_csv, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])

    # Ecrire l'en-tete si le fichier est nouveau
    if not already_done:
        writer.writeheader()

    total    = len(all_images)
    done     = len(already_done)
    skipped  = 0
    window   = "Plaque - appuie sur une touche apres avoir tape le label"

    print(f"[INFO] {total} images au total, {done} deja annotees")
    print("[INFO] Pour chaque plaque :")
    print("         - Tape le texte (ex: 90120ا72) et appuie sur ENTREE")
    print("         - Tape 's' pour sauter cette image")
    print("         - Tape 'q' pour quitter\n")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for i, img_path in enumerate(all_images):
        img_path_str = str(img_path)

        # Sauter les images deja annotees
        if img_path_str in already_done:
            continue

        img = cv2.imread(img_path_str)
        if img is None:
            continue

        # Detecter la plaque
        plate_crop = detect_plate_crop(img, plate_model)
        if plate_crop is None:
            print(f"  [{i+1}/{total}] {img_path.name} -> Pas de plaque, image sautee")
            skipped += 1
            continue

        # Afficher la plaque agrandie
        display_h = 120
        scale     = display_h / plate_crop.shape[0]
        display   = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        cv2.imshow(window, display)
        cv2.waitKey(1)  # Forcer l'affichage

        # Demander le label dans le terminal
        print(f"[{done+1}/{total}] {img_path.name}")
        label = input("  Label (ou 's'=sauter, 'q'=quitter) : ").strip()

        if label.lower() == "q":
            print("\n[INFO] Annotation arretee.")
            break

        if label.lower() == "s" or not label:
            print("  -> Saute")
            skipped += 1
            continue

        # Determiner le split train/val
        split = "val" if (done % 5 == 0) else "train"  # 20% val

        writer.writerow({
            "image_path": img_path_str,
            "label":      label,
            "split":      split,
        })
        csv_file.flush()  # Sauvegarder immediatement
        done += 1
        print(f"  -> '{label}' ({split}) [total annote: {done}]")

    cv2.destroyAllWindows()
    csv_file.close()

    print(f"\n[INFO] Annotation terminee !")
    print(f"       Annote  : {done} images")
    print(f"       Saute   : {skipped} images")
    print(f"       CSV     : {output_csv}")
    print(f"\n[NEXT] Lance l'entrainement :")
    print(f"  python 02_train.py --annotations {output_csv}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Outil d'annotation des plaques marocaines"
    )
    p.add_argument("--plates-dir",  required=True,
                   help="Dossier contenant les images de plaques")
    p.add_argument("--plate-model",
                   default="../yolo_existing_labels/runs/detect/runs/detect/"
                           "plate_maroc_existing_labels/weights/best.pt",
                   help="Modele YOLO de detection de plaque")
    p.add_argument("--output",      default="annotations.csv",
                   help="Fichier CSV de sortie")
    args = p.parse_args()

    annotate(args.plates_dir, args.plate_model, args.output)
