# -*- coding: utf-8 -*-
"""
05_evaluate.py
==============
Evalue le modele CRNN entraine sur le jeu de validation.

Metriques calculees :
  - CER  (Character Error Rate)  : % de caracteres mal predits
  - WER  (Word Error Rate)       : % de plaques entierement mal lues
  - Accuracy exacte              : % de plaques lues parfaitement
  - Matrice de confusion par caractere (optionnel)

Usage :
python evaluate.py --annotations annotations.csv --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --char-model  models/best_model.keras --show-errors --show-ok
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


# ------------------------------------------------------------------
# Import du vocabulaire depuis 02_model.py
# ------------------------------------------------------------------

def _load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_02", Path(__file__).parent / "model_02.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load_model_module()
VOCAB       = _mod.VOCAB
CHAR_TO_IDX = _mod.CHAR_TO_IDX
IDX_TO_CHAR = _mod.IDX_TO_CHAR
IMG_HEIGHT  = _mod.IMG_HEIGHT
IMG_WIDTH   = _mod.IMG_WIDTH
CTCLoss     = _mod.CTCLoss


# ------------------------------------------------------------------
# Preprocessing (identique a l'entrainement)
# ------------------------------------------------------------------

def preprocess_image(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
    return resized.astype(np.float32)[..., np.newaxis]


def detect_plate_crop(image, yolo_model, conf=0.25):
    dets = yolo_model(image, conf=conf, verbose=False)[0]
    if not dets.boxes:
        return None
    best = max(dets.boxes, key=lambda b: float(b.conf[0]))
    v    = [float(x) for x in best.xyxy[0]]
    h, w = image.shape[:2]
    crop = image[max(0,int(v[1])-4):min(h,int(v[3])+4),
                 max(0,int(v[0])-4):min(w,int(v[2])+4)]
    return crop if crop.size > 0 else None


# ------------------------------------------------------------------
# Decodage CTC
# ------------------------------------------------------------------

def decode_ctc(y_pred):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=input_len.astype(int),
        greedy=True,
    )
    results = []
    for seq in decoded[0].numpy():
        text = "".join(IDX_TO_CHAR.get(i, "") for i in seq if i >= 0)
        results.append(text)
    return results


# ------------------------------------------------------------------
# Metriques
# ------------------------------------------------------------------

def levenshtein(s1, s2):
    """Distance d'edition entre deux chaines."""
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def compute_metrics(true_texts, pred_texts):
    """
    Calcule CER, WER et accuracy exacte.

    CER = total_erreurs_chars / total_chars
    WER = plaques_pas_parfaites / total_plaques
    Acc = plaques_parfaites / total_plaques
    """
    total_chars  = 0
    total_edits  = 0
    perfect      = 0
    total        = len(true_texts)

    char_errors = defaultdict(lambda: {"ok": 0, "err": 0})

    for true, pred in zip(true_texts, pred_texts):
        edits = levenshtein(true, pred)
        total_edits += edits
        total_chars += max(len(true), 1)

        if true == pred:
            perfect += 1
            for c in true:
                char_errors[c]["ok"] += 1
        else:
            # Marquer les caracteres incorrects (approximation)
            for c in true:
                if c in pred:
                    char_errors[c]["ok"] += 1
                else:
                    char_errors[c]["err"] += 1

    cer = total_edits / total_chars
    wer = (total - perfect) / max(total, 1)
    acc = perfect / max(total, 1)

    return {
        "CER":      cer,
        "WER":      wer,
        "Accuracy": acc,
        "Perfect":  perfect,
        "Total":    total,
        "char_errors": dict(char_errors),
    }


# ------------------------------------------------------------------
# Evaluation principale
# ------------------------------------------------------------------

def evaluate(annotations_csv, plate_model_path, char_model_path,
             show_errors=True, show_ok=False, batch_size=32):

    # -- Charger le modele YOLO --
    print("[INFO] Chargement YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(plate_model_path)

    # -- Charger le modele CRNN --
    print("[INFO] Chargement CRNN...")
    model = keras.models.load_model(
        char_model_path,
        custom_objects={"CTCLoss": CTCLoss}
    )
    print(f"[INFO] Modele charge. Entree : {model.input_shape}")

    # -- Lire le CSV de validation --
    df = pd.read_csv(annotations_csv, encoding="utf-8")
    df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"[INFO] {len(df)} images dans le jeu de validation\n")

    # -- Preparer les images --
    images_proc = []
    labels_true = []
    image_paths = []
    skipped     = 0

    print("[INFO] Preparation des images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(row["image_path"])
        if img is None:
            skipped += 1
            continue

        crop = detect_plate_crop(img, yolo)
        if crop is None:
            skipped += 1
            continue

        proc = preprocess_image(crop)
        if proc is None:
            skipped += 1
            continue

        images_proc.append(proc)
        labels_true.append(row["label"])
        image_paths.append(row["image_path"])

    print(f"[INFO] {len(images_proc)} images preparees, {skipped} sautees\n")

    if not images_proc:
        print("[ERREUR] Aucune image a evaluer !")
        return

    # -- Inference par batch --
    print("[INFO] Inference en cours...")
    all_preds = []
    imgs_array = np.array(images_proc, dtype=np.float32)

    for i in tqdm(range(0, len(imgs_array), batch_size), desc="  Batch"):
        batch  = imgs_array[i:i+batch_size]
        y_pred = model.predict(batch, verbose=0)
        preds  = decode_ctc(y_pred)
        all_preds.extend(preds)

    # -- Calcul des metriques --
    metrics = compute_metrics(labels_true, all_preds)

    # -- Affichage du rapport --
    print("\n" + "="*55)
    print("        RAPPORT D'EVALUATION — CRNN Plaque Marocaine")
    print("="*55)
    print(f"  Images evaluees  : {metrics['Total']}")
    print(f"  Plaques parfaites: {metrics['Perfect']} / {metrics['Total']}")
    print()
    print(f"  Accuracy exacte  : {metrics['Accuracy']*100:.2f}%")
    print(f"    -> % de plaques lues PARFAITEMENT (100% des chars ok)")
    print()
    print(f"  CER              : {metrics['CER']*100:.2f}%")
    print(f"    -> % de caracteres incorrects en moyenne")
    print()
    print(f"  WER              : {metrics['WER']*100:.2f}%")
    print(f"    -> % de plaques avec au moins 1 erreur")
    print("="*55)

    # -- Caracteres les plus problematiques --
    char_err = metrics["char_errors"]
    problematic = []
    for c, counts in char_err.items():
        total_c = counts["ok"] + counts["err"]
        if total_c > 0:
            err_rate = counts["err"] / total_c
            problematic.append((c, err_rate, total_c, counts["err"]))

    problematic.sort(key=lambda x: -x[1])

    print("\n  Caracteres les plus difficiles :")
    print(f"  {'Char':<6} {'Erreurs':<10} {'Total':<8} {'Taux'}")
    print(f"  {'-'*35}")
    for c, rate, total_c, errors in problematic[:10]:
        if total_c >= 3:  # ignorer les chars trop rares
            bar = "█" * int(rate * 20)
            print(f"  '{c}'    {errors:<10} {total_c:<8} {rate*100:.1f}% {bar}")

    # -- Exemples d'erreurs --
    errors_list = [
        (p, t, pred)
        for p, t, pred in zip(image_paths, labels_true, all_preds)
        if t != pred
    ]
    ok_list = [
        (p, t, pred)
        for p, t, pred in zip(image_paths, labels_true, all_preds)
        if t == pred
    ]

    if show_errors and errors_list:
        print(f"\n  Exemples d'ERREURS ({len(errors_list)} au total) :")
        print(f"  {'Vrai':<15} {'Predit':<15} Fichier")
        print(f"  {'-'*60}")
        for path, true, pred in errors_list[:20]:
            fname = Path(path).name
            marker = ""
            # Identifier le type d'erreur
            if len(true) != len(pred):
                marker = "(longueur)"
            elif sum(a != b for a, b in zip(true, pred)) == 1:
                marker = "(1 char)"
            print(f"  {true:<15} {pred:<15} {fname}  {marker}")

    if show_ok and ok_list:
        print(f"\n  Exemples corrects ({len(ok_list)} au total) :")
        for path, true, pred in ok_list[:10]:
            print(f"  [OK] '{true}' <- {Path(path).name}")

    # -- Sauvegarder le rapport detaille en CSV --
    report_path = Path(char_model_path).parent / "evaluation_report.csv"
    report_df = pd.DataFrame({
        "image":   image_paths,
        "true":    labels_true,
        "pred":    all_preds,
        "correct": [t == p for t, p in zip(labels_true, all_preds)],
        "cer":     [levenshtein(t, p)/max(len(t),1)
                    for t, p in zip(labels_true, all_preds)],
    })
    report_df.to_csv(str(report_path), index=False, encoding="utf-8")
    print(f"\n[INFO] Rapport detaille sauvegarde : {report_path}")
    print(f"[NEXT] Ouvre ce CSV pour analyser chaque erreur.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluation CRNN plaques marocaines")
    p.add_argument("--annotations",  default="annotations.csv")
    p.add_argument("--plate-model",
                   default="../yolo_existing_labels/runs/detect/runs/detect/"
                           "plate_maroc_existing_labels/weights/best.pt")
    p.add_argument("--char-model",   default="models/best_model.keras")
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--show-errors",  action="store_true", default=True,
                   help="Afficher les exemples d'erreurs")
    p.add_argument("--show-ok",      action="store_true", default=False,
                   help="Afficher aussi les exemples corrects")
    args = p.parse_args()

    evaluate(
        args.annotations,
        args.plate_model,
        args.char_model,
        show_errors=args.show_errors,
        show_ok=args.show_ok,
        batch_size=args.batch_size,
    )