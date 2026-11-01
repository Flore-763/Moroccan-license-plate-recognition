# -*- coding: utf-8 -*-
"""
03_train.py
===========
ETAPE 3 : Entrainement du modele CRNN sur les plaques marocaines.

Ce script :
  1. Lit le CSV d'annotations produit par 01_annotate.py
  2. Detecte les plaques avec YOLO et prepare les crops
  3. Definit le pipeline de donnees avec augmentation
  4. Entraine le modele CRNN avec CTC loss
  5. Sauvegarde le meilleur modele dans models/

Usage :
python 03_train.py --annotations annotations.csv --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --output-dir models --epochs 100
Necessite :
  pip install tensorflow opencv-python pandas tqdm
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tqdm import tqdm

# Importer notre modele et vocabulaire
from model_02 import (
    build_crnn_model, CTCLoss, decode_predictions,
    VOCAB, CHAR_TO_IDX, IDX_TO_CHAR,
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
)

# Renommer pour correspondre au nom de fichier reel
# (Python importe sans le prefixe numerique)
import importlib, sys

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BATCH_SIZE    = 16    # Reduit car plaques en petit nombre
EPOCHS        = 100
LEARNING_RATE = 1e-3
MAX_LABEL_LEN = 10    # Max caracteres dans une plaque (ex: "90120ا72" = 8)


# ------------------------------------------------------------------
# ETAPE 1 : Preparation des images
# ------------------------------------------------------------------

def detect_plate_crop(image, model, conf=0.25):
    """Detecte et retourne le crop de la plaque."""
    dets = model(image, conf=conf, verbose=False)[0]
    if not dets.boxes:
        return None
    best = max(dets.boxes, key=lambda b: float(b.conf[0]))
    v = [float(x) for x in best.xyxy[0]]
    h, w = image.shape[:2]
    x1, y1 = max(0, int(v[0])-4), max(0, int(v[1])-4)
    x2, y2 = min(w, int(v[2])+4), min(h, int(v[3])+4)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def preprocess_image(img_bgr):
    """
    Transforme une image couleur en image prete pour le modele.
    
    Etapes :
      1. Conversion en niveaux de gris (la couleur n'apporte rien pour une plaque)
      2. Redimensionnement en 32x128 (hauteur x largeur fixe)
      3. Normalisation 0-255 -> 0-1
      4. Ajout de la dimension canal : (32, 128) -> (32, 128, 1)
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32)          # garde les valeurs 0-255
    return normalized[..., np.newaxis]               # (32, 128, 1)


_NORMALIZE_CHARS = {
    "\u0647": "\u06BE",  # ه -> ھ
}

def normalize_label(text):
    return "".join(_NORMALIZE_CHARS.get(c, c) for c in text)

def label_to_indices(text, max_len=MAX_LABEL_LEN):
    """
    Convertit un label texte en sequence d'indices.
    
    Exemple : "90120ا72" -> [10, 1, 2, 3, 1, 12, 8, 3]
              (indices dans le vocabulaire, 1-base car 0=blank CTC)
    
    Les positions non utilisees sont remplies avec -1 (padding).
    Le padding sera ignore par la CTC loss.
    """
    text = normalize_label(text)  # <- ajoute cette ligne
    indices = []
    for c in text:
        if c not in CHAR_TO_IDX:
            print(f"[WARN] caractere inconnu : '{c}' (U+{ord(c):04X})")
            continue
        indices.append(CHAR_TO_IDX[c])
    indices = indices[:max_len]
    indices += [-1] * (max_len - len(indices))
    return np.array(indices, dtype=np.int32)

# ------------------------------------------------------------------
# ETAPE 2 : Augmentation des donnees
# ------------------------------------------------------------------

def augment_image(img):
    """
    Augmentation legere adaptee aux plaques.
    
    But : augmenter artificiellement la diversite du dataset
    pour eviter le surapprentissage (overfitting).
    
    Transformations appliquees aleatoirement :
      - Bruit gaussien (simule les images floues ou de mauvaise qualite)
      - Variation de luminosite (eclairage different)
      - Legere rotation (plaque pas parfaitement a l'horizontale)
      - Flou gaussien (mouvement, mise au point imparfaite)
    
    IMPORTANT : les transformations sont DOUCES car les plaques
    ont un format tres structure et on ne veut pas deformer les chiffres.
    """
    img = img.copy()

    # Bruit gaussien
    if random.random() < 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img   = np.clip(img + noise, 0, 255)

    # Variation de luminosite
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img    = np.clip(img * factor, 0, 255)

    # Legere rotation (max 3 degres)
    if random.random() < 0.3:
        angle  = random.uniform(-3, 3)
        M      = cv2.getRotationMatrix2D(
            (IMG_WIDTH // 2, IMG_HEIGHT // 2), angle, 1.0
        )
        img    = cv2.warpAffine(
            img.squeeze(), M, (IMG_WIDTH, IMG_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )[..., np.newaxis]

    # Flou gaussien
    if random.random() < 0.3:
        img_2d = cv2.GaussianBlur(img.squeeze(), (3, 3), 0)
        img    = img_2d[..., np.newaxis]

    return img.astype(np.float32)


# ------------------------------------------------------------------
# ETAPE 3 : Pipeline de donnees TensorFlow
# ------------------------------------------------------------------

class PlateDataset:
    """
    Charge et prepare les donnees d'entrainement.
    
    Chaque element du dataset est un tuple :
      - image  : (32, 128, 1) float32, pixels 0-255
      - label  : (MAX_LABEL_LEN,) int32, indices des caracteres (-1 = padding)
    """

    def __init__(self, csv_path, plate_model_path, split="train", augment=True):
        self.augment = augment
        self.samples = []  # liste de (image_preprocessee, label_indices)

        print(f"\n[DATA] Chargement des donnees ({split})...")

        # Charger le modele YOLO
        yolo_model = None
        try:
            from ultralytics import YOLO
            m = Path(plate_model_path)
            if m.exists():
                yolo_model = YOLO(str(m))
                print(f"[DATA] Modele YOLO charge : {plate_model_path}")
        except Exception as e:
            print(f"[DATA] YOLO non disponible ({e}), image entiere utilisee")

        # Lire le CSV
        df = pd.read_csv(csv_path, encoding="utf-8")
        df = df[df["split"] == split].reset_index(drop=True)
        print(f"[DATA] {len(df)} images dans le split '{split}'")

        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Prep {split}"):
            img = cv2.imread(row["image_path"])
            if img is None:
                skipped += 1
                continue

            # Detecter la plaque
            if yolo_model is not None:
                crop = detect_plate_crop(img, yolo_model)
            else:
                crop = img

            if crop is None:
                skipped += 1
                continue

            # Preprocessing
            preprocessed = preprocess_image(crop)
            label_idx    = label_to_indices(row["label"])

            # Verifier que le label contient au moins un caractere connu
            if not any(idx > 0 for idx in label_idx):
                skipped += 1
                continue

            self.samples.append((preprocessed, label_idx))

        print(f"[DATA] {len(self.samples)} images chargees, {skipped} sautees")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.augment:
            img = augment_image(img)
        return img, label

    def as_tf_dataset(self, batch_size=BATCH_SIZE, shuffle=True):
        """
        Convertit en tf.data.Dataset optimise pour l'entrainement.
        tf.data gere le chargement parallele et le prefetch automatiquement.
        """
        imgs   = np.array([s[0] for s in self.samples], dtype=np.float32)
        labels = np.array([s[1] for s in self.samples], dtype=np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.samples), seed=42)

        if self.augment:
            # Appliquer l'augmentation via tf.py_function
            def aug_fn(img, label):
                img_aug = tf.py_function(
                    lambda x: augment_image(x.numpy()),
                    [img], tf.float32
                )
                img_aug.set_shape(img.shape)
                return img_aug, label
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# ------------------------------------------------------------------
# ETAPE 4 : Metrique CER (Character Error Rate)
# ------------------------------------------------------------------

def compute_cer(y_true_texts, y_pred_texts):
    """
    Calcule le Character Error Rate moyen.
    
    CER = (insertions + suppressions + substitutions) / nb_chars_total
    
    Exemple :
      vrai    : "90120ا72"
      predit  : "90120ا72"  -> CER = 0%  (parfait)
      predit  : "90120ا7"   -> CER = 1/8 = 12.5%  (un char manquant)
      predit  : "9012ا72"   -> CER = 1/8 = 12.5%  (un char manquant)
    """
    total_chars, total_errors = 0, 0
    for true, pred in zip(y_true_texts, y_pred_texts):
        # Distance d'edition (Levenshtein)
        n, m = len(true), len(pred)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if true[i-1] == pred[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        total_errors += dp[n][m]
        total_chars  += max(n, 1)

    return total_errors / total_chars if total_chars > 0 else 1.0


class CERCallback(keras.callbacks.Callback):
    """
    Callback qui calcule et affiche le CER apres chaque epoch.
    Sauvegarde le meilleur modele selon le val_CER.
    """
    def __init__(self, val_dataset_raw, output_dir):
        super().__init__()
        self.val_samples  = val_dataset_raw.samples
        self.val_texts    = []
        self.output_dir   = Path(output_dir)
        self.best_cer     = float("inf")

        # Recuperer les vrais labels texte
        # (on reconvertit les indices en texte)
        for _, label_idx in self.val_samples:
            text = "".join(
                IDX_TO_CHAR.get(idx, "") for idx in label_idx if idx > 0
            )
            self.val_texts.append(text)

    def on_epoch_end(self, epoch, logs=None):
        # Predictions sur le jeu de validation
        imgs = np.array([s[0] for s in self.val_samples], dtype=np.float32)
        preds = self.model.predict(imgs, verbose=0)

        # Decodage CTC
        pred_texts = decode_predictions(preds)

        # Calcul du CER
        cer = compute_cer(self.val_texts, pred_texts)

        print(f"  val_CER: {cer:.4f}", end="")

        # Sauvegarder si c'est le meilleur
        if cer < self.best_cer:
            self.best_cer = cer
            save_path = self.output_dir / "best_model.keras"
            self.model.save(str(save_path))
            print(f"  <- meilleur! Sauvegarde dans {save_path}", end="")

        # Afficher quelques exemples
        if epoch % 10 == 0:
            print()
            for true, pred in zip(self.val_texts[:3], pred_texts[:3]):
                status = "OK" if true == pred else "ERR"
                print(f"    [{status}] vrai='{true}' | predit='{pred}'")
        else:
            print()


# ------------------------------------------------------------------
# ETAPE 5 : Entrainement principal
# ------------------------------------------------------------------

def train(annotations_csv, plate_model_path, output_dir,
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Charger les donnees --
    train_ds_raw = PlateDataset(annotations_csv, plate_model_path,
                                split="train", augment=True)
    val_ds_raw   = PlateDataset(annotations_csv, plate_model_path,
                                split="val",   augment=False)

    if len(train_ds_raw) == 0:
        raise ValueError("Aucune image d'entrainement ! Verifie ton CSV.")
    if len(val_ds_raw) == 0:
        raise ValueError("Aucune image de validation ! Verifie ton CSV.")

    train_ds = train_ds_raw.as_tf_dataset(batch_size=batch_size, shuffle=True)
    val_ds   = val_ds_raw.as_tf_dataset(batch_size=batch_size, shuffle=False)

    print(f"\n[INFO] Dataset pret :")
    print(f"       Train : {len(train_ds_raw)} images")
    print(f"       Val   : {len(val_ds_raw)} images")
    print(f"       Vocab : {len(VOCAB)} classes")

    # -- Construire le modele --
    model = build_crnn_model(num_classes=NUM_CLASSES)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=CTCLoss(),
    )
    model.summary()

    # -- Callbacks --
    callbacks_list = [
        # Reduction du taux d'apprentissage si la loss stagne
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-6, verbose=1
        ),
        # Arret anticipé si aucune amelioration pendant 20 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1,
            restore_best_weights=True
        ),
        # Logs TensorBoard (optionnel)
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"), update_freq="epoch"
        ),
        # Notre metrique CER personnalisee
        CERCallback(val_ds_raw, output_dir),
    ]

    print(f"\n[INFO] Debut de l'entrainement ({epochs} epochs max)...")
    print(f"       Device : {tf.test.gpu_device_name() or 'CPU'}")
    print(f"       Pour visualiser : tensorboard --logdir {output_dir}/logs\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Sauvegarder le modele final
    final_path = output_dir / "final_model.keras"
    model.save(str(final_path))
    print(f"\n[INFO] Modele final sauvegarde : {final_path}")
    print(f"[INFO] Meilleur modele         : {output_dir}/best_model.keras")
    print(f"\n[NEXT] Lance l'inference :")
    print(f"  python 04_inference.py image.jpg --model {output_dir}/best_model.keras")

    return history


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    # # Hack pour permettre l'import du fichier modele avec prefixe numerique
    # import importlib.util, sys
    # spec = importlib.util.spec_from_file_location(
    #     "model_02", Path(__file__).parent / "02_model.py"
    # )
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # sys.modules["model_02"] = mod

    # from model_02 import (
    #     build_crnn_model, CTCLoss, decode_predictions,
    #     VOCAB, CHAR_TO_IDX, IDX_TO_CHAR,
    #     IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
    # )

    p = argparse.ArgumentParser(description="Entrainement CRNN plaques marocaines")
    p.add_argument("--annotations",  default="annotations.csv",
                   help="CSV produit par 01_annotate.py")
    p.add_argument("--plate-model",
                   default="../yolo_existing_labels/runs/detect/runs/detect/"
                           "plate_maroc_existing_labels/weights/best.pt",
                   help="Modele YOLO de detection")
    p.add_argument("--output-dir",   default="models")
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LEARNING_RATE)
    args = p.parse_args()

    train(
        args.annotations, args.plate_model, args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
