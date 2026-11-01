# -*- coding: utf-8 -*-
"""
04_inference.py
===============
ETAPE 4 : Utilisation du modele entraine pour lire les plaques.

Ce fichier contient :
  1. La classe PlateReader : interface simple pour lire une plaque
  2. L'integration avec le pipeline YOLO (training_me.py / util.py)
  3. Un mode test en ligne de commande
Usage autonome :
python 04_inference.py image.jpg --plate-model ../yolo_existing_labels/runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt --char-model  models/best_model.keras
Integration dans util.py :
  from inference_04 import PlateReader
  _reader = PlateReader("models/best_model.keras")

  def get_best_plate_text(frame, pairings, ocr_reader):
      for p in pairings:
          crop = crop_bbox(frame, p["plate"]["bbox"])
          p["ocr_text"], p["ocr_score"] = _reader.read(crop)
      return pairings
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ------------------------------------------------------------------
# Import du vocabulaire (meme que dans 02_model.py)
# ------------------------------------------------------------------

VOCAB = (
    "0123456789"
    "أابتثجحخدذرزسشصضطظعغفقكلمنهويىإآؤئء"
)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
IDX_TO_CHAR[0] = ""

IMG_HEIGHT = 32
IMG_WIDTH  = 128

# Regex de validation : 1-5 chiffres + lettre arabe + 1-2 chiffres
ARABIC_LETTERS = "أابتثجحخدذرزسشصضطظعغفقكلمنهويىإآؤئء"
_PLATE_REGEX = re.compile(
    r"(\d{1,5})\s*([" + ARABIC_LETTERS + r"])\s*(\d{1,2})"
)


# ------------------------------------------------------------------
# Preprocessing (identique a l'entrainement)
# ------------------------------------------------------------------

def preprocess_image(img_bgr):
    """
    Preparer une image pour le modele CRNN.
    Identique a la fonction dans 03_train.py pour la coherence.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
    return resized.astype(np.float32)[..., np.newaxis]  # (32, 128, 1)


# ------------------------------------------------------------------
# Decodage CTC
# ------------------------------------------------------------------

def decode_ctc(y_pred, idx_to_char=IDX_TO_CHAR):
    """
    Decode les sorties du modele en texte.

    Algorithme greedy (prend le caractere le plus probable a chaque pas) :
      1. argmax sur la dimension vocabulaire -> sequence d'indices
      2. Supprimer les repetitions consecutives
      3. Supprimer les blanks (indice 0)
    """
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded, log_probs = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=input_len.astype(int),
        greedy=True,
    )
    results = []
    confidences = []
    for i, sequence in enumerate(decoded[0].numpy()):
        text = "".join(
            idx_to_char.get(idx, "") for idx in sequence if idx >= 0
        )
        results.append(text)
        # Confiance approximative : exp(-log_prob / len)
        lp = float(-log_probs[i].numpy())
        conf = float(np.exp(-lp / max(1, len(text)))) if text else 0.0
        confidences.append(min(conf, 1.0))

    return results, confidences


# ------------------------------------------------------------------
# Classe principale : PlateReader
# ------------------------------------------------------------------

class PlateReader:
    """
    Interface simple pour lire les plaques marocaines avec le CRNN.

    Exemple d'utilisation dans util.py :

        from inference_04 import PlateReader

        # Charger une seule fois au demarrage
        _reader = PlateReader("models/best_model.keras")

        def get_best_plate_text(frame, pairings, ocr_reader):
            for p in pairings:
                crop = crop_bbox(frame, p["plate"]["bbox"])
                p["ocr_text"], p["ocr_score"] = _reader.read(crop)
            return pairings
    """

    def __init__(self, model_path: str):
        """
        Charge le modele CRNN entraine.

        Parametres :
          model_path : chemin vers best_model.keras
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele introuvable : {model_path}\n"
                f"Lance d'abord : python 03_train.py"
            )

        print(f"[PlateReader] Chargement du modele : {model_path}")
        # self.model = keras.models.load_model(
        #     str(model_path),
        #     # La CTC loss est une classe custom -> on la passe explicitement
        #     custom_objects={"CTCLoss": self._dummy_loss}
        # )
        from model_02 import CTCLoss   # import en haut du fichier
        self.model = keras.models.load_model(
        str(model_path),
        custom_objects={"CTCLoss": CTCLoss})

        print(f"[PlateReader] Modele pret. "
              f"Entree : {self.model.input_shape}, "
              f"Sortie : {self.model.output_shape}")

    @staticmethod
    def _dummy_loss(y_true, y_pred):
        """Loss factice pour le chargement (pas utilisee en inference)."""
        return tf.constant(0.0)

    def read(self, plate_crop_bgr, debug=False):
        """
        Lit le texte d'un crop de plaque.

        Parametres :
          plate_crop_bgr : image numpy BGR (sortie de YOLO)
          debug          : afficher les details

        Retourne :
          (texte, confiance) ou (None, None) si echec
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return None, None

        # Preprocessing
        img = preprocess_image(plate_crop_bgr)
        if img is None:
            return None, None

        # Inference : (1, 32, 128, 1) -> (1, 32, num_classes)
        batch   = img[np.newaxis, ...]  # ajouter dimension batch
        t0      = time.time()
        y_pred  = self.model.predict(batch, verbose=0)
        elapsed = time.time() - t0

        # Decodage CTC
        texts, confs = decode_ctc(y_pred)
        text, conf   = texts[0], confs[0]

        if debug:
            print(f"  [CRNN] brut='{text}' conf={conf:.3f} ({elapsed*1000:.0f}ms)")

        if not text:
            return None, None

        # Validation du format plaque marocaine
        m = _PLATE_REGEX.search(text)
        if m:
            validated = f"{m.group(1)} {m.group(2)} {m.group(3)}"
            if debug:
                print(f"  [CRNN] valide='{validated}'")
            return validated, conf

        # Retourner quand meme si on a au moins des chiffres
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 3:
            return text, conf * 0.7  # Confiance reduite car format invalide

        return None, None

    def read_batch(self, plate_crops, debug=False):
        """
        Lit plusieurs plaques en une seule passe (plus efficace).

        Parametres :
          plate_crops : liste d'images BGR

        Retourne :
          liste de (texte, confiance)
        """
        preprocessed = []
        valid_indices = []

        for i, crop in enumerate(plate_crops):
            img = preprocess_image(crop)
            if img is not None:
                preprocessed.append(img)
                valid_indices.append(i)

        if not preprocessed:
            return [(None, None)] * len(plate_crops)

        batch  = np.array(preprocessed, dtype=np.float32)
        y_pred = self.model.predict(batch, verbose=0)
        texts, confs = decode_ctc(y_pred)

        # Reconstruire les resultats dans l'ordre original
        results = [(None, None)] * len(plate_crops)
        for j, orig_idx in enumerate(valid_indices):
            text, conf = texts[j], confs[j]
            m = _PLATE_REGEX.search(text)
            if m:
                results[orig_idx] = (
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", conf
                )
            elif text:
                results[orig_idx] = (text, conf * 0.7)

        return results


# ------------------------------------------------------------------
# Integration dans le pipeline principal (training_me.py)
# ------------------------------------------------------------------
#
# Dans util.py, remplace load_ocr_reader() et get_best_plate_text() :
#
#   # En haut du fichier
#   from inference_04 import PlateReader
#
#   def load_ocr_reader():
#       """Charge le PlateReader CRNN a la place de EasyOCR."""
#       return PlateReader("models/best_model.keras")
#
#   def get_best_plate_text(frame, pairings, ocr_reader):
#       """ocr_reader est maintenant un PlateReader, pas EasyOCR."""
#       for pairing in pairings:
#           plate_crop = crop_bbox(frame, pairing["plate"]["bbox"])
#           text, score = ocr_reader.read(plate_crop)
#           pairing["ocr_text"]  = text
#           pairing["ocr_score"] = score
#       return pairings
#
# Dans training_me.py, rien ne change car l'interface est identique.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Visualisation des resultats
# ------------------------------------------------------------------

def annotate_image(frame, pairings):
    """
    Dessine les plaques detectees avec leur texte OCR sur l'image.
    Remplace la fonction annotate_frame dans util.py si tu veux.
    """
    annotated = frame.copy()
    for p in pairings:
        x1, y1, x2, y2 = map(int, p["plate"]["bbox"])
        text  = p.get("ocr_text") or "?"
        score = p.get("ocr_score") or 0.0

        # Rectangle rouge autour de la plaque
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Texte sous la plaque
        label = f"{text} ({score:.2f})"
        cv2.putText(annotated, label,
                    (x1, min(annotated.shape[0]-10, y2+20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated


# ------------------------------------------------------------------
# Test en ligne de commande
# ------------------------------------------------------------------

def _detect_plate(image, model_path, conf=0.25):
    """Detection YOLO pour le test CLI."""
    try:
        from ultralytics import YOLO
        m = Path(model_path)
        if not m.exists():
            return image, None
        dets = YOLO(str(m))(image, conf=conf, verbose=False)[0]
        if not dets.boxes:
            return None, None
        best = max(dets.boxes, key=lambda b: float(b.conf[0]))
        v    = [float(x) for x in best.xyxy[0]]
        h, w = image.shape[:2]
        crop = image[max(0,int(v[1])-4):min(h,int(v[3])+4),
                     max(0,int(v[0])-4):min(w,int(v[2])+4)]
        return crop, v
    except Exception as e:
        print(f"[WARN] {e}")
        return image, None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OCR CRNN plaque marocaine")
    p.add_argument("image",
                   help="Image source (plaque complete ou crop)")
    p.add_argument("--plate-model",
                   default="../yolo_existing_labels/runs/detect/runs/detect/"
                           "plate_maroc_existing_labels/weights/best.pt",
                   help="Modele YOLO de detection de plaque")
    p.add_argument("--char-model",  default="models/best_model.keras",
                   help="Modele CRNN entraine")
    p.add_argument("--no-detect",   action="store_true",
                   help="L'image est deja un crop de plaque")
    p.add_argument("--debug",       action="store_true",
                   help="Afficher les details de decodage")
    p.add_argument("--show",        action="store_true",
                   help="Afficher le resultat avec cv2.imshow")
    args = p.parse_args()

    # Charger l'image
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"[ERREUR] Impossible de lire : {args.image}")

    # Detecter la plaque
    if args.no_detect:
        plate_crop = img
        bbox       = None
    else:
        plate_crop, bbox = _detect_plate(img, args.plate_model)
        if plate_crop is None:
            raise SystemExit("[ERREUR] Aucune plaque detectee")
        print(f"[INFO] Plaque detectee")

    # Charger le reader et lire la plaque
    reader = PlateReader(args.char_model)
    text, score = reader.read(plate_crop, debug=args.debug)

    print(f"\nResultat  : {text!r}")
    print(f"Confiance : {0.0 if score is None else score:.3f}")

    # Affichage visuel
    if args.show:
        display = img.copy()
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{text or '?'} ({score or 0:.2f})"
            cv2.putText(display, label, (x1, max(30, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Resultat OCR CRNN", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
