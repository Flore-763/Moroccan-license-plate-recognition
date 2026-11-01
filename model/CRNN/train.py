"""
train.py — CRNN Plaque Marocaine (version finale)
===================================================
Compatible avec le format réel annotations.csv :
    image_path,label,split
    IMG-20200725-WA0000.jpg,75876أ40,train
    IMG-20200725-WA0000_aug01.jpg,75876أ40,train
    ...

Toutes les améliorations intégrées :
  ✅ Architecture CRNN : CNN 5 blocs + BatchNorm + 2x BiLSTM + Dropout
  ✅ Data augmentation en ligne (légère, pendant le training)
  ✅ CTC loss stable via couche Lambda
  ✅ Beam Search CTC à l'évaluation (callback par époque)
  ✅ Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
  ✅ Reprise d'entraînement (--resume)

Usage :
    python train.py --images data/images_augmented --annotations annotations.csv

    python train.py --images data/images_augmented \
                    --annotations annotations.csv \
                    --resume models/crnn_best.keras
"""

import os, csv, re, random, argparse
import numpy as np
from pathlib import Path, PureWindowsPath

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import Sequence
import cv2

from ctc_beam_search import batch_beam_search, batch_greedy_decode


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Alphabet : TOUS les caractères présents dans tes labels (sans doublons)
# Mis à jour à partir de l'analyse réelle de annotations.csv
CHARACTERS = "0123456789أبتثجحخدذرزسشصضطظعغفقكلمنهويءةوھٮں"
CHARACTERS = "".join(dict.fromkeys(CHARACTERS))   # déduplique sans changer l'ordre

IMG_H      = 32
IMG_W      = 128
MAX_LABEL  = 12    # longueur max d'une plaque marocaine (caractères)
BATCH_SIZE = 32
EPOCHS     = 100

CHAR2IDX   = {c: i for i, c in enumerate(CHARACTERS)}
IDX2CHAR   = {i: c for i, c in enumerate(CHARACTERS)}
BLANK_IDX  = len(CHARACTERS)
N_CLASSES  = len(CHARACTERS) + 1


def encode(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURE DU CSV
# ══════════════════════════════════════════════════════════════════════════════

AUG_RE = re.compile(r'_aug\d{2,}$', re.IGNORECASE)


def resolve_path(image_path: str, images_dir: Path) -> Path | None:
    """
    Cherche l'image dans images_dir en priorité,
    puis essaie le chemin direct (absolu ou relatif).
    Supporte les chemins Windows dans le CSV.
    """
    fname = PureWindowsPath(image_path).name   # nom de fichier seul
    p     = images_dir / fname
    if p.exists():
        return p
    p = Path(image_path)
    if p.exists():
        return p
    return None


def _assign_splits_auto(rows: list, val_ratio: float = 0.10, seed: int = 42) -> dict:
    """
    Attribue train/val au niveau des stems ORIGINAUX.
    Retourne { orig_stem → 'train' | 'val' }.
    """
    aug_re_local = re.compile(r'_aug\d{2,}$', re.IGNORECASE)

    def _orig_stem(image_path):
        name = PureWindowsPath(image_path).name
        stem = Path(name).stem
        return aug_re_local.sub('', stem)

    # Collecter les stems originaux (ceux sans _augNN)
    orig_stems = []
    seen = set()
    for r in rows:
        ip = r.get('image_path') or r.get('filename') or ''
        stem = Path(PureWindowsPath(ip).name).stem
        if not aug_re_local.search(stem):
            os_ = _orig_stem(ip)
            if os_ not in seen:
                seen.add(os_)
                orig_stems.append(os_)

    random.seed(seed)
    shuffled = orig_stems[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val_set = set(shuffled[:n_val])

    print(f"[INFO] Auto-split : {len(shuffled) - n_val} originaux train "
          f"/ {n_val} originaux val")

    return {s: ('val' if s in val_set else 'train') for s in orig_stems}, _orig_stem


def load_dataset(csv_path: str, images_dir: Path,
                 val_ratio: float = 0.10, seed: int = 42):
    """
    Retourne (train_items, val_items).
    Chaque item = (img_path: Path, label: str).

    Si le CSV n'a pas de splits remplis (colonne vide ou absente),
    le split est assigné automatiquement ici (90/10).
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if not rows:
        raise ValueError(f"CSV vide : {csv_path}")

    # Détecter si les splits sont remplis
    splits_found = [r.get('split', '').strip() for r in rows]
    n_valid_splits = sum(1 for s in splits_found if s in ('train', 'val'))

    if n_valid_splits < len(rows) * 0.5:
        # Moins de 50% des lignes ont un split valide → on les assigne ici
        print(f"[WARN] Seulement {n_valid_splits}/{len(rows)} lignes avec split valide.")
        print(f"[INFO] Attribution automatique des splits (val_ratio={val_ratio})...")
        stem_to_split, _orig_fn = _assign_splits_auto(rows, val_ratio, seed)

        for r in rows:
            ip = r.get('image_path') or r.get('filename') or ''
            r['split'] = stem_to_split.get(_orig_fn(ip), 'train')

        # Réécrire le CSV avec les splits pour les prochaines fois
        import shutil
        backup = Path(csv_path).with_name("annotations_backup.csv")
        shutil.copy2(csv_path, backup)
        fieldnames = list(rows[0].keys())
        if 'split' not in fieldnames:
            fieldnames.append('split')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
        print(f"[INFO] CSV mis à jour avec les splits → {csv_path}")
        print(f"[INFO] Backup → {backup}")

    train, val = [], []
    skipped = 0

    for r in rows:
        image_path = (r.get('image_path') or r.get('filename') or '').strip()
        label      = (r.get('label')      or r.get('text',  '') ).strip()
        split      = (r.get('split', 'train') or 'train').strip()

        if not image_path or not label:
            continue

        enc = encode(label)
        if len(enc) == 0 or len(enc) > MAX_LABEL:
            skipped += 1
            continue

        p = resolve_path(image_path, images_dir)
        if p is None:
            skipped += 1
            continue

        item = (p, label)
        if split == 'val':
            val.append(item)
        else:
            train.append(item)

    if skipped > 0:
        print(f"[WARN] {skipped} images ignorées (introuvables ou label trop long)")

    return train, val


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION EN LIGNE
# ══════════════════════════════════════════════════════════════════════════════

def online_augment(img: np.ndarray) -> np.ndarray:
    """Augmentation rapide pendant le training (sans trop ralentir)."""
    if random.random() < 0.5:
        img = cv2.convertScaleAbs(img,
            alpha=random.uniform(0.7, 1.3),
            beta=random.uniform(-30, 30))
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 15), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-6, 6), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class PlateGenerator(Sequence):
    """
    Keras Sequence generator.
    Sortie : (inputs_dict, outputs_dict) pour Model.fit() avec CTC loss.
    """

    def __init__(self, items, batch_size=BATCH_SIZE,
                 augment=False, shuffle=True):
        super().__init__()
        self.items      = list(items)
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.items) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.items)

    def __getitem__(self, idx):
        batch = self.items[idx * self.batch_size:(idx + 1) * self.batch_size]
        B     = len(batch)

        images        = np.zeros((B, IMG_H, IMG_W, 1), np.float32)
        labels        = np.full((B, MAX_LABEL), -1, np.int32)
        label_lengths = np.zeros((B, 1), np.int32)
        # T = nombre de timesteps en sortie du CNN → calculé dynamiquement
        # Pour (32, 128) avec la présente archi → T = 32
        input_lengths = np.full((B, 1), 32, np.int32)

        for i, (img_path, label) in enumerate(batch):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((IMG_H, IMG_W), np.uint8)
            img = cv2.resize(img, (IMG_W, IMG_H))

            if self.augment:
                img = online_augment(img)

            images[i, ..., 0] = img.astype(np.float32) / 255.0

            enc = encode(label)[:MAX_LABEL]
            labels[i, :len(enc)] = enc
            label_lengths[i, 0]  = len(enc)

        dummy = np.zeros((B, 1), dtype=np.float32)

        return (
            {
                'image': images,
                'label': labels,
                'input_length': input_lengths,
                'label_length': label_lengths
            },
            dummy
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE CRNN AMÉLIORÉE
# ══════════════════════════════════════════════════════════════════════════════

def build_crnn():
    """
    CNN  : 5 blocs Conv2D + BatchNorm + MaxPool
    RNN  : 2 couches BiLSTM (256) + Dropout 0.25
    Out  : Dense softmax (N_CLASSES)

    Retourne (training_model, inference_model).
    training_model  : prend [image, label, input_length, label_length] → ctc_loss
    inference_model : prend [image] → y_pred (batch, T, N_CLASSES)
    """

    image        = layers.Input(shape=(IMG_H, IMG_W, 1),  name='image')
    label        = layers.Input(shape=(None,),             name='label',        dtype='int32')
    input_length = layers.Input(shape=(1,),                name='input_length', dtype='int32')
    label_length = layers.Input(shape=(1,),                name='label_length', dtype='int32')

    def conv_block(x, filters, pool_h=2, pool_w=2):
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if pool_h > 0 or pool_w > 0:
            x = layers.MaxPooling2D((pool_h, pool_w))(x)
        return x

    # ── CNN ──────────────────────────────────────────────────────────────────
    # Entrée : (32, 128, 1)
    x = conv_block(image, 32,  2, 2)   # → (16, 64, 32)
    x = conv_block(x,     64,  2, 2)   # → (8,  32, 64)
    x = conv_block(x,     128, 0, 0)   # → (8,  32, 128)  pas de pool
    x = conv_block(x,     128, 2, 1)   # → (4,  32, 128)  pool H seul
    x = conv_block(x,     256, 0, 0)   # → (4,  32, 256)
    x = conv_block(x,     256, 2, 1)   # → (2,  32, 256)  pool H seul
    x = conv_block(x,     256, 2, 1)   # → (1,  32, 256)  pool H seul

    # ── CNN → séquence ───────────────────────────────────────────────────────
    # (batch, 1, 32, 256) → (batch, 32, 256)
    x = layers.Reshape((-1, 256))(x)   # T = 32 timesteps

    # ── BiLSTM ───────────────────────────────────────────────────────────────
    x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Dropout(0.25)(x)

    # ── Sortie ────────────────────────────────────────────────────────────────
    y_pred = layers.Dense(N_CLASSES, activation='softmax', name='y_pred')(x)

    # ── CTC Loss ─────────────────────────────────────────────────────────────
    def ctc_fn(args):
        y, lbl, il, ll = args
        return tf.keras.backend.ctc_batch_cost(lbl, y, il, ll)

    ctc_loss = layers.Lambda(ctc_fn, output_shape=(1,), name='ctc_loss')(
        [y_pred, label, input_length, label_length])

    training_model  = models.Model(
        inputs =[image, label, input_length, label_length],
        outputs=[ctc_loss])

    inference_model = models.Model(inputs=[image], outputs=[y_pred])

    return training_model, inference_model


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACK BEAM SEARCH
# ══════════════════════════════════════════════════════════════════════════════

class BeamSearchCallback(callbacks.Callback):
    """
    Calcule l'accuracy Beam Search sur le set de val à chaque époque.
    Affiche aussi 3 exemples (corrects et incorrects) pour suivre la progression.
    """

    def __init__(self, inference_model, val_items,
                 beam_width=10, max_samples=300, eval_every=1):
        super().__init__()
        self.inf_model   = inference_model
        self.val_items   = val_items
        self.beam_width  = beam_width
        self.max_samples = max_samples
        self.eval_every  = eval_every
        self.best_acc    = 0.0

    def _load_batch(self, items):
        imgs, gts = [], []
        for img_path, label in items:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((IMG_H, IMG_W), np.uint8)
            img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
            imgs.append(img[..., np.newaxis])
            gts.append(label)
        return np.stack(imgs), gts

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_every != 0:
            return

        sample = self.val_items[:self.max_samples]
        X, gts = self._load_batch(sample)
        y_pred = self.inf_model.predict(X, verbose=0)

        texts, _ = batch_beam_search(y_pred, CHARACTERS,
                                     beam_width=self.beam_width,
                                     blank_idx=BLANK_IDX)
        perfect = sum(t == g for t, g in zip(texts, gts))
        acc     = perfect / len(gts) * 100
        flag    = "  ★ meilleur" if acc > self.best_acc else ""
        if acc > self.best_acc:
            self.best_acc = acc

        print(f"\n  [BeamSearch val] Époque {epoch+1:3d} — "
              f"Accuracy : {acc:.2f}%  ({perfect}/{len(gts)}){flag}")

        # Quelques exemples
        errors   = [(g, t) for g, t in zip(gts, texts) if g != t][:2]
        corrects = [(g, t) for g, t in zip(gts, texts) if g == t][:2]
        for g, t in errors:
            print(f"    ✗ '{g}' → '{t}'")
        for g, t in corrects:
            print(f"    ✓ '{g}'")

        if logs is not None:
            logs['val_beam_acc'] = acc


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    random.seed(args.seed); np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)

    # ── Chargement du dataset ─────────────────────────────────────────────────
    train_items, val_items = load_dataset(args.annotations, images_dir)

    if not train_items:
        raise ValueError("Aucune image de train trouvée. "
                         "Vérifier --images et --annotations.")
    if not val_items:
        raise ValueError("Aucune image de val trouvée. "
                         "Vérifier que le CSV a bien des lignes split=val.")

    print(f"[INFO] Train : {len(train_items)} images")
    print(f"[INFO] Val   : {len(val_items)} images")
    print(f"[INFO] Alphabet : {len(CHARACTERS)} chars + 1 blank = {N_CLASSES} classes")
    print(f"[INFO] Alphabet  → {CHARACTERS}")

    train_gen = PlateGenerator(train_items, BATCH_SIZE, augment=True,  shuffle=True)
    val_gen   = PlateGenerator(val_items,   BATCH_SIZE, augment=False, shuffle=False)

    # ── Modèle ────────────────────────────────────────────────────────────────
    training_model, inference_model = build_crnn()

    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Reprise depuis : {args.resume}")
        try:
            # .keras charge architecture + poids + config en un seul appel propre
            training_model = tf.keras.models.load_model(
                args.resume, compile=False)
            inference_model = models.Model(
                inputs =[training_model.get_layer('image').input],
                outputs=[training_model.get_layer('y_pred').output])
            print("[INFO] Modèle chargé depuis .keras avec succès")
        except Exception as e:
            print(f"[WARN] Chargement complet échoué ({e}), tentative poids seuls...")
            inference_model.load_weights(args.resume, by_name=True, skip_mismatch=True)
    else:
        print("[INFO] Nouveau modèle")

    training_model.compile(
    optimizer=optimizers.Adam(
        learning_rate=args.lr,
        clipnorm=1.0
    ),
    loss=lambda y_true, y_pred: y_pred
)

    training_model.summary(line_length=90)

    # ── Chemins de sauvegarde (.keras — format moderne) ───────────────────────
    # .keras  = format natif Keras 3, ZIP contenant config JSON + poids
    #           → lisible, debuggable, pas de dépendance HDF5
    #           → tf.keras.models.load_model() sans custom_objects dans la plupart des cas
    # .h5     = format legacy HDF5, déprécié depuis Keras 3 (TF 2.12+)
    best_path = str(models_dir / "crnn_best.keras")       # meilleur val_loss
    last_path = str(models_dir / "crnn_last.keras")       # dernière époque
    inf_path  = str(models_dir / "crnn_inference.keras")  # inférence seule

    cb_list = [
        callbacks.ModelCheckpoint(
            best_path, monitor='val_loss',
            save_best_only=True, verbose=1),

        callbacks.ModelCheckpoint(
            last_path, monitor='val_loss',
            save_best_only=False, verbose=0),

        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1),

        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),

        callbacks.CSVLogger(
            str(models_dir / "training_log.csv"),
            append=bool(args.resume)),

        BeamSearchCallback(
            inference_model=inference_model,
            val_items=val_items,
            beam_width=args.beam_width,
            eval_every=1,
            max_samples=min(400, len(val_items))),
    ]

    # ── Lancement ─────────────────────────────────────────────────────────────
    print(f"\n[INFO] Démarrage de l'entraînement...")
    training_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1,
    )

    # Sauvegarder le modèle d'inférence séparément (.keras)
    inference_model.save(inf_path)
    print(f"\n[INFO] Modèle d'inférence → {inf_path}")
    print(f"[INFO] Meilleur modèle    → {best_path}")
    print(f"\n  → Évaluation :")
    print(f"    python evaluate.py --model {inf_path} "
          f"--images {args.images} --annotations {args.annotations}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement CRNN Plaque Marocaine")
    p.add_argument("--images",      default="data/images_augmented")
    p.add_argument("--annotations", default="annotations.csv")
    p.add_argument("--models_dir",  default="models")
    p.add_argument("--resume",      default=None,
                   help="Reprendre depuis un .keras existant (ex: models/crnn_best.keras)")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--beam_width",  type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())