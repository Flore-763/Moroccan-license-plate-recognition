"""
augment_dataset.py  (v3 — split-aware)
========================================
Augmente TOUT le dataset et maintient UN SEUL annotations.csv.

Règles garanties :
   Un original et TOUTES ses versions augmentées partagent le MÊME split
   Le split est attribué AVANT l'augmentation (au niveau des originaux)
   10% val / 90% train par défaut (modifiable via --val_ratio)
   Idempotent : relancer le script n'écrase pas les splits déjà attribués
   Compatible avec les chemins Windows (..\\maroc_car\\dataset\\...)

Usage (première fois — dataset sans split) :
    python augment_dataset.py --images_dir maroc_car/dataset --annotations annotations.csv --output_dir data/images_augmented --n_aug 6

Usage (relance sur un CSV déjà splitté) :
    python augment_dataset.py \
        --images_dir    data/images \
        --annotations   annotations.csv \
        --output_dir    data/images_augmented \
        --n_aug         6 \
        --keep_existing_split
"""

import cv2, csv, re, shutil, random, argparse
import numpy as np
from pathlib import Path, PureWindowsPath
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES CHEMINS
# ══════════════════════════════════════════════════════════════════════════════

AUG_RE = re.compile(r'_aug\d{2,}$', re.IGNORECASE)


def filename_only(image_path: str) -> str:
    """Extrait le nom de fichier depuis un chemin Windows ou Linux."""
    return PureWindowsPath(image_path).name   # 'IMG-20200725-WA0000.jpg'


def file_stem(image_path: str) -> str:
    return Path(filename_only(image_path)).stem   # 'IMG-20200725-WA0000'


def original_stem(image_path: str) -> str:
    """Retire le suffixe _augNN pour retrouver le stem de l'original."""
    return AUG_RE.sub('', file_stem(image_path))


def is_augmented(image_path: str) -> bool:
    return bool(AUG_RE.search(file_stem(image_path)))


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT (attribué au niveau des stems originaux)
# ══════════════════════════════════════════════════════════════════════════════

def assign_splits(orig_stems: list[str], val_ratio: float, seed: int) -> dict[str, str]:
    """
    Retourne { stem → 'train' | 'val' } pour chaque stem original.
    Le split est fait au niveau du stem → cohérence garantie.
    """
    random.seed(seed)
    stems = sorted(orig_stems)
    random.shuffle(stems)
    n_val = max(1, round(len(stems) * val_ratio))
    return {s: ('val' if i < n_val else 'train') for i, s in enumerate(stems)}


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def brightness_contrast(img):
    return cv2.convertScaleAbs(img,
        alpha=random.uniform(0.55, 1.45),
        beta=random.uniform(-45, 45))

def gaussian_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def motion_blur(img):
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k)); kernel[k//2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def gaussian_noise(img):
    noise = np.random.normal(0, random.uniform(5, 25), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def salt_pepper(img, amount=0.02):
    out = img.copy()
    n   = int(amount * img.size)
    for val in (255, 0):
        c = [np.random.randint(0, d - 1, n) for d in img.shape[:2]]
        out[c[0], c[1]] = val
    return out

def rotation(img):
    h, w  = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-12, 12), 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def perspective(img):
    h, w  = img.shape[:2]
    dx, dy = int(w*random.uniform(0.04, 0.10)), int(h*random.uniform(0.04, 0.10))
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [random.randint(0,dx),    random.randint(0,dy)],
        [w-random.randint(0,dx),  random.randint(0,dy)],
        [w-random.randint(0,dx),  h-random.randint(0,dy)],
        [random.randint(0,dx),    h-random.randint(0,dy)],
    ])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                               (w, h), borderMode=cv2.BORDER_REPLICATE)

def shadow(img):
    h, w  = img.shape[:2]
    mask  = np.zeros((h, w), np.uint8)
    x1, x2 = random.randint(0, w), random.randint(0, w)
    pts = np.array([[0,0],[w,0],[x2,h],[x1,h]], np.int32) \
          if random.random() < 0.5 else \
          np.array([[x1,0],[x2,0],[w,random.randint(0,h)],[w,random.randint(0,h)]], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    out = img.copy().astype(np.float32)
    out[mask == 1] *= random.uniform(0.35, 0.70)
    return np.clip(out, 0, 255).astype(np.uint8)

def jpeg_compression(img):
    ep = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 80)]
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    _, enc = cv2.imencode('.jpg', img, ep)
    return cv2.imdecode(enc, flag)

def random_erase(img):
    h, w = img.shape[:2]
    rh, rw = int(h*random.uniform(0.05,0.18)), int(w*random.uniform(0.03,0.15))
    x, y = random.randint(0, max(1,w-rw)), random.randint(0, max(1,h-rh))
    out = img.copy(); out[y:y+rh, x:x+rw] = random.randint(80, 200)
    return out

def clahe_enhance(img):
    c = cv2.createCLAHE(clipLimit=random.uniform(1.5, 4.0), tileGridSize=(4,4))
    return c.apply(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def downscale(img):
    h, w  = img.shape[:2]; s = random.uniform(0.4, 0.75)
    small = cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


TRANSFORMS = [brightness_contrast, gaussian_blur, motion_blur, gaussian_noise,
              salt_pepper, rotation, perspective, shadow, jpeg_compression,
              random_erase, clahe_enhance, downscale]


def augment_one(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for fn in random.sample(TRANSFORMS, random.randint(1, 4)):
        try: out = fn(out)
        except Exception: pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CSV I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows   = list(reader)
    return fields, rows


def save_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_path', 'label', 'split'])
        w.writeheader(); w.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",           default="../maroc_car/dataset",
                   help="Dossier des images originales")
    p.add_argument("--annotations",          default="annotations.csv",
                   help="CSV source (image_path, label[, split])")
    p.add_argument("--output_dir",           default="data/images_augmented",
                   help="Dossier de sortie pour toutes les images")
    p.add_argument("--n_aug",                type=int,   default=6,
                   help="Versions augmentées par image originale")
    p.add_argument("--val_ratio",            type=float, default=0.10)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--keep_existing_split",  action="store_true",
                   help="Conserver les splits déjà présents dans le CSV")
    p.add_argument("--no_copy_original",     action="store_true",
                   help="Ne pas copier les originaux dans output_dir")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {ann_path}")

    # Backup automatique
    backup = ann_path.with_name("annotations_backup.csv")
    shutil.copy2(ann_path, backup)
    print(f"[INFO] Backup → {backup}")

    fields, rows = load_csv(ann_path)
    print(f"[INFO] {len(rows)} lignes lues")

    # ── Séparer originaux et déjà-augmentés ───────────────────────────────────
    orig_rows = [r for r in rows if not is_augmented(r['image_path'])]
    aug_rows  = [r for r in rows if     is_augmented(r['image_path'])]
    print(f"[INFO] Originaux : {len(orig_rows)}  |  Déjà augmentés : {len(aug_rows)}")

    # ── Attribution du split ──────────────────────────────────────────────────
    if args.keep_existing_split and all(r.get('split','') in ('train','val') for r in orig_rows):
        # Conserver les splits existants
        stem_to_split = {original_stem(r['image_path']): r['split'] for r in orig_rows}
        print("[INFO] Splits existants conservés")
    else:
        # Attribuer de nouveaux splits sur les originaux
        all_orig_stems = [original_stem(r['image_path']) for r in orig_rows]
        stem_to_split  = assign_splits(all_orig_stems, args.val_ratio, args.seed)
        n_val  = sum(1 for s in stem_to_split.values() if s == 'val')
        n_train = len(stem_to_split) - n_val
        print(f"[INFO] Split attribué — train: {n_train} originaux | val: {n_val} originaux")

    # Appliquer le split sur tous les originaux
    for r in orig_rows:
        r['split'] = stem_to_split.get(original_stem(r['image_path']), 'train')

    # ── Génération des images augmentées ─────────────────────────────────────
    # Index des augmentées existantes pour éviter les doublons
    existing_aug_names = {filename_only(r['image_path']) for r in aug_rows}

    new_aug_rows = []
    skipped = 0

    for r in tqdm(orig_rows, desc="Augmentation"):
        img_path = images_dir / filename_only(r['image_path'])
        if not img_path.exists():
            img_path = Path(r['image_path'])      # chemin absolu direct
        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        # Copier l'original dans output_dir
        if not args.no_copy_original:
            dest = output_dir / filename_only(r['image_path'])
            if not dest.exists():
                shutil.copy2(img_path, dest)

        stem = file_stem(r['image_path'])
        ext  = Path(filename_only(r['image_path'])).suffix or '.jpg'
        split = r['split']   # ← héritage garanti

        for i in range(1, args.n_aug + 1):
            aug_name = f"{stem}_aug{i:02d}{ext}"

            if aug_name not in existing_aug_names:
                aug_path = output_dir / aug_name
                if not aug_path.exists():
                    cv2.imwrite(str(aug_path), augment_one(img))
                new_aug_rows.append({
                    'image_path': aug_name,
                    'label':      r['label'],
                    'split':      split,      # ← MÊME SPLIT que l'original
                })
                existing_aug_names.add(aug_name)

    # ── Mise à jour des splits sur les aug déjà existants ─────────────────────
    for r in aug_rows:
        r['split'] = stem_to_split.get(original_stem(r['image_path']), 'train')

    # ── Fusion finale et déduplication ────────────────────────────────────────
    all_rows = orig_rows + aug_rows + new_aug_rows
    seen, deduped = set(), []
    for r in all_rows:
        key = filename_only(r['image_path'])
        if key not in seen:
            seen.add(key); deduped.append(r)

    # Normaliser la colonne image_path → nom de fichier seulement
    for r in deduped:
        r['image_path'] = filename_only(r['image_path'])

    save_csv(ann_path, deduped)

    # ── Résumé ────────────────────────────────────────────────────────────────
    n_train = sum(1 for r in deduped if r['split'] == 'train')
    n_val   = sum(1 for r in deduped if r['split'] == 'val')
    print()
    print("═" * 60)
    print("   AUGMENTATION TERMINÉE")
    print("═" * 60)
    print(f"  Originaux traités        : {len(orig_rows) - skipped}")
    print(f"  Sautés (introuvables)    : {skipped}")
    print(f"  Nouvelles aug générées   : {len(new_aug_rows)}")
    print(f"  TOTAL lignes CSV         : {len(deduped)}")
    print(f"  → train                  : {n_train}  ({n_train/len(deduped)*100:.1f}%)")
    print(f"  → val                    : {n_val}   ({n_val/len(deduped)*100:.1f}%)")
    print(f"  CSV final                : {ann_path}")
    print(f"  Images dans              : {output_dir}")
    print("═" * 60)
    print(f"\n  → Lance l'entraînement :")
    print(f"    python train.py --images {output_dir} --annotations {ann_path}")


if __name__ == "__main__":
    main()