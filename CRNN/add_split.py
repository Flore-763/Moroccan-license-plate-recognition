"""
add_split.py
=============
À exécuter UNE SEULE FOIS dans ton dossier CRNN pour ajouter
la colonne split à mon annotations.csv existant.

Règles :
  • 90% train / 10% val
  • Le split est décidé AU NIVEAU DE L'IMAGE ORIGINALE
  • Toutes les versions augmentées (_aug01, _aug02...) héritent
    automatiquement du même split que leur original
  • Un backup annotations_backup.csv est créé avant modification

Usage :
    python add_split.py
    python add_split.py --annotations annotations.csv --val_ratio 0.10 --seed 42
"""

import csv, re, random, shutil, argparse
from pathlib import Path, PureWindowsPath


# ══════════════════════════════════════════════════════════════════════════════

AUG_RE = re.compile(r'_aug\d{2,}$', re.IGNORECASE)


def get_stem(image_path: str) -> str:
    """Nom de fichier sans extension, compatible chemins Windows."""
    name = PureWindowsPath(image_path).name   # ex: IMG-20200725-WA0000.jpg
    return Path(name).stem                    # ex: IMG-20200725-WA0000


def get_orig_stem(image_path: str) -> str:
    """Retire _augNN pour retrouver le stem de l'original."""
    return AUG_RE.sub('', get_stem(image_path))


def is_aug(image_path: str) -> bool:
    return bool(AUG_RE.search(get_stem(image_path)))


# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="annotations.csv")
    p.add_argument("--val_ratio",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {ann_path}")

    # Lecture
    with open(ann_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"[INFO] {len(rows)} lignes lues depuis '{ann_path}'")

    # Vérifier si des splits existent déjà
    has_split_col = 'split' in fieldnames
    splits_filled = sum(1 for r in rows if (r.get('split') or '').strip() in ('train', 'val'))
    print(f"[INFO] Colonne split présente : {has_split_col}")
    print(f"[INFO] Lignes avec split déjà rempli : {splits_filled}")

    if splits_filled == len(rows):
        print("[INFO] Tous les splits sont déjà remplis. Rien à faire.")
        return

    # Backup
    backup = ann_path.with_name("annotations_backup.csv")
    shutil.copy2(ann_path, backup)
    print(f"[INFO] Backup créé → {backup}")

    # Collecter les stems ORIGINAUX (pas les augmentés)
    orig_stems_ordered = []
    seen_stems = set()
    for r in rows:
        if not is_aug(r.get('image_path', r.get('filename', ''))):
            s = get_orig_stem(r.get('image_path', r.get('filename', '')))
            if s not in seen_stems:
                seen_stems.add(s)
                orig_stems_ordered.append(s)

    print(f"[INFO] Stems originaux uniques : {len(orig_stems_ordered)}")

    # Attribution du split au niveau des originaux
    random.seed(args.seed)
    stems_shuffled = orig_stems_ordered[:]
    random.shuffle(stems_shuffled)

    n_val = max(1, round(len(stems_shuffled) * args.val_ratio))
    val_stems = set(stems_shuffled[:n_val])

    print(f"[INFO] → val  : {n_val} originaux")
    print(f"[INFO] → train: {len(stems_shuffled) - n_val} originaux")

    # Injection du split dans chaque ligne (original ET augmentés)
    for r in rows:
        ip = r.get('image_path', r.get('filename', ''))
        og = get_orig_stem(ip)
        r['split'] = 'val' if og in val_stems else 'train'

    # Vérification
    n_train = sum(1 for r in rows if r['split'] == 'train')
    n_val_r = sum(1 for r in rows if r['split'] == 'val')
    print(f"[INFO] Résultat → train: {n_train}  val: {n_val_r}  "
          f"(ratio val: {n_val_r/len(rows)*100:.1f}%)")

    # Contrôle de cohérence : un original val → ses augmentés aussi
    sample_val = next((r for r in rows
                       if r['split'] == 'val' and not is_aug(r.get('image_path',''))), None)
    if sample_val:
        og = get_orig_stem(sample_val.get('image_path',''))
        group_splits = [r['split'] for r in rows if get_orig_stem(r.get('image_path','')) == og]
        assert all(s == 'val' for s in group_splits), "ERREUR : incohérence dans le groupe !"
        print(f"[INFO] Contrôle OK : '{og}' et ses {len(group_splits)} versions → toutes 'val'")

    # Écriture
    # S'assurer que 'split' est dans les fieldnames
    if 'split' not in fieldnames:
        fieldnames.append('split')

    # Détecter la colonne image_path réelle
    if 'image_path' not in fieldnames and 'filename' in fieldnames:
        # Renommer pour cohérence
        for r in rows:
            r['image_path'] = r.pop('filename', '')
        fieldnames = [f if f != 'filename' else 'image_path' for f in fieldnames]

    with open(ann_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] annotations.csv mis à jour avec les splits.")
    print(f"\n  → Lance maintenant l'entraînement :")
    print(f"     python train.py --images data/images_augmented --annotations annotations.csv")


if __name__ == "__main__":
    main()



