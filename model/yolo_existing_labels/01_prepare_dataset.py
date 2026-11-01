"""
Prepare un dataset YOLOv8 a partir des images et labels deja presents.

Source attendue:
    maroc_car/dataset/
        image1.jpg
        image1.txt
        image2.jpg
        image2.txt

Structure creee:
    data/images/train|val|test
    data/labels/train|val|test
"""

import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42


def collect_pairs(src: Path):
    pairs = []
    missing_labels = []

    for img_path in src.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    orphan_labels = []
    for label_path in src.glob("*.txt"):
        candidate_exists = any((src / f"{label_path.stem}{ext}").exists() for ext in IMAGE_EXTENSIONS)
        if not candidate_exists:
            orphan_labels.append(label_path.name)

    return pairs, missing_labels, orphan_labels


def prepare(src: str, dst: str):
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Dossier source introuvable: {src_path}")

    pairs, missing_labels, orphan_labels = collect_pairs(src_path)

    print(f"[INFO] {len(pairs)} paires image/label trouvees dans '{src_path}'")

    if missing_labels:
        print(f"[WARN] {len(missing_labels)} images sans label seront ignorees")
        for name in missing_labels[:10]:
            print(f"       - {name}")

    if orphan_labels:
        print(f"[WARN] {len(orphan_labels)} labels sans image seront ignores")
        for name in orphan_labels[:10]:
            print(f"       - {name}")

    if not pairs:
        raise FileNotFoundError("Aucune paire image/label valide trouvee.")

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_val = int(n * SPLIT["val"])
    n_test = int(n * SPLIT["test"])
    n_train = n - n_val - n_test

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    print(f"[INFO] Split -> train:{n_train}  val:{n_val}  test:{n_test}")

    for split_name in ("train", "val", "test"):
        (dst_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dst_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        img_dst = dst_path / "images" / split_name
        label_dst = dst_path / "labels" / split_name

        for img_path, label_path in tqdm(items, desc=f"Copie {split_name}", unit="sample"):
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(label_path, label_dst / label_path.name)

    print(f"\n[OK] Dataset prepare dans '{dst_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="../maroc_car/dataset",
                        help="Dossier source contenant images et labels YOLO")
    parser.add_argument("--dst", default="data",
                        help="Dossier de destination du dataset YOLOv8")
    args = parser.parse_args()
    prepare(args.src, args.dst)
