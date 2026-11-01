# Detection de Plaques Marocaines - Modele 1

Ce dossier contient le pipeline du premier modele:

- il utilise les labels YOLO deja presents dans `maroc_car/dataset`
- il reorganise les donnees en `train/val/test`
- il entraine ensuite un modele YOLOv8

## Structure attendue

Le script `01_prepare_dataset.py` cree:

data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/

## Source utilisee

- Images: `maroc_car/dataset/*.jpg`
- Labels: `maroc_car/dataset/*.txt`

Les labels orphelins sans image correspondante sont ignores.

## Utilisation

1. Preparation du dataset



2. Entrainement

Parametres utilises:

- `model`: version YOLOv8n la version nao c'est la plis petite donc la moins lourde a entrainer 
- `epochs`: nombre que l'on passe dans tous les donnees
- `imgsz`: taille d'image
- `batch`: taille de batch: nbre d'images traites ensemble
- `patience`: arret anticipé si le modele n'ameliore plus

3. Inference simple sur image ou video

```bash
python training_me.py --source ../yolo/video.mp4
python training_me.py --source ../maroc_car/dataset/IMG_0001.jpg
```

Le script:

- detecte les voitures avec `yolov8n.pt`
- detecte les plaques avec le modele entraine
- applique EasyOCR pour lire le texte
- sauvegarde le resultat dans `outputs/`

## Notes

- Classe unique: `license_plate`
- Le script de preparation copie l'image et son label ensemble
- Aucun auto-labeling n'est fait dans ce dossier
- `visualize.py` et `add_missing_data.py` ne sont pas necessaires pour le pipeline principal

