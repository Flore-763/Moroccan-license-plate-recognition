# Déployer la démo gratuitement — mode « 100 % navigateur »

Les deux modèles sont convertis en ONNX et s'exécutent **dans le navigateur du visiteur**
(onnxruntime-web). Pas de serveur, pas de base de données, pas de carte bancaire, et
aucune photo ne quitte l'ordinateur du visiteur. Hébergement : **GitHub Pages** (gratuit).

```
photo → YOLO (plaque) → recadrage → CRNN (lecture) → statut dans la base fictive
        └───────────── tout dans le navigateur ─────────────┘
```

## Étape 1 — Convertir vos modèles (une seule fois, sur votre PC)

Dans PowerShell, à la racine du projet (une commande par ligne ; `&&` ne marche pas dans l'ancien PowerShell) :

```powershell
py -3.12 -m venv .venv-convert312
.venv-convert312\Scripts\activate
pip install -r tools\requirements-convert.txt
```

Utilisez Python 3.12 ou 3.13 : TensorFlow 2.21.0, utilisé par l'environnement de conversion,
ne fournit pas de wheel pour Python 3.14. Si `py -3.12` n'est pas reconnu, installez Python 3.12
ou 3.13 et relancez ces commandes dans un nouveau terminal.

Placez vos fichiers dans `models\` : `plate_yolo.pt` et `crnn_best.keras`
(renommez `best_model.keras` en `crnn_best.keras`). Puis :

```powershell
python tools\convert_models.py
```

Résultat attendu :

```
[verif] écart max Keras↔ONNX = 3.xxe-05 | argmax identiques : True
[crnn] OK → frontend\public\models\crnn.onnx
[yolo] OK → frontend\public\models\plate_yolo.onnx
```

Le script **échoue volontairement** si le CRNN ONNX ne reproduit pas votre modèle Keras.

## Étape 2 — Ajouter des images d'exemple

Copiez 6 à 10 photos (JPG/PNG, < 1 Mo) dans `frontend\public\samples\`.
Le nom du fichier devient le titre (`dacia_logan.jpg` → « Dacia Logan »).
Utilisez des photos dont vous avez les droits.

## Étape 3 — Tester en local

```powershell
cd frontend
npm install
npm run dev:static
```

Ouvrez http://localhost:5173, cliquez un exemple, puis importez une photo de votre PC.
Au premier clic, comptez quelques secondes (téléchargement des modèles, ~15 Mo).

## Étape 4 — Publier sur GitHub Pages

1. Committez **les deux fichiers `.onnx`** (`frontend/public/models/`) et poussez sur GitHub.
2. Sur GitHub : **Settings → Pages → Build and deployment → Source : GitHub Actions**.
3. Le workflow `.github/workflows/deploy.yml` s'exécute à chaque push (onglet *Actions*).
4. Votre démo : `https://VOTRE_USER.github.io/NOM_DU_DEPOT/`

Le site fonctionne sur n'importe quel chemin (base relative), donc aussi sur Netlify,
Vercel ou Cloudflare Pages : build `npm run build:static`, dossier `frontend/dist`.

## Vérifier la fidélité du JavaScript

```powershell
python tools\make_pipeline_reference.py
cd frontend
npm test
```

Compare le pré/post-traitement JS à votre code Python (redimensionnement OpenCV, décodage CTC,
format de plaque, NMS YOLO).

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `No matching distribution found for tensorflow-cpu==2.21.0` | L'environnement utilise Python 3.14 ; recréez-le avec `py -3.12 -m venv .venv-convert312` |
| « Modèle introuvable » dans l'interface | `.onnx` absents de `frontend/public/models/` (ou non committés) |
| `ÉCHEC : le CRNN ONNX ne reproduit pas…` | `model_02.py` modifié après l'entraînement : remettez la version qui a produit le `.keras` |
| Plaque détectée mais texte faux | vérifier que `crnn_best.keras` vient de `model/CRNN` (blank = dernier index), pas de `CRNN_LSTM_CTC` |
| Page blanche sur GitHub Pages | Source Pages non réglée sur « GitHub Actions » |

## Limites assumées

- Images uniquement : la vidéo (suivi BoT-SORT) et la webcam demandent un vrai serveur Python.
  Elles restent disponibles en local (README) ou via le `Dockerfile` si vous déployez un jour un serveur.
- Les premiers chargements téléchargent ~15 Mo de modèles + ~15 Mo de moteur ONNX (mis en cache ensuite).
- La base de véhicules (`frontend/public/data/vehicules.json`) est fictive, générée avec la graine 42
  (`python tools\export_vehicules.py` pour la régénérer).
