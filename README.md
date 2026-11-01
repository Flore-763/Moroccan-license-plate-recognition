# Detection et reconnaissance de plaques d'immatriculation marocaines

Projet de fin d'annee (PFA) en vision par ordinateur pour la detection et la reconnaissance automatique de plaques d'immatriculation marocaines. Le systeme combine la detection d'objets avec YOLOv8n et un module OCR base sur CNN, BiLSTM et decodage CTC afin de lire des plaques contenant des caracteres latins et arabes.

L'application peut etre utilisee dans des contextes de securite routiere, de controle d'acces ou de detection de fraudes, en permettant d'identifier les vehicules detectes, d'enregistrer les passages et de generer des alertes lorsque la plaque est inconnue ou suspecte.

## Fonctionnalites principales

- Detection de plaques d'immatriculation sur image ou video avec YOLOv8n.
- Reconnaissance des caracteres de plaque avec un modele OCR CNN + BiLSTM + CTC.
- Prise en charge des plaques marocaines combinant caracteres latins et arabes.
- API backend pour gerer les vehicules, detections et alertes.
- Stockage des donnees dans PostgreSQL avec migrations Alembic.
- Interface web React pour interagir avec le systeme.
- Scripts d'entrainement, d'inference et d'evaluation des modeles.

## Architecture du projet

```text
PFA_SIDRPI/
|-- backend_pfa/          # API FastAPI, CRUD, schemas, base de donnees, Alembic
|-- frontend/             # Interface web React + Vite
|-- model/                # Modeles, scripts YOLO/OCR et experiments
|-- CRNN/                 # Scripts lies a la preparation/augmentation OCR
|-- CRNN_LSTM_CTC/        # Entrainement et inference OCR CNN + BiLSTM + CTC
|-- requirements.txt      # Dependances Python principales
`-- .gitignore
```

## Pipeline de traitement

1. Une image ou une video est fournie au systeme.
2. YOLOv8n detecte la zone de la plaque d'immatriculation.
3. La plaque detectee est recadree et pretraitee.
4. Le modele OCR extrait la sequence de caracteres.
5. Le texte reconnu est compare aux vehicules enregistres en base.
6. Une detection est sauvegardee et une alerte peut etre creee si necessaire.

## Stack technologique

- **Computer Vision** : OpenCV, YOLOv8n, Ultralytics
- **Deep Learning / OCR** : TensorFlow/Keras, CNN, BiLSTM, CTC decoding
- **Backend** : Python, FastAPI, SQLAlchemy
- **Base de donnees** : PostgreSQL
- **Migrations** : Alembic
- **Frontend** : React, Vite, JavaScript
- **Evaluation** : scripts d'analyse de performance, rapports JSON/CSV, visualisations

## Démo en ligne

La démo publique tourne **entièrement dans le navigateur** (modèles ONNX + onnxruntime-web, hébergée sur GitHub Pages) : aucune photo n'est envoyée à un serveur. Voir **[DEPLOIEMENT.md](DEPLOIEMENT.md)**.

## Installation locale

Aucune base de données à installer : SQLite est utilisé par défaut (PostgreSQL reste possible via `DATABASE_URL`).

Placer d'abord les deux modèles dans `models/` (voir `models/README.md`).

### Backend

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd backend_pfa
uvicorn main:app --reload
```

API : http://127.0.0.1:8000  (docs : `/docs`, état des modèles : `/health`)

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Variables d'environnement utiles

| Variable | Défaut | Rôle |
|---|---|---|
| `DEMO_MODE` | `0` (Docker : `1`) | lecture seule + rien n'est enregistré côté serveur |
| `ENABLE_VIDEO` | `1` hors démo | active l'onglet/route vidéo |
| `ENABLE_WEBCAM` | `0` | webcam live (trop lourde pour un CPU gratuit) |
| `DATABASE_URL` | SQLite local | ex. `postgresql://user:pass@host/db` |
| `RATE_LIMIT_PER_MIN` | `15` | analyses/minute/IP |
| `CORS_ORIGINS` | localhost:5173 | seulement si le front est hébergé ailleurs |

## Endpoints principaux

- `GET /health` : etat de l'API et des modeles.
- `GET /samples` : images d'exemple proposees dans l'interface.
- `POST /detect-image` : detecter et reconnaitre une plaque depuis une image.
- `GET /vehicules` : lister les vehicules connus.
- `POST /vehicules` : ajouter un vehicule.
- `GET /detections` : consulter les detections enregistrees.
- `GET /alerts` : consulter les alertes generees.

Le routeur video est egalement present dans `backend_pfa/routers/detect_video.py`.

## Modeles et donnees

Les fichiers de poids, sorties d'entrainement, videos de test et resultats experimentaux peuvent etre volumineux. Ils sont donc exclus du suivi Git par defaut via `.gitignore`. Pour reproduire completement le projet, placer les modeles entraines aux emplacements attendus par le backend, notamment pour YOLO et le modele OCR.

## Contexte

Ce projet a ete realise dans le cadre d'un Projet de Fin d'Annee. Il illustre l'integration d'un pipeline de vision par ordinateur complet, depuis la detection de plaques jusqu'a la reconnaissance des caracteres et la gestion applicative des resultats via une API, une base de donnees et une interface web.

---

## Quelques captures d'écran

![alt text](analyse_sur_video.PNG) ![alt text](<moduleimage2.png>) ![alt text](<fastapi_endpoints.PNG>) ![alt text](<vehicule.png>)