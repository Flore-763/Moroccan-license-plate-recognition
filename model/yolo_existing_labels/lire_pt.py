from ultralytics import YOLO

# Charger le modèle YOLO directement
model = YOLO('runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt')

# Afficher la structure et les informations du modèle
# print(model.info())



# Afficher les noms des classes (ex: plaques d'immatriculation)
# print("Classes détectées :", model.names)
results = model.predict('image.png', verbose=True)
