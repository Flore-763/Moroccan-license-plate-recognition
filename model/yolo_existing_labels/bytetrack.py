import cv2
import os
from ultralytics import YOLO

# Chargement du modèle
model_path = 'runs/detect/runs/detect/plate_maroc_existing_labels/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ Erreur : Le fichier de modèle '{model_path}' est introuvable.")
    exit()

model = YOLO(model_path)

# Ouverture vidéo
video_path = 'video_marrakech.mp4'

if not os.path.exists(video_path):
    print(f"❌ Erreur : La vidéo '{video_path}' est introuvable.")
    exit()

cap = cv2.VideoCapture(video_path)

# Fenêtre redimensionnable
nom_fenetre = "ByteTrack - Detection de Plaques"
cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
cv2.resizeWindow(nom_fenetre, 800, 600)

print("🚀 Lancement... Appuyez sur 'q' pour quitter.")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    # Affichage des boîtes + IDs
    annotated_frame = results[0].plot()

    cv2.imshow(nom_fenetre, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Session interrompue par l'utilisateur.")
        break

cap.release()
cv2.destroyAllWindows()