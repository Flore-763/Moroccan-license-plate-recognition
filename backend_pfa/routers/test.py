import json
import csv

# Vérifier les bounding boxes du CSV
with open("video3_annotatee.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    gt_boxes = []
    for row in reader:
        if int(row['frame']) == 259 and row['plate_id'] == '7':
            gt_boxes.append([float(row['x1']), float(row['y1']), 
                             float(row['x2']), float(row['y2'])])
            break
print("GT bbox (plaque 7, frame 259):", gt_boxes)

# Vérifier les bounding boxes des prédictions
with open("video3_pred.json", "r", encoding="utf-8") as f:
    pred = json.load(f)
    for det in pred['detections']:
        if det.get('frame') == 259 and '60084' in det.get('immatriculation', ''):
            print("Pred bbox (frame 259):", det.get('bbox'))
            break