
"""
evaluate_video.py
Évaluation complète du pipeline vidéo avec DEUX approches :
1. Association spatiale (uniquement sur les frames avec prédictions)
2. Qualité OCR (vérification indépendante des lectures)
"""

import json
import csv
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

def normalize_plate(text: str) -> str:
    """Normalise les plaques pour comparaison."""
    if not text:
        return ""
    text = text.replace(" ", "")
    
    arabic_map = {
        'ﺃ': 'أ', 'ﺄ': 'أ',
        'ﺏ': 'ب', 'ﺐ': 'ب', 'ﺑ': 'ب',
        'ﺩ': 'د', 'ﺪ': 'د',
        'ﻭ': 'و', 'ﻮ': 'و',
        'ﻉ': 'ع', 'ﻊ': 'ع', 'ﻋ': 'ع', 'ﻌ': 'ع',
    }
    for old, new in arabic_map.items():
        text = text.replace(old, new)
    
    return text


def plate_similarity(p1: str, p2: str) -> float:
    """Similarité entre deux plaques normalisées."""
    if not p1 or not p2:
        return 0.0
    return SequenceMatcher(None, normalize_plate(p1), normalize_plate(p2)).ratio()


def iou(box1, box2):
    """Calcul de l'Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


class VideoEvaluator:
    def __init__(self, ground_truth_path: str, predictions_path: str):
        self.gt_by_frame = self._load_ground_truth(ground_truth_path)
        self.pred_by_frame = self._load_predictions(predictions_path)
        self.all_gt_plates = self._load_unique_gt_plates(ground_truth_path)
        
    def _load_ground_truth(self, path: str) -> dict:
        """Charge les annotations CVAT par frame."""
        gt_by_frame = defaultdict(list)
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['frame'])
                gt_by_frame[frame].append({
                    'plate_id': int(row['plate_id']),
                    'immatriculation': row['immatriculation'],
                    'bbox': [float(row['x1']), float(row['y1']), 
                             float(row['x2']), float(row['y2'])]
                })
        return gt_by_frame
    
    def _load_predictions(self, path: str) -> dict:
        """Charge les prédictions du pipeline par frame."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pred_by_frame = defaultdict(list)
        for det in data.get('detections', []):
            frame = det.get('frame')
            if frame is not None:
                pred_by_frame[frame].append({
                    'plate_track_id': det.get('plate_track_id'),
                    'car_track_id': det.get('car_track_id'),
                    'immatriculation': det.get('immatriculation', ''),
                    'confiance': det.get('confiance', 0),
                    'bbox': det.get('bbox', [0, 0, 0, 0])
                })
        return pred_by_frame
    
    def _load_unique_gt_plates(self, path: str) -> set:
        """Charge toutes les plaques uniques du CSV."""
        gt_plates = set()
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_plates.add(row['immatriculation'])
        return gt_plates
    
    # =========================================================
    # APPROCHE 1 : Association spatiale (frames avec prédictions)
    # =========================================================
    def evaluate_spatial_on_pred_frames(self, iou_threshold: float = 0.5, sim_threshold: float = 0.85) -> dict:
        """
        Évalue UNIQUEMENT sur les frames où le pipeline a fait des prédictions.
        """
        prediction_frames = list(self.pred_by_frame.keys())
        
        if not prediction_frames:
            return {'error': 'Aucune prédiction trouvée'}
        
        print(f"\n Frames avec prédictions: {sorted(prediction_frames)}")
        
        total_tp = 0
        total_fn = 0
        total_fp = 0
        total_predictions = 0
        total_gt_in_pred_frames = 0
        
        frame_results = []
        
        for frame in sorted(prediction_frames):
            gt_list = self.gt_by_frame.get(frame, [])
            pred_list = self.pred_by_frame.get(frame, [])
            
            total_gt_in_pred_frames += len(gt_list)
            total_predictions += len(pred_list)
            
            used_preds = set()
            frame_associations = []
            tp_frame = 0
            fn_frame = 0
            fp_frame = 0
            
            # Associer chaque GT à la meilleure prédiction
            for gt in gt_list:
                best_pred = None
                best_iou = 0.0
                best_idx = -1
                
                for idx, pred in enumerate(pred_list):
                    if idx in used_preds:
                        continue
                    if pred.get('bbox') and len(pred['bbox']) == 4:
                        iou_score = iou(gt['bbox'], pred['bbox'])
                        if iou_score > best_iou and iou_score >= iou_threshold:
                            best_iou = iou_score
                            best_pred = pred
                            best_idx = idx
                
                if best_pred:
                    used_preds.add(best_idx)
                    sim = plate_similarity(gt['immatriculation'], best_pred['immatriculation'])
                    is_correct = sim >= sim_threshold
                    
                    if is_correct:
                        tp_frame += 1
                        total_tp += 1
                    else:
                        fn_frame += 1
                        total_fn += 1
                    
                    frame_associations.append({
                        'gt_plate_id': gt['plate_id'],
                        'gt_text': gt['immatriculation'],
                        'pred_text': best_pred['immatriculation'],
                        'iou': best_iou,
                        'similarity': sim,
                        'correct': is_correct
                    })
                else:
                    fn_frame += 1
                    total_fn += 1
                    frame_associations.append({
                        'gt_plate_id': gt['plate_id'],
                        'gt_text': gt['immatriculation'],
                        'pred_text': '',
                        'correct': False,
                        'not_detected': True
                    })
            
            # Prédictions sans GT (faux positifs)
            for idx, pred in enumerate(pred_list):
                if idx not in used_preds:
                    fp_frame += 1
                    total_fp += 1
                    frame_associations.append({
                        'pred_text': pred['immatriculation'],
                        'is_false_positive': True
                    })
            
            frame_results.append({
                'frame': frame,
                'n_gt': len(gt_list),
                'n_pred': len(pred_list),
                'tp': tp_frame,
                'fn': fn_frame,
                'fp': fp_frame,
                'precision': tp_frame / (tp_frame + fp_frame) if (tp_frame + fp_frame) > 0 else 0,
                'recall': tp_frame / (tp_frame + fn_frame) if (tp_frame + fn_frame) > 0 else 0,
                'associations': frame_associations
            })
        
        precision = total_tp / total_predictions if total_predictions > 0 else 0
        recall = total_tp / total_gt_in_pred_frames if total_gt_in_pred_frames > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'frames_analyzed': prediction_frames,
            'total_frames': len(prediction_frames),
            'total_predictions': total_predictions,
            'total_gt_in_these_frames': total_gt_in_pred_frames,
            'tp': total_tp,
            'fn': total_fn,
            'fp': total_fp,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'frame_details': frame_results
        }
    
    # =========================================================
    # APPROCHE 2 : Qualité OCR (vérification indépendante)
    # =========================================================
    def evaluate_ocr_quality(self, similarity_threshold: float = 0.85) -> dict:
        """
        Évalue la qualité de l'OCR : vérifie si les plaques lues existent dans le CSV.
        """
        # Collecter toutes les prédictions uniques (par track_id pour éviter doublons)
        predictions_by_track = {}
        for frame, preds in self.pred_by_frame.items():
            for pred in preds:
                tid = pred.get('plate_track_id')
                if tid is not None:
                    if tid not in predictions_by_track or pred['confiance'] > predictions_by_track[tid]['confiance']:
                        predictions_by_track[tid] = {
                            'immatriculation': pred['immatriculation'],
                            'confiance': pred['confiance'],
                            'frame': frame,
                            'track_id': tid
                        }
        
        all_predictions = list(predictions_by_track.values())
        
        print(f"\n Prédictions uniques (par track_id): {len(all_predictions)}")
        
        correct_predictions = []
        incorrect_predictions = []
        
        for pred in all_predictions:
            best_match = None
            best_score = 0.0
            best_gt = None
            
            for gt in self.all_gt_plates:
                score = plate_similarity(pred['immatriculation'], gt)
                if score > best_score:
                    best_score = score
                    best_gt = gt
            
            if best_score >= similarity_threshold:
                correct_predictions.append({
                    'track_id': pred['track_id'],
                    'prediction': pred['immatriculation'],
                    'matched_with': best_gt,
                    'similarity': best_score,
                    'confiance': pred['confiance'],
                    'frame': pred['frame']
                })
            else:
                incorrect_predictions.append({
                    'track_id': pred['track_id'],
                    'prediction': pred['immatriculation'],
                    'best_match': best_gt if best_gt else '(aucune)',
                    'similarity': best_score,
                    'confiance': pred['confiance'],
                    'frame': pred['frame']
                })
        
        total = len(all_predictions)
        correct = len(correct_predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Plaques GT non détectées
        detected_plates = set()
        for pred in all_predictions:
            for gt in self.all_gt_plates:
                if plate_similarity(pred['immatriculation'], gt) >= similarity_threshold:
                    detected_plates.add(gt)
                    break
        
        undetected = self.all_gt_plates - detected_plates
        
        return {
            'total_ocr_readings': total,
            'correct_readings': correct,
            'incorrect_readings': len(incorrect_predictions),
            'ocr_accuracy': accuracy,
            'total_gt_plates': len(self.all_gt_plates),
            'detected_gt_plates': len(detected_plates),
            'undetected_gt_plates': list(undetected),
            'correct_details': correct_predictions,
            'incorrect_details': incorrect_predictions
        }
    
    # =========================================================
    # RAPPORT COMPLET
    # =========================================================
    def generate_report(self, output_path: str = None) -> dict:
        """Génère un rapport complet avec les deux approches."""
        
        print("\n" + "="*60)
        print("ÉVALUATION COMPLÈTE DU PIPELINE VIDÉO")
        print("="*60)
        
        # APPROCHE 1 : Association spatiale
        print("\n" + "="*60)
        print("APPROCHE 1 : ASSOCIATION SPATIALE")
        print("(Uniquement sur les frames avec prédictions)")
        print("="*60)
        
        spatial_results = self.evaluate_spatial_on_pred_frames()
        
        if 'error' not in spatial_results:
            print(f"\n Résultats:")
            print(f"   Frames analysées: {spatial_results['frames_analyzed']}")
            print(f"   Total prédictions: {spatial_results['total_predictions']}")
            print(f"   Total GT sur ces frames: {spatial_results['total_gt_in_these_frames']}")
            print(f"   TP: {spatial_results['tp']}, FN: {spatial_results['fn']}, FP: {spatial_results['fp']}")
            print(f"   Précision: {spatial_results['precision']:.2%}")
            print(f"   Rappel: {spatial_results['recall']:.2%}")
            print(f"   F1-score: {spatial_results['f1_score']:.2%}")
        
        # APPROCHE 2 : Qualité OCR
        print("\n" + "="*60)
        print("APPROCHE 2 : QUALITÉ OCR")
        print("(Vérification indépendante des lectures)")
        print("="*60)
        
        ocr_results = self.evaluate_ocr_quality()
        
        print(f"\n Résultats:")
        print(f"   Total lectures OCR uniques: {ocr_results['total_ocr_readings']}")
        print(f"   Lectures correctes: {ocr_results['correct_readings']}")
        print(f"   Lectures incorrectes: {ocr_results['incorrect_readings']}")
        print(f"   Accuracy OCR: {ocr_results['ocr_accuracy']:.2%}")
        print(f"   Plaques GT totales: {ocr_results['total_gt_plates']}")
        print(f"   Plaques GT détectées: {ocr_results['detected_gt_plates']}")
        
        if ocr_results['incorrect_details']:
            print(f"\n Lectures incorrectes:")
            for inc in ocr_results['incorrect_details']:
                print(f"   '{inc['prediction']}' → meilleur match: '{inc['best_match']}' (sim: {inc['similarity']:.2%})")
        
        if ocr_results['undetected_gt_plates']:
            print(f"\n Plaques GT non détectées:")
            for u in ocr_results['undetected_gt_plates']:
                print(f"   - {u}")
        
        # Rapport consolidé
        report = {
            'spatial_metrics': spatial_results if 'error' not in spatial_results else {'error': spatial_results.get('error')},
            'ocr_metrics': ocr_results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n Rapport sauvegardé: {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Évaluation complète du pipeline vidéo")
    parser.add_argument("--gt", required=True, help="Chemin du CSV d'annotations (video3_annotatee.csv)")
    parser.add_argument("--pred", required=True, help="Chemin du JSON des prédictions (video3_pred.json)")
    parser.add_argument("--output", default="evaluation_report.json", help="Fichier de sortie JSON")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="Seuil IoU pour association spatiale")
    parser.add_argument("--sim_thresh", type=float, default=0.85, help="Seuil de similarité pour OCR")
    
    args = parser.parse_args()
    
    evaluator = VideoEvaluator(args.gt, args.pred)
    report = evaluator.generate_report(args.output)
    
    return report


if __name__ == "__main__":
    main()

















# """
# evaluate_video_pipeline.py
# Évaluation complète du pipeline vidéo avec annotations CVAT
# """

# import json
# import csv
# from collections import defaultdict
# from typing import List, Dict, Tuple, Optional
# import numpy as np
# from difflib import SequenceMatcher
# import argparse
# from pathlib import Path
# import unicodedata

# def normalize_plate(text: str) -> str:
#     """Normalise les plaques pour comparaison."""
#     if not text:
#         return ""
#     # Supprimer les espaces
#     text = text.replace(" ", "")
    
#     # Normalisation arabe (formes isolées → standard)
#     arabic_map = {
#         'ﺃ': 'أ', 'ﺄ': 'أ',
#         'ﺏ': 'ب', 'ﺐ': 'ب', 'ﺑ': 'ب',
#         'ﺩ': 'د', 'ﺪ': 'د',
#         'ﻭ': 'و', 'ﻮ': 'و',
        
#         'ﻉ': 'ع', 'ﻊ': 'ع', 'ﻋ': 'ع', 'ﻌ': 'ع',
#     }
#     for old, new in arabic_map.items():
#         text = text.replace(old, new)
    
#     return text


# def iou(box1, box2):
#     """Calcul de l'Intersection over Union entre deux boîtes [x1,y1,x2,y2]"""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
    
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area1 + area2 - inter
    
#     return inter / union if union > 0 else 0.0


# def plate_similarity(p1: str, p2: str) -> float:
#     """Similarité entre deux plaques (normalisées)."""
#     def normalize(s):
#         return s.replace(" ", "").strip()
#     return SequenceMatcher(None, normalize_plate(p1), normalize_plate(p2)).ratio()


# class VideoEvaluator:
#     def __init__(self, ground_truth_path: str, predictions_path: str):
#         self.gt_by_frame = self._load_ground_truth(ground_truth_path)
#         self.pred_by_frame = self._load_predictions(predictions_path)
        
#     def _load_ground_truth(self, path: str) -> Dict[int, List[Dict]]:
#         """Charge les annotations CVAT par frame."""
#         gt_by_frame = defaultdict(list)
#         with open(path, 'r', encoding='utf-8-sig') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 frame = int(row['frame'])
#                 gt_by_frame[frame].append({
#                     'plate_id': int(row['plate_id']),
#                     'immatriculation': row['immatriculation'],
#                     'bbox': [float(row['x1']), float(row['y1']), 
#                              float(row['x2']), float(row['y2'])]
#                 })
#         return gt_by_frame
    
#     def _load_predictions(self, path: str) -> Dict[int, List[Dict]]:
#         """Charge les prédictions du pipeline par frame."""
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         pred_by_frame = defaultdict(list)
#         for det in data.get('detections', []):
#             frame = det.get('frame')
#             if frame is not None:
#                 pred_by_frame[frame].append({
#                     'plate_track_id': det.get('plate_track_id'),
#                     'car_track_id': det.get('car_track_id'),
#                     'immatriculation': det.get('immatriculation', ''),
#                     'confiance': det.get('confiance', 0),
#                     'bbox': det.get('bbox', [0, 0, 0, 0])  # Si disponible
#                 })
#         return pred_by_frame
    
#     def associate_by_iou(self, frame: int, iou_threshold: float = 0.5) -> List[Dict]:
#         """
#         Associe les prédictions aux GT sur une frame donnée par IoU.
#         Retourne la liste des associations avec IoU.
#         """
#         gt_list = self.gt_by_frame.get(frame, [])
#         pred_list = self.pred_by_frame.get(frame, [])
        
#         associations = []
        
#         # Pour chaque GT, trouver la meilleure prédiction
#         used_preds = set()
#         for gt in gt_list:
#             best_pred = None
#             best_iou = 0.0
            
#             for idx, pred in enumerate(pred_list):
#                 if idx in used_preds:
#                     continue
#                 # Si la prédiction n'a pas de bbox, ignorer
#                 if pred.get('bbox') and len(pred['bbox']) == 4:
#                     iou_score = iou(gt['bbox'], pred['bbox'])
#                     if iou_score > best_iou and iou_score >= iou_threshold:
#                         best_iou = iou_score
#                         best_pred = (idx, pred)
            
#             if best_pred:
#                 used_preds.add(best_pred[0])
#                 associations.append({
#                     'frame': frame,
#                     'gt_plate_id': gt['plate_id'],
#                     'gt_text': gt['immatriculation'],
#                     'pred_text': best_pred[1]['immatriculation'],
#                     'confiance': best_pred[1]['confiance'],
#                     'iou': best_iou,
#                     'plate_track_id': best_pred[1].get('plate_track_id'),
#                     'car_track_id': best_pred[1].get('car_track_id')
#                 })
#             else:
#                 # GT non détecté
#                 associations.append({
#                     'frame': frame,
#                     'gt_plate_id': gt['plate_id'],
#                     'gt_text': gt['immatriculation'],
#                     'pred_text': '',
#                     'confiance': 0,
#                     'iou': 0,
#                     'matched': False
#                 })
        
#         # Prédictions sans GT (fausses détections)
#         for idx, pred in enumerate(pred_list):
#             if idx not in used_preds:
#                 associations.append({
#                     'frame': frame,
#                     'gt_plate_id': None,
#                     'gt_text': None,
#                     'pred_text': pred['immatriculation'],
#                     'confiance': pred['confiance'],
#                     'iou': 0,
#                     'is_false_positive': True
#                 })
        
#         return associations
    
#     def evaluate_frame_by_frame(self, iou_threshold: float = 0.5) -> Dict:
#         """
#         Évaluation frame par frame : précision de détection + reconnaissance.
#         """
#         all_frames = set(self.gt_by_frame.keys()) | set(self.pred_by_frame.keys())
        
#         frame_results = []
#         total_tp = 0  # True Positives (GT détecté ET texte correct)
#         total_fn = 0  # False Negatives (GT non détecté)
#         total_fp = 0  # False Positives (détection sans GT)
        
#         for frame in sorted(all_frames):
#             associations = self.associate_by_iou(frame, iou_threshold)
            
#             tp_frame = 0
#             fn_frame = 0
#             fp_frame = 0
            
#             for assoc in associations:
#                 if assoc.get('is_false_positive'):
#                     fp_frame += 1
#                 elif assoc.get('gt_plate_id') is None:
#                     fp_frame += 1
#                 elif not assoc.get('pred_text'):
#                     fn_frame += 1
#                 else:
#                     # Vérifier si l'OCR est correct
#                     sim = plate_similarity(assoc['gt_text'], assoc['pred_text'])
#                     if sim >= 0.85:
#                         tp_frame += 1
#                         assoc['ocr_correct'] = True
#                     else:
#                         fn_frame += 1  # Détecté mais mal reconnu
#                         assoc['ocr_correct'] = False
#                         assoc['similarity'] = sim
            
#             total_tp += tp_frame
#             total_fn += fn_frame
#             total_fp += fp_frame
            
#             frame_results.append({
#                 'frame': frame,
#                 'tp': tp_frame,
#                 'fn': fn_frame,
#                 'fp': fp_frame,
#                 'precision': tp_frame / (tp_frame + fp_frame) if (tp_frame + fp_frame) > 0 else 0,
#                 'recall': tp_frame / (tp_frame + fn_frame) if (tp_frame + fn_frame) > 0 else 0,
#                 'associations': associations
#             })
        
#         # Métriques globales frame par frame
#         precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#         recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         return {
#             'frame_by_frame': frame_results,
#             'total_tp': total_tp,
#             'total_fn': total_fn,
#             'total_fp': total_fp,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }
    
#     def evaluate_aggregated_plates(self, similarity_threshold: float = 0.85) -> Dict:
#         """
#         Évaluation par plaque unique (après fusion/vote du pipeline).
#         Utilise la meilleure association par plaque_id.
#         """
#         # Récupérer toutes les associations par plaque GT
#         plate_associations = defaultdict(list)
        
#         # Parcourir toutes les frames
#         for frame in self.gt_by_frame.keys():
#             associations = self.associate_by_iou(frame, iou_threshold=0.5)
#             for assoc in associations:
#                 if assoc.get('gt_plate_id') is not None and assoc.get('pred_text'):
#                     plate_associations[assoc['gt_plate_id']].append(assoc)
        
#         # Pour chaque plaque GT, prendre la meilleure prédiction (confiance max ou plus fréquente)
#         results = []
#         correct = 0
        
#         for plate_id, assocs in plate_associations.items():
#             # Texte GT (tous identiques pour une même plaque_id)
#             gt_text = assocs[0]['gt_text']
            
#             # Meilleure prédiction : par confiance, puis par similarité
#             best = max(assocs, key=lambda a: (a.get('confiance', 0), 
#                                                plate_similarity(gt_text, a['pred_text'])))
#             pred_text = best['pred_text']
#             sim = plate_similarity(gt_text, pred_text)
#             is_correct = sim >= similarity_threshold
            
#             if is_correct:
#                 correct += 1
            
#             results.append({
#                 'plate_id': plate_id,
#                 'gt_text': gt_text,
#                 'pred_text': pred_text,
#                 'similarity': sim,
#                 'correct': is_correct,
#                 'best_confiance': best.get('confiance', 0),
#                 'n_detections': len(assocs)
#             })
        
#         total_plates = len(plate_associations)
        
#         # Identifier les plaques jamais détectées
#         all_gt_plates = set()
#         for frame, items in self.gt_by_frame.items():
#             for item in items:
#                 all_gt_plates.add(item['plate_id'])
        
#         # Ligne à corriger : detected_plates est un set, pas un list de résultats
#         detected_plates_ids = {r['plate_id'] for r in results}
#         undetected = all_gt_plates - detected_plates_ids  # ← Utiliser detected_plates_ids
        
#         return {
#             'plate_accuracy': correct / total_plates if total_plates > 0 else 0,
#             'total_plates_gt': len(all_gt_plates),
#             'detected_plates': len(results),
#             'correctly_recognized': correct,
#             'undetected_plates_count': len(undetected),
#             'undetected_plates': list(undetected),
#             'per_plate_results': results
#         }
    
#     def evaluate_tracking_consistency(self) -> Dict:
#         """
#         Évalue la stabilité des track_ids (ID switches).
#         """
#         # Suivre pour chaque GT les changements de track_id
#         track_history = defaultdict(lambda: {'last_track_id': None, 'switches': 0, 'frames': []})
        
#         frames = sorted(self.gt_by_frame.keys())
        
#         for frame in frames:
#             associations = self.associate_by_iou(frame, iou_threshold=0.5)
#             for assoc in associations:
#                 if assoc.get('gt_plate_id') is not None and assoc.get('plate_track_id'):
#                     plate_id = assoc['gt_plate_id']
#                     track_id = assoc['plate_track_id']
                    
#                     track_history[plate_id]['frames'].append(frame)
                    
#                     if track_history[plate_id]['last_track_id'] is not None:
#                         if track_history[plate_id]['last_track_id'] != track_id:
#                             track_history[plate_id]['switches'] += 1
                    
#                     track_history[plate_id]['last_track_id'] = track_id
        
#         total_switches = sum(h['switches'] for h in track_history.values())
#         n_tracks_with_switches = sum(1 for h in track_history.values() if h['switches'] > 0)
        
#         return {
#             'total_id_switches': total_switches,
#             'tracks_with_switches': n_tracks_with_switches,
#             'total_tracks': len(track_history),
#             'per_plate_history': dict(track_history)
#         }
    
#     def generate_report(self, output_path: Optional[str] = None) -> Dict:
#         """Génère un rapport complet."""
        
#         print("\n" + "="*60)
#         print("ÉVALUATION DU PIPELINE VIDÉO")
#         print("="*60)
        
#         # 1. Évaluation frame par frame
#         print("\n 1. Évaluation frame par frame (détection + OCR)")
#         frame_results = self.evaluate_frame_by_frame()
#         print(f"   Précision: {frame_results['precision']:.2%}")
#         print(f"   Rappel:    {frame_results['recall']:.2%}")
#         print(f"   F1-score:  {frame_results['f1_score']:.2%}")
#         print(f"   TP: {frame_results['total_tp']}, FN: {frame_results['total_fn']}, FP: {frame_results['total_fp']}")
        
#         # 2. Évaluation par plaque unique
#         print("\n 2. Évaluation par plaque unique (après vote)")
#         plate_results = self.evaluate_aggregated_plates()
#         print(f"   Accuracy plaque: {plate_results['plate_accuracy']:.2%}")
#         print(f"   Plaques GT: {plate_results['total_plates_gt']}")
#         print(f"   Détectées: {plate_results['detected_plates']}")
#         print(f"   Correctement reconnues: {plate_results['correctly_recognized']}")
#         if plate_results['undetected_plates']:
#             print(f"   ⚠️ Plaques non détectées: {plate_results['undetected_plates']}")
        
#         # 3. Stabilité du tracking
#         print("\n 3. Qualité du tracking")
#         tracking_results = self.evaluate_tracking_consistency()
#         print(f"   ID switches totaux: {tracking_results['total_id_switches']}")
#         print(f"   Tracks avec switches: {tracking_results['tracks_with_switches']}/{tracking_results['total_tracks']}")
        
#         report = {
#             'frame_metrics': {
#                 'precision': frame_results['precision'],
#                 'recall': frame_results['recall'],
#                 'f1_score': frame_results['f1_score'],
#                 'tp': frame_results['total_tp'],
#                 'fn': frame_results['total_fn'],
#                 'fp': frame_results['total_fp']
#             },
#             'plate_metrics': plate_results,
#             'tracking_metrics': tracking_results,
#             'frame_details': frame_results['frame_by_frame'],
#             'plate_details': plate_results['per_plate_results']
#         }
        
#         if output_path:
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 json.dump(report, f, indent=2, ensure_ascii=False)
#             print(f"\n Rapport sauvegardé: {output_path}")
        
#         return report


# def main():
#     parser = argparse.ArgumentParser(description="Évaluation du pipeline vidéo")
#     parser.add_argument("--gt", required=True, default="video3_annotatee.csv", help="Chemin du CSV d'annotations (video3_annotatee.csv)")
#     parser.add_argument("--pred", required=True, default="video3_pred.json", help="Chemin du JSON des prédictions (sortie /detect-video)")
#     parser.add_argument("--output", default="evaluation_report.json", help="Fichier de sortie JSON")
    
#     args = parser.parse_args()
    
#     evaluator = VideoEvaluator(args.gt, args.pred)
#     report = evaluator.generate_report(args.output)
    
#     return report



# if __name__ == "__main__":
#     main()