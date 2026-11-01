"""
evaluate_performance.py
Mesure les performances de traitement vidéo
"""

import time
import json
import requests
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def measure_processing_time(video_path: str, api_url: str = "http://localhost:8000/detect-video"):
    """Mesure le temps de traitement complet."""
    
    print("\n" + "="*60)
    print("ÉVALUATION DES PERFORMANCES TEMPORELLES")
    print("="*60)
    
    # 1. Récupérer les infos de la vidéo
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    
    print(f"\n Infos vidéo:")
    print(f"   Fichier: {video_path}")
    print(f"   Frames totales: {total_frames}")
    print(f"   FPS source: {fps:.2f}")
    print(f"   Durée: {duration:.2f} secondes")
    
    # 2. Mesurer le temps de traitement
    print(f"\n Mesure du traitement...")
    
    start = time.time()
    
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_url, files=files)
    
    end = time.time()
    
    if response.status_code != 200:
        print(f" Erreur API: {response.status_code}")
        return None
    
    result = response.json()
    processing_time = end - start
    
    # 3. Afficher les résultats
    print(f"\n Résultats:")
    print(f"   Temps total de traitement: {processing_time:.2f} secondes")
    print(f"   Temps / frame: {(processing_time / total_frames) * 1000:.2f} ms")
    print(f"   FPS de traitement: {total_frames / processing_time:.2f} f/s")
    print(f"   Ratio temps réel: {processing_time / duration:.2f}x")
    
    # Interprétation
    ratio = processing_time / duration
    if ratio <= 1.0:
        print(f"\n Traitement en temps réel (x{ratio:.2f})")
    elif ratio <= 2.0:
        print(f"\n Traitement 2x plus lent que la vidéo (x{ratio:.2f})")
    else:
        print(f"\n Traitement très lent (x{ratio:.2f})")
    
    return {
        'video_duration': duration,
        'total_frames': total_frames,
        'processing_time_sec': processing_time,
        'ms_per_frame': (processing_time / total_frames) * 1000,
        'processing_fps': total_frames / processing_time,
        'real_time_ratio': processing_time / duration
    }


def run_benchmark(video_path: str, n_runs: int = 3):
    """Exécute plusieurs mesures pour obtenir une moyenne."""
    
    times = []
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        result = measure_processing_time(video_path)
        if result:
            times.append(result['processing_time_sec'])
    
    if times:
        avg_time = sum(times) / len(times)
        print("\n" + "="*60)
        print("RÉSUMÉ DU BENCHMARK")
        print("="*60)
        print(f"   Moyenne sur {len(times)} runs: {avg_time:.2f} secondes")
        print(f"   Min: {min(times):.2f}s, Max: {max(times):.2f}s")
        
        return {
            'avg_processing_time': avg_time,
            'min_time': min(times),
            'max_time': max(times),
            'n_runs': len(times)
        }
    
    return None


if __name__ == "__main__":
    result = run_benchmark("video3.mp4", n_runs=3)
    
    # Sauvegarder les résultats
    if result:
        with open("performance_report.json", "w") as f:
            json.dump(result, f, indent=2)