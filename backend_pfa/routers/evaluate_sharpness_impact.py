"""
evaluate_sharpness_impact.py
Mesure l'impact du seuil de netteté sur la qualité et la vitesse
"""

import json
import time
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_sharpness_threshold(video_path: str, thresholds: list, api_url: str = "http://localhost:8000/detect-video"):
    """
    Teste différents seuils de netteté et mesure leur impact.
    
    Args:
        video_path: Chemin de la vidéo
        thresholds: Liste des seuils à tester (ex: [30, 50, 70, 100, 150])
        api_url: URL de l'API
    """
    
    results = []
    
    print("\n" + "="*60)
    print("IMPACT DU SEUIL DE NETTETÉ")
    print("="*60)
    
    for sharp_thresh in thresholds:
        print(f"\n Test avec SHARPNESS_MIN = {sharp_thresh}")
        print("-" * 40)
        
        # Appel API avec paramètre personnalisé
        # Note: Vous devez modifier votre API pour accepter ce paramètre
        start = time.time()
        
        with open(video_path, 'rb') as f:
            files = {'file': f}
            # Ajouter le paramètre sharpness_threshold
            data = {'sharpness_threshold': sharp_thresh}
            response = requests.post(api_url, files=files, data=data)
        
        processing_time = time.time() - start
        
        if response.status_code != 200:
            print(f"   Erreur: {response.status_code}")
            continue
        
        result = response.json()
        
        # Collecter les métriques
        detections = result.get('detections', [])
        avg_confiance = sum(d.get('confiance', 0) for d in detections) / len(detections) if detections else 0
        avg_sharpness = sum(d.get('sharpness_max', 0) for d in detections) / len(detections) if detections else 0
        total_votes = sum(d.get('votes', 0) for d in detections)
        
        print(f"    Détections: {len(detections)}")
        print(f"    Crops analysés: {total_votes}")
        print(f"    Netteté moyenne: {avg_sharpness:.1f}")
        print(f"    Confiance moyenne: {avg_confiance:.4f}")
        print(f"    Temps traitement: {processing_time:.2f}s")
        
        results.append({
            'sharpness_threshold': sharp_thresh,
            'detections_count': len(detections),
            'total_crops_analyzed': total_votes,
            'avg_sharpness': avg_sharpness,
            'avg_confiance': avg_confiance,
            'processing_time': processing_time
        })
    
    return results


def plot_sharpness_impact(results: list):
    """Visualise l'impact du seuil de netteté."""
    
    thresholds = [r['sharpness_threshold'] for r in results]
    detections = [r['detections_count'] for r in results]
    times = [r['processing_time'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Graphique 1: Détections vs seuil
    axes[0].plot(thresholds, detections, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Seuil de netteté (SHARPNESS_MIN)')
    axes[0].set_ylabel('Nombre de détections')
    axes[0].set_title('Impact sur le nombre de détections')
    axes[0].grid(True, alpha=0.3)
    
    # Graphique 2: Temps de traitement vs seuil
    axes[1].plot(thresholds, times, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Seuil de netteté (SHARPNESS_MIN)')
    axes[1].set_ylabel('Temps de traitement (s)')
    axes[1].set_title('Impact sur le temps de traitement')
    axes[1].grid(True, alpha=0.3)
    
    # Graphique 3: Compromis (détections / temps)
    efficiency = [d/t for d, t in zip(detections, times)]
    axes[2].plot(thresholds, efficiency, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Seuil de netteté (SHARPNESS_MIN)')
    axes[2].set_ylabel('Efficacité (détections/seconde)')
    axes[2].set_title('Compromis qualité/vitesse')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sharpness_impact.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n Graphiques sauvegardés: sharpness_impact.png")


def find_optimal_threshold(results: list) -> dict:
    """Trouve le seuil optimal selon différents critères."""
    
    if not results:
        return {}
    
    # Meilleur compromis (score = détections / temps)
    for r in results:
        r['score'] = r['detections_count'] / r['processing_time']
    
    best_compromise = max(results, key=lambda x: x['score'])
    
    # Plus de détections
    best_detection = max(results, key=lambda x: x['detections_count'])
    
    # Plus rapide
    fastest = min(results, key=lambda x: x['processing_time'])
    
    print("\n" + "="*60)
    print(" SEUIL OPTIMAL")
    print("="*60)
    print(f"\n Meilleur compromis (détections/vitesse):")
    print(f"   Seuil = {best_compromise['sharpness_threshold']}")
    print(f"   Détections: {best_compromise['detections_count']}")
    print(f"   Temps: {best_compromise['processing_time']:.2f}s")
    
    print(f"\n Maximum de détections:")
    print(f"   Seuil = {best_detection['sharpness_threshold']}")
    print(f"   Détections: {best_detection['detections_count']}")
    
    print(f"\n Plus rapide:")
    print(f"   Seuil = {fastest['sharpness_threshold']}")
    print(f"   Temps: {fastest['processing_time']:.2f}s")
    
    return {
        'best_compromise': best_compromise,
        'best_detection': best_detection,
        'fastest': fastest
    }


if __name__ == "__main__":
    # Tests avec différents seuils
    thresholds_to_test = [30, 50, 70, 100, 120, 150]
    
    results = test_sharpness_threshold("video3.mp4", thresholds_to_test)
    
    if results:
        plot_sharpness_impact(results)
        optimal = find_optimal_threshold(results)
        
        # Sauvegarder
        with open("sharpness_analysis.json", "w") as f:
            json.dump(results, f, indent=2)