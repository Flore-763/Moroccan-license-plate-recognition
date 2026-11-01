# generate_report_charts.py
import matplotlib.pyplot as plt
import numpy as np

# Données
configs = ['CPU (actuel)', 'GTX 1060', 'RTX 2060', 'RTX 3060', 'RTX 4080']
times = [110, 38, 23, 14, 7]
ratios = [8.0, 2.8, 1.7, 1.0, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Temps de traitement
colors = ['#ff6666', '#ffaa66', '#ffee66', '#66ff66', '#66ffaa']
bars1 = axes[0].bar(configs, times, color=colors)
axes[0].set_ylabel('Temps de traitement (secondes)')
axes[0].set_title('Temps de traitement par configuration')
axes[0].axhline(y=13.65, color='red', linestyle='--', label='Durée vidéo (13.65s)')
axes[0].legend()

for bar, time in zip(bars1, times):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{time}s', ha='center', va='bottom')

# Graphique 2 : Ratio temps réel
bars2 = axes[1].bar(configs, ratios, color=colors)
axes[1].set_ylabel('Ratio (temps traitement / durée vidéo)')
axes[1].set_title('Ratio temps réel')
axes[1].axhline(y=1.0, color='green', linestyle='--', label='Temps réel (1x)')
axes[1].legend()

for bar, ratio in zip(bars2, ratios):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{ratio}x', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Graphiques générés: performance_comparison.png")