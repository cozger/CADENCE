"""Plot Emotiv EPOC 14-ch electrode positions with ROI coloring."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
pos = np.array([
    [-0.31, 0.95],  # AF3
    [-0.81, 0.59],  # F7
    [-0.39, 0.69],  # F3
    [-0.67, 0.35],  # FC5
    [-1.00, 0.00],  # T7
    [-0.81,-0.59],  # P7
    [-0.31,-0.95],  # O1
    [ 0.31,-0.95],  # O2
    [ 0.81,-0.59],  # P8
    [ 1.00, 0.00],  # T8
    [ 0.67, 0.35],  # FC6
    [ 0.39, 0.69],  # F4
    [ 0.81, 0.59],  # F8
    [ 0.31, 0.95],  # AF4
])

rois = {
    'frontal': [0, 2, 11, 13],
    'left_temp': [1, 3, 4],
    'right_temp': [10, 12, 9],
    'posterior': [5, 6, 7, 8],
}
colors = {
    'frontal': '#e74c3c',
    'left_temp': '#3498db',
    'right_temp': '#2ecc71',
    'posterior': '#9b59b6',
}

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Head outline
head = plt.Circle((0, 0), 1.1, fill=False, linewidth=2, color='gray')
ax.add_patch(head)
# Nose
ax.plot([-0.08, 0, 0.08], [1.1, 1.22, 1.1], 'k-', lw=2)
# Ears
ax.plot([-1.15, -1.1], [0, 0], 'k-', lw=2)
ax.plot([1.1, 1.15], [0, 0], 'k-', lw=2)

# Adjacency lines
dist = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(axis=2))
for i in range(14):
    for j in range(i + 1, 14):
        if dist[i, j] < 0.65:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', alpha=0.15, lw=1, zorder=1)

# Electrodes
for roi, idxs in rois.items():
    c = colors[roi]
    for i in idxs:
        circle = plt.Circle(pos[i], 0.07, color=c, alpha=0.3, zorder=2)
        ax.add_patch(circle)
        ax.plot(pos[i][0], pos[i][1], 'o', color=c, markersize=14, zorder=3)
        ax.annotate(f'{names[i]} ({i})', pos[i], textcoords='offset points',
                    xytext=(10, 8), fontsize=10, fontweight='bold', zorder=4)

# Legend
legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
              markersize=10, label=r) for r, c in colors.items()]
ax.legend(handles=legend_els, loc='lower left', fontsize=11)

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.3, 1.4)
ax.set_aspect('equal')
ax.set_title('Emotiv EPOC 14-ch: Electrode Positions & ROIs\n(nose up, left=left)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig('results/epoc_electrode_map.png', dpi=150, bbox_inches='tight')
print('Saved to results/epoc_electrode_map.png')
