import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from matplotlib.lines import Line2D

def build_mst(x):
    N = x.shape[0]
    src = np.repeat(np.arange(N), N)
    dst = np.tile(np.arange(N), N)
    dist = np.linalg.norm(x[src] - x[dst], axis=1)
    mat = csr_matrix((dist, (src, dst)), shape=(N,N))
    return minimum_spanning_tree(mat).tocoo()

def draw_panel(ax, x, val, mst, inlet, outlet, cmap_name, vmin, vmax, title):
    ax.set_facecolor('white')
    ax.axis('off')
    cmap = plt.cm.get_cmap(cmap_name)
    for s, d in zip(mst.row, mst.col):
        mv = (val[s] + val[d]) / 2
        c = cmap((mv - vmin) / (vmax - vmin + 1e-8))
        ax.plot([x[s,0], x[d,0]], [x[s,1], x[d,1]],
                color=c, lw=2.5, zorder=2, solid_capstyle='round')
    sc = ax.scatter(x[:,0], x[:,1], c=val, cmap=cmap_name,
                    vmin=vmin, vmax=vmax, s=18, zorder=3, edgecolors='none')
    im = inlet.flatten().astype(bool)
    om = outlet.flatten().astype(bool)
    if im.sum() > 0:
        ax.scatter(x[im,0], x[im,1], s=130, c='#22d3ee', marker='^',
                   zorder=6, edgecolors='black', linewidths=0.8)
    if om.sum() > 0:
        ax.scatter(x[om,0], x[om,1], s=80, c='#f97316', marker='v',
                   zorder=6, edgecolors='black', linewidths=0.6)
    ax.set_title(title, fontsize=10, fontweight='bold', color='#0f172a', pad=6)
    ax.set_aspect('equal')
    return sc

fig, axes = plt.subplots(3, 3, figsize=(15, 13), facecolor='white')
fig.suptitle('Pressure Distribution: CFD vs PC Model Prediction',
             fontsize=14, fontweight='bold', color='#0f172a', y=1.01)
current_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(3):
    x      = np.load(os.path.join(current_dir, f'coords_{i}.npy'))
    true_p = np.load(os.path.join(current_dir, f'true_p_{i}.npy'))[:, 0, :]
    pred_p = np.load(os.path.join(current_dir, f'pred_p_{i}.npy'))
    inlet  = np.load(os.path.join(current_dir, f'inlet_{i}.npy'))
    outlet = np.load(os.path.join(current_dir, f'outlet_{i}.npy'))

    t = true_p.shape[1] // 2
    tp = true_p[:, t]
    pp = pred_p[:, t]
    err = np.abs(tp - pp)
    mst = build_mst(x)
    vmin = min(tp.min(), pp.min())
    vmax = max(tp.max(), pp.max())

    sc1 = draw_panel(axes[i,0], x, tp,  mst, inlet, outlet, 'plasma', vmin, vmax, f'Graph {i+1} — CFD (Ground Truth)')
    sc2 = draw_panel(axes[i,1], x, pp,  mst, inlet, outlet, 'plasma', vmin, vmax, f'Graph {i+1} — PC Model Prediction')
    sc3 = draw_panel(axes[i,2], x, err, mst, inlet, outlet, 'Reds',   0, err.max(), f'Graph {i+1} — Absolute Error')

    for ax, sc, label in zip(axes[i], [sc1,sc2,sc3], ['Pressure','Pressure','|Error|']):
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(label, fontsize=8, color='#475569')
        cbar.ax.tick_params(labelsize=7, colors='#475569')
        cbar.outline.set_edgecolor('#e2e8f0')

legend_elements = [
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#22d3ee',
           markeredgecolor='black', markersize=10, label='Inlet'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor='#f97316',
           markeredgecolor='black', markersize=9, label='Outlet'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=11, facecolor='white', edgecolor='#cbd5e1',
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('/home/claude/pressure_all.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('done')
