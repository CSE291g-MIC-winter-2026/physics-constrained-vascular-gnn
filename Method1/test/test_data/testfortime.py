import sys, os
sys.path.append(r'C:\Users\30691\Desktop\cse291\code\gROM-main')
import dgl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

graphs_dir = r'C:\Users\30691\Desktop\cse291\code\gROM-main\graphs\\'
graphs, _ = dgl.load_graphs(graphs_dir + '0003_0001.1.0.grph')
graph = graphs[0]

x      = graph.ndata['x'].numpy()
inlet  = graph.ndata['inlet_mask'].numpy().astype(bool)
outlet = graph.ndata['outlet_mask'].numpy().astype(bool)
degrees = (graph.in_degrees() + graph.out_degrees()).numpy()
junction = degrees > 6
area   = graph.ndata['area'].numpy().flatten()
area_norm = (area - area.min()) / (area.max() - area.min() + 1e-8)

src_all, dst_all = graph.edges()
src_all, dst_all = src_all.numpy(), dst_all.numpy()
N = graph.num_nodes()

# 建稀疏距离矩阵，用最小生成树提取骨架
edge_len = np.linalg.norm(x[src_all] - x[dst_all], axis=1)
mat = csr_matrix((edge_len, (src_all, dst_all)), shape=(N, N))
mst = minimum_spanning_tree(mat)
mst = mst + mst.T  # 对称化
cx = mst.tocoo()

fig, ax = plt.subplots(figsize=(8, 12), facecolor='#0a0e1a')
ax.set_facecolor('#0a0e1a')

# 画边，粗细 = 截面积
for s, d in zip(cx.row, cx.col):
    if s >= d: continue
    w = 1.0 + area_norm[s] * 6
    ax.plot([x[s,0], x[d,0]], [x[s,1], x[d,1]],
            color='#38bdf8', linewidth=w, alpha=0.8, solid_capstyle='round')

# 节点
ax.scatter(x[~junction & ~inlet & ~outlet, 0],
           x[~junction & ~inlet & ~outlet, 1],
           c='#93c5fd', s=6, zorder=3, alpha=0.6)
if junction.sum() > 0:
    ax.scatter(x[junction,0], x[junction,1],
               c='#f59e0b', s=80, marker='D', zorder=5,
               label=f'Junction ({junction.sum()})', edgecolors='white', lw=0.5)
ax.scatter(x[inlet,0], x[inlet,1],
           c='#ef4444', s=200, marker='o', zorder=6,
           label='Inlet', edgecolors='white', lw=1)
ax.scatter(x[outlet,0], x[outlet,1],
           c='#fbbf24', s=120, marker='o', zorder=6,
           label=f'Outlet ({outlet.sum()})', edgecolors='white', lw=0.5)

ax.set_title('0003_0001.1.0\nVascular Network', color='white',
             fontsize=13, fontweight='bold')
ax.tick_params(colors='#333', labelsize=7)
for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2d45')
for sp in ['top','right']:   ax.spines[sp].set_visible(False)
ax.legend(fontsize=9, facecolor='#111827', edgecolor='#1e2d45', labelcolor='white')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('vascular_tree.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('saved: vascular_tree.png')
plt.show()