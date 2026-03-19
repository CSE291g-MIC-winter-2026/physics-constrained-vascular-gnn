import sys
sys.path.append(r'C:\Users\30691\Desktop\cse291\code\gROM-main')
import dgl
import torch as th
import numpy as np
from network1d.meshgraphnet import MeshGraphNet
from network1d.rollout import rollout
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import json

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# 加载参数
with open(r'C:\Users\30691\Desktop\cse291\code\gROM-main\models\11.03.2026_22.05.27\parameters.json') as f:
    params = json.load(f)

# 加载模型
model = MeshGraphNet(params).to(device)
model.load_state_dict(th.load(
    r'C:\Users\30691\Desktop\cse291\code\gROM-main\models\11.03.2026_22.05.27\trained_gnn.pms',
    map_location=device))
model.eval()

# 用和训练完全一样的方式生成数据集
input_dir = r'C:\Users\30691\Desktop\cse291\code\gROM-main\graphs\\'
norm_type = {'features': 'normal', 'labels': 'normal'}
info = json.load(open(input_dir + 'dataset_info.json'))

types_to_keep = ['synthetic_aorta_coarctation',
                 'synthetic_pulmonary',
                 'synthetic_aortofemoral']

graphs, _ = gng.generate_normalized_graphs(
    input_dir, norm_type, 'physiological',
    {'dataset_info': info, 'types_to_keep': types_to_keep})

datasets = dset.generate_dataset(graphs, params, info, nchunks=5)
dataset = datasets[0]
for i in range(3):
    i= i * 40
    graph = dataset['test'].graphs[i]
    graph_name = dataset['test'].graph_names[i]
    print(f'Graph {i}:', graph_name)
    
    pred_traj, errors, _, _, _ = rollout(model, params, graph)
    
    true_p = graph.ndata['pressure'].numpy()
    pred_p = pred_traj[:, 0, :]
    inlet  = graph.ndata['inlet_mask'].numpy()
    outlet = graph.ndata['outlet_mask'].numpy()
    
    graphs_raw, _ = dgl.load_graphs(input_dir + graph_name)
    x = graphs_raw[0].ndata['x'].numpy()
    
    np.save(f'true_p_{i}.npy', true_p)
    np.save(f'pred_p_{i}.npy', pred_p)
    np.save(f'inlet_{i}.npy',  inlet)
    np.save(f'outlet_{i}.npy', outlet)
    np.save(f'coords_{i}.npy', x)
    print(f'  saved graph {i}')
    
    