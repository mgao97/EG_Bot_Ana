import networkx as nx
import torch
import numpy as np
import time
import pandas as pd

def load_edges(path):
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj):
        edge_index = obj
    elif isinstance(obj, dict):
        edge_index = obj.get('edge_index', None)
        if edge_index is None:
            for v in obj.values():
                if torch.is_tensor(v) and v.ndim == 2:
                    edge_index = v
                    break
    else:
        raise ValueError('unsupported pt format')
    if edge_index.ndim != 2 or edge_index.shape[0] not in (2,):
        if edge_index.shape[1] == 2:
            edge_index = edge_index.T
        else:
            raise ValueError('edge_index shape invalid')
    return edge_index

def build_graph(edge_index):
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = nx.DiGraph()
    G.add_edges_from(zip(u, v))
    return G

def main():
    path = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.pt'
    edge_index = load_edges(path)
    G = build_graph(edge_index)
    n = G.number_of_nodes()
    k = min(1000, n) if n > 0 else 0
    np.random.seed(2026)
    sample_nodes = np.random.choice(list(G.nodes()), size=k, replace=False) if k > 0 else []
    rows = []
    for node in sample_nodes:
        deg = G.degree(node)
        t0 = time.perf_counter()
        _ = nx.closeness_centrality(G, u=node)
        t1 = time.perf_counter()
        rows.append({'node_id': int(node), 'degree': int(deg), 'runtime': float(t1 - t0)})
    df = pd.DataFrame(rows)
    df.to_csv('cc_runtime_stats.csv', index=False)
    if len(rows) > 0:
        avg_rt = float(np.mean([r['runtime'] for r in rows]))
    else:
        avg_rt = 0.0
    print({'nodes': n, 'edges': G.number_of_edges(), 'k': k, 'seed': 2026, 'avg_runtime_sec': avg_rt})

if __name__ == '__main__':
    main()
