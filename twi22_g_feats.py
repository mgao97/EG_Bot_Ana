import torch
import networkx as nx
import numpy as np
import os
import csv

def load_edges(path):
    if path.endswith('.csv'):
        with open(path, 'r') as f:
            first_line = f.readline().strip()
        header_tokens = [t.strip().lower() for t in first_line.split(',')]
        if 'relation' in header_tokens or 'source' in header_tokens or 'source_id' in header_tokens:
            u_list = []
            v_list = []
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    s = row[0].strip()
                    t = row[2].strip()
                    if s.startswith('u'):
                        s = s[1:]
                    if t.startswith('u'):
                        t = t[1:]
                    try:
                        u = int(s)
                        v = int(t)
                    except ValueError:
                        continue
                    u_list.append(u)
                    v_list.append(v)
            edge_index = torch.stack([torch.tensor(u_list, dtype=torch.long),
                                      torch.tensor(v_list, dtype=torch.long)], dim=0)
            return edge_index
        else:
            arr = np.loadtxt(path, delimiter=',')
            if arr.ndim != 2:
                raise ValueError('edge_index csv format invalid')
            if arr.shape[0] == 2 and arr.shape[1] != 2:
                edge_index = torch.as_tensor(arr, dtype=torch.long)
            elif arr.shape[1] == 2:
                edge_index = torch.as_tensor(arr.T, dtype=torch.long)
            else:
                raise ValueError('edge_index shape invalid')
            return edge_index
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

def build_nx(edge_index):
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = nx.Graph()
    G.add_edges_from(zip(u, v))
    return G

def _features_matrix(G):
    nodes = list(G.nodes())
    deg = dict(G.degree())
    clust = nx.clustering(G)
    close = nx.closeness_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    pr = nx.pagerank(G)
    core = nx.core_number(G)
    andeg = nx.average_neighbor_degree(G)
    feats = []
    for n in nodes:
        feats.append([
            float(deg.get(n, 0)),
            float(clust.get(n, 0.0)),
            float(close.get(n, 0.0)),
            float(btw.get(n, 0.0)),
            float(pr.get(n, 0.0)),
            float(core.get(n, 0)),
            float(andeg.get(n, 0.0)),
        ])
    return nodes, np.asarray(feats, dtype=np.float32)

def main():
    path = os.path.join(os.path.dirname(__file__), '../dataset/TwiBot22/edge_index.csv')
    path = os.path.normpath(path)
    edge_index = load_edges(path)
    G = build_nx(edge_index)
    nodes, feats = _features_matrix(G)
    out = {
        'nodes': torch.as_tensor(nodes, dtype=torch.long),
        'feats': torch.as_tensor(feats),
    }
    torch.save(out, 'twibot22_node_feats.pt')

if __name__ == '__main__':
    main()
