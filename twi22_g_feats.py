import torch
import easygraph as eg
import numpy as np
import os

def get_val(container, idx, default=0.0):
    if isinstance(container, dict):
        return container.get(idx, default)
    elif isinstance(container, (list, np.ndarray)):
        if 0 <= idx < len(container):
            return container[idx]
    return default

def _features_matrix(G):
    nodes = list(G.nodes)
    
    # EasyGraph functions
    deg = G.degree()
    clust = eg.clustering(G)
    close = eg.closeness_centrality(G)
    btw = eg.betweenness_centrality(G)
    pr = eg.pagerank(G)
    
    # Core number
    core = None
    
    # Check for k_core_decomposition
    if hasattr(eg, 'k_core_decomposition'):
        try:
            core = eg.k_core_decomposition(G)
        except TypeError:
            pass # Exists but None/not callable
            
    # Check for core_decomposition if still None
    if core is None and hasattr(eg, 'core_decomposition'):
        try:
            core = eg.core_decomposition(G)
        except TypeError:
            pass # Exists but None/not callable

    # Check for k_core if still None
    if core is None and hasattr(eg, 'k_core'):
        try:
            core = eg.k_core(G)
        except TypeError:
            pass

    # Check for core_number if still None
    if core is None and hasattr(eg, 'core_number'):
        try:
            core = eg.core_number(G)
        except TypeError:
            pass
            
    # Fallback to zeros if everything failed
    if core is None:
        core = {n: 0 for n in nodes}

    # Average neighbor degree
    andeg = None
    if hasattr(eg, 'average_neighbor_degree'):
        try:
            andeg = eg.average_neighbor_degree(G)
        except TypeError:
            pass # Exists but None/not callable
            
    if andeg is None:
        # Fallback calculation if function missing
        andeg = {}
        for n in nodes:
            nbrs = list(G.neighbors(n))
            if len(nbrs) > 0:
                avg = sum(deg[nbr] for nbr in nbrs) / len(nbrs)
                andeg[n] = avg
            else:
                andeg[n] = 0.0

    feats = []
    for n in nodes:
        feats.append([
            float(get_val(deg, n, 0)),
            float(get_val(clust, n, 0.0)),
            float(get_val(close, n, 0.0)),
            float(get_val(btw, n, 0.0)),
            float(get_val(pr, n, 0.0)),
            float(get_val(core, n, 0)),
            float(get_val(andeg, n, 0.0)),
        ])
    return nodes, np.asarray(feats, dtype=np.float32)

def main():
    # 优先使用绝对路径
    path = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.csv'
    if not os.path.exists(path):
        # Fallback
        path = os.path.join(os.path.dirname(__file__), '../dataset/TwiBot22/edge_index.csv')
    
    print(f"Loading graph from {path}...")
    G = eg.GraphC()
    try:
        G.add_edges_from_file(path, weighted=False, is_transform=True)
    except Exception as e:
        print(f"Error loading file directly: {e}")
        pass

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Computing features...")
    nodes, feats = _features_matrix(G)
    
    out = {
        'nodes': torch.as_tensor(nodes, dtype=torch.long),
        'feats': torch.as_tensor(feats),
    }
    torch.save(out, 'twibot22_node_feats.pt')
    print("Features saved to twibot22_node_feats.pt")

if __name__ == '__main__':
    main()
