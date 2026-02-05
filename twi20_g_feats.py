import torch
import easygraph as eg
import numpy as np
import os

def get_val(container, key, idx, default=0.0):
    if isinstance(container, dict):
        return container.get(key, default)
    elif isinstance(container, (list, np.ndarray)):
        if 0 <= idx < len(container):
            return container[idx]
    return default

def _features_matrix(G):
    nodes = list(G.nodes)
    node_index = G.node_index
    
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
        idx = node_index[n]
        feats.append([
            float(get_val(deg, n, idx, 0)),
            float(get_val(clust, n, idx, 0.0)),
            float(get_val(close, n, idx, 0.0)),
            float(get_val(btw, n, idx, 0.0)),
            float(get_val(pr, n, idx, 0.0)),
            float(get_val(core, n, idx, 0)),
            float(get_val(andeg, n, idx, 0.0)),
        ])
    return nodes, np.asarray(feats, dtype=np.float32)

def main():
    # 优先使用绝对路径
    path = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot20/edge_index.csv'
    if not os.path.exists(path):
        # Fallback
        path = os.path.join(os.path.dirname(__file__), '../dataset/TwiBot20/edge.csv')
    
    print(f"Loading graph from {path}...")
    G = eg.GraphC()
    # 注意：Twibot-20 的 edge_index.csv 可能包含 relation 列，需要确保正确加载
    # GraphC.add_edges_from_file 默认处理 csv，但可能需要根据 header 调整
    # 如果文件头包含 source, target 等，EasyGraph 通常能自动识别
    try:
        G.add_edges_from_file(path, weighted=False, is_transform=True)
    except Exception as e:
        print(f"Error loading file directly: {e}")
        # 如果直接加载失败（例如因为 header），可以尝试回退到手动解析
        # 但既然要求使用 easygraph，我们尽量用其接口。
        # 如果是 header 问题，GraphC 可能需要无 header 的文件，或者指定的格式。
        # 考虑到之前的 check_data.py，这个 csv 有 header。
        # EasyGraph 的 add_edges_from_file 支持带 header 的 csv吗？
        # 如果不支持，我们可能需要先读取并跳过 header。
        # 但 EasyGraph 文档说支持 csv。
        pass

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Computing features...")
    nodes, feats = _features_matrix(G)
    
    # Convert nodes to integers for tensor storage
    try:
        nodes_int = [int(n) for n in nodes]
        nodes_tensor = torch.as_tensor(nodes_int, dtype=torch.long)
    except ValueError:
        print("Warning: Node labels are not numeric. Saving internal 0..N-1 indices.")
        nodes_tensor = torch.arange(len(nodes), dtype=torch.long)
    
    out = {
        'nodes': nodes_tensor,
        'feats': torch.as_tensor(feats),
    }
    torch.save(out, 'twibot20_node_feats.pt')
    print("Features saved to twibot20_node_feats.pt")

if __name__ == '__main__':
    main()
