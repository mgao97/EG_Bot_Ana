import timeit
import statistics
import torch
import networkx as nx
import easygraph as eg
import numpy as np
import igraph as ig


def benchmark(stmt, n, globals):
    times = timeit.repeat(stmt, number=1, globals=globals, repeat=n)
    print('Function:', stmt)
    print('  --------------')
    print(f'  Min:      {min(times)}')
    print(f'  Median:   {statistics.median(times)}')
    print(f'  Mean:     {statistics.mean(times)}')
    print(f'  Stdev:    {statistics.stdev(times) if len(times) > 1 else "N.A."}')
    print(f'  Max:      {max(times)}')
    print('  --------------')
    print(f'  samples:  {len(times)}')
    print()

def benchmark_autorange(stmt, globals, n):
    timer = timeit.Timer(stmt, globals=globals)
    if n is None:
        count, total_time = timer.autorange()
    else:
        count = n
        total_time = timer.timeit(number=n)
    print('Function:', stmt)
    print('  --------------')
    print(f'  Total time: {total_time}')
    print(f'  Count:      {count}')
    print(f'  Mean:       {(avg_time := total_time / count)}')
    return avg_time

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

def build_nx(edge_index):
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = nx.Graph()
    G.add_edges_from(zip(u, v))
    return G

def build_eg(edge_index):
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = eg.Graph()
    G.add_edges_from(zip(u, v))
    return G

def build_ig(edge_index):
    if ig is None:
        return None, None
    u = edge_index[0].cpu().numpy()
    v = edge_index[1].cpu().numpy()
    nodes = np.unique(np.concatenate([u, v]))
    id2idx = {int(n): i for i, n in enumerate(nodes.tolist())}
    pairs = [(id2idx[int(a)], id2idx[int(b)]) for a, b in zip(u.tolist(), v.tolist())]
    G = ig.Graph(directed=False)
    G.add_vertices(len(nodes))
    G.add_edges(pairs)
    return G, id2idx

def sample_nodes(nodes, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, len(nodes))
    return rng.choice(nodes, size=k, replace=False).tolist()

def algo_cc_nx(G, nodes):
    for n in nodes:
        nx.closeness_centrality(G, u=n)
    return 0

def algo_cc_eg(G, nodes):
    for n in nodes:
        eg.closeness_centrality(G, n)
    return 0

def algo_cc_ig(G, idx_list):
    if G is None:
        return 0
    G.closeness(vertices=idx_list)
    return 0

def algo_bc_nx(G, nodes):
    nx.betweenness_centrality_subset(G, sources=nodes, targets=list(G.nodes()), normalized=True)
    return 0

def algo_bc_eg(G, nodes):
    eg.betweenness_centrality(G)
    return 0

def algo_bc_ig(G, idx_list):
    if G is None:
        return 0
    G.betweenness(vertices=idx_list)
    return 0

def algo_pr_nx(G):
    nx.pagerank(G)
    return 0

def algo_pr_eg(G):
    eg.pagerank(G)
    return 0

def algo_pr_ig(G):
    if G is None:
        return 0
    G.pagerank()
    return 0

def algo_kcore_nx(G):
    nx.core_number(G)
    return 0

def algo_kcore_eg(G):
    eg.core_number(G)
    return 0

def algo_kcore_ig(G):
    if G is None:
        return 0
    G.coreness()
    return 0

def algo_hierarchy_nx(G):
    nx.degree_assortativity_coefficient(G)
    return 0

def algo_hierarchy_eg(G):
    eg.degree_assortativity_coefficient(G)
    return 0

def algo_hierarchy_ig(G):
    if G is None:
        return 0
    G.assortativity_degree()
    return 0

def run_dataset(name, path):
    print(f'===== Dataset: {name} =====')
    edge_index = load_edges(path)
    G_nx = build_nx(edge_index)
    G_eg = build_eg(edge_index)
    G_ig, id2idx = build_ig(edge_index)
    nodes = list(G_nx.nodes())
    sample = sample_nodes(nodes, 1000, 2026)
    sample_idx = [id2idx[s] for s in sample] if id2idx is not None else []
    g = {
        'G_nx': G_nx,
        'G_eg': G_eg,
        'G_ig': G_ig,
        'sample': sample,
        'sample_idx': sample_idx,
        'algo_cc_nx': algo_cc_nx,
        'algo_cc_eg': algo_cc_eg,
        'algo_cc_ig': algo_cc_ig,
        'algo_bc_nx': algo_bc_nx,
        'algo_bc_eg': algo_bc_eg,
        'algo_bc_ig': algo_bc_ig,
        'algo_pr_nx': algo_pr_nx,
        'algo_pr_eg': algo_pr_eg,
        'algo_pr_ig': algo_pr_ig,
        'algo_kcore_nx': algo_kcore_nx,
        'algo_kcore_eg': algo_kcore_eg,
        'algo_kcore_ig': algo_kcore_ig,
        'algo_hierarchy_nx': algo_hierarchy_nx,
        'algo_hierarchy_eg': algo_hierarchy_eg,
        'algo_hierarchy_ig': algo_hierarchy_ig,
    }
    benchmark_autorange('algo_cc_nx(G_nx, sample)', g, None)
    benchmark_autorange('algo_cc_eg(G_eg, sample)', g, None)
    if ig is not None:
        benchmark_autorange('algo_cc_ig(G_ig, sample_idx)', g, None)
    benchmark_autorange('algo_bc_nx(G_nx, sample)', g, None)
    benchmark_autorange('algo_bc_eg(G_eg, sample)', g, None)
    if ig is not None:
        benchmark_autorange('algo_bc_ig(G_ig, sample_idx)', g, None)
    benchmark_autorange('algo_pr_nx(G_nx)', g, None)
    benchmark_autorange('algo_pr_eg(G_eg)', g, None)
    if ig is not None:
        benchmark_autorange('algo_pr_ig(G_ig)', g, None)
    benchmark_autorange('algo_kcore_nx(G_nx)', g, None)
    benchmark_autorange('algo_kcore_eg(G_eg)', g, None)
    if ig is not None:
        benchmark_autorange('algo_kcore_ig(G_ig)', g, None)
    benchmark_autorange('algo_hierarchy_nx(G_nx)', g, None)
    benchmark_autorange('algo_hierarchy_eg(G_eg)', g, None)
    if ig is not None:
        benchmark_autorange('algo_hierarchy_ig(G_ig)', g, None)

def main():
    datasets = [
        ('TwiBot-22', '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.pt'),
        ('TwiBot-20', '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot20/edge_index.pt'),
        ('MGTAB', '/NVMeDATA/gxj_data/hyperscan_cikm25/mgtab/edge_index.pt'),
    ]
    for name, path in datasets:
        run_dataset(name, path)

if __name__ == '__main__':
    main()
