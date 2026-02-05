import timeit
import statistics
import torch
import networkx as nx
import easygraph as eg
import numpy as np
import igraph as ig
import os
import csv
# from shs_eg import run_structural_holes_twibot22

def _make_progress(prefix, total):
    state = {'cur': 0, 'total': max(0, int(total))}
    def _render():
        t = state['total']
        c = state['cur']
        width = 30
        filled = int(width * (0 if t == 0 else c / t))
        bar = '[' + '#' * filled + '-' * (width - filled) + ']'
        pct = 0 if t == 0 else int(100 * c / t)
        print(f'\r{prefix} {bar} {c}/{t} {pct}%', end='', flush=True)
    def advance():
        state['cur'] += 1
        _render()
        if state['cur'] >= state['total']:
            print()
    _render()
    return advance


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

def benchmark_runs(stmt, globals, runs=5):
    times = timeit.repeat(stmt, number=1, globals=globals, repeat=runs)
    print('Function:', stmt)
    print('  --------------')
    print(f'  Min:      {min(times)}')
    print(f'  Median:   {statistics.median(times)}')
    print(f'  Mean:     {statistics.mean(times)}')
    print(f'  Stdev:    {statistics.stdev(times) if len(times) > 1 else 0.0}')
    print(f'  Max:      {max(times)}')
    print('  --------------')
    print(f'  samples:  {len(times)}')
    print()
    return times
    
def _results_dir():
    return os.path.join(os.path.dirname(__file__), 'results')
def _append_rows_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)
def save_benchmark_results(dataset, algo, lib, times):
    raw_path = os.path.join(_results_dir(), 'bench_raw.csv')
    summary_path = os.path.join(_results_dir(), 'bench_summary.csv')
    raw_rows = [(dataset, algo, lib, i + 1, t) for i, t in enumerate(times)]
    _append_rows_csv(raw_path, ['dataset', 'algo', 'lib', 'run', 'time'], raw_rows)
    m = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    srow = [(dataset, algo, lib, len(times), m, sd, min(times), max(times))]
    _append_rows_csv(summary_path, ['dataset', 'algo', 'lib', 'count', 'mean', 'stdev', 'min', 'max'], srow)

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

def save_edges_csv(edge_index, csv_path, relation='friend', id_prefix='u'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    def _to_list(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return np.asarray(x).tolist()
    u = _to_list(edge_index[0])
    v = _to_list(edge_index[1])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'relation', 'target_id'])
        for su, tv in zip(u, v):
            writer.writerow([f'{id_prefix}{su}', relation, f'{id_prefix}{tv}'])
def save_edges_edgelist(edge_index, txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    def _to_list(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return np.asarray(x).tolist()
    u = _to_list(edge_index[0])
    v = _to_list(edge_index[1])
    with open(txt_path, 'w') as f:
        for a, b in zip(u, v):
            f.write(f'{int(a)} {int(b)}\n')

def convert_labels_in_dir(dir_path):
    for fname in os.listdir(dir_path):
        if fname.startswith('label') and fname.endswith('.pt'):
            fpath = os.path.join(dir_path, fname)
            obj = torch.load(fpath, map_location='cpu')
            if torch.is_tensor(obj):
                lab = obj
            elif isinstance(obj, dict):
                lab = None
                for k in ('label', 'y', 'labels'):
                    v = obj.get(k, None)
                    if torch.is_tensor(v):
                        lab = v
                        break
                if lab is None:
                    for v in obj.values():
                        if torch.is_tensor(v):
                            lab = v
                            break
            else:
                continue
            if lab.ndim > 1 and lab.shape[1] == 1:
                lab = lab.view(-1)
            if lab.ndim != 1:
                continue
            idx = torch.arange(lab.shape[0], dtype=torch.long)
            arr = torch.stack([idx, lab.long()], dim=1).cpu().numpy()
            out_name = fname[:-3] + 'csv'
            out_path = os.path.join(dir_path, out_name)
            np.savetxt(out_path, arr, fmt='%d', delimiter=',')

def build_nx(edge_index):
    def _to_list(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return np.asarray(x).tolist()
    u = _to_list(edge_index[0])
    v = _to_list(edge_index[1])
    G = nx.Graph()
    G.add_edges_from(zip(u, v))
    # Remove self-loops to avoid errors in algorithms that don't support them
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def build_eg(edge_index):
    def _to_list(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return np.asarray(x).tolist()
    u = _to_list(edge_index[0])
    v = _to_list(edge_index[1])
    G = eg.Graph()
    G.add_edges_from(zip(u, v))
    return G

def build_ig(edge_index):
    if ig is None:
        return None, None
    def _to_np(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)
    u = _to_np(edge_index[0])
    v = _to_np(edge_index[1])
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
_MP_G_NX = None
def _init_mp_nx(G):
    global _MP_G_NX
    _MP_G_NX = G
def _work_cc_nx(n):
    nx.closeness_centrality(_MP_G_NX, u=n)
    return 0
def algo_cc_nx_mp(G, nodes):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(processes=os.cpu_count(), initializer=_init_mp_nx, initargs=(G,)) as pool:
        list(pool.imap_unordered(_work_cc_nx, nodes))
    return 0

def algo_cc_eg(G, nodes):
    for n in nodes:
        eg.closeness_centrality(G, n)
    return 0
_MP_G_EG = None
def _init_mp_eg(G):
    global _MP_G_EG
    _MP_G_EG = G
def _work_cc_eg(n):
    eg.closeness_centrality(_MP_G_EG, n)
    return 0
def algo_cc_eg_mp(G, nodes):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(processes=os.cpu_count(), initializer=_init_mp_eg, initargs=(G,)) as pool:
        list(pool.imap_unordered(_work_cc_eg, nodes))
    return 0

def algo_cc_ig(G, idx_list):
    if G is None:
        return 0
    G.closeness(vertices=idx_list)
    return 0

def algo_bc_nx(G, nodes):
    nx.betweenness_centrality_subset(G, sources=nodes, targets=list(G.nodes()), normalized=True)
    return 0
def _work_bc_nx(args):
    G, src = args
    nx.betweenness_centrality_subset(G, sources=[src], targets=list(G.nodes()), normalized=True)
    return 0
def algo_bc_nx_mp(G, nodes):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(processes=os.cpu_count()) as pool:
        list(pool.imap_unordered(_work_bc_nx, [(G, s) for s in nodes]))
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

def algo_cc_nx_all(G):
    """Compute closeness centrality for all nodes using NetworkX."""
    # returns dict but we ignore the result for benchmarking
    nx.closeness_centrality(G)
    return 0

def algo_all_pairs_shortest_paths_nx(G):
    """Compute all-pairs shortest path lengths using NetworkX."""
    # returns dict-of-dict: {source: {target: distance}}
    dict(nx.all_pairs_shortest_path_length(G))
    return 0

def run_dataset(name, path):
    print(f'===== Dataset: {name} =====')
    if not os.path.exists(path):
        print(f'WARNING: missing path {path}, skip')
        return
    dir_path = os.path.dirname(path)
    if path.endswith('.pt'):
        edge_index_pt = load_edges(path)
        edge_csv = os.path.join(dir_path, 'edge_index.csv')
        nm = name.lower()
        if 'twibot' in nm:
            relation, id_prefix = 'friend', 'u'
        elif 'mgtab' in nm:
            relation, id_prefix = 'friend', ''
        else:
            relation, id_prefix = 'friend', ''
        save_edges_csv(edge_index_pt, edge_csv, relation=relation, id_prefix=id_prefix)
        path = edge_csv
    else:
        pass
    edge_index = load_edges(path)
    G_nx = build_nx(edge_index)
    G_eg = build_eg(edge_index)
    G_ig, id2idx = build_ig(edge_index)
    edgelist_path = os.path.join(dir_path, 'edge_index.edgelist')
    save_edges_edgelist(edge_index, edgelist_path)
    gC = None
    try:
        gC = eg.GraphC()
        gC.add_edges_from_file(edgelist_path, weighted=False, is_transform=True)
    except Exception:
        gC = None
    g_ig_edgelist = None
    try:
        g_ig_edgelist = ig.Graph.Read_Edgelist(edgelist_path, False)
    except Exception:
        g_ig_edgelist = None
    nodes = list(G_nx.nodes())
    sample = sample_nodes(nodes, 1000, 2026)
    sample_idx = [id2idx[s] for s in sample] if id2idx is not None else []
    eg_nodes = list(G_eg.nodes())
    eg_node_list = []
    if len(eg_nodes) > 0:
        idxs = sample_nodes(list(range(len(eg_nodes))), min(1000, len(eg_nodes)), 2026)
        for i in idxs:
            eg_node_list.append(eg_nodes[i])
    ig_node_list = []
    if g_ig_edgelist is not None:
        vs = list(range(g_ig_edgelist.vcount()))
        ig_node_list = sample_nodes(vs, min(1000, len(vs)), 2026)
    g = {
        'G_nx': G_nx,
        'G_eg': G_eg,
        'G_ig': G_ig,
        'gC': gC,
        'g_ig_edgelist': g_ig_edgelist,
        'eg_node_list': eg_node_list,
        'ig_node_list': ig_node_list,
        'n_workers': os.cpu_count(),
        'sample': sample,
        'sample_idx': sample_idx,
        'algo_cc_nx': algo_cc_nx,
        'algo_cc_eg': algo_cc_eg,
        'algo_cc_nx_mp': algo_cc_nx_mp,
        'algo_cc_eg_mp': algo_cc_eg_mp,
        'algo_cc_ig': algo_cc_ig,
        'algo_bc_nx': algo_bc_nx,
        'algo_bc_eg': algo_bc_eg,
        'algo_bc_nx_mp': algo_bc_nx_mp,
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
    tasks = [
        ('cc', 'nx', 'algo_cc_nx(G_nx, sample)'),
        ('cc', 'eg', 'algo_cc_eg(G_eg, sample)'),
        ('cc', 'nx_mp', 'algo_cc_nx_mp(G_nx, sample)'),
        ('cc', 'eg_mp', 'algo_cc_eg_mp(G_eg, sample)'),
    ]
    if eg is not None:
        tasks.append(('cc', 'eg', 'algo_cc_eg(G_eg, sample)'))
    if ig is not None:
        tasks.append(('cc', 'ig', 'algo_cc_ig(G_ig, sample_idx)'))
    tasks += [
        ('bc', 'nx', 'algo_bc_nx(G_nx, sample)'),
        ('bc', 'nx_mp', 'algo_bc_nx_mp(G_nx, sample)'),
    ]
    if eg is not None:
        tasks.append(('bc', 'eg', 'algo_bc_eg(G_eg, sample)'))
    if ig is not None:
        tasks.append(('bc', 'ig', 'algo_bc_ig(G_ig, sample_idx)'))
    if eg is not None:
        tasks += [
            ('hierarchy', 'eg_workers', 'eg.hierarchy(G_eg, n_workers=n_workers)'),
            ('clustering', 'eg_workers', 'eg.clustering(G_eg, n_workers=n_workers)'),
            ('cc', 'eg_workers', 'eg.closeness_centrality(G_eg, n_workers=n_workers)'),
            ('bc', 'eg_workers', 'eg.betweenness_centrality(G_eg, n_workers=n_workers)'),
        ]
    if gC is not None:
        tasks += [
            ('dijkstra', 'eg_graphc', 'eg.multi_source_dijkstra(gC, sources=eg_node_list)'),
            ('kcore', 'eg_graphc', 'eg.k_core(gC)'),
            ('bc', 'eg_graphc', 'eg.betweenness_centrality(gC)'),
            ('cc', 'eg_graphc', 'eg.closeness_centrality(gC)'),
        ]
    if g_ig_edgelist is not None:
        tasks += [
            ('distances', 'ig_read', 'g_ig_edgelist.distances(source=ig_node_list, weights=[1]*len(g_ig_edgelist.es))'),
            ('kcore', 'ig_read', 'g_ig_edgelist.coreness()'),
            ('bc', 'ig_read', 'g_ig_edgelist.betweenness(directed=False, weights=[1]*len(g_ig_edgelist.es))'),
            ('cc', 'ig_read', 'g_ig_edgelist.closeness(weights=[1]*len(g_ig_edgelist.es))'),
        ]
    tasks += [
        ('pr', 'nx', 'algo_pr_nx(G_nx)'),
    ]
    if eg is not None:
        tasks.append(('pr', 'eg', 'algo_pr_eg(G_eg)'))
    if ig is not None:
        tasks.append(('pr', 'ig', 'algo_pr_ig(G_ig)'))
    tasks += [
        ('kcore', 'nx', 'algo_kcore_nx(G_nx)'),
    ]
    if eg is not None:
        tasks.append(('kcore', 'eg', 'algo_kcore_eg(G_eg)'))
    if ig is not None:
        tasks.append(('kcore', 'ig', 'algo_kcore_ig(G_ig)'))
    tasks += [
        ('hierarchy', 'nx', 'algo_hierarchy_nx(G_nx)'),
    ]
    if eg is not None:
        tasks.append(('hierarchy', 'eg', 'algo_hierarchy_eg(G_eg)'))
    if ig is not None:
        tasks.append(('hierarchy', 'ig', 'algo_hierarchy_ig(G_ig)'))
    task_progress = _make_progress(f'[{name}] Running tasks', len(tasks))
    for algo, lib, stmt in tasks:
        times = benchmark_runs(stmt, g, runs=5)
        save_benchmark_results(name, algo, lib, times)
        task_progress()

def ensure_csv_in_dir(dir_path):
    if not os.path.isdir(dir_path):
        return
    edge_csv = os.path.join(dir_path, 'edge_index.csv')
    edge_pt = os.path.join(dir_path, 'edge_index.pt')
    if not os.path.exists(edge_csv) and os.path.exists(edge_pt):
        edge_index_pt = load_edges(edge_pt)
        base = os.path.basename(dir_path).lower()
        if 'twibot' in base:
            relation, id_prefix = 'friend', 'u'
        elif 'mgtab' in base:
            relation, id_prefix = 'friend', ''
        else:
            relation, id_prefix = 'friend', ''
        save_edges_csv(edge_index_pt, edge_csv, relation=relation, id_prefix=id_prefix)

def run_dataset_nx_ig_only(name, path):
    """Run only NetworkX and igraph algorithms for the dataset"""
    print(f'===== Dataset: {name} (NX & iGraph only) =====')
    if not os.path.exists(path):
        print(f'WARNING: missing path {path}, skip')
        return
    dir_path = os.path.dirname(path)
    if path.endswith('.pt'):
        edge_index_pt = load_edges(path)
        edge_csv = os.path.join(dir_path, 'edge_index.csv')
        nm = name.lower()
        if 'twibot' in nm:
            relation, id_prefix = 'friend', 'u'
        elif 'mgtab' in nm:
            relation, id_prefix = 'friend', ''
        else:
            relation, id_prefix = 'friend', ''
        save_edges_csv(edge_index_pt, edge_csv, relation=relation, id_prefix=id_prefix)
        path = edge_csv
    else:
        pass
    edge_index = load_edges(path)
    G_nx = build_nx(edge_index)
    G_ig, id2idx = build_ig(edge_index)
    edgelist_path = os.path.join(dir_path, 'edge_index.edgelist')
    save_edges_edgelist(edge_index, edgelist_path)
    g_ig_edgelist = None
    try:
        g_ig_edgelist = ig.Graph.Read_Edgelist(edgelist_path, False)
    except Exception:
        g_ig_edgelist = None
    nodes = list(G_nx.nodes())
    sample = sample_nodes(nodes, 1000, 2026)
    sample_idx = [id2idx[s] for s in sample] if id2idx is not None else []
    ig_node_list = []
    if g_ig_edgelist is not None:
        vs = list(range(g_ig_edgelist.vcount()))
        ig_node_list = sample_nodes(vs, min(1000, len(vs)), 2026)
    g = {
        'G_nx': G_nx,
        'G_ig': G_ig,
        'g_ig_edgelist': g_ig_edgelist,
        'ig_node_list': ig_node_list,
        'sample': sample,
        'sample_idx': sample_idx,
        'algo_cc_nx': algo_cc_nx,
        'algo_cc_nx_mp': algo_cc_nx_mp,
        'algo_cc_ig': algo_cc_ig,
        'algo_bc_nx': algo_bc_nx,
        'algo_bc_nx_mp': algo_bc_nx_mp,
        'algo_bc_ig': algo_bc_ig,
        'algo_pr_nx': algo_pr_nx,
        'algo_pr_ig': algo_pr_ig,
        'algo_cc_nx_all': algo_cc_nx_all,
        'algo_all_pairs_shortest_paths_nx': algo_all_pairs_shortest_paths_nx,
        'algo_kcore_nx': algo_kcore_nx,
        'algo_kcore_ig': algo_kcore_ig,
        'algo_hierarchy_nx': algo_hierarchy_nx,
        'algo_hierarchy_ig': algo_hierarchy_ig,
    }
    # Only NetworkX and igraph tasks
    tasks = [
        ('cc', 'nx', 'algo_cc_nx(G_nx, sample)'),
        ('cc', 'nx_mp', 'algo_cc_nx_mp(G_nx, sample)'),
        ('cc_full', 'nx', 'algo_cc_nx_all(G_nx)'),
        ('all_pairs_sp', 'nx', 'algo_all_pairs_shortest_paths_nx(G_nx)'),
    ]
    if ig is not None:
        tasks.append(('cc', 'ig', 'algo_cc_ig(G_ig, sample_idx)'))
    tasks += [
        ('bc', 'nx', 'algo_bc_nx(G_nx, sample)'),
        ('bc', 'nx_mp', 'algo_bc_nx_mp(G_nx, sample)'),
    ]
    if ig is not None:
        tasks.append(('bc', 'ig', 'algo_bc_ig(G_ig, sample_idx)'))
    if g_ig_edgelist is not None:
        tasks += [
            ('distances', 'ig_read', 'g_ig_edgelist.distances(source=ig_node_list, weights=[1]*len(g_ig_edgelist.es))'),
            ('kcore', 'ig_read', 'g_ig_edgelist.coreness()'),
            ('bc', 'ig_read', 'g_ig_edgelist.betweenness(directed=False, weights=[1]*len(g_ig_edgelist.es))'),
            ('cc', 'ig_read', 'g_ig_edgelist.closeness(weights=[1]*len(g_ig_edgelist.es))'),
        ]
    tasks += [
        ('pr', 'nx', 'algo_pr_nx(G_nx)'),
    ]
    if ig is not None:
        tasks.append(('pr', 'ig', 'algo_pr_ig(G_ig)'))
    tasks += [
        ('kcore', 'nx', 'algo_kcore_nx(G_nx)'),
    ]
    if ig is not None:
        tasks.append(('kcore', 'ig', 'algo_kcore_ig(G_ig)'))
    tasks += [
        ('hierarchy', 'nx', 'algo_hierarchy_nx(G_nx)'),
    ]
    if ig is not None:
        tasks.append(('hierarchy', 'ig', 'algo_hierarchy_ig(G_ig)'))
    task_progress = _make_progress(f'[{name}] Running tasks', len(tasks))
    for algo, lib, stmt in tasks:
        times = benchmark_runs(stmt, g, runs=5)
        save_benchmark_results(name, algo, lib, times)
        task_progress()

def build_eg(edge_index):
    def _to_list(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return np.asarray(x).tolist()
    u = _to_list(edge_index[0])
    v = _to_list(edge_index[1])
    G = eg.Graph()
    G.add_edges_from(zip(u, v))
    return G

def run_dataset_eg_only(name, path):
    """使用多进程运行 EasyGraph 相关的算法测试"""
    print(f'===== Dataset: {name} (EasyGraph Multiprocessing) =====')
    if not os.path.exists(path):
        print(f'WARNING: missing path {path}, skip')
        return

    # 1. 加载数据并构建 EasyGraph 图对象 (从文件)
    dir_path = os.path.dirname(path)
    edgelist_path = os.path.join(dir_path, 'edge_index.edgelist')
    
    # 确保有 edgelist 文件
    if not os.path.exists(edgelist_path):
        edge_index = load_edges(path)
        save_edges_edgelist(edge_index, edgelist_path)
    
    # 从 edgelist 文件加载图（推荐方式）
    g = None
    try:
        g = eg.Graph()
        g.add_edges_from_file(edgelist_path, weighted=False)
    except Exception as e:
        print(f"Error loading graph from file: {e}")
        return
    
    if g is None:
        print("WARNING: Failed to create EasyGraph, skipping benchmarks")
        return

    # 2. 定义要测试的 n_workers 值
    worker_list = [8, 16, 32]
    
    # 3. 定义 EasyGraph 多进程算法任务
    algorithms = [
        ('clustering', 'eg.clustering(g, n_workers=n_workers)'),
        ('hierarchy', 'eg.hierarchy(g, n_workers=n_workers)'),
        ('cc', 'eg.closeness_centrality(g, n_workers=n_workers)'),
        ('bc', 'eg.betweenness_centrality(g, n_workers=n_workers)'),
    ]
    
    # 4. 循环不同的 worker 数量进行基准测试
    for n_workers in worker_list:
        print(f"\n========== n_workers = {n_workers} ==========")
        
        # 为每个 n_workers 值创建上下文
        g_context = {
            'eg': eg,
            'g': g,
            'n_workers': n_workers,
        }
        
        for algo_name, stmt in algorithms:
            print(f"\n========{algo_name}========")
            try:
                times = benchmark_runs(stmt, g_context, runs=5)
                save_benchmark_results(name, algo_name, f'eg_mp_{n_workers}', times)
            except Exception as e:
                print(f"Error running {stmt}: {e}")

def main():
    # 设定数据集路径
    dataset_name = 'TwiBot20'
    dataset_path = '../dataset/TwiBot20/edge.csv'
    
    # 确保 CSV 存在 (如果原始是 .pt)
    ensure_csv_in_dir('../dataset/TwiBot20')
    
    # 仅运行 EasyGraph 测试
    run_dataset_eg_only(dataset_name, dataset_path)

if __name__ == '__main__':
    main()
