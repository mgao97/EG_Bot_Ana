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
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'relation', 'target_id'])
        for su, tv in zip(u, v):
            writer.writerow([f'{id_prefix}{su}', relation, f'{id_prefix}{tv}'])

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
    tasks = [
        ('cc', 'nx', 'algo_cc_nx(G_nx, sample)'),
    ]
    if eg is not None:
        tasks.append(('cc', 'eg', 'algo_cc_eg(G_eg, sample)'))
    if ig is not None:
        tasks.append(('cc', 'ig', 'algo_cc_ig(G_ig, sample_idx)'))
    tasks += [
        ('bc', 'nx', 'algo_bc_nx(G_nx, sample)'),
    ]
    if eg is not None:
        tasks.append(('bc', 'eg', 'algo_bc_eg(G_eg, sample)'))
    if ig is not None:
        tasks.append(('bc', 'ig', 'algo_bc_ig(G_ig, sample_idx)'))
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
    for algo, lib, stmt in tasks:
        times = benchmark_runs(stmt, g, runs=5)
        save_benchmark_results(name, algo, lib, times)

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

def main():
    for d in ['../dataset/TwiBot22', '../dataset/TwiBot20', '../dataset/MGTAB']:
        ensure_csv_in_dir(d)
    datasets = [
        ('TwiBot-22', '../dataset/TwiBot22/edge_index.csv'),
        ('TwiBot-20', '../dataset/TwiBot20/edge_index.csv'),
        ('MGTAB', '../dataset/MGTAB/edge_index.csv'),
    ]
    # for name, path in datasets:
    #     run_dataset(name, path)
    # run_structural_holes_twibot22()

if __name__ == '__main__':
    main()
