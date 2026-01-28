import os
import csv
import numpy as np
import easygraph as eg
import torch
try:
    from easygraph.functions.community import greedy_modularity_communities
except Exception:
    greedy_modularity_communities = None

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

def _resolve_structural_holes_func(name):
    try:
        import easygraph.functions.structural_holes as sh
    except Exception:
        return None
    candidates = [name, name.lower(), name.upper()]
    if name.lower() == 'ap_greedy':
        candidates += ['ap_greedy', 'AP_Greedy', 'apgreedy']
    if name.lower() == 'maxd':
        candidates += ['maxd', 'MaxD', 'get_structural_holes_MaxD']
    if name.lower() == 'his':
        candidates += ['his', 'HIS', 'get_structural_holes_HIS']
    if name.lower() == 'greedy':
        candidates += ['greedy']
    if name.lower() == 'nobe_ga':
        candidates += ['nobe_ga', 'NOBE_GA', 'NOBE_GA_SH']
    if name.lower() == 'nobe':
        candidates += ['nobe', 'NOBE_SH']
    if name.lower() == 'bicc':
        candidates += ['bicc', 'BICC']
    for c in candidates:
        fn = getattr(sh, c, None)
        if callable(fn):
            return fn
    return None

def _communities_for(G):
    if greedy_modularity_communities is None:
        return []
    try:
        comms = list(greedy_modularity_communities(G))
        return [frozenset(c) for c in comms]
    except Exception:
        return []

def _reindex_graph(G, begin_index=1):
    try:
        to_idx = getattr(G, 'to_index_node_graph', None)
        if callable(to_idx):
            Gi, _, _ = to_idx(begin_index=begin_index)
            return Gi
    except Exception:
        pass
    mapping = {}
    idx = begin_index
    for n in G.nodes:
        mapping[n] = idx
        idx += 1
    Gi = eg.Graph()
    for a, b, _ in G.edges:
        Gi.add_edge(mapping[a], mapping[b])
    return Gi

def _call_structural_holes(fn, label, Gs):
    nm = getattr(fn, '__name__', '').lower()
    if 'bicc' in nm:
        return fn(Gs, 5, 10, 4)
    if 'ap_greedy' in nm:
        return fn(Gs, 5)
    if 'nobe_ga_sh' in nm:
        Gi = _reindex_graph(Gs, begin_index=1)
        return fn(Gi, 8, 5)
    if 'nobe_sh' in nm:
        Gi = _reindex_graph(Gs, begin_index=1)
        return fn(Gi, 8, 5)
    if 'get_structural_holes_maxd' in nm or label.lower() == 'maxd':
        C = _communities_for(Gs)
        if not C:
            return []
        return fn(Gs, 5, C)
    if 'get_structural_holes_his' in nm or label.lower() == 'his':
        C = _communities_for(Gs)
        if not C:
            return {}
        S, I, H = fn(Gs, C)
        scores = {}
        for n, d in H.items():
            try:
                scores[n] = max(d.values()) if isinstance(d, dict) and len(d) > 0 else 0
            except Exception:
                scores[n] = 0
        return scores
    return fn(Gs)

def _top5_from_result(res):
    if isinstance(res, dict):
        items = list(res.items())
        try:
            items.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            pass
        pairs = [(k, v) for k, v in items[:5]]
        return pairs
    if isinstance(res, (list, tuple)):
        if len(res) == 0:
            return []
        first = res[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                tmp = sorted(res, key=lambda x: x[1], reverse=True)
            except Exception:
                tmp = res
            return [(x[0], x[1]) for x in tmp[:5]]
        else:
            return [(x, None) for x in res[:5]]
    return []

def _save_structural_holes_top5(dataset, algo, rows):
    path = os.path.join(_results_dir(), 'structural_holes_top5.csv')
    out = [(dataset, algo, i + 1, str(n), '' if s is None else s) for i, (n, s) in enumerate(rows)]
    _append_rows_csv(path, ['dataset', 'algo', 'rank', 'node', 'score'], out)

def _to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.cpu().numpy().tolist()
    return np.asarray(x).tolist()

def _build_eg_from_csv(path):
    if eg is None:
        return None
    with open(path, 'r') as f:
        first = f.readline().strip()
    toks = [t.strip().lower() for t in first.split(',')]
    G = eg.Graph()
    if 'relation' in toks or 'source' in toks or 'source_id' in toks:
        with open(path, 'r') as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in f if header is None else r:
                if isinstance(row, str):
                    parts = [p.strip() for p in row.split(',')]
                else:
                    parts = [p.strip() for p in row]
                if len(parts) < 3:
                    continue
                s = parts[0]
                t = parts[2]
                if s.startswith('u'):
                    s = s[1:]
                if t.startswith('u'):
                    t = t[1:]
                try:
                    su = int(s)
                    tv = int(t)
                except ValueError:
                    continue
                G.add_edge(su, tv)
    else:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim != 2:
            return None
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            u = arr[0].tolist()
            v = arr[1].tolist()
        elif arr.shape[1] == 2:
            u = arr[:, 0].tolist()
            v = arr[:, 1].tolist()
        else:
            return None
        for a, b in zip(u, v):
            G.add_edge(int(a), int(b))
    return G

def _induce_subgraph(G, nodes):
    s = set(nodes)
    H = eg.Graph()
    for a, b, _ in G.edges:
        if a in s and b in s:
            H.add_edge(a, b)
    return H

def _degree_map(G):
    d = {}
    for a, b, _ in G.edges:
        d[a] = d.get(a, 0) + 1
        d[b] = d.get(b, 0) + 1
    return d

def _neighbors_of(G, seed):
    ns = set()
    for a, b, _ in G.edges:
        if a == seed:
            ns.add(b)
        elif b == seed:
            ns.add(a)
    return list(ns)

def _subgraph_via_api(G, nodes):
    try:
        sub = getattr(G, 'subgraph', None)
        if callable(sub):
            return sub(nodes)
    except Exception:
        pass
    try:
        api = getattr(eg, 'subgraph', None)
        if callable(api):
            return api(G, nodes)
    except Exception:
        pass
    return _induce_subgraph(G, nodes)

def sample_nodes(nodes, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, len(nodes))
    return rng.choice(nodes, size=k, replace=False).tolist()

def get_twibot22_subgraph():
    if eg is None:
        return eg.Graph()
    path = os.path.join(os.path.dirname(__file__), '../dataset/TwiBot22/edge_index.csv')
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return eg.Graph()
    G_full = _build_eg_from_csv(path)
    if G_full is None:
        return eg.Graph()
    nodes = list(G_full.nodes)
    deg = _degree_map(G_full)
    degree_min = 20
    candidates = [n for n in nodes if deg.get(n, 0) >= degree_min]
    if len(candidates) == 0:
        sample = sample_nodes(nodes, 50, 2026)
        Gs = _induce_subgraph(G_full, sample)
    else:
        seed = max(candidates, key=lambda n: deg.get(n, 0))
        neigh = _neighbors_of(G_full, seed)
        limit = 50
        chosen = neigh[:limit]
        sub_nodes = [seed] + chosen
        Gs = _subgraph_via_api(G_full, sub_nodes)
    return Gs

def run_structural_holes_twibot22():
    if eg is None:
        return
    path = os.path.join(os.path.dirname(__file__), '../dataset/TwiBot22/edge_index.csv')
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return
    G_full = _build_eg_from_csv(path)
    if G_full is None:
        return
    nodes = list(G_full.nodes)
    deg = _degree_map(G_full)
    degree_min = 20
    candidates = [n for n in nodes if deg.get(n, 0) >= degree_min]
    if len(candidates) == 0:
        sample = sample_nodes(nodes, 50, 2026)
        Gs = _induce_subgraph(G_full, sample)
    else:
        seed = max(candidates, key=lambda n: deg.get(n, 0))
        neigh = _neighbors_of(G_full, seed)
        limit = 50
        chosen = neigh[:limit]
        sub_nodes = [seed] + chosen
        Gs = _subgraph_via_api(G_full, sub_nodes)
    algos = [('HIS', 'his'), ('MaxD', 'maxd'), ('AP_Greedy', 'ap_greedy'), ('NOBE_GA', 'nobe_ga'), ('NOBE', 'nobe'), ('BICC', 'bicc')]
    for label, name in algos:
        fn = _resolve_structural_holes_func(name)
        if fn is None:
            continue
        res = _call_structural_holes(fn, label, Gs)
        rows = _top5_from_result(res)
        _save_structural_holes_top5('TwiBot-22', label, rows)
        for i, (n, s) in enumerate(rows, 1):
            print(f'{label} Top{i}: {n}')

if __name__ == '__main__':
    run_structural_holes_twibot22()
