import os
import csv
import numpy as np
try:
    import easygraph as eg
except Exception:
    eg = None
try:
    import torch
except Exception:
    torch = None

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
        candidates += ['maxd', 'MaxD']
    if name.lower() == 'his':
        candidates += ['his', 'HIS']
    if name.lower() == 'greedy':
        candidates += ['greedy']
    if name.lower() == 'nobe_ga':
        candidates += ['nobe_ga', 'NOBE_GA']
    if name.lower() == 'bicc':
        candidates += ['bicc', 'BICC']
    for c in candidates:
        fn = getattr(sh, c, None)
        if callable(fn):
            return fn
    return None

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
    for a, b in G.edges():
        if a in s and b in s:
            H.add_edge(a, b)
    return H

def sample_nodes(nodes, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, len(nodes))
    return rng.choice(nodes, size=k, replace=False).tolist()

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
    sample = sample_nodes(nodes, 50, 2026)
    Gs = _induce_subgraph(G_full, sample)
    algos = [('HIS', 'his'), ('MaxD', 'maxd'), ('Greedy', 'greedy'), ('AP_Greedy', 'ap_greedy'), ('NOBE_GA', 'nobe_ga'), ('BICC', 'bicc')]
    for label, name in algos:
        fn = _resolve_structural_holes_func(name)
        if fn is None:
            continue
        try:
            res = fn(Gs)
        except TypeError:
            try:
                res = fn(Gs, 5)
            except Exception:
                res = fn(Gs)
        rows = _top5_from_result(res)
        _save_structural_holes_top5('TwiBot-22', label, rows)
        for i, (n, s) in enumerate(rows, 1):
            print(f'{label} Top{i}: {n} {"" if s is None else s}')

if __name__ == '__main__':
    run_structural_holes_twibot22()
