
import csv
import time
import random

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import torch
except Exception:
    torch = None
import os

from easygraph.functions.community import greedy_modularity_communities
from easygraph.functions.community import modularity
from easygraph.functions.graph_embedding import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from datetime import datetime
import argparse

warnings.filterwarnings("ignore")
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/sim/simsun.ttc')

CHN_FONT = 'SimSun'
ROMAN_FONT = 'Times New Roman'

def _emb_matrix(embeddings, nodes_order, dim_hint=None):
    rows = []
    if isinstance(embeddings, dict):
        for n in nodes_order:
            v = embeddings.get(n)
            if v is None:
                v = embeddings.get(str(n))
            if v is None:
                if dim_hint is None:
                    dim_hint = len(next(iter(embeddings.values())))
                v = np.zeros(dim_hint)
            rows.append(np.asarray(v))
        return np.vstack(rows)
    try:
        if dim_hint is None and len(embeddings) > 0:
            try:
                dim_hint = len(embeddings[0])
            except Exception:
                dim_hint = None
        if set(nodes_order) == set(range(len(embeddings))):
            for n in nodes_order:
                rows.append(np.asarray(embeddings[n]))
            return np.vstack(rows)
        return np.asarray(list(embeddings))
    except Exception:
        return np.asarray(list(embeddings))


def _build_eg_from_csv(path):
    G = eg.Graph()
    with open(path, 'r') as f:
        first = f.readline().strip()
    toks = [t.strip().lower() for t in first.split(',')]
    if 'relation' in toks or 'source' in toks or 'source_id' in toks:
        with open(path, 'r') as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row or len(row) < 3:
                    continue
                s = row[0].strip()
                t = row[2].strip()
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
            return G
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            u = arr[0].tolist()
            v = arr[1].tolist()
        elif arr.shape[1] == 2:
            u = arr[:, 0].tolist()
            v = arr[:, 1].tolist()
        else:
            return G
        for a, b in zip(u, v):
            G.add_edge(int(a), int(b))
    return G

def _build_eg_from_pt(path):
    obj = torch.load(path, map_location='cpu') if torch is not None else None
    if obj is None:
        return eg.Graph()
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
        return eg.Graph()
    if edge_index.ndim != 2:
        return eg.Graph()
    if edge_index.shape[0] == 2:
        u = edge_index[0].cpu().numpy().tolist()
        v = edge_index[1].cpu().numpy().tolist()
    elif edge_index.shape[1] == 2:
        u = edge_index[:, 0].cpu().numpy().tolist()
        v = edge_index[:, 1].cpu().numpy().tolist()
    else:
        return eg.Graph()
    G = eg.Graph()
    for a, b in zip(u, v):
        G.add_edge(int(a), int(b))
    return G

def _load_labels_for_graph(G, path):
    import os
    if not os.path.exists(path):
        return np.zeros(len(G.nodes), dtype=int)
    try:
        rows = []
        with open(path, 'r') as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row or len(row) < 2:
                    continue
                rows.append(row)
        idx_vals = []
        label_vals = []
        for row in rows:
            a = row[0].strip()
            b = row[1].strip()
            if a.startswith('u'):
                a = a[1:]
            try:
                ai = int(a)
            except ValueError:
                continue
            lb = b.strip().lower()
            if lb in ('bot', 'human'):
                bi = 1 if lb == 'bot' else 0
            else:
                try:
                    bi = int(float(b))
                except ValueError:
                    continue
            idx_vals.append(ai)
            label_vals.append(bi)
        nodes_order = list(G.nodes)
        if len(idx_vals) == 0:
            return np.zeros(len(nodes_order), dtype=int)
        if max(idx_vals) > len(nodes_order) - 1:
            d = {idx_vals[i]: label_vals[i] for i in range(len(idx_vals))}
            out = [d.get(n, 0) for n in nodes_order]
            return np.asarray(out, dtype=int)
        else:
            d = {idx_vals[i]: label_vals[i] for i in range(len(idx_vals))}
            out = [d.get(i, 0) for i in range(len(nodes_order))]
            return np.asarray(out, dtype=int)
    except Exception:
        return np.zeros(len(G.nodes), dtype=int)

def _load_labels_pt(path, n_nodes):
    if torch is None or not os.path.exists(path):
        return np.zeros(n_nodes, dtype=int)
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj):
        arr = obj.view(-1).cpu().numpy()
    elif isinstance(obj, dict):
        arr = None
        for k in ('label', 'labels', 'y'):
            v = obj.get(k, None)
            if torch.is_tensor(v):
                arr = v.view(-1).cpu().numpy()
                break
        if arr is None:
            for v in obj.values():
                if torch.is_tensor(v):
                    arr = v.view(-1).cpu().numpy()
                    break
    else:
        arr = None
    if arr is None:
        return np.zeros(n_nodes, dtype=int)
    if len(arr) < n_nodes:
        pad = np.zeros(n_nodes - len(arr), dtype=int)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:n_nodes]
    return arr.astype(int)

if __name__ == "__main__":
    device = torch.device('cuda:0' if (torch is not None and torch.cuda.is_available()) else 'cpu') if torch is not None else 'cpu'
    base = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot20'
    edge_csv = os.path.join(base, 'edge.csv')
    edge_idx_csv = os.path.join(base, 'edge_index.csv')
    edge_pt = os.path.join(base, 'edge.pt')
    edge_index_pt = os.path.join(base, 'edge_index.pt')
    if os.path.exists(edge_csv):
        g = _build_eg_from_csv(edge_csv)
    elif os.path.exists(edge_idx_csv):
        g = _build_eg_from_csv(edge_idx_csv)
    elif os.path.exists(edge_pt):
        g = _build_eg_from_pt(edge_pt)
    elif os.path.exists(edge_index_pt):
        g = _build_eg_from_pt(edge_index_pt)
    else:
        g = eg.Graph()
    label_csv = os.path.join(base, 'label.csv')
    label_pt = os.path.join(base, 'label.pt')
    if os.path.exists(label_csv):
        labels = _load_labels_for_graph(g, label_csv)
    else:
        labels = _load_labels_pt(label_pt, len(g.nodes))
    nodes_order = list(g.nodes)
    if torch is not None:
        torch.save(torch.as_tensor(labels), 'twibot20_labels.pt')

    print("Graph embedding via DeepWalk...........")
    deepwalk_emb, _ = deepwalk(g, dimensions=128, walk_length=80, num_walks=10)
    dw_emb = _emb_matrix(deepwalk_emb, nodes_order, dim_hint=128)
    if torch is not None:
        torch.save(dw_emb,'dw_twibot20_emb.pt')
    print(dw_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    max_n = min(1000, len(dw_emb))
    indices = random.sample(range(len(dw_emb)), max_n)
    dw_emb_sub = dw_emb[indices]
    labels_sub = labels[indices]
    z = tsne.fit_transform(dw_emb_sub)
    z_data = np.vstack((z.T, labels_sub)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    plt.savefig("figs/dw_twibot20.pdf", bbox_inches="tight")
    plt.xlabel('横坐标', fontname=CHN_FONT, fontsize=18)
    plt.ylabel('纵坐标', fontname=CHN_FONT, fontsize=18)
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.show()

    print("Graph embedding via Node2Vec..............")
    node2vec_emb, _ = node2vec(
        g, dimensions=128, walk_length=80, num_walks=10, p=4, q=0.25
    )
    n2v_emb = _emb_matrix(node2vec_emb, nodes_order, dim_hint=128)
    if torch is not None:
        torch.save(n2v_emb,'n2v_twibot20_emb.pt')

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    max_n = min(1000, len(n2v_emb))
    indices = random.sample(range(len(n2v_emb)), max_n)
    n2v_emb_sub = n2v_emb[indices]
    labels_sub = labels[indices]
    z = tsne.fit_transform(n2v_emb_sub)
    z_data = np.vstack((z.T, labels_sub)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
    plt.savefig("figs/n2v_twibot20.pdf", bbox_inches="tight")
    plt.xlabel('横坐标', fontname=CHN_FONT, fontsize=18)
    plt.ylabel('纵坐标', fontname=CHN_FONT, fontsize=18)
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.show()

    print("Graph embedding via LINE........")
    
    model = LINE(dimension=128, walk_length=80, walk_num=10, negative=5, batch_size=128, init_alpha=0.025, order=2)

    model.train()
    line_emb = model(g, return_dict=True)
    l_emb = _emb_matrix(line_emb, nodes_order, dim_hint=128)
    if torch is not None:
        torch.save(l_emb,'line_twibot20_emb.pt')

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    max_n = min(1000, len(l_emb))
    indices = random.sample(range(len(l_emb)), max_n)
    l_emb_sub = l_emb[indices]
    labels_sub = labels[indices]
    z = tsne.fit_transform(l_emb_sub)
    z_data = np.vstack((z.T, labels_sub)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
    plt.savefig("figs/line_twibot20.pdf", bbox_inches="tight")
    plt.xlabel('横坐标', fontname=CHN_FONT, fontsize=18)
    plt.ylabel('纵坐标', fontname=CHN_FONT, fontsize=18)
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.show()

    print("Graph embedding via SDNE...........")
    node_size_val = (max(g.nodes) + 1) if len(g.nodes) > 0 else 0
    model = eg.SDNE(g, node_size=node_size_val, nhid0=200, nhid1=100, dropout=0.25, alpha=3e-2, beta=5)
    sdne_emb = model.train(model)

    sd_emb = _emb_matrix(sdne_emb, nodes_order, dim_hint=100)
    if torch is not None:
        torch.save(sd_emb,'sd_twibot20_emb.pt')
    print(sd_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    max_n = min(1000, len(sd_emb))
    indices = random.sample(range(len(sd_emb)), max_n)
    sd_emb_sub = sd_emb[indices]
    labels_sub = labels[indices]
    z = tsne.fit_transform(sd_emb_sub)
    z_data = np.vstack((z.T, labels_sub)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
    plt.savefig("figs/sdne_twibot20.pdf", bbox_inches="tight")
    plt.xlabel('横坐标', fontname=CHN_FONT, fontsize=18)
    plt.ylabel('纵坐标', fontname=CHN_FONT, fontsize=18)
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.show()
