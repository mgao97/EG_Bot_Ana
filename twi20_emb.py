
import csv
import time

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import torch
except Exception:
    torch = None

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


def _build_eg_from_csv(path):
    with open(path, 'r') as f:
        first = f.readline().strip()
    toks = [t.strip().lower() for t in first.split(',')]
    G = eg.Graph()
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

if __name__ == "__main__":
    device = torch.device('cuda:0' if (torch is not None and torch.cuda.is_available()) else 'cpu') if torch is not None else 'cpu'
    g = _build_eg_from_csv('../dataset/TwiBot20/edge.csv')
    labels = _load_labels_for_graph(g, '../dataset/TwiBot20/label.csv')
    if torch is not None:
        torch.save(torch.as_tensor(labels), 'twibot20_labels.pt')

    print("Graph embedding via DeepWalk...........")
    deepwalk_emb, _ = deepwalk(g, dimensions=128, walk_length=80, num_walks=10)
    # print(deepwalk_emb, len(deepwalk_emb))

    dw_emb = []
    for i in range(0, len(deepwalk_emb)):
        dw_emb.append(list(deepwalk_emb[i]))
    #   print(len(dw_emb))
    if torch is not None:
        torch.save(dw_emb,'dw_twibot20_emb.pt')
    dw_emb = np.array(dw_emb)
    print(dw_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(dw_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df_tsne['Class'] = df_tsne['Class'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    plt.legend(loc='upper right')
    #increase font size of all elements
    
    sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black'])
    plt.savefig("emb_figs/dw_twibot20.pdf", bbox_inches="tight")
    plt.savefig("emb_figs/dw_twibot20.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via Node2Vec..............")
    node2vec_emb, _ = node2vec(
        g, dimensions=128, walk_length=80, num_walks=10, p=4, q=0.25
    )
    # print(node2vec_emb, len(node2vec_emb))

    n2v_emb = []
    for i in range(0, len(node2vec_emb)):
        n2v_emb.append(list(node2vec_emb[i]))
    # print(len(n2v_emb))
    if torch is not None:
        torch.save(n2v_emb,'n2v_twibot20_emb.pt')
    n2v_emb = np.array(n2v_emb)
    # print(n2v_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(n2v_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df_tsne['Class'] = df_tsne['Class'].astype(int)
    plt.figure(figsize=(8, 8))
    plt.legend(loc='upper right')
    sns.set(font_scale=1.5)
    sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black'])
    
    plt.savefig("emb_figs/n2v_twibot20.pdf", bbox_inches="tight")
    plt.savefig("emb_figs/n2v_twibot20.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via LINE........")
    
    model = LINE(dimension=128, walk_length=80, walk_num=10, negative=5, batch_size=128, init_alpha=0.025, order=2)

    model.train()
    line_emb = model(g, return_dict=True)

    l_emb = []
    for i in range(0, len(line_emb)):
        l_emb.append(list(line_emb[i]))
    #   print(len(l_emb))
    if torch is not None:
        torch.save(l_emb,'line_twibot20_emb.pt')
    l_emb = np.array(l_emb)
    # print(l_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(l_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df_tsne['Class'] = df_tsne['Class'].astype(int)
    plt.figure(figsize=(8, 8))
    plt.legend(loc='upper right')
    sns.set(font_scale=1.5)
    sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black'])
    
    plt.savefig("emb_figs/line_twibot20.pdf", bbox_inches="tight")
    plt.savefig("emb_figs/line_twibot20.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via SDNE...........")
    model = eg.SDNE(g, node_size=len(g.nodes), nhid0=200, nhid1=100, dropout=0.25, alpha=3e-2, beta=5)
    sdne_emb = model.train(model)

    sd_emb = []
    for i in range(0, len(sdne_emb)):
        sd_emb.append(list(sdne_emb[i]))
    #   print(len(sd_emb))
    if torch is not None:
        torch.save(sd_emb,'sd_twibot20_emb.pt')
    sd_emb = np.array(sd_emb)
    print(sd_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(sd_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df_tsne['Class'] = df_tsne['Class'].astype(int)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    # plt.legend(loc='upper right')
    
    sns.scatterplot(data=df_tsne, hue='Class', x='Dimension 1', y='Dimension 2', palette=['green','orange','brown','red', 'blue','black'])
    
    plt.savefig("emb_figs/sdne_twibot20.pdf", bbox_inches="tight")
    plt.savefig("emb_figs/sdne_twibot20.png", bbox_inches="tight")
    plt.show()
