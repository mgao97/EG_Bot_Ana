
import csv
import time
import random
import os

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


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
times_path = '/usr/share/fonts/Times/times.ttf'     # 新罗马字体 
simsun_path = '/usr/share/fonts/sim/simsun.ttc'   # 宋体 
font_manager.fontManager.addfont(simsun_path)
font_manager.fontManager.addfont(times_path)

CHN_FONT = 'SimSun'
ROMAN_FONT = 'Times New Roman'

def _load_labels_pt(path):
    try:
        t = torch.load(path, map_location='cpu')
        if isinstance(t, torch.Tensor):
            arr = t.cpu().numpy()
        else:
            arr = np.asarray(t)
        arr = np.asarray(arr, dtype=int).flatten()
        return arr
    except Exception:
        return np.asarray([], dtype=int)

def _align_labels(labels, n):
    if len(labels) >= n:
        return labels[:n]
    if len(labels) == 0:
        return np.zeros(n, dtype=int)
    pad = np.zeros(n - len(labels), dtype=int)
    return np.concatenate([labels, pad])

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

if __name__ == "__main__":
    device = torch.device('cuda:0' if (torch is not None and torch.cuda.is_available()) else 'cpu') if torch is not None else 'cpu'
    g = _build_eg_from_csv('/NVMeDATA/gxj_data/hyperscan_cikm25/mgtab/edge_index.csv')
    labels_raw = _load_labels_pt('/NVMeDATA/gxj_data/hyperscan_cikm25/mgtab/labels_bot.pt')
    
    # 自动检测标签格式：如果是 indices (max > 1)，则转换为 0/1 向量
    if len(labels_raw) > 0 and labels_raw.max() > 1:
        print("Detected label file contains indices, converting to binary labels...")
        # 假设 g.nodes 的最大 ID 覆盖了所有节点，或者使用 len(g.nodes) 如果 ID 是连续的
        # 安全起见，创建一个足够大的数组
        max_id = max(max(g.nodes), labels_raw.max()) + 1
        new_labels = np.zeros(max_id, dtype=int)
        new_labels[labels_raw] = 1
        # 如果需要与 g.nodes 对齐
        labels_raw = np.array([new_labels[n] for n in list(g.nodes)])
    
    nodes_order = list(g.nodes)

    # Pre-calculate sampling indices to ensure consistent node selection across all methods
    # Assuming labels_raw aligns with nodes_order or is sufficiently long
    n_nodes = len(nodes_order)
    max_n = min(1000, n_nodes)
    rng = np.random.default_rng(seed=42)
    common_indices = rng.choice(n_nodes, max_n, replace=False)

    print("Graph embedding via DeepWalk...........")
    dw_path = 'graph_embs/dw_mgtab_emb.pt'
    if os.path.exists(dw_path):
        print(f"Loading DeepWalk embeddings from {dw_path}...")
        dw_emb = torch.load(dw_path, map_location='cpu')
    else:
        deepwalk_emb, _ = deepwalk(g, dimensions=128, walk_length=80, num_walks=10)
        dw_emb = _emb_matrix(deepwalk_emb, nodes_order, dim_hint=128)
        if torch is not None:
            if not os.path.exists('graph_embs'):
                os.makedirs('graph_embs')
            torch.save(dw_emb, dw_path)
    print(dw_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    dw_emb_sub = dw_emb[common_indices]
    labels_dw = _align_labels(labels_raw, len(dw_emb))[common_indices]
    z = tsne.fit_transform(dw_emb_sub)
    z_data = np.vstack((z.T, labels_dw)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    # 将数值类别映射为中文标签
    label_map = {0: "人类", 1: "社交机器人"}
    df_tsne['类别'] = df_tsne['类别'].map(label_map)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    # 设置横纵轴刻度为 Times New Roman
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    plt.xlabel('')
    plt.ylabel('')
    legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    plt.savefig("figs/dw_mgtab.pdf", bbox_inches="tight")
    plt.show()

    print("Graph embedding via Node2Vec..............")
    n2v_path = 'graph_embs/n2v_mgtab_emb.pt'
    if os.path.exists(n2v_path):
        print(f"Loading Node2Vec embeddings from {n2v_path}...")
        n2v_emb = torch.load(n2v_path, map_location='cpu')
    else:
        node2vec_emb, _ = node2vec(
            g, dimensions=128, walk_length=80, num_walks=10, p=4, q=0.25
        )
        n2v_emb = _emb_matrix(node2vec_emb, nodes_order, dim_hint=128)
        if torch is not None:
            if not os.path.exists('graph_embs'):
                os.makedirs('graph_embs')
            torch.save(n2v_emb, n2v_path)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    n2v_emb_sub = n2v_emb[common_indices]
    labels_n2v = _align_labels(labels_raw, len(n2v_emb))[common_indices]
    z = tsne.fit_transform(n2v_emb_sub)
    z_data = np.vstack((z.T, labels_n2v)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    # 将数值类别映射为中文标签
    label_map = {0: "人类", 1: "社交机器人"}
    df_tsne['类别'] = df_tsne['类别'].map(label_map)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    # 设置横纵轴刻度为 Times New Roman
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    plt.xlabel('')
    plt.ylabel('')
    legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    plt.savefig("figs/n2v_mgtab.pdf", bbox_inches="tight")
    plt.show()

    print("Graph embedding via LINE........")
    l_path = 'graph_embs/line_mgtab_emb.pt'
    if os.path.exists(l_path):
        print(f"Loading LINE embeddings from {l_path}...")
        l_emb = torch.load(l_path, map_location='cpu')
    else:
        model = LINE(dimension=128, walk_length=80, walk_num=10, negative=5, batch_size=128, init_alpha=0.025, order=2)

        model.train()
        line_emb = model(g, return_dict=True)

        l_emb = _emb_matrix(line_emb, nodes_order, dim_hint=128)
        if torch is not None:
            if not os.path.exists('graph_embs'):
                os.makedirs('graph_embs')
            torch.save(l_emb, l_path)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    l_emb_sub = l_emb[common_indices]
    labels_line = _align_labels(labels_raw, len(l_emb))[common_indices]
    z = tsne.fit_transform(l_emb_sub)
    z_data = np.vstack((z.T, labels_line)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    # 将数值类别映射为中文标签
    label_map = {0: "人类", 1: "社交机器人"}
    df_tsne['类别'] = df_tsne['类别'].map(label_map)
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    # 设置横纵轴刻度为 Times New Roman
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(ROMAN_FONT)
        lbl.set_fontsize(18)
    plt.xlabel('')
    plt.ylabel('')
    legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    plt.savefig("figs/line_mgtab.pdf", bbox_inches="tight")
    plt.show()

    if torch is not None:
        print("Graph embedding via SDNE...........")
        sd_path = 'graph_embs/sd_mgtab_emb.pt'
        if os.path.exists(sd_path):
            print(f"Loading SDNE embeddings from {sd_path}...")
            sd_emb = torch.load(sd_path, map_location='cpu')
        else:
            node_size_val = (max(g.nodes) + 1) if len(g.nodes) > 0 else 0
            model = eg.SDNE(g, node_size=node_size_val, nhid0=200, nhid1=100, dropout=0.25, alpha=3e-2, beta=5)
            sdne_emb = model.train(model)
            sd_emb = _emb_matrix(sdne_emb, nodes_order, dim_hint=100)
            if not os.path.exists('graph_embs'):
                os.makedirs('graph_embs')
            torch.save(sd_emb, sd_path)
        print(sd_emb)
        tsne = TSNE(n_components=2, verbose=1, random_state=0)
        sd_emb_sub = sd_emb[common_indices]
        labels_sdne = _align_labels(labels_raw, len(sd_emb))[common_indices]
        z = tsne.fit_transform(sd_emb_sub)
        z_data = np.vstack((z.T, labels_sdne)).T
        df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
        df_tsne['类别'] = df_tsne['类别'].astype(int)
        # 将数值类别映射为中文标签
        label_map = {0: "人类", 1: "社交机器人"}
        df_tsne['类别'] = df_tsne['类别'].map(label_map)
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.5)
        ax = plt.gca()
        sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
        # 设置横纵轴刻度为 Times New Roman
        for lbl in ax.get_xticklabels():
            lbl.set_fontname(ROMAN_FONT)
            lbl.set_fontsize(18)
        for lbl in ax.get_yticklabels():
            lbl.set_fontname(ROMAN_FONT)
            lbl.set_fontsize(18)
        plt.xlabel('')
        plt.ylabel('')
        legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
        plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
        plt.savefig("figs/sdne_mgtab.pdf", bbox_inches="tight")
        plt.show()
