
import csv
import time
import random
import os

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import torch
except Exception:
    torch = None
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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


def get_sparse_adj(g):
    # Map nodes to 0..N-1 is already done in main, so g.nodes are integers 0..N-1
    nodes = list(g.nodes)
    num_nodes = len(nodes)
    
    row = []
    col = []
    data = []
    
    # Iterate over edges. EasyGraph edges are (u, v, data)
    for u, v, _ in g.edges:
        row.append(u)
        col.append(v)
        data.append(1)
        # Add symmetric edge
        row.append(v)
        col.append(u)
        data.append(1)
        
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj, num_nodes

class SparseSDNEDataset(Dataset):
    def __init__(self, adj, num_nodes):
        self.adj = adj
        self.num_nodes = num_nodes
        
    def __len__(self):
        return self.num_nodes
        
    def __getitem__(self, index):
        return index

class SparseSDNE(nn.Module):
    def __init__(self, graph, node_size, nhid0, nhid1, dropout=0.06, alpha=2e-2, beta=10.0):
        super(SparseSDNE, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.droput = dropout
        self.alpha = alpha
        self.beta = beta
        self.graph = graph

    def forward(self, adj_batch, adj_mat, b_mat):
        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)
        L_1st = torch.sum(
            adj_mat
            * (
                embedding_norm
                - 2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                + torch.transpose(embedding_norm, dim0=0, dim1=1)
            )
        )
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def train_model(
        self,
        epochs=100,
        lr=0.006,
        bs=100,
        step_size=10,
        gamma=0.9,
        nu1=1e-5,
        nu2=1e-4,
        device="cpu",
    ):
        # Use sparse adjacency matrix
        Adj, Node = get_sparse_adj(self.graph)
        
        self.to(device)
        opt = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=step_size, gamma=gamma
        )
        
        dataset = SparseSDNEDataset(Adj, Node)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        for epoch in range(1, epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            for index in dataloader:
                # index is a tensor of indices
                indices = index.numpy()
                
                # Slicing sparse matrix -> csr_matrix
                adj_batch_sparse = Adj[indices]
                # Convert to dense numpy array -> Tensor
                adj_batch = torch.FloatTensor(adj_batch_sparse.toarray()).to(device)
                
                # Slicing columns for adj_mat
                # adj_batch is [bs, N], we want adj_mat [bs, bs] which is Adj[indices][:, indices]
                # But adj_batch is already Adj[indices], so we slice columns from it
                adj_mat = adj_batch[:, indices]
                
                b_mat = torch.ones_like(adj_batch)
                b_mat[adj_batch != 0] = self.beta

                opt.zero_grad()
                L_1st, L_2nd, L_all = self(adj_batch, adj_mat, b_mat)
                L_reg = 0
                for param in self.parameters():
                    L_reg += nu1 * torch.sum(torch.abs(param)) + nu2 * torch.sum(
                        param * param
                    )
                Loss = L_all + L_reg
                Loss.backward()
                opt.step()
                
                loss_sum += Loss.item()
                loss_L1 += L_1st.item()
                loss_L2 += L_2nd.item()
                loss_reg += L_reg.item()
                
            scheduler.step()
            print("loss for epoch %d is: %f" % (epoch, loss_sum))

        # Generate embeddings
        # Process in batches to avoid OOM during inference
        self.eval()
        embeddings = []
        # Create a sequential dataloader
        seq_loader = DataLoader(range(Node), batch_size=bs, shuffle=False)
        
        with torch.no_grad():
            for index in seq_loader:
                indices = index.numpy()
                adj_batch_sparse = Adj[indices]
                adj_batch = torch.FloatTensor(adj_batch_sparse.toarray()).to(device)
                t0 = self.encode0(adj_batch)
                t0 = self.encode1(t0)
                embeddings.append(t0.cpu().numpy())
                
        return np.vstack(embeddings)

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
        pad = np.full(n_nodes - len(arr), -1, dtype=int)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:n_nodes]
    return arr.astype(int)


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
    # device = torch.device('cuda:0' if (torch is not None and torch.cuda.is_available()) else 'cpu') if torch is not None else 'cpu'
    device = 'cpu'
    g = _build_eg_from_csv('/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.csv')
    label_pt = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/label.pt'
    labels = _load_labels_pt(label_pt, len(g.nodes))
    
    # 节点重映射：确保节点ID是连续的 0..N-1
    print("Remapping nodes to consecutive integers...")
    old_nodes = list(g.nodes)
    node_mapping = {old_node: i for i, old_node in enumerate(old_nodes)}
    
    g_new = eg.Graph()
    new_edges = []
    for u, v, _ in g.edges:
        new_edges.append((node_mapping[u], node_mapping[v]))
    g_new.add_edges(new_edges, [{} for _ in range(len(new_edges))])
    
    # 替换 g
    g = g_new
    nodes_order = list(g.nodes)

    if torch is not None:
        torch.save(torch.as_tensor(labels), 'twibot22_labels.pt')

    # Pre-calculate sampling indices to ensure consistent node selection across all methods
    valid_indices = np.where(labels != -1)[0]
    max_n = min(1000, len(valid_indices))
    # Use a fixed random seed for reproducibility of sampling
    rng = np.random.default_rng(seed=42)
    common_indices = rng.choice(valid_indices, max_n, replace=False)

    # print("Graph embedding via DeepWalk...........")
    # dw_path = 'graph_embs/dw_twibot22_emb.pt'
    # if os.path.exists(dw_path):
    #     print(f"Loading DeepWalk embeddings from {dw_path}...")
    #     dw_emb = torch.load(dw_path, map_location='cpu')
    # else:
    #     deepwalk_emb, _ = deepwalk(g, dimensions=128, walk_length=80, num_walks=10)
    #     dw_emb = _emb_matrix(deepwalk_emb, nodes_order, dim_hint=128)
    #     if torch is not None:
    #         if not os.path.exists('graph_embs'):
    #             os.makedirs('graph_embs')
    #         torch.save(dw_emb, dw_path)
    # print(dw_emb)

    # tsne = TSNE(n_components=2, verbose=1, random_state=0)
    # dw_emb_sub = dw_emb[common_indices]
    # labels_sub = labels[common_indices]
    # z = tsne.fit_transform(dw_emb_sub)
    # z_data = np.vstack((z.T, labels_sub)).T
    # df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    # df_tsne['类别'] = df_tsne['类别'].astype(int)
    # # 将数值类别映射为中文标签
    # label_map = {0: "人类", 1: "社交机器人"}
    # df_tsne['类别'] = df_tsne['类别'].map(label_map)
    # # 处理未映射的标签（如果有）
    # df_tsne['类别'] = df_tsne['类别'].fillna('未知')
    # plt.figure(figsize=(8, 8))
    # sns.set(font_scale=1.5)
    # ax = plt.gca()
    # sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    # for lbl in ax.get_xticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # for lbl in ax.get_yticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # plt.xlabel('')
    # plt.ylabel('')
    # legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    # plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    # plt.savefig("figs/dw_twibot22.pdf", bbox_inches="tight")
    # plt.show()

    # print("Graph embedding via Node2Vec..............")
    # n2v_path = 'graph_embs/n2v_twibot22_emb.pt'
    # if os.path.exists(n2v_path):
    #     print(f"Loading Node2Vec embeddings from {n2v_path}...")
    #     n2v_emb = torch.load(n2v_path, map_location='cpu')
    # else:
    #     node2vec_emb, _ = node2vec(
    #         g, dimensions=128, walk_length=80, num_walks=10, p=4, q=0.25
    #     )
    #     n2v_emb = _emb_matrix(node2vec_emb, nodes_order, dim_hint=128)
    #     if torch is not None:
    #         if not os.path.exists('graph_embs'):
    #             os.makedirs('graph_embs')
    #         torch.save(n2v_emb, n2v_path)

    # tsne = TSNE(n_components=2, verbose=1, random_state=0)
    # n2v_emb_sub = n2v_emb[common_indices]
    # labels_sub = labels[common_indices]
    # z = tsne.fit_transform(n2v_emb_sub)
    # z_data = np.vstack((z.T, labels_sub)).T
    # df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    # df_tsne['类别'] = df_tsne['类别'].astype(int)
    # # 将数值类别映射为中文标签
    # label_map = {0: "人类", 1: "社交机器人"}
    # df_tsne['类别'] = df_tsne['类别'].map(label_map)
    # # 处理未映射的标签（如果有）
    # df_tsne['类别'] = df_tsne['类别'].fillna('未知')
    # plt.figure(figsize=(8, 8))
    # sns.set(font_scale=1.5)
    # ax = plt.gca()
    # sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
    # for lbl in ax.get_xticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # for lbl in ax.get_yticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # plt.xlabel('')
    # plt.ylabel('')
    # legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    # plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    # plt.savefig("figs/n2v_twibot22.pdf", bbox_inches="tight")
    # plt.show()

    # print("Graph embedding via LINE........")
    # l_path = 'graph_embs/line_twibot22_emb.pt'
    # if os.path.exists(l_path):
    #     print(f"Loading LINE embeddings from {l_path}...")
    #     l_emb = torch.load(l_path, map_location='cpu')
    # else:
    #     model = LINE(dimension=128, walk_length=80, walk_num=10, negative=5, batch_size=128, init_alpha=0.025, order=2)

    #     model.train()
    #     line_emb = model(g, return_dict=True)

    #     l_emb = _emb_matrix(line_emb, nodes_order, dim_hint=128)
    #     if torch is not None:
    #         if not os.path.exists('graph_embs'):
    #             os.makedirs('graph_embs')
    #         torch.save(l_emb, l_path)

    # tsne = TSNE(n_components=2, verbose=1, random_state=0)
    # l_emb_sub = l_emb[common_indices]
    # labels_sub = labels[common_indices]
    # z = tsne.fit_transform(l_emb_sub)
    # z_data = np.vstack((z.T, labels_sub)).T
    # df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    # df_tsne['类别'] = df_tsne['类别'].astype(int)
    # # 将数值类别映射为中文标签
    # label_map = {0: "人类", 1: "社交机器人"}
    # df_tsne['类别'] = df_tsne['类别'].map(label_map)
    # # 处理未映射的标签（如果有）
    # df_tsne['类别'] = df_tsne['类别'].fillna('未知')
    # plt.figure(figsize=(8, 8))
    # sns.set(font_scale=1.5)
    # ax = plt.gca()
    # sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
    # for lbl in ax.get_xticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # for lbl in ax.get_yticklabels():
    #     lbl.set_fontname(ROMAN_FONT)
    #     lbl.set_fontsize(18)
    # plt.xlabel('')
    # plt.ylabel('')
    # legend = ax.legend(loc='upper right', prop={'family':CHN_FONT,'size':18}, title='类别')
    # plt.setp(legend.get_title(), fontname=CHN_FONT, fontsize=18)
    # plt.savefig("figs/line_twibot22.pdf", bbox_inches="tight")
    # plt.show()

    print("Graph embedding via SDNE...........")
    sd_path = 'graph_embs/sd_twibot22_emb.pt'
    if os.path.exists(sd_path):
        print(f"Loading SDNE embeddings from {sd_path}...")
        sd_emb = torch.load(sd_path, map_location='cpu')
    else:
        # Use our custom SparseSDNE to avoid OOM
        node_size_val = (max(g.nodes) + 1) if len(g.nodes) > 0 else 0
        # Use smaller hidden dims for memory efficiency if needed, but 200/100 should be fine with sparse
        model = SparseSDNE(g, node_size=node_size_val, nhid0=128, nhid1=64, dropout=0.25, alpha=3e-2, beta=5)
        
        # Train
        sdne_emb = model.train_model(epochs=40, bs=100, device=device) 

        sd_emb = _emb_matrix(sdne_emb, nodes_order, dim_hint=64)
        if torch is not None:
            if not os.path.exists('graph_embs'):
                os.makedirs('graph_embs')
            torch.save(sd_emb, sd_path)
    print(sd_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    sd_emb_sub = sd_emb[common_indices]
    labels_sub = labels[common_indices]
    z = tsne.fit_transform(sd_emb_sub)
    z_data = np.vstack((z.T, labels_sub)).T
    df_tsne = pd.DataFrame(z_data, columns=['x', 'y', '类别'])
    df_tsne['类别'] = df_tsne['类别'].astype(int)
    # 将数值类别映射为中文标签
    label_map = {0: "人类", 1: "社交机器人"}
    df_tsne['类别'] = df_tsne['类别'].map(label_map)
    # 处理未映射的标签（如果有）
    df_tsne['类别'] = df_tsne['类别'].fillna('未知')
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    ax = plt.gca()
    sns.scatterplot(data=df_tsne, hue='类别', x='x', y='y', palette=sns.color_palette("Set2"))
    
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
    plt.savefig("figs/sdne_twibot22.pdf", bbox_inches="tight")
    plt.show()
