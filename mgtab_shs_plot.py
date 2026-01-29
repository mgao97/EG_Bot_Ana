import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import bezier
import numpy as np
import easygraph as eg
from easygraph.functions.community import louvain_communities
import torch

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

colors = ['#F49193', '#9ED8EC']

def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)
    
    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse = True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:,0,:]
    coords_node2 = coords[:,1,:]
    
    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:,0] > coords_node2[:,0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]
    
    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
    m2 = -1/m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l),m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)),m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:,i,:].T
        curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)
    # Return an array of these curves
    curves = np.array(curveplots)
    return curves

def _build_eg_from_csv(path):
    with open(path, 'r') as f:
        first = f.readline().strip()
    toks = [t.strip().lower() for t in first.split(',')]
    G = eg.Graph()
    if 'relation' in toks or 'source' in toks or 'source_id' in toks:
        with open(path, 'r') as f:
            import csv
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
        if arr.ndim == 2:
            if arr.shape[0] == 2 and arr.shape[1] != 2:
                u = arr[0].tolist()
                v = arr[1].tolist()
            elif arr.shape[1] == 2:
                u = arr[:, 0].tolist()
                v = arr[:, 1].tolist()
            else:
                u, v = [], []
            for a, b in zip(u, v):
                G.add_edge(int(a), int(b))
    return G

def _load_labels_pt(path):
    try:
        t = torch.load(path, map_location='cpu')
        if isinstance(t, torch.Tensor):
            arr = t.cpu().numpy()
        else:
            arr = np.asarray(t)
        return np.asarray(arr, dtype=int).flatten()
    except Exception:
        return np.asarray([], dtype=int)

def _pick_connected_union(G_eg, comm_a, comm_b, limit=30):
    nodes_union = list(set(comm_a) | set(comm_b))
    H = nx.Graph()
    for n in nodes_union:
        H.add_node(n)
    for a, b, _ in G_eg.edges:
        if a in nodes_union and b in nodes_union:
            H.add_edge(a, b)
    if H.number_of_nodes() == 0:
        return H
    comps = sorted(nx.connected_components(H), key=lambda c: len(c), reverse=True)
    nodes = list(comps[0])
    if len(nodes) > limit:
        # BFS sample to preserve connectivity
        seed = nodes[0]
        visited = set([seed])
        queue = [seed]
        while queue and len(visited) < limit:
            cur = queue.pop(0)
            for nbr in H.neighbors(cur):
                if nbr in nodes and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
                if len(visited) >= limit:
                    break
        nodes = list(visited)
    S = H.subgraph(nodes).copy()
    return S

# Build graph from MGTAB and run Louvain
G_eg = _build_eg_from_csv('/NVMeDATA/gxj_data/hyperscan_cikm25/mgtab/edge_index.csv')
comms = list(louvain_communities(G_eg))
comms.sort(key=lambda c: len(c), reverse=True)
comm_a = comms[0] if len(comms) > 0 else set()
comm_b = comms[1] if len(comms) > 1 else set()
G = _pick_connected_union(G_eg, comm_a, comm_b, limit=30)

# Load labels and color mapping: red=bot(1), green=human(0)
labels = _load_labels_pt('/NVMeDATA/gxj_data/hyperscan_cikm25/mgtab/label.pt')
def _node_color(n):
    try:
        return 'red' if labels[int(n)] == 1 else 'green'
    except Exception:
        return 'green'

# layout - 减少迭代次数显著提升速度
pos = nx.spring_layout(G, iterations=15, seed=42)  # 从50降到15，添加seed保证可复现

# 预计算一次即可（不要在循环中重复计算！）
node_colors = [_node_color(n) for n in G.nodes()]
curves = curved_edges(G, pos)
# lc = LineCollection(curves, color='#999999', alpha=0.4)

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(8, 6))

# 遍历所有子图 - 使用预计算的结果
for i, ax in enumerate(axes.flatten()):
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color=node_colors, alpha=0.8)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_family=ROMAN_FONT, font_color='black')

    # 直接添加预计算的连线集合
    lc = LineCollection(curves, color='#999999', alpha=0.4)
    ax.add_collection(lc)

    # 设置坐标轴参数
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 调整子图之间的间距
plt.tight_layout()

# 保存为 PDF 和 PNG 图片
plt.savefig('figs/mgtab_shs.pdf')

# 显示图形
plt.show()
