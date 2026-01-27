import networkx as nx
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# 保持你原有的字体设置
try:
    font_manager.fontManager.addfont('/usr/share/fonts/sim/simsun.ttc')
    mpl.rcParams['font.family'] = 'SimSun'
except:
    pass 
mpl.rcParams['axes.unicode_minus'] = False

def load_edges(path):
    # ... 保持你原有的 load_edges 不变 ...
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj): edge_index = obj
    elif isinstance(obj, dict):
        edge_index = obj.get('edge_index', None)
        if edge_index is None:
            for v in obj.values():
                if torch.is_tensor(v) and v.ndim == 2:
                    edge_index = v
                    break
    return edge_index

def build_graph(edge_index):
    # ... 保持你原有的 build_graph 不变 ...
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = nx.DiGraph()
    G.add_edges_from(zip(u, v))
    return G

def main():
    path = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.pt'
    edge_index = load_edges(path)
    G = build_graph(edge_index)
    
    n = G.number_of_nodes()
    k = 1000 # 采样1000个节点
    
    # 随机采样 k 个种子节点
    np.random.seed(2026)
    sample_nodes = np.random.choice(list(G.nodes()), size=k, replace=False)
    
    node_stats = []
    print(f"开始逐节点分析，共 {k} 个任务...")

    for i, node in enumerate(sample_nodes):
        # 获取节点的度（入度+出度）
        degree = G.degree(node)
        
        # 计时开始
        t0 = time.perf_counter()
        # 计算以该节点为源点的近似贡献（BC算法核心步骤）
        _ = nx.betweenness_centrality_subset(G, sources=[node], targets=list(G.nodes), normalized=True)
        t1 = time.perf_counter()
        
        node_stats.append({
            'node_id': node,
            'degree': degree,
            'runtime': t1 - t0
        })
        
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{k}")

    # 转化为 DataFrame
    df = pd.DataFrame(node_stats)
    df.to_csv('performance_stats.csv', index=False)
    
    # --- 绘图部分 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 图1：散点图 - 度数与耗时的关系
    ax1.scatter(df['degree'], df['runtime'], alpha=0.5, color='royalblue', label='节点任务')
    ax1.set_xlabel('节点度数')
    ax1.set_ylabel('计算耗时')
    # ax1.set_title(f'TwiBot-22 节点度数与 BC 计算耗时分布 ($\alpha=1.59$)')
    ax1.set_xscale('log') # 社交网络必须用对数坐标
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('bottleneck_analysis.pdf', dpi=300)
    print("分析完成，图表已保存为 bottleneck_analysis.pdf")

if __name__ == '__main__':
    main()