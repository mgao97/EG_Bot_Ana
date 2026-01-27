import easygraph as eg
import torch
import easygraph as eg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/sim/simsun.ttc')
mpl.rcParams['font.family'] = 'SimSun'
mpl.rcParams['axes.unicode_minus'] = False

def load_edges(path):
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

def build_graph(edge_index):
    u = edge_index[0].cpu().numpy().tolist()
    v = edge_index[1].cpu().numpy().tolist()
    G = eg.DiGraph()
    G.add_edges_from(zip(u, v))
    return G

def fit_powerlaw(deg):
    counts = np.bincount(deg)
    x = np.arange(len(counts))
    mask = (x > 0) & (counts > 0)
    lx = np.log10(x[mask])
    ly = np.log10(counts[mask])
    p = np.polyfit(lx, ly, 1)
    slope, intercept = p[0], p[1]
    y_pred = slope * lx + intercept
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    alpha = -slope
    return x[mask], counts[mask], slope, intercept, alpha, r2

def plot_powerlaw(x, y, slope, intercept, title, out_prefix):
    fig = plt.figure(figsize=(4, 3))
    plt.loglog(x, y, marker='*', linestyle='none', markersize=6, color='#4C72B0', label='度分布')
    lx = np.log10(x)
    ly_fit = slope * lx + intercept
    y_fit = 10 ** ly_fit
    plt.loglog(x, y_fit, color='#55A868', linewidth=2.0, label='幂律拟合')
    plt.xlabel('度')
    plt.ylabel('计数')
    plt.legend(loc='best', fontsize=10)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.pdf', dpi=300)
    plt.savefig(f'{out_prefix}.png', dpi=300)

def main():
    path = '/NVMeDATA/gxj_data/hyperscan_cikm25/twibot22/edge_index.pt'
    edge_index = load_edges(path)
    G = build_graph(edge_index)
    deg_data = G.degree()
    if isinstance(deg_data, dict):
        deg = np.array(list(deg_data.values()))
    else:
        try:
            deg = np.array([d for _, d in deg_data])
        except Exception:
            deg = np.array(list(deg_data))
    x, y, slope, intercept, alpha, r2 = fit_powerlaw(deg)
    title = f'度分布幂律拟合 α={alpha:.2f}, R^2={r2:.3f}'
    plot_powerlaw(x, y, slope, intercept, title, 'twibot22_powerlaw_deg')
    print({'alpha': float(alpha), 'r2': float(r2), 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges()})

if __name__ == '__main__':
    main()
