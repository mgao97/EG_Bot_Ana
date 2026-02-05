"""
绘制不同库实现的算法运行时间对比图
支持 NetworkX 和 igraph 的性能比较
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_benchmark_data(results_dir='results'):
    """加载基准测试数据"""
    summary_path = os.path.join(results_dir, 'bench_summary.csv')
    raw_path = os.path.join(results_dir, 'bench_raw.csv')
    
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found")
        return None, None
    
    summary_df = pd.read_csv(summary_path)
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    
    return summary_df, raw_df

def plot_algorithm_comparison(summary_df, output_dir='figs'):
    """
    绘制每个算法的不同库实现对比图
    支持 networkx (nx) 和 igraph (ig) 的比较
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有算法
    algos = summary_df['algo'].unique()
    
    for algo in algos:
        algo_data = summary_df[summary_df['algo'] == algo]
        
        # 提取 nx 和 ig 的实现
        nx_data = algo_data[algo_data['lib'].str.contains('nx', na=False, case=False)]
        ig_data = algo_data[algo_data['lib'].str.contains('ig', na=False, case=False)]
        
        if len(nx_data) == 0 and len(ig_data) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 合并两个数据框进行比较
        comparison_data = pd.concat([nx_data, ig_data])
        
        if len(comparison_data) > 0:
            x = np.arange(len(comparison_data))
            width = 0.6
            
            colors = ['#FF6B6B' if 'nx' in lib else '#4ECDC4' for lib in comparison_data['lib']]
            bars = ax.bar(x, comparison_data['mean'], width, 
                         yerr=comparison_data['stdev'], 
                         capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('运行时间 (秒)', fontsize=12, fontweight='bold')
            ax.set_title(f'算法: {algo.upper()} - 不同库实现的运行时间对比 (MGTAB数据集)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_data['lib'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF6B6B', alpha=0.7, edgecolor='black', label='NetworkX (nx)'),
                Patch(facecolor='#4ECDC4', alpha=0.7, edgecolor='black', label='iGraph (ig)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
            
            # 在柱子上添加具体数值
            for i, (bar, val) in enumerate(zip(bars, comparison_data['mean'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + comparison_data['stdev'].iloc[i],
                       f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'algo_{algo}_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存: {output_path}")
            plt.close()

def plot_lib_comparison(summary_df, output_dir='figs'):
    """
    绘制 NetworkX vs iGraph 的整体性能对比
    按算法分组展示
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 只对比单线程版本的 nx 和 ig
    nx_simple = summary_df[(summary_df['lib'] == 'nx') | (summary_df['lib'] == 'ig')]
    
    if len(nx_simple) == 0:
        print("WARNING: No NX/IG data found for library comparison")
        return
    
    algos = nx_simple['algo'].unique()
    nx_times = []
    ig_times = []
    algo_names = []
    
    for algo in algos:
        nx_row = nx_simple[(nx_simple['algo'] == algo) & (nx_simple['lib'] == 'nx')]
        ig_row = nx_simple[(nx_simple['algo'] == algo) & (nx_simple['lib'] == 'ig')]
        
        if len(nx_row) > 0 and len(ig_row) > 0:
            nx_times.append(nx_row['mean'].values[0])
            ig_times.append(ig_row['mean'].values[0])
            algo_names.append(algo.upper())
    
    if len(algo_names) == 0:
        print("WARNING: No matching algorithms for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(algo_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nx_times, width, label='NetworkX', 
                   color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ig_times, width, label='iGraph', 
                   color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('运行时间 (秒)', fontsize=12, fontweight='bold')
    ax.set_title('NetworkX vs iGraph - 算法运行时间对比 (MGTAB数据集)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加性能加速比标签
    for i, (nx_t, ig_t) in enumerate(zip(nx_times, ig_times)):
        speedup = nx_t / ig_t if ig_t > 0 else 0
        mid_x = i
        max_y = max(nx_t, ig_t)
        ax.text(mid_x, max_y * 1.05, f'Speed up: {speedup:.1f}x', 
               ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E74C3C')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'nx_vs_ig_overall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()

def plot_variability_analysis(raw_df, output_dir='figs'):
    """
    绘制运行时间的变异性分析（箱线图）
    显示每个库实现的稳定性
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if raw_df is None or len(raw_df) == 0:
        print("WARNING: No raw data for variability analysis")
        return
    
    algos = raw_df['algo'].unique()
    
    for algo in algos:
        algo_data = raw_df[raw_df['algo'] == algo]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按库分组
        libs = sorted(algo_data['lib'].unique())
        data_by_lib = [algo_data[algo_data['lib'] == lib]['time'].values for lib in libs]
        
        bp = ax.boxplot(data_by_lib, labels=libs, patch_artist=True)
        
        # 着色
        colors = ['#FF6B6B' if 'nx' in lib else '#4ECDC4' for lib in libs]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('运行时间 (秒)', fontsize=12, fontweight='bold')
        ax.set_title(f'算法: {algo.upper()} - 运行时间稳定性分析 (5次运行)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'algo_{algo}_variability.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

def print_summary_statistics(summary_df):
    """打印汇总统计信息"""
    print("\n" + "="*80)
    print("MGTAB 数据集 - 基准测试汇总统计")
    print("="*80)
    
    # 按库统计
    print("\n[按库统计]")
    lib_stats = summary_df.groupby('lib').agg({
        'mean': ['count', 'mean', 'min', 'max'],
        'stdev': 'mean'
    }).round(4)
    print(lib_stats)
    
    # NetworkX vs iGraph 对比
    print("\n[NetworkX vs iGraph 对比]")
    nx_data = summary_df[summary_df['lib'] == 'nx']
    ig_data = summary_df[summary_df['lib'] == 'ig']
    
    if len(nx_data) > 0:
        print(f"\nNetworkX (nx):")
        print(f"  平均运行时间: {nx_data['mean'].mean():.2f}s")
        print(f"  最快: {nx_data['mean'].min():.2f}s ({nx_data.loc[nx_data['mean'].idxmin(), 'algo']})")
        print(f"  最慢: {nx_data['mean'].max():.2f}s ({nx_data.loc[nx_data['mean'].idxmax(), 'algo']})")
    
    if len(ig_data) > 0:
        print(f"\niGraph (ig):")
        print(f"  平均运行时间: {ig_data['mean'].mean():.2f}s")
        print(f"  最快: {ig_data['mean'].min():.2f}s ({ig_data.loc[ig_data['mean'].idxmin(), 'algo']})")
        print(f"  最慢: {ig_data['mean'].max():.2f}s ({ig_data.loc[ig_data['mean'].idxmax(), 'algo']})")
        
        # 性能加速比
        if len(nx_data) > 0:
            common_algos = set(nx_data['algo']) & set(ig_data['algo'])
            if common_algos:
                print(f"\niGraph vs NetworkX 性能加速比:")
                for algo in sorted(common_algos):
                    nx_time = nx_data[nx_data['algo'] == algo]['mean'].values[0]
                    ig_time = ig_data[ig_data['algo'] == algo]['mean'].values[0]
                    speedup = nx_time / ig_time
                    print(f"  {algo.upper()}: {speedup:.2f}x (NX: {nx_time:.2f}s, IG: {ig_time:.2f}s)")
    
    print("\n" + "="*80)

def main():
    # 加载数据
    summary_df, raw_df = load_benchmark_data()
    
    if summary_df is None:
        print("Failed to load benchmark data")
        return
    
    print("数据加载成功")
    print(f"包含 {len(summary_df)} 条汇总记录")
    if raw_df is not None:
        print(f"包含 {len(raw_df)} 条原始记录")
    
    # 打印统计信息
    print_summary_statistics(summary_df)
    
    # 绘制各种对比图
    print("\n开始绘制对比图...")
    plot_algorithm_comparison(summary_df)
    plot_lib_comparison(summary_df)
    plot_variability_analysis(raw_df)
    
    print("\n✓ 所有图表已保存到 figs 目录")

if __name__ == '__main__':
    main()
