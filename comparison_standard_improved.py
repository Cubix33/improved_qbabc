import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
from scipy.interpolate import make_interp_spline


# Import standard QBABC (from your provided code)
from standard_qbabc import QBABC as StandardQBABC


# Import improved QBABC and its components from more.py
from improved_qbabc import ImprovedQBABC, fuzzy_cmeans_clustering, CryptoComm, ContextualAgenticAI


def create_sample_vanet(nnodes=20, seed=42):
    """Create smaller VANET for faster testing"""
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(nnodes):
        pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
        G.add_node(i, pos=pos)
    for i in range(nnodes):
        for j in range(i + 1, nnodes):
            dist = np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))
            if dist < 450:
                G.add_edge(i, j)
    for u, v in G.edges:
        G[u][v]['weight'] = np.linalg.norm(
            np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])
        )
    return G


def run_standard_qbabc(G):
    print("\n" + "="*60)
    print("RUNNING STANDARD QBABC (No Improvements)")
    print("="*60)
    start_time = time.time()
    qbabc = StandardQBABC(G, population_size=20, max_iterations=30)
    fitness_history = qbabc.optimize()
    best_solutions = qbabc.get_best_solutions(1)
    end_time = time.time()
    if best_solutions:
        best = best_solutions[0]
        best_fitness = best['fitness']
        print(f"\n✓ Standard QBABC Best Fitness: {best_fitness:.4f}")
        print(f"✓ Time Taken: {end_time - start_time:.2f} seconds")
        print(f"✓ Edges Selected: {len(best['selected_edges'])}")
    else:
        best_fitness = float('inf')
        print("\n✗ No feasible solution found")
    return {
        'fitness_history': fitness_history,
        'best_fitness': best_fitness,
        'time': end_time - start_time,
        'name': 'Standard QBABC'
    }


def run_improved_qbabc(G):
    print("\n" + "="*60)
    print("RUNNING IMPROVED QBABC (FCM + Contextual AI + Crypto)")
    print("="*60)
    start_time = time.time()
    # Step 1: FCM Clustering
    clusters, memberships = fuzzy_cmeans_clustering(G, n_clusters=3)
    print(f"\n✓ FCM Clustering created {len(clusters)} clusters")
    cluster_sizes = [len(cluster) for cluster in clusters]  # ADDED: Store cluster sizes
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} nodes")
    # Step 2: Run Improved QBABC on each cluster
    cluster_histories = []
    final_fitness_vals = []
    total_best_fitness = 0
    context_ai = ContextualAgenticAI()
    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        suggested_params = context_ai.suggest_parameters(len(nodes), len(subgraph.edges))
        qb = ImprovedQBABC(subgraph,
                           populationsize=suggested_params['population_size'],
                           maxiterations=suggested_params['max_iterations'],
                           scoutlimit=suggested_params['scout_limit'])
        hist = qb.optimize()
        _, bestfit = qb.get_best_solution()
        cluster_histories.append(hist)
        final_fitness_vals.append(bestfit)
        total_best_fitness += bestfit
        print(f"  Cluster {idx + 1} best fitness: {bestfit:.4f}")
        # Record learning for agentic AI
        context_ai.record_optimization(len(nodes), len(subgraph.edges),
                                      suggested_params['population_size'],
                                      suggested_params['max_iterations'],
                                      suggested_params['scout_limit'],
                                      bestfit, qb.get_convergence_speed())
    end_time = time.time()
    # Step 3: Crypto test
    c = CryptoComm()
    x = np.random.rand(10)
    en = c.encrypt(x)
    de_bytes = c.decrypt(en)
    decrypted_x = np.frombuffer(de_bytes, dtype=x.dtype)
    assert np.allclose(decrypted_x, x, atol=1e-3)
    print("\n✓ Cryptography test passed")
    print(f"\n✓ Improved QBABC Total Best Fitness: {total_best_fitness:.4f}")
    print(f"✓ Time Taken: {end_time - start_time:.2f} seconds")
    # Average fitness history across clusters
    max_len = max(len(h) for h in cluster_histories)
    avg_history = []
    for i in range(max_len):
        vals = [h[i] if i < len(h) else h[-1] for h in cluster_histories]
        avg_history.append(np.mean(vals))
    return {
        'fitness_history': avg_history,
        'cluster_histories': cluster_histories,
        'cluster_sizes': cluster_sizes,  # ADDED: Include cluster sizes in return
        'best_fitness': total_best_fitness,
        'cluster_fitness': final_fitness_vals,
        'time': end_time - start_time,
        'clusters': clusters,
        'name': 'Improved QBABC (FCM+AI+Crypto)'
    }


def plot_comparison(standard_results, improved_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Standard QBABC convergence
    ax1 = axes[0]
    x = np.arange(len(standard_results['fitness_history']))
    y = np.array(standard_results['fitness_history'])
    if len(x) > 3:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Standard QBABC')
    else:
        ax1.plot(x, y, 'b-', linewidth=2, label='Standard QBABC')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Standard QBABC Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    # Improved QBABC convergence (all clusters)
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(improved_results['cluster_histories'])))
    cluster_sizes = improved_results['cluster_sizes']  # ADDED: Get cluster sizes
    for idx, hist in enumerate(improved_results['cluster_histories']):
        x = np.arange(len(hist))
        y = np.array(hist)
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            ax2.plot(x_smooth, y_smooth, linewidth=2,
                     color=colors[idx], label=f'Cluster {idx+1} ({cluster_sizes[idx]} nodes)')  # MODIFIED: Added cluster size
        else:
            ax2.plot(x, y, linewidth=2, color=colors[idx], label=f'Cluster {idx+1} ({cluster_sizes[idx]} nodes)')  # MODIFIED: Added cluster size
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best Fitness', fontsize=12)
    ax2.set_title('Improved QBABC Convergence (per Cluster)', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('comparison_convergence.png', dpi=300)
    plt.show()
    # Performance Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Best Fitness Comparison
    ax1 = axes[0]
    methods = ['Standard\nQBABC', 'Improved\nQBABC']
    fitness_vals = [standard_results['best_fitness'], improved_results['best_fitness']]
    colors_bar = ['#3498db', "#d94a3a"]
    bars = ax1.bar(methods, fitness_vals, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Best Fitness Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar, val in zip(bars, fitness_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Execution Time Comparison
    ax2 = axes[1]
    time_vals = [standard_results['time'], improved_results['time']]
    bars = ax2.bar(methods, time_vals, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for bar, val in zip(bars, time_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Improvement Percentage
    ax3 = axes[2]
    fitness_improvement = ((standard_results['best_fitness'] - improved_results['best_fitness'])
                          / standard_results['best_fitness'] * 100)
    metric_names = ['Fitness\nImprovement (%)']
    metric_vals = [fitness_improvement]
    color_imp = '#2ecc71' if fitness_improvement > 0 else '#e74c3c'
    bars = ax3.bar(metric_names, metric_vals, color=color_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    for bar, val in zip(bars, metric_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=300)
    plt.show()
    # Summary Table
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'Standard':<15} {'Improved':<15}")
    print("-"*60)
    print(f"{'Best Fitness':<30} {standard_results['best_fitness']:<15.4f} {improved_results['best_fitness']:<15.4f}")
    print(f"{'Execution Time (s)':<30} {standard_results['time']:<15.2f} {improved_results['time']:<15.2f}")
    print(f"{'Fitness Improvement (%)':<30} {'-':<15} {fitness_improvement:<15.2f}")
    print("="*60)


def main():
    print("="*60)
    print("QBABC ALGORITHM COMPARISON - FAST MODE")
    print("Standard QBABC vs Improved QBABC (FCM+AI+Crypto)")
    print("="*60)
    G = create_sample_vanet(nnodes=20, seed=42)
    print(f"\nCreated VANET: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    standard_results = run_standard_qbabc(G)
    improved_results = run_improved_qbabc(G)
    plot_comparison(standard_results, improved_results)
    print("\n✓ Comparison complete! Plots saved as:")
    print("  - comparison_convergence.png")
    print("  - comparison_metrics.png")


if __name__ == "__main__":
    main()
