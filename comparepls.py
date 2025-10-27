import matplotlib.pyplot as plt
import numpy as np

# --- Import Improved QBABC (VANET) ---
from improved_qbabc import ImprovedQBABC, fuzzy_cmeans_clustering, ContextualAgenticAI
import networkx as nx

def run_improved_qbabc_comparison(nnodes=20, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(nnodes):
        pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
        G.add_node(i, pos=pos)
    for i in range(nnodes):
        for j in range(i+1, nnodes):
            dist = np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))
            if dist < 450:
                G.add_edge(i, j)
    clusters, _ = fuzzy_cmeans_clustering(G, n_clusters=2)
    context_ai = ContextualAgenticAI()
    cluster_sizes = [len(cluster) for cluster in clusters]
    convergence_speeds = []
    final_fitness_vals = []
    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        params = context_ai.suggest_parameters(len(nodes), len(subgraph.edges))
        qb = ImprovedQBABC(subgraph,
                           populationsize=params['population_size'],
                           maxiterations=params['max_iterations'],
                           scoutlimit=params['scout_limit'])
        hist = qb.optimize()
        _, bestfit = qb.get_best_solution()
        try:
            conv_speed = qb.get_convergence_speed()
        except:
            conv_speed = len(hist)
        final_fitness_vals.append(bestfit)
        convergence_speeds.append(conv_speed)
    avg_cluster_size = np.mean(cluster_sizes)
    avg_convergence_speed = np.mean(convergence_speeds)
    avg_best_fitness = np.mean(final_fitness_vals)
    return {
        'avg_cluster_size': avg_cluster_size,
        'convergence_speed': avg_convergence_speed,
        'best_fitness': avg_best_fitness,
        'cluster_sizes': cluster_sizes,
        'name': 'Improved QBABC (VANET)'
    }

# --- Import ABC+ACO WSN ---
from opti_WSN import run_until_first_node_death

def run_abc_aco_wsn_comparison():
    # Modify run_until_first_node_death to return history and final_stats
    # If not, copy the function and add: return history, final_stats at the end
    history, final_stats = run_until_first_node_death(
        n=100, area=100.0, k=8, E0=0.5, pkt_bits=4000, seed=42, visualize=False)
    avg_cluster_size = np.mean([len(members) for members in final_stats['clusters'].values()])
    rounds_to_fnd = len(history)
    total_energy_at_fnd = final_stats['total_energy']
    return {
        'avg_cluster_size': avg_cluster_size,
        'rounds_to_fnd': rounds_to_fnd,
        'total_energy_at_fnd': total_energy_at_fnd,
        'name': 'ABC+ACO (WSN)'
    }
    

# --- Plot Comparison ---
def plot_comparison(qbabc_results, wsn_results):
    labels = [qbabc_results['name'], wsn_results['name']]
    # Cluster size
    plt.figure(figsize=(6,5))
    vals = [qbabc_results['avg_cluster_size'], wsn_results['avg_cluster_size']]
    plt.bar(labels, vals, color=['#3498db','#e67e22'], alpha=0.85)
    plt.ylabel("Average Cluster Size")
    plt.title("Average Cluster Size Comparison")
    plt.tight_layout()
    plt.show()
    # Convergence speed / rounds to FND
    plt.figure(figsize=(6,5))
    vals = [qbabc_results['convergence_speed'], wsn_results['rounds_to_fnd']]
    plt.bar(labels, vals, color=['#1abc9c','#c0392b'], alpha=0.85)
    plt.ylabel("Convergence Speed / Rounds to FND")
    plt.title("Convergence / Lifetime Comparison")
    plt.tight_layout()
    plt.show()
    # Best fitness / total energy at FND
    plt.figure(figsize=(6,5))
    vals = [qbabc_results['best_fitness'], wsn_results['total_energy_at_fnd']]
    plt.bar(labels, vals, color=['#34495e','#e74c3c'], alpha=0.85)
    plt.ylabel("Best Fitness / Energy at FND")
    plt.title("Best Fitness vs. Energy at FND")
    plt.tight_layout()
    plt.show()
    print("\n| Metric            | Improved QBABC (VANET) | ABC+ACO (WSN) |")
    print("|-------------------|-----------------------|---------------|")
    print(f"| Cluster Size      | {qbabc_results['avg_cluster_size']:.2f}              | {wsn_results['avg_cluster_size']:.2f}        |")
    print(f"| Convergence Speed | {qbabc_results['convergence_speed']:.2f}             | {wsn_results['rounds_to_fnd']}       |")
    print(f"| Best Fitness      | {qbabc_results['best_fitness']:.2f}                  | {wsn_results['total_energy_at_fnd']:.2f}            |")

# --- Main ---
if __name__ == "__main__":
    qbabc_results = run_improved_qbabc_comparison(nnodes=20, seed=42)
    wsn_results = run_abc_aco_wsn_comparison()
    plot_comparison(qbabc_results, wsn_results)
