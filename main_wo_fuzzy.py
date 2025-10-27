import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qbabc import QBABC
from fuzzy import build_fuzzy_param_controller, fuzzy_adapt_fis, fuzzy_cmeans_clustering
from crypto import CryptoComm
from context_ai import ContextHistory
from benchmarks import cec_benchmark_function
from scipy.interpolate import make_interp_spline


def create_sample_vanet(nnodes=30, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(nnodes):
        pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
        G.add_node(i, pos=pos)
    for i in range(nnodes):
        for j in range(i + 1, nnodes):
            if np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos'])) < 400:
                G.add_edge(i, j)
    for u, v in G.edges:
        G[u][v]['weight'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G


def plot_clusters(G, clusters, title="Cluster Visualization"):
    plt.figure(figsize=(8, 8))
    colors = plt.colormaps.get_cmap('tab10')
    for i, cluster in enumerate(clusters):
        xy = np.array([G.nodes[n]['pos'] for n in cluster])
        plt.scatter(xy[:, 0], xy[:, 1], label=f'Cluster {i+1} ({len(cluster)} nodes)', color=colors(i % 10))
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(clusters):
    sizes = [len(cluster) for cluster in clusters]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(sizes) + 1), sizes, tick_label=[f"Cluster {i+1}" for i in range(len(sizes))])
    plt.ylabel("Number of Nodes")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.show()


def plot_final_fitness_boxplot(fitness_vals):
    plt.figure(figsize=(6, 4))
    plt.boxplot([[v] for v in fitness_vals], tick_labels=[f'Cluster {i+1}' for i in range(len(fitness_vals))])
    plt.ylabel("Final Fitness")
    plt.title("Final Fitness Across Clusters")
    plt.tight_layout()
    plt.show()


def plot_treebank(G, clusters, qb_solutions):
    for idx, (nodes, (bee, _)) in enumerate(zip(clusters, qb_solutions)):
        subg = G.subgraph(nodes)
        # Find correct edges for the tree representation by matching indices
        if len(subg.edges) == 0:
            continue  # skip empty subgraphs
        edge_list = list(subg.edges)
        edges_in_tree = [edge_list[i] for i, v in enumerate(bee) if v]
        H = nx.Graph()
        H.add_nodes_from(subg.nodes)
        H.add_edges_from(edges_in_tree)
        pos = {n: subg.nodes[n]['pos'] for n in subg.nodes}
        plt.figure(figsize=(5, 5))
        nx.draw(H, pos, node_color='lightblue', edge_color='green', node_size=50, width=2)
        plt.title(f"Optimized Tree for Cluster {idx+1}")
        plt.tight_layout()
        plt.show()


def plot_network(G):
    pos = {n: G.nodes[n]['pos'] for n in G.nodes}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=30, alpha=0.7, with_labels=False)
    plt.title("VANET Network Topology")
    plt.tight_layout()
    plt.show()


def plot_fitness_diversity(diversity_histories):
    plt.figure(figsize=(8, 4))
    for label, diversity in diversity_histories:
        plt.plot(diversity, label=f'{label} Diversity')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Diversity (Std. Dev.)")
    plt.title("Population Diversity Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


ctx = ContextHistory()
param_grid = []
for pop in [10, 20, 30]:
    for limit in [5, 10, 15]:
        param_grid.append([pop, limit])
param_grid = np.array(param_grid)


def main():
    G = create_sample_vanet(nnodes=30)
    clusters, memberships = fuzzy_cmeans_clustering(G, n_clusters=3)
    print("Fuzzy clusters:", clusters)

    histories = []
    final_fitness_vals = []
    qb_solutions = []        # Store (bee, fitness) for each cluster
    diversity_histories = []  # Optional, if diversity track is implemented

    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        qb = QBABC(subgraph, populationsize=max(10, len(nodes) * 3), maxiterations=100)
        hist = qb.optimize()
        histories.append((f'Cluster: {len(nodes)} nodes', hist))
        bee, bestfit = qb.get_best_solution()
        final_fitness_vals.append(bestfit)
        qb_solutions.append((bee, bestfit))
        print(f"Cluster {idx + 1} best fitness: {bestfit:.5f}")
        # If QBABC supports returning population snapshot each iteration, collect diversity here:
        # (Uncomment if supported)
        # diversity_histories.append((f'Cluster {idx+1}', qb.diversity_track))

    # Cryptography test with numpy arrays
    c = CryptoComm()
    x = np.random.rand(10)
    en = c.encrypt(x)
    de_bytes = c.decrypt(en)
    decrypted_x = np.frombuffer(de_bytes, dtype=x.dtype)
    assert np.allclose(decrypted_x, x, atol=1e-3)
    print("Crypto test passed.")

    # --- Visualizations ---

    # 1. Smoother convergence curves (cubic spline)
    plt.figure(figsize=(12, 6))
    plt.title("Convergence Curve")
    for label, hist in histories:
        x = np.arange(len(hist))
        y = np.array(hist)
        # Skip spline if too few points
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            plt.plot(x_smooth, y_smooth, label=label, linewidth=2)
        else:
            plt.plot(x, y, label=label, linewidth=2)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Cluster scatter plot
    plot_clusters(G, clusters)
    # 3. Cluster size bar chart
    plot_cluster_sizes(clusters)
    # 4. Network topology plot
    plot_network(G)
    # 5. Final fitness boxplot
    plot_final_fitness_boxplot(final_fitness_vals)
    # 6. Plot the tree bank (all optimized final trees)
    plot_treebank(G, clusters, qb_solutions)
    # 7. Fitness diversity plot (if you recorded diversity traces)
    ## plot_fitness_diversity(diversity_histories)


if __name__ == "__main__":
    main()
