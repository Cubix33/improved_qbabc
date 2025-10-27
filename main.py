import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qbabc import QBABC
from fuzzy import build_fuzzy_param_controller, fuzzy_adapt_fis, fuzzy_cmeans_clustering
from crypto import CryptoComm
from scipy.interpolate import make_interp_spline
from skfuzzy import control as ctrl

def create_sample_vanet(nnodes=30, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(nnodes):
        pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
        G.add_node(i, pos=pos)
    for i in range(nnodes):
        for j in range(i + 1, nnodes):
            dist = np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))
            if dist < 400:
                G.add_edge(i, j)
    for u, v in G.edges:
        G[u][v]['weight'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G

def plot_cluster_sizes(clusters):
    sizes = [len(cluster) for cluster in clusters]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(sizes) + 1), sizes, tick_label=[f"Cluster {i+1}" for i in range(len(sizes))])
    plt.ylabel("Number of Nodes")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.show()

def main():
    G = create_sample_vanet(nnodes=30)
    clusters, memberships = fuzzy_cmeans_clustering(G, n_clusters=3)
    print("Fuzzy clusters:", clusters)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} has {len(cluster)} nodes: {cluster}")

    # Plot cluster size distribution
    plot_cluster_sizes(clusters)

    histories = []
    final_fitness_vals = []
    qb_solutions = []

    # Build the fuzzy control system ONCE (not simulation)
    fuzzy_ctrl_system = build_fuzzy_param_controller().ctrl

    for idx, nodes in enumerate(clusters):
        if len(nodes) < 2:
            print(f"Skipping cluster {idx+1} with fewer than 2 nodes")
            continue
        subgraph = G.subgraph(nodes).copy()
        qb = QBABC(subgraph, populationsize=max(10, len(nodes) * 3), maxiterations=50)
        qb.initialize_population()
        params = {'beta': 0.5, 'limit': 10}
        hist = []
        for iter in range(qb.maxiterations):
            pop_fit = [qb.calculate_fitness(ind) for ind in qb.population] if qb.population else []
            diversity = np.std(pop_fit)/np.mean(pop_fit) if pop_fit and np.mean(pop_fit) > 0 else 0.5
            if len(hist) > 10:
                conv = (np.mean(hist[-5:])-np.mean(hist[-10:-5]))/max(np.mean(hist[-10:-5]),1e-6)
            else:
                conv = 0.5
            metrics = {'diversity':diversity,'convergence':abs(conv)}
            # Create a new simulation object for each call
            fuzzy_ctrl_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl_system)
            try:
                params = fuzzy_adapt_fis(params, metrics, fuzzy_ctrl_sim)
            except KeyError:
                pass
            qb.beta = params['beta']
            qb.scoutlimit = params['limit']
            qb.employed_bee_phase()
            qb.onlooker_bee_phase()
            qb.scout_bee_phase()
            bestfit = min([qb.calculate_fitness(ind) for ind in qb.population])
            hist.append(bestfit)
            if iter % 10 == 0 or iter==(qb.maxiterations-1):
                print(f"Cluster {idx+1} Iter {iter:2d} beta={qb.beta:.2f} scoutlimit={qb.scoutlimit:2d} best={bestfit:.2f}")
        bee, bestfit = qb.get_best_solution()
        histories.append((f'Cluster {idx+1}', hist))
        final_fitness_vals.append(bestfit)
        qb_solutions.append((bee, bestfit))

    # Crypto test
    c = CryptoComm()
    x = np.random.rand(10)
    en = c.encrypt(x)
    de_bytes = c.decrypt(en)
    decrypted_x = np.frombuffer(de_bytes, dtype=x.dtype)
    assert np.allclose(decrypted_x, x, atol=1e-3)
    print("Crypto test passed.")

    # Plot convergence
    plt.figure(figsize=(12, 6))
    plt.title("Convergence Curve (with fuzzy param)")
    for label, hist in histories:
        x = np.arange(len(hist))
        y = np.array(hist)
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 200)
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

    # Plot clusters (scatter)
    plt.figure(figsize=(8, 8))
    colors = plt.colormaps.get_cmap('tab10')
    for i, cluster_nodes in enumerate(clusters):
        xy = np.array([G.nodes[n]['pos'] for n in cluster_nodes])
        plt.scatter(xy[:, 0], xy[:, 1], label=f"Cluster {i+1} ({len(cluster_nodes)} nodes)", color=colors(i % 10))
    plt.title("Cluster Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally: Plot network topology
    pos = {n: G.nodes[n]['pos'] for n in G.nodes}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=30, alpha=0.7, with_labels=False)
    plt.title("VANET Network Topology")
    plt.tight_layout()
    plt.show()

    # Optionally: plot final fitness boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot([[v] for v in final_fitness_vals], tick_labels=[f"Cluster {i+1}" for i in range(len(final_fitness_vals))])
    plt.ylabel("Final Fitness")
    plt.title("Final Fitness Across Clusters")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
