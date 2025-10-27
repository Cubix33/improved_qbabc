import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Import Improved QBABC (VANET) ---
from improved_qbabc import ImprovedQBABC, fuzzy_cmeans_clustering, ContextualAgenticAI

# --- Import ABC+ACO WSN ---
from opti_WSN import ABCClustering, ACORouting, WSN, Node, RadioModel

def create_common_network(nnodes=20, area=1000, comm_range=450, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    positions = {}
    for i in range(nnodes):
        pos = (np.random.uniform(0, area), np.random.uniform(0, area))
        G.add_node(i, pos=pos)
        positions[i] = pos
    for i in range(nnodes):
        for j in range(i+1, nnodes):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            if dist < comm_range:
                G.add_edge(i, j, weight=dist)
    return G, positions

def qbabc_total_tree_cost(G):
    clusters, _ = fuzzy_cmeans_clustering(G, n_clusters=2)
    context_ai = ContextualAgenticAI()
    total_cost = 0
    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        params = context_ai.suggest_parameters(len(nodes), len(subgraph.edges))
        qb = ImprovedQBABC(subgraph,
                           populationsize=params['population_size'],
                           maxiterations=params['max_iterations'],
                           scoutlimit=params['scout_limit'])
        qb.optimize()
        best_ind, bestfit = qb.get_best_solution()
        total_cost += bestfit
    return total_cost

def abc_aco_total_tree_cost(G, positions, k=2, pop=20, limit=10, iters=25, seed=42):
    # Convert networkx graph to WSN object
    nodes = [Node(i, positions[i][0], positions[i][1], E=0.5) for i in G.nodes]
    bs = (np.mean([p[0] for p in positions.values()]), np.mean([p[1] for p in positions.values()]) + 200)
    wsn = WSN(nodes=nodes, bs=bs, radio=RadioModel())
    abc = ABCClustering(wsn, k=k, pop=pop, limit=limit, iters=iters, seed=seed)
    CH, clusters = abc.run()
    aco = ACORouting(wsn, CH.tolist(), alpha=1.0, beta=2.0, rho=0.4, ants=15, iters=15, seed=seed)
    tree = aco.run()
    # Calculate total cost of the routing tree (sum of edge weights)
    total_cost = 0
    for u, v in tree.items():
        if v is None:
            # CH to BS
            ch_pos = positions[u]
            total_cost += np.linalg.norm(np.array(ch_pos) - np.array(bs))
        else:
            total_cost += np.linalg.norm(np.array(positions[u]) - np.array(positions[v]))
    return total_cost

def main():
    G, positions = create_common_network(nnodes=20, area=1000, comm_range=450, seed=42)
    qbabc_cost = qbabc_total_tree_cost(G)
    aco_cost = abc_aco_total_tree_cost(G, positions, k=2, pop=20, limit=10, iters=25, seed=42)
    print(f"QBABC Total Tree Cost: {qbabc_cost:.2f}")
    print(f"ABC+ACO Total Routing Tree Cost: {aco_cost:.2f}")

    # Bar plot for comparison
    labels = ['Improved QBABC', 'ABC+ACO']
    vals = [qbabc_cost, aco_cost]
    plt.figure(figsize=(6,5))
    plt.bar(labels, vals, color=['#3498db','#e67e22'], alpha=0.85)
    plt.ylabel("Total Tree/Routing Cost")
    plt.title("Spanning Tree/Routing Cost Comparison")
    plt.tight_layout()
    plt.show()

    print("\n| Metric         | Improved QBABC | ABC+ACO |")
    print("|---------------|----------------|---------|")
    print(f"| Total Cost     | {qbabc_cost:.2f}         | {aco_cost:.2f}   |")

if __name__ == "__main__":
    main()
