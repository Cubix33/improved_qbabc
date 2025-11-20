import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import skfuzzy as fuzz
from cryptography.fernet import Fernet
import pickle
from sklearn.linear_model import RidgeCV
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

# ===========================
# FUZZY C-MEANS CLUSTERING
# ===========================
def fuzzy_cmeans_clustering(graph, n_clusters):
    node_positions = np.array([graph.nodes[n]['pos'] for n in graph.nodes])
    data = node_positions.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000)
    cluster_assignments = np.argmax(u, axis=0)
    clusters = []
    node_ids = list(graph.nodes)
    for i in range(n_clusters):
        idx = np.where(cluster_assignments == i)[0]
        clusters.append([node_ids[j] for j in idx])
    return clusters, u

# ===========================
# CRYPTOGRAPHY MODULE
# ===========================
class CryptoComm:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    def encrypt(self, data):
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)
    def decrypt(self, token):
        return self.cipher.decrypt(token)

# ===========================
# CONTEXTUAL AGENTIC AI MODULE
# ===========================
class ContextualAgenticAI:
    def __init__(self, filename='enhanced_context_history.pkl'):
        self.filename = filename
        self.history = self._load_history()
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.q_table = {}

    def _load_history(self):
        try:
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        except:
            return []

    def _save_history(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)

    def _get_state(self, cluster_size, n_edges):
        size_bin = cluster_size // 5
        edge_bin = n_edges // 10
        density = cluster_size / n_edges if n_edges > 0 else 0
        density_bin = int(density * 10)
        return (size_bin, edge_bin, density_bin)

    def _encode_action(self, params):
        pop_bin = params['population_size'] // 10
        iter_bin = params['max_iterations'] // 25
        scout_bin = params['scout_limit'] // 2
        return (pop_bin, iter_bin, scout_bin)

    def _decode_action(self, action):
        pop_bin, iter_bin, scout_bin = action
        return {
            'population_size': pop_bin * 10,
            'max_iterations': iter_bin * 25,
            'scout_limit': scout_bin * 2
        }

    def suggest_parameters(self, cluster_size, n_edges):
        state = self._get_state(cluster_size, n_edges)
        if len(self.history) < 5 or random.random() < self.exploration_rate:
            # Exploration
            params = self._explore_new_params(cluster_size, n_edges)
            print(f"  [Agentic AI] Exploring: {params}")
        else:
            # Exploitation
            if state in self.q_table and self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                params = self._decode_action(best_action)
                print(f"  [Agentic AI] Exploiting from Q-table: {params}")
            else:
                params = self._ml_predict(cluster_size, n_edges)
                print(f"  [Agentic AI] Exploiting ML prediction: {params}")

        # Decay exploration rate
        self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        return params
    
    def get_statistics(self):
        
        """Provide insights about learning progress"""
        if not self.history:
            return "No learning data yet"
        fitness_vals = [h['fitness'] for h in self.history]
        rewards = [h.get('reward', 0) for h in self.history]
        return {
            'total_runs': len(self.history),
            'best_fitness': min(fitness_vals),
            'avg_fitness': np.mean(fitness_vals),
            'learning_progress': np.mean(rewards[-5:]) if len(rewards) >= 5 else np.mean(rewards),
            'exploration_rate': self.exploration_rate,
            'q_table_size': sum(len(v) for v in self.q_table.values())
        }
        
    def _explore_new_params(self, cluster_size, n_edges):
        base_pop = cluster_size * 3
        return {
            'population_size': int(np.clip(base_pop + random.randint(-10, 20), 10, 100)),
            'max_iterations': random.choice([50, 75, 100, 125, 150]),
            'scout_limit': random.choice([5, 8, 10, 12, 15])
        }

    def _ml_predict(self, cluster_size, n_edges):
        if len(self.history) < 3:
            return {'population_size': cluster_size * 3, 'max_iterations': 100, 'scout_limit': 10}
        X = np.array([h['cluster_features'] for h in self.history])
        y_pop = np.array([h['params'][0] for h in self.history])
        y_iter = np.array([h['params'][1] for h in self.history])
        y_scout = np.array([h['params'][2] for h in self.history])
        density = cluster_size / n_edges if n_edges > 0 else 0
        features = np.array([[cluster_size, n_edges, density]])
        model = RidgeCV(alphas=[1.0, 10.0, 100.0])

        model.fit(X, y_pop)
        pred_pop = int(np.clip(model.predict(features)[0], 10, 100))
        model.fit(X, y_iter)
        pred_iter = int(np.clip(model.predict(features)[0], 50, 200))
        model.fit(X, y_scout)
        pred_scout = int(np.clip(model.predict(features)[0], 5, 20))
        return {'population_size': pred_pop, 'max_iterations': pred_iter, 'scout_limit': pred_scout}

    def record_optimization(self, cluster_size, n_edges, population_size, max_iterations, scout_limit, final_fitness, convergence_speed):
        state = self._get_state(cluster_size, n_edges)
        action = self._encode_action({'population_size': population_size, 'max_iterations': max_iterations, 'scout_limit': scout_limit})
        avg_fitness = np.mean([h['fitness'] for h in self.history]) if self.history else final_fitness
        reward = (avg_fitness - final_fitness) / avg_fitness if avg_fitness != 0 else final_fitness
        if state not in self.q_table:
            self.q_table[state] = {}
        old_q = self.q_table[state].get(action, 0)
        self.q_table[state][action] = old_q + self.learning_rate * (reward - old_q)
        record = {'cluster_features':[cluster_size,n_edges,cluster_size/n_edges if n_edges>0 else 0],
                  'params':[population_size,max_iterations,scout_limit],
                  'fitness':final_fitness,
                  'convergence_speed':convergence_speed,
                  'reward':reward}
        self.history.append(record)
        self._save_history()
        print(f"  [Agentic AI] Learned fitness={final_fitness:.4f}, reward={reward:.4f}")

# ===========================
# IMPROVED QBABC ALGORITHM
# ===========================
class ImprovedQBABC:
    def __init__(self, graph, populationsize=100, maxiterations=100, scoutlimit=10):
        self.graph = graph
        self.nnodes = len(graph.nodes)
        self.nedges = len(graph.edges)
        self.populationsize = populationsize
        self.maxiterations = maxiterations
        self.scoutlimit = scoutlimit
        self.qosweights = (0.3, 0.2, 0.2, 0.15, 0.15)
        self.edges = list(graph.edges)
        self.edgeweights = self.calculate_edge_weights()
        self.population = []
        self.fitnessvalues = []
        self.limit_counters = np.zeros(self.populationsize)
        self.history = []

    def calculate_edge_weights(self):
        weights = {}
        for edge in self.edges:
            u, v = edge
            posu = self.graph.nodes[u].get('pos', (0,0))
            posv = self.graph.nodes[v].get('pos', (0,0))
            dist = np.linalg.norm(np.array(posu)-np.array(posv))
            cost = dist
            delay = dist / (1.5 * 3e8) + np.random.uniform(0.01, 0.1)
            packetloss = np.random.uniform(0,1)
            jitter = np.random.uniform(5,18)
            bandwidth = np.random.uniform(20,280)
            weight = (self.qosweights[0]*cost + self.qosweights[1]*delay + self.qosweights[2]*packetloss
                      + self.qosweights[3]*jitter - self.qosweights[4]*bandwidth)
            weights[edge] = weight
        return weights

    def qrng_initialize(self):
        """Use actual quantum random number generation"""
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
    
        n_qubits = int(np.ceil(np.log2(self.nedges)))
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(range(n_qubits))  # Hadamard gates
        qc.measure(range(n_qubits), range(n_qubits))
    
        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=self.nnodes-1).result()
    
    # Convert quantum measurements to edge indices
        counts = result.get_counts()
        indices = [int(bitstring, 2) % self.nedges for bitstring in counts.keys()]
    
        v = np.zeros(self.nedges, dtype=int)
        v[indices[:self.nnodes-1]] = 1
        return v


    def is_feasible(self, individual):
        if np.sum(individual) != self.nnodes -1:
            return False
        edges_in = [self.edges[i] for i in range(len(individual)) if individual[i]==1]
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(edges_in)
        return nx.is_connected(temp_graph) and len(temp_graph.nodes)==self.nnodes

    def calculate_fitness(self, individual):
        if self.is_feasible(individual):
            return sum(self.edgeweights[self.edges[i]] for i in range(len(individual)) if individual[i]==1)
        else:
            return sum(self.edgeweights.values())*10

    def initialize_population(self):
        self.population = []
        for _ in range(self.populationsize):
            while True:
                ind=self.qrng_initialize()
                if self.is_feasible(ind):
                    self.population.append(ind)
                    break
        self.fitnessvalues = [self.calculate_fitness(ind) for ind in self.population]
        self.limit_counters = np.zeros(self.populationsize)

    def generate_candidate(self, bee, peer):
        candidate = bee.copy()
        pos = np.random.randint(len(bee))
        candidate[pos] = 1 - candidate[pos]
        if not self.is_feasible(candidate):
            candidate[pos] = bee[pos]
        return candidate

    def calculate_probabilities(self):
        fit_inv = np.max(self.fitnessvalues) - np.array(self.fitnessvalues) + 1e-10
        prob = fit_inv / np.sum(fit_inv)
        return prob

    def employed_bee_phase(self):
        for idx in range(self.populationsize):
            k = np.random.choice([i for i in range(self.populationsize) if i != idx])
            candidate=self.generate_candidate(self.population[idx], self.population[k])
            candidate_fitness=self.calculate_fitness(candidate)
            if candidate_fitness < self.fitnessvalues[idx]:
                self.population[idx] = candidate
                self.fitnessvalues[idx] = candidate_fitness
                self.limit_counters[idx] = 0
            else:
                self.limit_counters[idx] += 1

    def onlooker_bee_phase(self):
        prob = self.calculate_probabilities()
        for idx in range(self.populationsize):
            if np.random.rand() < prob[idx]:
                k = np.random.choice([i for i in range(self.populationsize) if i != idx])
                candidate=self.generate_candidate(self.population[idx], self.population[k])
                candidate_fitness=self.calculate_fitness(candidate)
                if candidate_fitness < self.fitnessvalues[idx]:
                    self.population[idx]=candidate
                    self.fitnessvalues[idx]=candidate_fitness
                    self.limit_counters[idx] =0
                else:
                    self.limit_counters[idx]+=1

    def scout_bee_phase(self):
        for idx in range(self.populationsize):
            if self.limit_counters[idx]>self.scoutlimit:
                new_ind=None
                for _ in range(100):
                    candidate=self.qrng_initialize()
                    if self.is_feasible(candidate):
                        new_ind=candidate
                        break
                if new_ind is not None:
                    self.population[idx]=new_ind
                    self.fitnessvalues[idx]=self.calculate_fitness(new_ind)
                    self.limit_counters[idx]=0

    def optimize(self, verbose=True):
        self.initialize_population()
        history=[]
        best_so_far=float('inf')
        for iteration in range(self.maxiterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            current_best=np.min(self.fitnessvalues)
            if current_best<best_so_far:
                best_so_far=current_best
            history.append(best_so_far)
            if verbose and (iteration%20==0 or iteration==self.maxiterations-1):
                print(f'Iteration {iteration+1}/{self.maxiterations} Best-so-far: {best_so_far:.5f}')
        self.history=history
        return history

    def get_best_solution(self):
        min_idx=np.argmin(self.fitnessvalues)
        return self.population[min_idx], self.fitnessvalues[min_idx]

    def get_convergence_speed(self):
        if not self.history:
            return self.maxiterations
        final = self.history[-1]
        target = final * 1.1
        for i, fitness in enumerate(self.history):
            if fitness <= target:
                return i
        return self.maxiterations

# ===========================
# VANET CREATION
# ===========================
def create_sample_vanet(nnodes=30, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(nnodes):
        pos = (np.random.uniform(0,1000), np.random.uniform(0,1000))
        G.add_node(i, pos=pos)
    for i in range(nnodes):
        for j in range(i+1, nnodes):
            if np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))<400:
                G.add_edge(i,j)
    for u,v in G.edges:
        G[u][v]['weight'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G

# ===========================
# PLOTTING UTILITIES
# ===========================
def plot_clusters(G, clusters):
    plt.figure(figsize=(8,8))
    colors = plt.get_cmap('tab10')
    for i, cluster in enumerate(clusters):
        xy = np.array([G.nodes[n]['pos'] for n in cluster])
        plt.scatter(xy[:,0], xy[:,1], label=f'Cluster {i+1} ({len(cluster)} nodes)', color=colors(i%10))
    plt.title("Cluster Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cluster_sizes(clusters):
    sizes = [len(c) for c in clusters]
    plt.figure(figsize=(6,4))
    plt.bar(range(1,len(sizes)+1), sizes, tick_label=[f'Cluster {i+1}' for i in range(len(sizes))])
    plt.ylabel("Number of Nodes")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.show()

def plot_convergence(histories):
    plt.figure(figsize=(12,6))
    for label, hist in histories:
        x = np.arange(len(hist))
        y = np.array(hist)
        if len(x)>3:
            x_smooth = np.linspace(x.min(),x.max(),300)
            spl=make_interp_spline(x,y,k=3)
            y_smooth= spl(x_smooth)
            plt.plot(x_smooth, y_smooth, label=label, linewidth=2)
        else:
            plt.plot(x,y,label=label, linewidth=2)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===========================
# MAIN EXECUTION WITH AGENTIC AI
# ===========================
def main():
    print("IMPROVED QBABC WITH CONTEXTUAL AGENTIC AI\n"+"="*50)
    context_ai = ContextualAgenticAI()

    G = create_sample_vanet(nnodes=25)
    clusters, _ = fuzzy_cmeans_clustering(G, n_clusters=3)

    print(f"[FCM] Created {len(clusters)} clusters")
    for i,cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {len(cluster)} nodes")

    histories = []
    final_fitness_vals = []

    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        cluster_size = len(nodes)
        n_edges = len(subgraph.edges)

        print(f"\nCluster {idx+1} optimization:")
        suggested_params = context_ai.suggest_parameters(cluster_size, n_edges)

        qb = ImprovedQBABC(subgraph, 
                          populationsize=suggested_params['population_size'],
                          maxiterations=suggested_params['max_iterations'],
                          scoutlimit=suggested_params['scout_limit'])

        hist = qb.optimize()
        _, bestfit = qb.get_best_solution()
        conv_speed = qb.get_convergence_speed()

        context_ai.record_optimization(cluster_size, n_edges, suggested_params['population_size'],
                                       suggested_params['max_iterations'], suggested_params['scout_limit'],
                                       bestfit, conv_speed)

        histories.append((f"Cluster {idx+1} ({cluster_size} nodes)", hist))
        final_fitness_vals.append(bestfit)
        print(f"Final fitness: {bestfit:.4f}; Convergence speed: {conv_speed} iterations")

    print("\nCryptography test:")
    crypto = CryptoComm()
    data = np.random.rand(10)
    encrypted = crypto.encrypt(data)
    decrypted = crypto.decrypt(encrypted)
    decrypted_data = np.frombuffer(decrypted, dtype=data.dtype)
    assert np.allclose(data, decrypted_data, atol=1e-3)
    print("Cryptography test passed")

    total_fitness = sum(final_fitness_vals)
    print(f"\nOptimization summary: Total fitness: {total_fitness:.4f}")
    for i, f in enumerate(final_fitness_vals, 1):
        print(f"  Cluster {i}: {f:.4f}")
    plot_convergence(histories)
    plot_clusters(G, clusters)
    plot_cluster_sizes(clusters)

    stats = context_ai.get_statistics()
    if isinstance(stats, dict):
        print("Context AI stats:")
        for k,v in stats.items():
            if isinstance(v, float):
                print(f" {k}: {v:.4f}")
            else:
                print(f" {k}: {v}")
    else:
        print(f"Context AI: {stats}")

if __name__=="__main__":
    main()
