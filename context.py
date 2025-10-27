# ========================================
# COMPLETE IMPROVED QBABC WITH CONTEXTUAL AGENTIC AI
# ========================================

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import make_interp_spline
from cryptography.fernet import Fernet
import pickle
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings('ignore')

# ========================================
# FUZZY MODULE
# ========================================

def build_fuzzy_param_controller():
    diversity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'diversity')
    convergence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'convergence')
    beta = ctrl.Consequent(np.arange(0.3, 0.81, 0.01), 'beta')
    scout_limit = ctrl.Consequent(np.arange(8, 16, 1), 'scout_limit')

    diversity['low'] = fuzz.trimf(diversity.universe, [0, 0, 0.25])
    diversity['medium'] = fuzz.trimf(diversity.universe, [0.15, 0.5, 0.85])
    diversity['high'] = fuzz.trimf(diversity.universe, [0.75, 1, 1])
    convergence['slow'] = fuzz.trimf(convergence.universe, [0, 0, 0.25])
    convergence['medium'] = fuzz.trimf(convergence.universe, [0.15, 0.5, 0.85])
    convergence['fast'] = fuzz.trimf(convergence.universe, [0.75, 1, 1])
    beta['low'] = fuzz.trimf(beta.universe, [0.3, 0.3, 0.45])
    beta['medium'] = fuzz.trimf(beta.universe, [0.4, 0.55, 0.7])
    beta['high'] = fuzz.trimf(beta.universe, [0.65, 0.8, 0.8])
    scout_limit['small'] = fuzz.trimf(scout_limit.universe, [8, 8, 11])
    scout_limit['normal'] = fuzz.trimf(scout_limit.universe, [10, 13, 15])
    scout_limit['large'] = fuzz.trimf(scout_limit.universe, [14, 16, 16])

    rules = [
        ctrl.Rule(diversity['low'] & convergence['slow'], (beta['high'], scout_limit['small'])),
        ctrl.Rule(diversity['high'] & convergence['fast'], (beta['low'], scout_limit['large'])),
        ctrl.Rule(diversity['medium'] | convergence['medium'], (beta['medium'], scout_limit['normal'])),
        ctrl.Rule(diversity['low'] & convergence['medium'], (beta['medium'], scout_limit['small'])),
        ctrl.Rule(diversity['high'] & convergence['medium'], (beta['low'], scout_limit['normal'])),
        ctrl.Rule(diversity['medium'] & convergence['fast'], (beta['medium'], scout_limit['normal'])),
        ctrl.Rule(diversity['medium'] & convergence['slow'], (beta['high'], scout_limit['normal'])),
    ]
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim

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

# ========================================
# CRYPTO MODULE
# ========================================

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

# ========================================
# CONTEXTUAL AGENTIC AI MODULE (ENHANCED)
# ========================================

class ContextualAgenticAI:
    """
    Contextual Agentic AI that learns from past optimizations and suggests
    optimal parameters for future runs based on cluster characteristics
    """
    def __init__(self, filename='context_history.pkl'):
        self.filename = filename
        self.history = self._load_history()
        self.model = None
        
    def _load_history(self):
        """Load historical optimization data"""
        try:
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        except:
            return []
    
    def _save_history(self):
        """Save optimization history to disk"""
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)
    
    def record_optimization(self, cluster_size, n_edges, population_size, max_iterations, 
                           scout_limit, final_fitness, convergence_speed):
        """
        Record the results of an optimization run
        
        Args:
            cluster_size: Number of nodes in cluster
            n_edges: Number of edges in cluster
            population_size: Population size used
            max_iterations: Iterations used
            scout_limit: Scout limit parameter
            final_fitness: Final fitness achieved
            convergence_speed: How fast it converged (iterations to 90% of final)
        """
        record = {
            'cluster_features': [cluster_size, n_edges, cluster_size/n_edges if n_edges > 0 else 0],
            'params': [population_size, max_iterations, scout_limit],
            'fitness': final_fitness,
            'convergence_speed': convergence_speed
        }
        self.history.append(record)
        self._save_history()
        print(f"  [Context AI] Recorded optimization: Cluster size={cluster_size}, Fitness={final_fitness:.4f}")
    
    def suggest_parameters(self, cluster_size, n_edges):
        """
        Suggest optimal parameters based on cluster characteristics using ML
        
        Args:
            cluster_size: Number of nodes in the new cluster
            n_edges: Number of edges in the new cluster
            
        Returns:
            dict: Suggested parameters {population_size, max_iterations, scout_limit}
        """
        # Default parameters if no history
        if len(self.history) < 3:
            default = {
                'population_size': max(10, cluster_size * 3),
                'max_iterations': 100,
                'scout_limit': 10
            }
            print(f"  [Context AI] Not enough history ({len(self.history)} records). Using defaults.")
            return default
        
        # Extract features and targets from history
        X = np.array([h['cluster_features'] for h in self.history])
        y_pop = np.array([h['params'][0] for h in self.history])
        y_iter = np.array([h['params'][1] for h in self.history])
        y_scout = np.array([h['params'][2] for h in self.history])
        
        # Current cluster features
        density = cluster_size / n_edges if n_edges > 0 else 0
        current_features = np.array([[cluster_size, n_edges, density]])
        
        # Train models and predict
        try:
            # Population size predictor
            model_pop = RidgeCV(alphas=[1.0, 10.0, 100.0])
            model_pop.fit(X, y_pop)
            pred_pop = int(np.clip(model_pop.predict(current_features)[0], 10, 100))
            
            # Max iterations predictor
            model_iter = RidgeCV(alphas=[1.0, 10.0, 100.0])
            model_iter.fit(X, y_iter)
            pred_iter = int(np.clip(model_iter.predict(current_features)[0], 50, 200))
            
            # Scout limit predictor
            model_scout = RidgeCV(alphas=[1.0, 10.0, 100.0])
            model_scout.fit(X, y_scout)
            pred_scout = int(np.clip(model_scout.predict(current_features)[0], 5, 20))
            
            suggested = {
                'population_size': pred_pop,
                'max_iterations': pred_iter,
                'scout_limit': pred_scout
            }
            
            print(f"  [Context AI] Suggested params based on {len(self.history)} past optimizations:")
            print(f"    Population: {pred_pop}, Iterations: {pred_iter}, Scout Limit: {pred_scout}")
            
            return suggested
            
        except Exception as e:
            print(f"  [Context AI] Prediction failed: {e}. Using defaults.")
            return {
                'population_size': max(10, cluster_size * 3),
                'max_iterations': 100,
                'scout_limit': 10
            }
    
    def get_best_historical_params(self):
        """Get parameters that achieved best fitness in history"""
        if not self.history:
            return None
        best_record = min(self.history, key=lambda x: x['fitness'])
        return {
            'population_size': best_record['params'][0],
            'max_iterations': best_record['params'][1],
            'scout_limit': best_record['params'][2],
            'fitness': best_record['fitness']
        }
    
    def get_statistics(self):
        """Get statistics about historical optimizations"""
        if not self.history:
            return "No history available"
        
        fitness_vals = [h['fitness'] for h in self.history]
        return {
            'total_runs': len(self.history),
            'best_fitness': min(fitness_vals),
            'avg_fitness': np.mean(fitness_vals),
            'worst_fitness': max(fitness_vals),
            'std_fitness': np.std(fitness_vals)
        }

# ========================================
# QBABC MODULE (WITH CONTEXT AI INTEGRATION)
# ========================================

class QBABC:
    def __init__(self, graph, populationsize=100, maxiterations=500, scoutlimit=10, qosweights=None):
        self.graph = graph
        self.nnodes = len(graph.nodes)
        self.nedges = len(graph.edges)
        self.populationsize = populationsize
        self.maxiterations = maxiterations
        self.scoutlimit = scoutlimit
        self.qosweights = qosweights or (0.3, 0.2, 0.2, 0.15, 0.15)
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
            posu = self.graph.nodes[u].get('pos', (0, 0))
            posv = self.graph.nodes[v].get('pos', (0, 0))
            dist = np.linalg.norm(np.array(posu) - np.array(posv))
            cost = dist
            delay = dist / (1.5 * 3e8) + np.random.uniform(0.01, 0.1)
            packetloss = np.random.uniform(0, 1)
            jitter = np.random.uniform(5, 18)
            bandwidth = np.random.uniform(20, 280)
            weight = (self.qosweights[0] * cost + self.qosweights[1] * delay +
                      self.qosweights[2] * packetloss + self.qosweights[3] * jitter -
                      self.qosweights[4] * bandwidth)
            weights[edge] = weight
        return weights

    def qrng_initialize(self):
        v = np.zeros(self.nedges, dtype=int)
        indices = np.random.choice(self.nedges, self.nnodes - 1, replace=False)
        v[indices] = 1
        return v

    def is_feasible(self, individual):
        if np.sum(individual) != self.nnodes - 1:
            return False
        edges_in = [self.edges[i] for i in range(len(individual)) if individual[i] == 1]
        tempgraph = nx.Graph()
        tempgraph.add_edges_from(edges_in)
        return nx.is_connected(tempgraph) and len(tempgraph.nodes) == self.nnodes

    def calculate_fitness(self, individual):
        if self.is_feasible(individual):
            return sum(self.edgeweights[self.edges[i]] for i in range(len(individual)) if individual[i] == 1)
        else:
            return sum(self.edgeweights.values()) * 10

    def initialize_population(self):
        self.population = []
        for _ in range(self.populationsize):
            while True:
                ind = self.qrng_initialize()
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
            candidate = self.generate_candidate(self.population[idx], self.population[k])
            candidate_fitness = self.calculate_fitness(candidate)
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
                candidate = self.generate_candidate(self.population[idx], self.population[k])
                candidate_fitness = self.calculate_fitness(candidate)
                if candidate_fitness < self.fitnessvalues[idx]:
                    self.population[idx] = candidate
                    self.fitnessvalues[idx] = candidate_fitness
                    self.limit_counters[idx] = 0
                else:
                    self.limit_counters[idx] += 1

    def scout_bee_phase(self):
        for idx in range(self.populationsize):
            if self.limit_counters[idx] > self.scoutlimit:
                new_ind = None
                for _ in range(100):
                    candidate = self.qrng_initialize()
                    if self.is_feasible(candidate):
                        new_ind = candidate
                        break
                if new_ind is not None:
                    self.population[idx] = new_ind
                    self.fitnessvalues[idx] = self.calculate_fitness(new_ind)
                    self.limit_counters[idx] = 0

    def optimize(self, verbose=True):
        self.initialize_population()
        history = []
        best_so_far = float('inf')
        for iteration in range(self.maxiterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            current_best = np.min(self.fitnessvalues)
            if current_best < best_so_far:
                best_so_far = current_best
            history.append(best_so_far)
            if verbose and (iteration % 20 == 0 or iteration == self.maxiterations - 1):
                print(f"    Iteration {iteration+1}/{self.maxiterations}, Best-so-far: {best_so_far:.5f}")
        self.history = history
        return history

    def get_best_solution(self):
        min_idx = np.argmin(self.fitnessvalues)
        return self.population[min_idx], self.fitnessvalues[min_idx]
    
    def get_convergence_speed(self):
        """Calculate how fast the algorithm converged (iterations to reach 90% of final fitness)"""
        if not self.history:
            return self.maxiterations
        final_fitness = self.history[-1]
        target = final_fitness * 1.1  # 90% improvement
        for i, fitness in enumerate(self.history):
            if fitness <= target:
                return i
        return self.maxiterations

# ========================================
# MAIN MODULE WITH CONTEXTUAL AI
# ========================================

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

def plot_convergence_comparison(histories, title="Convergence Curves"):
    plt.figure(figsize=(12, 6))
    for label, hist in histories:
        x = np.arange(len(hist))
        y = np.array(hist)
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
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("="*80)
    print("IMPROVED QBABC WITH CONTEXTUAL AGENTIC AI")
    print("="*80)
    
    # Initialize Contextual AI
    context_ai = ContextualAgenticAI()
    
    # Show historical statistics if available
    stats = context_ai.get_statistics()
    if isinstance(stats, dict):
        print(f"\n[Context AI] Historical Statistics:")
        print(f"  Total past runs: {stats['total_runs']}")
        print(f"  Best fitness ever: {stats['best_fitness']:.4f}")
        print(f"  Average fitness: {stats['avg_fitness']:.4f}")
    else:
        print(f"\n[Context AI] {stats}")
    
    # Create network
    G = create_sample_vanet(nnodes=30)
    clusters, memberships = fuzzy_cmeans_clustering(G, n_clusters=3)
    print(f"\n[FCM] Created {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} nodes")
    
    histories = []
    final_fitness_vals = []
    qb_solutions = []
    
    # Optimize each cluster with Context AI suggestions
    for idx, nodes in enumerate(clusters):
        subgraph = G.subgraph(nodes).copy()
        cluster_size = len(nodes)
        n_edges = len(subgraph.edges)
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {idx + 1} OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Cluster size: {cluster_size} nodes, {n_edges} edges")
        
        # Get parameter suggestions from Context AI
        suggested_params = context_ai.suggest_parameters(cluster_size, n_edges)
        
        # Create and run QBABC with suggested parameters
        qb = QBABC(
            subgraph, 
            populationsize=suggested_params['population_size'],
            maxiterations=suggested_params['max_iterations'],
            scoutlimit=suggested_params['scout_limit']
        )
        
        hist = qb.optimize(verbose=True)
        bee, bestfit = qb.get_best_solution()
        convergence_speed = qb.get_convergence_speed()
        
        # Record optimization results in Context AI
        context_ai.record_optimization(
            cluster_size=cluster_size,
            n_edges=n_edges,
            population_size=suggested_params['population_size'],
            max_iterations=suggested_params['max_iterations'],
            scout_limit=suggested_params['scout_limit'],
            final_fitness=bestfit,
            convergence_speed=convergence_speed
        )
        
        histories.append((f'Cluster {idx+1} ({cluster_size} nodes)', hist))
        final_fitness_vals.append(bestfit)
        qb_solutions.append((bee, bestfit))
        
        print(f"\n  ✓ Final Fitness: {bestfit:.5f}")
        print(f"  ✓ Convergence Speed: {convergence_speed} iterations")
    
    # Cryptography test
    print(f"\n{'='*80}")
    print("CRYPTOGRAPHY TEST")
    print(f"{'='*80}")
    c = CryptoComm()
    x = np.random.rand(10)
    en = c.encrypt(x)
    de_bytes = c.decrypt(en)
    decrypted_x = np.frombuffer(de_bytes, dtype=x.dtype)
    assert np.allclose(decrypted_x, x, atol=1e-3)
    print("✓ Cryptography test passed")
    
    # Final summary
    total_fitness = sum(final_fitness_vals)
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Best Fitness: {total_fitness:.4f}")
    for i, fitness in enumerate(final_fitness_vals):
        print(f"  Cluster {i+1}: {fitness:.4f}")
    
    # Plot results
    plot_convergence_comparison(histories, "Convergence Curves with Context AI")
    plot_clusters(G, clusters)
    plot_cluster_sizes(clusters)
    
    # Show updated Context AI statistics
    print(f"\n{'='*80}")
    print("UPDATED CONTEXTUAL AI STATISTICS")
    print(f"{'='*80}")
    stats = context_ai.get_statistics()
    print(f"Total optimizations recorded: {stats['total_runs']}")
    print(f"Best fitness achieved: {stats['best_fitness']:.4f}")
    print(f"Average fitness: {stats['avg_fitness']:.4f}")
    print(f"Std deviation: {stats['std_fitness']:.4f}")

if __name__ == "__main__":
    main()
