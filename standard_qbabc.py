import numpy as np
import random
import networkx as nx
from math import comb
from typing import List, Dict, Tuple, Any
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
 

class QBABC:
    """
    Quantum Binary Improved Artificial Bee Colony Algorithm
    
    Based on the mathematical formulations in Pan et al. (2024):
    - QoS weight calculation (Equations 1-6)
    - Spanning Tree Hit Rate (Equations 15-16)
    - Quantum Random Number Generator simulation
    - Fitness function (Equation 12)
    - Search strategies (Equations 20-25)
    """
    
    def __init__(self, graph: nx.Graph, population_size: int = 100,
                 max_iterations: int = 500, employed_ratio: float = 0.5,
                 onlooker_ratio: float = 0.5, scout_limit: int = 10):
        """
        Initialize QBABC algorithm with parameters from Table II
        
        Args:
            graph: Input VANET graph
            population_size: Population size (sn = 100)
            max_iterations: Maximum iterations (500)
            employed_ratio: Employed bee ratio (0.5)
            onlooker_ratio: Onlooker bee ratio (0.5)
            scout_limit: Scout bee limit (10)
        """
        self.graph = graph
        self.n_nodes = len(graph.nodes())
        self.n_edges = len(graph.edges())
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.scout_limit = scout_limit
        
        # QoS parameters (α1 to α5) from Equation 1
        self.qos_weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Σαi = 1
        
        # Initialize edge list and weights
        self.edges = list(graph.edges())
        self.edge_weights = self._calculate_edge_weights()
        
        # Population and counters
        self.population = []
        self.counters = []
        self.fitness_values = []
    
    def _calculate_edge_weights(self) -> Dict[Tuple, float]:
        """
        Calculate edge weights using QoS metrics (Equations 1-6)
        
        Returns:
            Dictionary mapping edges to their QoS-based weights
        """
        weights = {}
        c = 3e8  # speed of light (m/s)
        beta = 1.0  # coefficient for cost calculation
        
        for edge in self.edges:
            u, v = edge
            
            # Calculate Euclidean distance
            if ('pos' in self.graph.nodes[u] and
                'pos' in self.graph.nodes[v]):
                pos_u = self.graph.nodes[u]['pos']
                pos_v = self.graph.nodes[v]['pos']
                dist = np.sqrt((pos_u[0] - pos_v[0])**2 +
                             (pos_u[1] - pos_v[1])**2)
            else:
                dist = random.uniform(50, 200)  # Default random distance
            
            # Calculate QoS components (Equations 2-6)
            cost = beta * dist  # Equation 2
            delay = (dist / (1.5 * c) +
                    random.uniform(0, 1) * 90 + 10)  # Equation 3
            packet_loss = random.uniform(0, 1)  # Equation 4
            jitter = random.uniform(0, 1) * 13 + 5  # Equation 5
            bandwidth = random.uniform(0, 1) * 260 + 20  # Equation 6
            
            # Equation 1: Combined weight calculation
            weight = (self.qos_weights[0] * cost +
                     self.qos_weights[1] * delay +
                     self.qos_weights[2] * packet_loss +
                     self.qos_weights[3] * jitter -
                     self.qos_weights[4] * bandwidth)
            
            weights[edge] = weight
        
        return weights
    
    def _spanning_tree_hit_rate(self) -> Tuple[float, int, int]:
        """
        Calculate Spanning Tree Hit Rate (Equations 15-16) using Kirchhoff's Theorem
        
        Returns:
            Tuple of (SHR, total_combinations, number_of_spanning_trees)
        """
        # Calculate C^(n-1)_l combinations
        if self.n_nodes <= 1: return 0, 0, 0
        combinations = comb(self.n_edges, self.n_nodes - 1)
        
        # Calculate the number of spanning trees using Kirchhoff's Theorem
        if not nx.is_connected(self.graph):
            number_of_spanning_trees = 0
        else:
            laplacian = nx.laplacian_matrix(self.graph).toarray()
            cofactor = laplacian[1:, 1:]
            try:
                number_of_spanning_trees = int(round(abs(np.linalg.det(cofactor))))
            except np.linalg.LinAlgError:
                number_of_spanning_trees = 0
        
        shr = number_of_spanning_trees / combinations if combinations > 0 else 0
        return shr, combinations, number_of_spanning_trees
    
    def _qrng_initialize(self) -> np.ndarray:
        """
        QRNG initialization using quantum circuits
        """
        individual = np.zeros(self.n_edges, dtype=int)
        if self.n_nodes <= 1: return individual
        
        n = int(np.ceil(np.log2(self.n_edges))) if self.n_edges > 0 else 0
        if n == 0: return individual
        
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        qc.measure(range(n), range(n))
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=100)
        result = job.result()
        counts = result.get_counts()
        l = list(counts.keys())
        
        count, idx = 0, 0
        while count < self.n_nodes - 1 and idx < len(l):
            s = int(l[idx], 2)
            idx += 1
            if s < self.n_edges and individual[s] == 0:
                individual[s] = 1
                count += 1
        
        current_ones = np.sum(individual)
        if current_ones < self.n_nodes - 1:
            zero_indices = [i for i in range(self.n_edges) if individual[i] == 0]
            needed = self.n_nodes - 1 - current_ones
            additional = random.sample(zero_indices, min(needed, len(zero_indices)))
            for idx in additional:
                individual[idx] = 1
        elif current_ones > self.n_nodes - 1:
            one_indices = [i for i in range(self.n_edges) if individual[i] == 1]
            excess = current_ones - (self.n_nodes - 1)
            to_remove = random.sample(one_indices, min(excess, len(one_indices)))
            for idx in to_remove:
                individual[idx] = 0
        
        return individual
    
    def _is_feasible(self, individual: np.ndarray) -> bool:
        """
        Check feasibility constraints (Equation 11)
        """
        if self.n_nodes <= 1: return np.sum(individual) == 0
        
        if np.sum(individual) != self.n_nodes - 1:
            return False
        
        selected_edges = [self.edges[i] for i in range(len(individual))
                         if individual[i] == 1]
        
        if not selected_edges:
            return False
        
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(selected_edges)
        
        try:
            is_connected = nx.is_connected(temp_graph)
            has_all_nodes = len(temp_graph.nodes()) == self.n_nodes
            return is_connected and has_all_nodes
        except:
            return False
    
    def _calculate_fitness(self, individual: np.ndarray) -> float:
        """
        Calculate fitness value using Equation 12
        """
        if self._is_feasible(individual):
            fitness = sum(self.edge_weights[self.edges[i]] * individual[i]
                         for i in range(len(individual)))
        else:
            fitness = sum(self.edge_weights.values())
        
        return fitness
    
    def _improved_replacement_strategy(self, original: np.ndarray,
                                      new_solution: np.ndarray) -> np.ndarray:
        """
        Improved Replacement Search Strategy (Equation 20)
        """
        if self._is_feasible(new_solution):
            infeasible_exists = any(not self._is_feasible(ind)
                                   for ind in self.population)
            if infeasible_exists:
                return new_solution
            
            new_fitness = self._calculate_fitness(new_solution)
            original_fitness = self._calculate_fitness(original)
            population_tuples = {tuple(ind) for ind in self.population}
            is_duplicate = tuple(new_solution) in population_tuples
            
            if new_fitness < original_fitness and not is_duplicate:
                return new_solution
            else:
                return original
        else:
            try:
                original_idx = -1
                for i, ind in enumerate(self.population):
                    if np.array_equal(ind, original):
                        original_idx = i
                        break
                if original_idx == -1: return original
                counter = self.counters[original_idx]
            except (ValueError, IndexError):
                return original
            
            sorted_indices = np.argsort(counter)
            for i in range(len(sorted_indices) - 1):
                j, k = sorted_indices[i], sorted_indices[i + 1]
                if original[j] != original[k]:
                    new_modified = original.copy()
                    new_modified[j], new_modified[k] = new_modified[k], new_modified[j]
                    if self._is_feasible(new_modified):
                        new_fitness = self._calculate_fitness(new_modified)
                        original_fitness = self._calculate_fitness(original)
                        if new_fitness < original_fitness:
                            return new_modified
            return original
    
    def _employed_bee_search(self):
        """
        Employed Bee Search Strategy (Equation 21)
        """
        new_population = []
        for i in range(len(self.population)):
            individual = self.population[i]
            k = random.randint(0, len(self.population) - 1)
            while k == i and len(self.population) > 1:
                k = random.randint(0, len(self.population) - 1)
            partner = self.population[k] if k < len(self.population) else individual
            new_solution = self._create_offspring(individual, partner, 'AND')
            result = self._improved_replacement_strategy(individual, new_solution)
            new_population.append(result)
        self.population = new_population
    
    def _onlooker_bee_search(self):
        """
        Onlooker Bee Search Strategy (Equations 23-24)
        """
        if not self.population: return
        fitness_values = [self._calculate_fitness(ind) for ind in self.population]
        best_idx = np.argmin(fitness_values)
        queen = self.population[best_idx]
        new_population = []
        for individual in self.population:
            solution_or = self._create_offspring(individual, queen, 'OR')
            solution_and = self._create_offspring(individual, queen, 'AND')
            fitness_or = self._calculate_fitness(solution_or)
            fitness_and = self._calculate_fitness(solution_and)
            new_solution = solution_or if fitness_or <= fitness_and else solution_and
            result = self._improved_replacement_strategy(individual, new_solution)
            new_population.append(result)
        self.population = new_population
    
    def _scouter_bee_search(self):
        """
        Scouter Bee Search Strategy (Equation 25)
        """
        for i in range(len(self.population)):
            if not self._is_feasible(self.population[i]):
                self.population[i] = self._qrng_initialize()
        
        feasible_solutions_tuples = set()
        new_population = []
        for individual in self.population:
            if self._is_feasible(individual):
                individual_tuple = tuple(individual)
                if individual_tuple in feasible_solutions_tuples:
                    new_population.append(self._qrng_initialize())
                else:
                    feasible_solutions_tuples.add(individual_tuple)
                    new_population.append(individual)
            else:
                new_population.append(individual)
        self.population = new_population
    
    def _create_offspring(self, parent1: np.ndarray, parent2: np.ndarray,
                         operation: str) -> np.ndarray:
        """
        Create offspring using genetic operations
        """
        if self.n_nodes <= 1: return parent1.copy()
        if operation == 'AND':
            result = parent1 & parent2
        elif operation == 'OR':
            result = parent1 | parent2
        else:
            result = parent1.copy()
        
        current_ones = np.sum(result)
        if current_ones < self.n_nodes - 1:
            zero_indices = [i for i in range(self.n_edges) if result[i] == 0]
            if zero_indices:
                needed = self.n_nodes - 1 - current_ones
                additional = random.sample(zero_indices,
                                         min(needed, len(zero_indices)))
                for idx in additional:
                    result[idx] = 1
        elif current_ones > self.n_nodes - 1:
            one_indices = [i for i in range(self.n_edges) if result[i] == 1]
            if one_indices:
                excess = current_ones - (self.n_nodes - 1)
                to_remove = random.sample(one_indices,
                                        min(excess, len(one_indices)))
                for idx in to_remove:
                    result[idx] = 0
        return result
    
    def _initialize_population(self):
        """Initialize population using QRNG"""
        self.population = []
        self.counters = []
        for _ in range(self.population_size):
            individual = self._qrng_initialize()
            self.population.append(individual)
            counter = np.ones(self.n_edges)
            self.counters.append(counter)
    
    def _update_counters(self):
        """Update counters based on current population (Equation 19)"""
        for i in range(min(len(self.population), len(self.counters))):
            individual = self.population[i]
            for j in range(self.n_edges):
                if self._is_feasible(individual):
                    if individual[j] == 0:
                        self.counters[i][j] = max(0, self.counters[i][j] - 1)
                    else:
                        self.counters[i][j] += 1
                else:
                    self.counters[i][j] = max(0, self.counters[i][j] - 1)
    
    def optimize(self) -> List[float]:
        """
        Main QBABC optimization algorithm (Figure 3 pseudocode)
        """
        print("Initializing QBABC Algorithm...")
        self._initialize_population()
        self.fitness_values = [self._calculate_fitness(ind) for ind in self.population]
        print(f"Initial population: {len(self.population)} individuals")
        print(f"Network: {self.n_nodes} nodes, {self.n_edges} edges")
        shr, total_combinations, num_spanning_trees = self._spanning_tree_hit_rate()
        print(f"Number of spanning trees (Kirchhoff's): {num_spanning_trees}")
        print(f"Spanning Tree Hit Rate: {shr*100:.4f}%")
        
        best_fitness_history = []
        best_so_far = float('inf')
        
        for iteration in range(self.max_iterations):
            self._update_counters()
            self._employed_bee_search()
            self._onlooker_bee_search()
            self._scouter_bee_search()
            self.fitness_values = [self._calculate_fitness(ind) for ind in self.population]
            best_fitness = min(self.fitness_values) if self.fitness_values else float('inf')
            
            # Track best-so-far
            if best_fitness < best_so_far:
                best_so_far = best_fitness
            best_fitness_history.append(best_so_far)
            
            if iteration % 50 == 0 or iteration == self.max_iterations - 1:
                feasible_count = sum(1 for ind in self.population if self._is_feasible(ind))
                print(f"Iter {iteration:3d}: Best={best_so_far:.4f}, Feasible={feasible_count}/{self.population_size}")
        
        return best_fitness_history
    
    def get_best_solutions(self, n_solutions: int = 1) -> List[Dict[str, Any]]:
        """Get n best feasible solutions from population"""
        solutions_with_fitness = []
        for i, individual in enumerate(self.population):
            if self._is_feasible(individual):
                fitness = self._calculate_fitness(individual)
                selected_edges = [self.edges[j] for j in range(len(individual)) if individual[j] == 1]
                solutions_with_fitness.append({
                    'fitness': fitness, 'individual': individual.copy(),
                    'selected_edges': selected_edges, 'index': i
                })
        solutions_with_fitness.sort(key=lambda x: x['fitness'])
        return solutions_with_fitness[:n_solutions]


def create_sample_vanet(n_nodes: int = 30, seed: int = 42) -> nx.Graph:
    """
    Create a sample VANET topology for testing
    """
    np.random.seed(seed)
    random.seed(seed)
    
    G = nx.Graph()
    
    # Add nodes with random positions
    for i in range(n_nodes):
        pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
        G.add_node(i, pos=pos)
    
    # Add edges based on communication range
    communication_range = 400
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(np.array(G.nodes[i]['pos']) - np.array(G.nodes[j]['pos']))
            if dist <= communication_range:
                G.add_edge(i, j)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            comp1 = list(components[i])
            comp2 = list(components[i + 1])
            # Connect closest nodes from different components
            min_dist = float('inf')
            best_edge = None
            for n1 in comp1:
                for n2 in comp2:
                    dist = np.linalg.norm(np.array(G.nodes[n1]['pos']) - np.array(G.nodes[n2]['pos']))
                    if dist < min_dist:
                        min_dist = dist
                        best_edge = (n1, n2)
            if best_edge:
                G.add_edge(*best_edge)
    
    return G


if __name__ == "__main__":
    # Create VANET
    vanet = create_sample_vanet(20)
    print(f"VANET: {vanet.number_of_nodes()} nodes, {vanet.number_of_edges()} edges")
    
    # Run QBABC
    qbabc = QBABC(vanet, population_size=60, max_iterations=100)
    fitness_history = qbabc.optimize()
    
    # Get best solution
    best_solutions = qbabc.get_best_solutions(1)
    if best_solutions:
        best = best_solutions[0]
        print(f"\nBest fitness found: {best['fitness']:.4f}")
        print(f"Edges in best solution: {best['selected_edges']}")
    else:
        print("\nNo feasible solution found.")
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label="Best Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title("QBABC Convergence Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
