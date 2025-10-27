import numpy as np
import networkx as nx

class QBABC:
    """
    Quantum-Behaved Artificial Bee Colony (QBABC) Algorithm
    for optimization applied to Vehicular Ad Hoc Networks (VANETs).
    """

    def __init__(self, graph, populationsize=100, maxiterations=500, employedratio=0.5, onlookerratio=0.5, scoutlimit=10, qosweights=None):
        """
        Initialize the QBABC with network graph and parameters.
        """
        self.graph = graph
        self.nnodes = len(graph.nodes)
        self.nedges = len(graph.edges)
        self.populationsize = populationsize
        self.maxiterations = maxiterations
        self.employedratio = employedratio
        self.onlookerratio = onlookerratio
        self.scoutlimit = scoutlimit
        self.qosweights = qosweights or (0.3, 0.2, 0.2, 0.15, 0.15)
        self.edges = list(graph.edges)
        self.edgeweights = self.calculate_edge_weights()
        self.population = []
        self.fitnessvalues = []
        self.counters = []
        self.limit_counters = np.zeros(self.populationsize)  # For scout limit tracking
        self.history = []

    def calculate_edge_weights(self):
        """
        Calculate weights for edges using QoS measures.
        """
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
            weight = (self.qosweights[0] * cost +
                      self.qosweights[1] * delay +
                      self.qosweights[2] * packetloss +
                      self.qosweights[3] * jitter -
                      self.qosweights[4] * bandwidth)
            weights[edge] = weight
        return weights

    def qrng_initialize(self):
        """
        Quantum-inspired initializer for creating a feasible individual.
        Generates a candidate binary solution vector of edges representing a spanning tree.
        """
        v = np.zeros(self.nedges, dtype=int)
        indices = np.random.choice(self.nedges, self.nnodes - 1, replace=False)
        v[indices] = 1
        return v

    def is_feasible(self, individual):
        """
        Check if the individual represents a valid spanning tree in the graph.
        """
        if np.sum(individual) != self.nnodes - 1:
            return False
        edges_in = [self.edges[i] for i in range(len(individual)) if individual[i] == 1]
        tempgraph = nx.Graph()
        tempgraph.add_edges_from(edges_in)
        return nx.is_connected(tempgraph) and len(tempgraph.nodes) == self.nnodes

    def calculate_fitness(self, individual):
        """
        Calculate fitness for an individual solution.
        Lower fitness is better (minimization).
        """
        if self.is_feasible(individual):
            return sum(self.edgeweights[self.edges[i]] for i in range(len(individual)) if individual[i] == 1)
        else:
            # Penalize infeasible solutions heavily
            return sum(self.edgeweights.values()) * 10

    def initialize_population(self):
        """
        Initialize population with feasible solutions.
        """
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
        """
        Generate a new candidate solution by bitwise mutation between a bee and peer.
        """
        candidate = bee.copy()
        pos = np.random.randint(len(bee))
        candidate[pos] = 1 - candidate[pos]
        # Ensure feasibility - if not feasible, revert change (repair)
        if not self.is_feasible(candidate):
            candidate[pos] = bee[pos]
        return candidate

    def calculate_probabilities(self):
        """
        Calculate selection probabilities for onlooker bees based on inverse fitness.
        """
        fit_inv = np.max(self.fitnessvalues) - np.array(self.fitnessvalues) + 1e-10
        prob = fit_inv / np.sum(fit_inv)
        return prob

    def employed_bee_phase(self):
        """
        Employed bee phase: each employed bee generates a candidate solution and replaces if better.
        """
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
        """
        Onlooker bee phase: probabilistically select employed bees and try to improve solutions.
        """
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
        """
        Scout bee phase: replace solutions that have not improved for 'limit' iterations.
        """
        for idx in range(self.populationsize):
            if self.limit_counters[idx] > self.scoutlimit:
                new_ind = None
                # Try create feasible new individual
                for _ in range(100):
                    candidate = self.qrng_initialize()
                    if self.is_feasible(candidate):
                        new_ind = candidate
                        break
                if new_ind is not None:
                    self.population[idx] = new_ind
                    self.fitnessvalues[idx] = self.calculate_fitness(new_ind)
                    self.limit_counters[idx] = 0

    def optimize(self):
        """
        Main optimization loop performing all phases iteratively.
        Returns history of the best fitness values found.
        """
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
            print(f"Iteration {iteration+1}/{self.maxiterations}, Best-so-far Fitness: {best_so_far:.5f}")
        self.history = history
        return history

    def get_best_solution(self):
        """
        Retrieve the best solution and its fitness after optimization.
        """
        min_idx = np.argmin(self.fitnessvalues)
        return self.population[min_idx], self.fitnessvalues[min_idx]