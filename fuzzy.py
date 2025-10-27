import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def build_fuzzy_param_controller():
    diversity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'diversity')
    convergence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'convergence')
    beta = ctrl.Consequent(np.arange(0.3, 0.81, 0.01), 'beta')  # Tighter, less extreme
    scout_limit = ctrl.Consequent(np.arange(8, 16, 1), 'scout_limit')  # More stable range

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

def fuzzy_adapt_fis(params, metrics, fuzzy_ctrl_sim):
    diversity = np.clip(metrics.get('diversity', 0.5), 0, 1)
    convergence = np.clip(metrics.get('convergence', 0.5), 0, 1)
    fuzzy_ctrl_sim.input['diversity'] = diversity
    fuzzy_ctrl_sim.input['convergence'] = convergence
    fuzzy_ctrl_sim.compute()
    adapted = params.copy()
    adapted['beta'] = float(fuzzy_ctrl_sim.output['beta'])
    adapted['limit'] = int(round(fuzzy_ctrl_sim.output['scout_limit']))
    return adapted

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
