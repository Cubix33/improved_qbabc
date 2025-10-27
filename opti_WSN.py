import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Config & Models ----------------------------- #
class RadioModel:
    def __init__(self):
        self.E_elec = 50e-9  # J/bit
        self.E_fs = 10e-12   # J/bit/m^2
        self.E_mp = 0.0013e-12  # J/bit/m^4
        self.E_da = 5e-9     # J/bit for data aggregation at CH
        self.d0 = math.sqrt(self.E_fs / self.E_mp)

    def tx_cost(self, k_bits, d):
        if d < self.d0:
            return k_bits * (self.E_elec + self.E_fs * d * d)
        return k_bits * (self.E_elec + self.E_mp * (d ** 4))

    def rx_cost(self, k_bits):
        return k_bits * self.E_elec

class Node:
    def __init__(self, idx, x, y, E):
        self.idx = idx
        self.x = x
        self.y = y
        self.E = E

    def pos(self):
        return (self.x, self.y)

class WSN:
    def __init__(self, nodes, bs, radio):
        self.nodes = nodes
        self.bs = bs
        self.radio = radio

    def alive_nodes(self):
        return [n for n in self.nodes if n.E > 0]

    def dist(self, i, j):
        xi, yi = self.nodes[i].pos()
        xj, yj = self.nodes[j].pos()
        return math.hypot(xi - xj, yi - yj)

    def dist_pt(self, i, pt):
        xi, yi = self.nodes[i].pos()
        xj, yj = pt
        return math.hypot(xi - xj, yi - yj)

# ----------------------------- ABC: CH Selection --------------------------- #
class ABCClustering:
    def __init__(self, wsn, k, pop=20, limit=10, iters=25, w_intra=0.6, w_energy=0.3, w_bs=0.1, seed=42):
        self.wsn = wsn
        self.k = k
        self.pop = pop
        self.limit = limit
        self.iters = iters
        self.w_intra = w_intra
        self.w_energy = w_energy
        self.w_bs = w_bs
        random.seed(seed)
        np.random.seed(seed)
        alive_idx = [n.idx for n in self.wsn.alive_nodes()]
        self.foods = [np.array(random.sample(alive_idx, k)) for _ in range(pop)]
        self.fitness = np.array([self._fitness(sol) for sol in self.foods], dtype=float)
        self.trials = np.zeros(pop, dtype=int)

    def _assign_clusters(self, CH):
        clusters = {int(c): [] for c in CH}
        for n in self.wsn.alive_nodes():
            best = min(CH, key=lambda c: self.wsn.dist(n.idx, int(c)))
            clusters[int(best)].append(n.idx)
        return clusters

    def _fitness(self, CH):
        clusters = self._assign_clusters(CH)
        intra = []
        for c, members in clusters.items():
            cx, cy = self.wsn.nodes[c].pos()
            for m in members:
                mx, my = self.wsn.nodes[m].pos()
                intra.append(math.hypot(mx - cx, my - cy))
        intra_term = np.mean(intra) if intra else 1e9
        energies = [self.wsn.nodes[int(c)].E for c in CH]
        energy_term = 1.0 / (np.mean(energies) + 1e-9)
        bs_d = np.mean([self.wsn.dist_pt(int(c), self.wsn.bs) for c in CH])
        return self.w_intra * intra_term + self.w_energy * energy_term + self.w_bs * bs_d

    def _neighbour(self, CH):
        alive_idx = [n.idx for n in self.wsn.alive_nodes()]
        CH_new = CH.copy()
        drop = random.randrange(self.k)
        candidates = list(set(alive_idx) - set(CH_new.tolist()))
        if not candidates:
            return CH_new
        CH_new[drop] = random.choice(candidates)
        return CH_new

    def run(self):
        for _ in range(self.iters):
            for i in range(self.pop):
                cand = self._neighbour(self.foods[i])
                f = self._fitness(cand)
                if f < self.fitness[i]:
                    self.foods[i], self.fitness[i] = cand, f
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
            probs = (self.fitness.max() - self.fitness + 1e-9)
            probs = probs / probs.sum()
            for _o in range(self.pop):
                i = np.random.choice(self.pop, p=probs)
                cand = self._neighbour(self.foods[i])
                f = self._fitness(cand)
                if f < self.fitness[i]:
                    self.foods[i], self.fitness[i] = cand, f
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
            alive_idx = [n.idx for n in self.wsn.alive_nodes()]
            for i in range(self.pop):
                if self.trials[i] > self.limit and len(alive_idx) >= self.k:
                    self.foods[i] = np.array(random.sample(alive_idx, self.k))
                    self.fitness[i] = self._fitness(self.foods[i])
                    self.trials[i] = 0
        best_i = int(np.argmin(self.fitness))
        best_CH = self.foods[best_i]
        clusters = self._assign_clusters(best_CH)
        return best_CH, clusters

# ----------------------------- ACO: Routing among CHs ---------------------- #
class ACORouting:
    def __init__(self, wsn, CH, alpha=1.0, beta=3.0, rho=0.5, ants=20, iters=20, seed=42):
        self.wsn = wsn
        self.CH = list(map(int, CH))
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ants = ants
        self.iters = iters
        random.seed(seed)
        np.random.seed(seed)
        self.m = len(self.CH)
        self.tau = np.full((self.m, self.m), 1e-3, dtype=float)
        self.eta = np.zeros((self.m, self.m), dtype=float)
        self._build_heuristic()

    def _idx(self, node_idx):
        return self.CH.index(int(node_idx))

    def _build_heuristic(self):
        for i, ci in enumerate(self.CH):
            for j, cj in enumerate(self.CH):
                if i == j:
                    self.eta[i, j] = 0.0
                else:
                    d = self.wsn.dist(ci, cj)
                    e = max(self.wsn.nodes[cj].E, 1e-6)
                    self.eta[i, j] = (1.0 / (d + 1e-6)) * (e)

    def _p_select(self, current, visited):
        i = self._idx(current)
        candidates = [j for j in range(self.m) if j not in visited and j != i]
        if not candidates:
            return -1
        weights = []
        for j in candidates:
            w = (self.tau[i, j] ** self.alpha) * (self.eta[i, j] ** self.beta)
            weights.append(w)
        weights = np.array(weights)
        if weights.sum() == 0:
            return candidates[random.randrange(len(candidates))]
        probs = weights / weights.sum()
        return candidates[int(np.random.choice(len(candidates), p=probs))]

    def _construct_tree(self):
        best_tree = {}
        best_score = float("inf")
        for _ in range(self.ants):
            visited = set()
            parents = {c: None for c in self.CH}
            order = sorted(self.CH, key=lambda c: self.wsn.dist_pt(c, self.wsn.bs), reverse=True)
            for start in order:
                visited_local = set([self._idx(start)])
                current = start
                while True:
                    if len(visited_local) >= self.m - 1:
                        parents[current] = None
                        break
                    d_bs = self.wsn.dist_pt(current, self.wsn.bs)
                    p_direct = min(0.9, 1.0 / (1.0 + d_bs / 20.0))
                    if random.random() < p_direct:
                        parents[current] = None
                        break
                    nxt_idx = self._p_select(current, visited_local)
                    if nxt_idx == -1:
                        parents[current] = None
                        break
                    nxt = self.CH[nxt_idx]
                    parents[current] = nxt
                    current = nxt
                    visited_local.add(nxt_idx)
            score = 0.0
            for u, v in parents.items():
                if v is None:
                    score += self.wsn.dist_pt(u, self.wsn.bs)
                else:
                    score += self.wsn.dist(u, v)
            if score < best_score:
                best_score = score
                best_tree = parents
        return best_tree

    def _evaporate(self):
        self.tau *= (1.0 - self.rho)
        self.tau = np.clip(self.tau, 1e-6, None)

    def _deposit(self, tree):
        for u, v in tree.items():
            if v is None:
                continue
            i, j = self._idx(u), self._idx(v)
            d = self.wsn.dist(u, v)
            self.tau[i, j] += 1.0 / (d + 1e-6)

    def run(self):
        best = {}
        best_score = float("inf")
        for _ in range(self.iters):
            tree = self._construct_tree()
            score = 0.0
            for u, v in tree.items():
                if v is None:
                    score += self.wsn.dist_pt(u, self.wsn.bs)
                else:
                    score += self.wsn.dist(u, v)
            if score < best_score:
                best, best_score = tree, score
            self._evaporate()
            self._deposit(tree)
        return best

# ----------------------------- Simulation --------------------------------- #
def generate_wsn(n=100, area=100.0, E0=0.5, bs=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    nodes = []
    for i in range(n):
        x, y = random.random() * area, random.random() * area
        nodes.append(Node(i, x, y, E=E0))
    if bs is None:
        bs = (area / 2.0, 1.2 * area)
    return WSN(nodes=nodes, bs=bs, radio=RadioModel())

def simulate_round(wsn, k, pkt_bits=4000, abc_kwargs=None, aco_kwargs=None, visualize=False, title="Round"):
    abc_kwargs = abc_kwargs or {}
    aco_kwargs = aco_kwargs or {}
    abc = ABCClustering(wsn, k=k, **abc_kwargs)
    CH, clusters = abc.run()
    aco = ACORouting(wsn, CH.tolist(), **aco_kwargs)
    tree = aco.run()
    radio = wsn.radio
    for ch in CH:
        members = clusters[int(ch)]
        for m in members:
            if m == int(ch):
                continue
            d = wsn.dist(m, int(ch))
            tx = radio.tx_cost(pkt_bits, d)
            rx = radio.rx_cost(pkt_bits)
            wsn.nodes[m].E -= tx
            wsn.nodes[int(ch)].E -= rx
        wsn.nodes[int(ch)].E -= radio.E_da * pkt_bits
    for ch in CH:
        current = int(ch)
        while True:
            parent = tree.get(current, None)
            if parent is None:
                d = wsn.dist_pt(current, wsn.bs)
                wsn.nodes[current].E -= radio.tx_cost(pkt_bits, d)
                break
            else:
                d = wsn.dist(current, int(parent))
                wsn.nodes[current].E -= radio.tx_cost(pkt_bits, d)
                wsn.nodes[int(parent)].E -= radio.rx_cost(pkt_bits)
                current = int(parent)
    alive = sum(1 for n in wsn.nodes if n.E > 0)
    dead = len(wsn.nodes) - alive
    total_energy = sum(max(n.E, 0.0) for n in wsn.nodes)
    return {
        "alive": alive,
        "dead": dead,
        "total_energy": total_energy,
        "CH": CH,
        "clusters": clusters,
        "tree": tree,
    }

def plot_final_network(wsn, CH, clusters, tree, title="Final Network Topology"):
    plt.figure(figsize=(8, 8))
    # plot members
    for ch, members in clusters.items():
        xs = [wsn.nodes[m].x for m in members if m != ch]
        ys = [wsn.nodes[m].y for m in members if m != ch]
        plt.scatter(xs, ys, s=16, alpha=0.7, label=f"Cluster {ch}")
    # plot CHs
    ch_x = [wsn.nodes[int(c)].x for c in CH]
    ch_y = [wsn.nodes[int(c)].y for c in CH]
    plt.scatter(ch_x, ch_y, s=80, marker='^', color='red', label="CHs")
    # plot BS
    plt.scatter([wsn.bs[0]], [wsn.bs[1]], s=120, marker='*', color='black', label="BS")
    # draw member→CH links
    for ch, members in clusters.items():
        cx, cy = wsn.nodes[int(ch)].pos()
        for m in members:
            if m == int(ch):
                continue
            mx, my = wsn.nodes[m].pos()
            plt.plot([mx, cx], [my, cy], linewidth=0.5, color='gray')
    # draw CH routing tree
    for u, v in tree.items():
        ux, uy = wsn.nodes[int(u)].pos()
        if v is None:
            bx, by = wsn.bs
            plt.plot([ux, bx], [uy, by], color='blue')
        else:
            vx, vy = wsn.nodes[int(v)].pos()
            plt.plot([ux, vx], [uy, vy], color='blue')
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_until_first_node_death(n=100, area=100.0, k=8, E0=0.5, pkt_bits=4000, seed=42, visualize=False):
    wsn = generate_wsn(n=n, area=area, E0=E0, seed=seed)
    history = []
    round_num = 0
    first_node_dead = False
    cluster_size_trend = []
    final_stats = None

    while not first_node_dead:
        round_num += 1
        stats = simulate_round(
            wsn, k=k, pkt_bits=pkt_bits,
            abc_kwargs=dict(pop=20, limit=10, iters=20),
            aco_kwargs=dict(alpha=1.0, beta=2.0, rho=0.4, ants=15, iters=15),
            visualize=visualize,
            title=f"Round {round_num}"
        )
        avg_cluster_size = np.mean([len(members) for members in stats['clusters'].values()])
        cluster_size_trend.append(avg_cluster_size)
        history.append((round_num, stats['alive'], stats['dead'], stats['total_energy']))
        print(f"Round {round_num}: alive={stats['alive']}, dead={stats['dead']}, E_total={stats['total_energy']:.3f} J")
        if stats['dead'] > 0:
            print(f"\nFirst node death at round {round_num}!")
            final_stats = stats
            break

    rounds = [h[0] for h in history]
    alive = [h[1] for h in history]
    energy = [h[3] for h in history]

    # Alive nodes trend
    plt.figure(figsize=(7,4))
    plt.plot(rounds, alive, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Alive Nodes')
    plt.title('Network Survival (Until First Node Death)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Energy trend
    plt.figure(figsize=(7,4))
    plt.plot(rounds, energy, marker='s')
    plt.xlabel('Round')
    plt.ylabel('Total Residual Energy (J)')
    plt.title('Energy Over Time (Until First Node Death)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Cluster size trend
    plt.figure(figsize=(7,4))
    plt.plot(rounds, cluster_size_trend, marker='^')
    plt.xlabel('Round')
    plt.ylabel('Average Cluster Size')
    plt.title('Average Cluster Size Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Final network topology
    if final_stats:
        plot_final_network(wsn, final_stats['CH'], final_stats['clusters'], final_stats['tree'],
                           title=f"Final Network Topology (Round {round_num})")

        # Histogram of final node energies
        energies = [n.E for n in wsn.nodes]
        plt.figure(figsize=(7,4))
        plt.hist(energies, bins=20, color='orange', edgecolor='black')
        plt.xlabel('Node Residual Energy (J)')
        plt.ylabel('Count')
        plt.title('Node Energy Distribution at First Node Death')
        plt.tight_layout()
        plt.show()

        # Scatter plot of node positions colored by energy
        xs = [n.x for n in wsn.nodes]
        ys = [n.y for n in wsn.nodes]
        plt.figure(figsize=(7,7))
        plt.scatter(xs, ys, c=energies, cmap='viridis', s=40)
        plt.colorbar(label='Residual Energy (J)')
        plt.scatter([wsn.bs[0]], [wsn.bs[1]], s=120, marker='*', color='black', label="BS")
        plt.title('Node Positions Colored by Energy (at FND)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return history, final_stats

if __name__ == '__main__':
    run_until_first_node_death(n=100, area=100.0, k=8, E0=0.5, pkt_bits=4000, seed=42, visualize=False)
