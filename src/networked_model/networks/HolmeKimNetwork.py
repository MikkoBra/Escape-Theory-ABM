import numpy as np
import networkx as nx
import random
from networked_model.networks.AbstractNetwork import Network


class HolmeKimNetwork(Network):
    def __init__(self, model):
        """
        model must provide:
            - model.num_agents (number of nodes, n)
            - model.k (number of edges to attach per new node, m)
        """
        self.type = "hk"
        self.model = model
        self.edges = None
        self.nodes = None
        self.adjacency = None
        self.default_weight = 0.3

    def generate_network(self):
        """
        Generates a more realistic social network with:
        1. Triadic closure (powerlaw_cluster_graph)
        2. Initial attractiveness (fitness model)
        3. Node deletion/pruning
        4. Randomized edge removal for degree heterogeneity
        """

        n = int(self.model.num_agents)
        m = int(self.model.m)
        p = float(self.model.cluster_prob)
        
        # Get optional parameters
        removal_rate = getattr(self.model, 'node_removal_rate', 0.0)
        initial_attr = getattr(self.model, 'initial_attractiveness', 1)
        edge_removal_prob = getattr(self.model, 'edge_removal_prob', 0.0)
        hub_count = getattr(self.model, 'hub_count', 0)
        hub_degree = getattr(self.model, 'hub_degree', 0)

        if m < 1 or m >= n:
            raise ValueError("Network requires 1 <= m < n")

        # Step 1: Generate base network with triadic closure
        # This already helps with the "arched" appearance
        G = nx.powerlaw_cluster_graph(n=n, m=m, p=p)

        # Step 2: Add initial attractiveness (shifts from pure Π(k)~k to Π(k)=A+k)
        # This creates more realistic attachment for low-degree nodes
        if initial_attr > 0:
            G = self._add_initial_attractiveness(G, n, m, initial_attr)

        # Step 3: Node deletion/aging (creates nodes with degree < m)
        if removal_rate > 0:
            G = self._apply_node_removal(G, removal_rate)

        # Step 4: Random edge removal (creates more degree heterogeneity)
        if edge_removal_prob > 0:
            G = self._apply_edge_removal(G, edge_removal_prob, min_degree=1)
        
        # Step 5: create hubs
        if hub_count > 0 and hub_degree > 0:
            G = self._create_hubs(G, hub_count, hub_degree)
        
        # Step 6: assign Dunbar weights
        if getattr(self.model, "weighted_network", False):
            G = self._apply_weight_distribution(G)

        # Store edges as numpy array
        self.edges = np.array(G.edges(), dtype=np.int32)

        if self.edges.ndim == 1:
            self.edges = self.edges.reshape(1, 2)

        self.nodes = np.array(G.nodes(), dtype=np.int32)

        # Build adjacency dict
        self.adjacency = {int(node): {} for node in self.nodes}

        for u, v in self.edges:
            u, v = int(u), int(v)
            w = G[u][v].get("weight", self.default_weight)
            self.adjacency[u][v] = float(w)
            self.adjacency[v][u] = float(w)

    def _add_initial_attractiveness(self, G, n, m, A):
        """
        Optimized preferential attachment using a repeated-node list.
        Approximation of Π(k) = A + k without recomputing probabilities.
        """
        if A == 0:
            return
        G_new = nx.Graph()
        nodes = list(G.nodes())

        # Initial complete graph
        initial_nodes = nodes[:m+1]
        G_new.add_edges_from(
            (i, j) for i in initial_nodes for j in initial_nodes if i < j
        )

        # Repeated node list: node appears (degree + A) times
        repeated_nodes = []

        for node in initial_nodes:
            repeated_nodes.extend([node] * (G_new.degree(node) + A))

        for new_node in nodes[m+1:]:
            # Sample targets from repeated list
            targets = set()
            while len(targets) < m:
                targets.add(random.choice(repeated_nodes))

            G_new.add_edges_from((new_node, t) for t in targets)

            # Update repeated list
            repeated_nodes.extend([new_node] * A)
            for t in targets:
                repeated_nodes.append(t)
                repeated_nodes.append(new_node)

        return G_new

    def _apply_node_removal(self, G, removal_rate):
        """
        Removes random nodes to simulate churn/aging.
        This creates nodes with degree < m and more realistic degree distribution.
        From diagram: this shifts the network towards stretched exponential.
        """
        if removal_rate == 0:
            return
        nodes = list(G.nodes())
        num_to_remove = int(len(nodes) * removal_rate)
        
        if num_to_remove > 0:
            # Remove random nodes (preferably lower degree to maintain connectivity)
            degrees = dict(G.degree())
            # Weight removal probability inversely to degree (remove low-degree more)
            weights = 1.0 / (np.array([degrees[n] for n in nodes]) + 1)
            weights = weights / weights.sum()
            
            to_remove = np.random.choice(nodes, size=num_to_remove, 
                                        replace=False, p=weights)
            G.remove_nodes_from(to_remove)
            
            # Relabel nodes to be consecutive
            G = nx.convert_node_labels_to_integers(G)
        
        return G

    def _apply_edge_removal(self, G, removal_prob, min_degree=1):
        """
        Randomly remove edges while maintaining minimum degree.
        This creates the "arched" appearance in log-log plots.
        """
        if removal_prob == 0:
            return
        edges = list(G.edges())
        edges_to_check = np.random.permutation(edges)
        
        for u, v in edges_to_check:
            if np.random.random() < removal_prob:
                # Only remove if both nodes stay above min_degree
                if G.degree(u) > min_degree and G.degree(v) > min_degree:
                    G.remove_edge(u, v)
        
        return G
    
    def _create_hubs(self, G, hub_count, target_degree):
        """
        Select hub_count random nodes and increase their degree
        until they reach target_degree by adding random edges.

        Existing edges are preserved.
        """
        if hub_count <= 0:
            return G

        nodes = list(G.nodes())

        if hub_count > len(nodes):
            hub_count = len(nodes)

        hubs = random.sample(nodes, hub_count)

        for i, hub in enumerate(hubs):
            print(f"Creating hub {i+1} out of {len(hubs)}")
            # Maximum possible degree
            max_degree = len(nodes) - 1
            desired_degree = min(target_degree, max_degree)

            while G.degree(hub) < desired_degree:

                # Nodes not already connected to hub
                candidates = [
                    n for n in nodes
                    if n != hub and not G.has_edge(hub, n)
                ]

                if not candidates:
                    break

                # Preferential attachment
                degrees = np.array([G.degree(n) + 1 for n in candidates], dtype=float)
                probs = degrees / degrees.sum()
                target = np.random.choice(candidates, p=probs)
                G.add_edge(hub, target)
        return G
    
    def _apply_weight_distribution(self, G):
        """
        Assigns weighted edges according to a 3-component mixture:
        
        - 15/150 edges: U(0.99, 1.0)
        - 35/150 edges: U(0.4, 0.6)
        - 100/150 edges: U(0.05, 0.1)
        
        Applied globally across edges.
        """
        if not getattr(self.model, "weighted_network", False):
            return G

        edges = list(G.edges())
        rng = np.random.default_rng()
        rng.shuffle(edges)

        n = len(edges)

        n_high = int(n * 15 / 150)
        n_mid  = int(n * 35 / 150)
        n_low  = n - n_high - n_mid

        for i, (u, v) in enumerate(edges):
            if i < n_high:
                w = np.random.uniform(0.99, 1.0)
            elif i < n_high + n_mid:
                w = np.random.uniform(0.4, 0.6)
            else:
                w = np.random.uniform(0.05, 0.1)

            # IMPORTANT: write directly to graph edge attribute
            G[u][v]["weight"] = float(w)

        return G
    
    def visualize(self, output_file="src/output/hk_network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):

        return super().visualize(output_file=output_file, notebook=notebook, node_size=node_size,
                                 edge_width=edge_width, num_samples=num_samples)
    
    def plot_degree_distribution(self, net_type="Holme-Kim Network", xmax=2000, plot_now=True, bin_count=20):
        
        return super().plot_degree_distribution(net_type=net_type, xmax=xmax, plot_now=plot_now, bin_count=bin_count)