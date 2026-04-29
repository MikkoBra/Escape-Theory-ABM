import numpy as np
import networkx as nx
import random
from iccs_model.networked_agents.networks.AbstractNetwork import Network


class HolmeKimNetwork(Network):
    def __init__(self, model):
        """
        model must provide:
            - model.num_agents (number of nodes, n)
            - model.k (number of edges to attach per new node, m)
        """

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

        # Store edges as numpy array
        self.edges = np.array(G.edges(), dtype=np.int32)

        if self.edges.ndim == 1:
            self.edges = self.edges.reshape(1, 2)

        self.nodes = np.array(G.nodes(), dtype=np.int32)

        # Build adjacency dict
        self.adjacency = {int(node): {} for node in self.nodes}

        for u, v in self.edges:
            u, v = int(u), int(v)
            self.adjacency[u][v] = self.default_weight
            self.adjacency[v][u] = self.default_weight

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
    
    def visualize(self, output_file="src/output/hk_network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):

        return super().visualize(output_file=output_file, notebook=notebook, node_size=node_size,
                                 edge_width=edge_width, num_samples=num_samples)
    
    def plot_degree_distribution(self, net_type="Holme-Kim Network", xmax=2000, plot_now=True):
        
        return super().plot_degree_distribution(net_type=net_type, xmax=xmax, plot_now=plot_now)