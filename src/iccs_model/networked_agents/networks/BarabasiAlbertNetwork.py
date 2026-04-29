import numpy as np
import networkx as nx
from iccs_model.networked_agents.networks.AbstractNetwork import Network


class BarabasiAlbertNetwork(Network):
    def __init__(self, model):
        """
        model must provide:
            - model.num_agents (number of nodes, n)
            - model.m (number of edges to attach per new node, m)
        """

        self.model = model
        self.edges = None
        self.nodes = None
        self.adjacency = None
        self.default_weight = 0.3

    def generate_network(self):
        """
        Generates a Barabási-Albert network using NetworkX.
        Produces:
        - edges (Nx2 numpy array)
        - nodes (unique node ids)
        - adjacency dict
        """

        n = int(self.model.num_agents)
        m = int(self.model.m)

        if m < 1 or m >= n:
            raise ValueError("Barabási-Albert requires 1 <= m < n")

        # Generate graph
        G = nx.barabasi_albert_graph(n=n, m=m)

        # Store edges as numpy array
        self.edges = np.array(G.edges(), dtype=np.int32)

        # Handle edge case (very small graph)
        if self.edges.ndim == 1:
            self.edges = self.edges.reshape(1, 2)

        self.nodes = np.array(G.nodes(), dtype=np.int32)

        # Build adjacency dict
        self.adjacency = {int(node): {} for node in self.nodes}

        for u, v in self.edges:
            u, v = int(u), int(v)
            self.adjacency[u][v] = self.default_weight
            self.adjacency[v][u] = self.default_weight
    
    def visualize(self, output_file="src/output/ba_network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):

        return super().visualize(output_file=output_file, notebook=notebook, node_size=node_size,
                                 edge_width=edge_width, num_samples=num_samples)
    
    def plot_degree_distribution(self, net_type="Barabási Albert Network", xmax=2000, plot_now=True):
        
        return super().plot_degree_distribution(net_type=net_type, xmax=xmax, plot_now=plot_now)