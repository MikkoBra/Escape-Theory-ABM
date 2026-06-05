import numpy as np
import pandas as pd
from pathlib import Path
from networked_model.networks.AbstractNetwork import Network

class FacebookSubNetwork(Network):
    def __init__(self, file_path=None, model=None):
        # If no path is provided, use default file in facebook_clean_data directory
        if file_path is None:
            file_path = Path(__file__).parent.parent / "facebook_clean_data" / "artist_edges.csv"

        self.type = "artist"
        self.file_path = Path(file_path)
        self.edges = None
        self.nodes = None
        self.adjacency = None
        self.default_weight = 0.3

    def generate_network(self):
        """
        Reads edge list from CSV file and constructs:
        - edges (Nx2 numpy array)
        - nodes (unique node ids)
        - adjacency list (optional, dict)
        """
        # Read CSV file
        df = pd.read_csv(self.file_path)
        
        # Convert to numpy array (assuming first two columns are the edges)
        self.edges = df.iloc[:, :2].to_numpy(dtype=np.int32)

        # Ensure shape is correct even if single edge
        if self.edges.ndim == 1:
            self.edges = self.edges.reshape(1, 2)

        self.nodes = np.unique(self.edges)

        self.adjacency = {node: {} for node in self.nodes}

        # Efficient iteration
        for u, v in self.edges:
            self.adjacency[u][v] = self.default_weight
            self.adjacency[v][u] = self.default_weight  # undirected graph
    
    def visualize(self, output_file="src/output/artist_network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):

        return super().visualize(output_file=output_file, notebook=notebook, node_size=node_size,
                                 edge_width=edge_width, num_samples=num_samples)
    
    def plot_degree_distribution(self, net_type="Network", xmax=2000, plot_now=True, bin_count=20):
        
        return super().plot_degree_distribution(net_type=net_type, xmax=xmax, plot_now=plot_now, bin_count=bin_count)