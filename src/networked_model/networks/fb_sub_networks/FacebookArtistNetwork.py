from pathlib import Path
from networked_model.networks.fb_sub_networks.FacebookSubNetwork import FacebookSubNetwork

class ArtistNetwork(FacebookSubNetwork):
    def __init__(self, file_path=None, model=None):
        # If no path is provided, use default file in facebook_clean_data directory
        if file_path is None:
            file_path = Path(__file__).parent.parent / "facebook_clean_data" / "artist_edges.csv"
        super().__init__(file_path=file_path)
        self.type = "artist"
        self.default_weight = 0.3
    
    def visualize(self, output_file="src/output/artist_network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):

        return super().visualize(output_file=output_file, notebook=notebook, node_size=node_size,
                                 edge_width=edge_width, num_samples=num_samples)
    
    def plot_degree_distribution(self, net_type="Artist Network", xmax=2000, plot_now=True, bin_count=20):
        
        return super().plot_degree_distribution(net_type=net_type, xmax=xmax, plot_now=plot_now, bin_count=bin_count)