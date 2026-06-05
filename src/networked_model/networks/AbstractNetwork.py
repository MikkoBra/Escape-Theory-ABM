from abc import ABCMeta, abstractmethod
from pathlib import Path
import networkx as nx
import pyvis.network as pynet
import plotly.graph_objects as go
import io
from PIL import Image
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

class Network(metaclass=ABCMeta):
    def __init__(self, file_path=None, model=None):
        if file_path is None:
            file_path = Path(__file__).parent / "facebook_combined.txt"

        self.model = model
        self.file_path = Path(file_path)

        self.edges = None
        self.nodes = None
        self.adjacency = None
        self.default_weight = 0.3
    
    @abstractmethod
    def generate_network():
        pass

    def compute_clustering_coefficients(self):
        """
        Compute weighted clustering coefficient for all nodes, scaled by
        mean neighbor weight. Mirrors the deprecated per-agent calculation:
            weighted_clustering = nx.clustering(G, node, weight="strength") * avg_weight
        
        Returns a numpy array indexed by position in self.nodes.
        """
        if self.adjacency is None:
            raise ValueError("Adjacency not built. Call generate_network() first.")

        G = nx.Graph()
        for u in self.adjacency:
            for v, weight in self.adjacency[u].items():
                G.add_edge(int(u), int(v), strength=weight)

        clustering_map = nx.clustering(G, weight="strength")

        num_nodes = len(self.nodes)
        result = np.zeros(num_nodes, dtype=np.float32)

        for i, node in enumerate(self.nodes):
            node = int(node)
            neighbors = self.adjacency.get(node, {})
            if neighbors:
                avg_weight = np.mean(list(neighbors.values()))
                result[i] = clustering_map[node] * avg_weight
            else:
                result[i] = 0.0

        return result

    def set_edge_weight(self, u, v, weight):
        if self.adjacency is None:
            raise ValueError("Adjacency not built. Call generate_network(build_adjacency=True)")

        if u not in self.adjacency or v not in self.adjacency:
            raise ValueError(f"Edge ({u}, {v}) contains unknown node(s)")

        if v not in self.adjacency[u]:
            raise ValueError(f"Edge ({u}, {v}) does not exist")

        self.adjacency[u][v] = weight
        self.adjacency[v][u] = weight

    def get_neighbors(self, node_id):
        if self.adjacency is None:
            raise ValueError("Adjacency not built.")
        return self.adjacency.get(node_id, {})
    
    def visualize(self, output_file="src/output/network.html", notebook=False,
                node_size=50, edge_width=20, num_samples=1000):
        """
        Creates an html file that visualizes the structure of the network.
        """

        if self.adjacency is None:
            raise ValueError("Adjacency not built. Call generate_network() first.")

        # Build NetworkX graph
        G = nx.Graph()
        for u in self.adjacency:
            for v, weight in self.adjacency[u].items():
                G.add_edge(int(u), int(v))

        # Sample nodes
        if num_samples < len(self.nodes):
            nodes_sample = random.sample(list(G.nodes), num_samples)
            G = G.subgraph(nodes_sample)

        net = pynet.Network(notebook=notebook)
        net.from_nx(G)
        
        # Set node sizes AFTER conversion to PyVis
        for node in net.nodes:
            node["size"] = node_size
        
        # Set edge widths AFTER conversion to PyVis
        for edge in net.edges:
            edge["width"] = edge_width

        net.barnes_hut()
        net.write_html(output_file)

        return output_file
    
    def plot_degree_distribution(self, net_type="Network", xmax=1500, plot_now=True, bin_count=20):
        """
        Compute degree distribution using finer log-binned PDF estimation
        and optionally plot it.
        """
        if self.adjacency is None:
            raise ValueError("Adjacency not built. Call generate_network() first.")

        # Build graph
        G = nx.Graph()
        for u in self.adjacency:
            for v in self.adjacency[u]:
                G.add_edge(int(u), int(v))

        degrees = np.array([deg for _, deg in G.degree()])

        # --- LOG-SPACED FINER BINNING ---
        min_deg = max(1, degrees.min())
        max_deg = degrees.max()
        
        bin_count = min(
                bin_count,
                len(degrees),
                len(np.unique(degrees)) + 1
            )
        bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bin_count)
        hist, bin_edges = np.histogram(degrees, bins=bins, density=True)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        deg = bin_centers
        pdf = hist

        # remove zeros for log-log stability
        mask = pdf > 0
        deg = deg[mask]
        pdf = pdf[mask]

        if plot_now:
            fontsize = 16
            fig, ax = plt.subplots()

            ax.plot(deg, pdf, '-')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("Degree (k)", fontsize=fontsize)
            ax.set_ylabel("P(X = k)", fontsize=fontsize)
            ax.set_title(f"Degree Distribution of {net_type}", fontsize=fontsize, fontweight="bold")
            ax.set_xlim(right=xmax)
            ax.tick_params(axis="both", labelsize=fontsize)

            output_path = Path("src/output") / f"{net_type.replace(' ', '_').lower()}_degree_dist.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fig.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()

            return None

        else:
            return {
                "degrees": deg,
                "pdf": pdf,
                "name": net_type,
                "size": len(self.nodes)
            }
    
    @staticmethod
    def plot_combined_distributions(
        networks_data,
        output_file="src/output/combined_degree_dist.png",
        xmax=2000,
        title="Degree Distributions",
        bin_count=20,
        include_size=True
    ):
        """
        Plot multiple degree distributions using consistent log-binned PDFs.
        """
        fontsize = 20
        fig, ax = plt.subplots(figsize=(10, 6))

        for data in networks_data:
            degrees = np.array(data["degrees"])

            # --- re-bin each dataset consistently ---
            min_deg = max(1, degrees.min())
            max_deg = degrees.max()

            bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bin_count)
            hist, bin_edges = np.histogram(degrees, bins=bins, density=True)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            pdf = hist

            mask = pdf > 0
            size_label = f" (size = {data["size"]})" if include_size else ""
            ax.plot(bin_centers[mask], pdf[mask], '-', label=data["name"] + size_label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Degree (k)", fontsize=fontsize)
        ax.set_ylabel("P(X = k)", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize, fontweight="bold")
        ax.set_xlim(right=xmax)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.legend(fontsize=fontsize - 2, frameon=True)
        ax.grid(True, alpha=0.3)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_plotly(self, node_size=10, edge_width=2, width=1200, height=800):
        # Build NetworkX graph
        G = nx.Graph()
        for u in self.adjacency:
            for v, weight in self.adjacency[u].items():
                G.add_edge(int(u), int(v))

        # Sample nodes
        nodes_sample = random.sample(list(G.nodes), 500)
        G = G.subgraph(nodes_sample)

        pos = nx.spring_layout(G)
        
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=edge_width, color='#888'),
            hoverinfo='none')
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        node_trace = go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=node_size, color='lightblue', line=dict(width=0.5, color='darkblue')),
            hoverinfo='text')
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height
        )
        
        # Convert to image
        img_bytes = fig.to_image(format="png", width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))
        
        return img
    
    def network_summary(self):
        """
        Prints comprehensive statistics essential for modeling realistic social networks.
        
        Key metrics include:
        - Basic properties (nodes, edges, density)
        - Degree distribution characteristics (power-law behavior)
        - Clustering coefficient (triadic closure, community structure)
        - Path lengths (small-world property)
        - Connectivity and component structure
        - Centrality measures (influential nodes)
        - Assortativity (homophily patterns)
        """
        
        if self.adjacency is None:
            raise ValueError("Adjacency not built. Call generate_network() first.")
        
        # Build NetworkX graph
        G = nx.Graph()
        for u in self.adjacency:
            for v in self.adjacency[u]:
                G.add_edge(int(u), int(v))
        
        print("=" * 70)
        print("SOCIAL NETWORK SUMMARY STATISTICS")
        print("=" * 70)
        
        # === BASIC PROPERTIES ===
        print("\n📊 BASIC PROPERTIES")
        print("-" * 70)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        print(f"  Nodes (users):              {num_nodes:,}")
        print(f"  Edges (connections):        {num_edges:,}")
        print(f"  Network density:            {density:.6f}")
        print(f"  Sparsity:                   {1-density:.6f}")
        
        # === DEGREE DISTRIBUTION ===
        print("\n📈 DEGREE DISTRIBUTION (Power-law / Scale-free properties)")
        print("-" * 70)
        degrees = [deg for _, deg in G.degree()]
        avg_degree = np.mean(degrees)
        median_degree = np.median(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        std_degree = np.std(degrees)
        
        print(f"  Average degree:             {avg_degree:.2f}")
        print(f"  Median degree:              {median_degree:.1f}")
        print(f"  Std deviation:              {std_degree:.2f}")
        print(f"  Min degree:                 {min_degree}")
        print(f"  Max degree (hub):           {max_degree}")
        
        # Degree percentiles
        p90 = np.percentile(degrees, 90)
        p95 = np.percentile(degrees, 95)
        p99 = np.percentile(degrees, 99)
        print(f"  90th percentile:            {p90:.1f}")
        print(f"  95th percentile:            {p95:.1f}")
        print(f"  99th percentile:            {p99:.1f}")
        
        # === CLUSTERING (Triadic Closure) ===
        print("\n🔺 CLUSTERING COEFFICIENT (Triadic closure, local structure)")
        print("-" * 70)
        avg_clustering = nx.average_clustering(G)
        transitivity = nx.transitivity(G)
        
        print(f"  Average clustering coeff:   {avg_clustering:.4f}")
        print(f"  Global transitivity:        {transitivity:.4f}")
        print(f"  Random network expected:    {density:.4f}")
        print(f"  Clustering ratio (C/p):     {avg_clustering/density if density > 0 else 0:.2f}x")
        
        # === SMALL-WORLD PROPERTIES ===
        # print("\n🌐 PATH LENGTHS (Small-world property)")
        # print("-" * 70)
        
        # if nx.is_connected(G):
        #     avg_shortest_path = nx.average_shortest_path_length(G)
        #     diameter = nx.diameter(G)
        #     print(f"  Average shortest path:      {avg_shortest_path:.2f}")
        #     print(f"  Network diameter:           {diameter}")
        #     print(f"  Expected (random network):  {np.log(num_nodes)/np.log(avg_degree) if avg_degree > 1 else 'N/A'}")
        # else:
        #     # Use largest component
        #     largest_cc = max(nx.connected_components(G), key=len)
        #     G_largest = G.subgraph(largest_cc)
        #     avg_shortest_path = nx.average_shortest_path_length(G_largest)
        #     diameter = nx.diameter(G_largest)
        #     print(f"  Network is disconnected - using largest component:")
        #     print(f"  Largest component size:     {len(largest_cc):,} ({100*len(largest_cc)/num_nodes:.1f}%)")
        #     print(f"  Average shortest path:      {avg_shortest_path:.2f}")
        #     print(f"  Component diameter:         {diameter}")
        
        # === CONNECTIVITY ===
        print("\n🔗 CONNECTIVITY & COMPONENTS")
        print("-" * 70)
        num_components = nx.number_connected_components(G)
        is_connected = nx.is_connected(G)
        
        print(f"  Is connected:               {is_connected}")
        print(f"  Number of components:       {num_components}")
        
        if num_components > 1:
            component_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
            print(f"  Largest component:          {component_sizes[0]:,} nodes ({100*component_sizes[0]/num_nodes:.1f}%)")
            if len(component_sizes) > 1:
                print(f"  2nd largest component:      {component_sizes[1]:,} nodes")
            print(f"  Isolated nodes:             {sum(1 for s in component_sizes if s == 1)}")
        
        # === CENTRALITY ===
        print("\n⭐ CENTRALITY (Influential nodes)")
        print("-" * 70)
        
        # Degree centrality (most connections)
        degree_centrality = nx.degree_centrality(G)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 by degree centrality:")
        for node, cent in top_degree:
            print(f"    Node {node}: {cent:.4f} ({dict(G.degree())[node]} connections)")
        
        # Betweenness centrality (brokers/bridges)
        # Sample for large networks to speed up computation
        if num_nodes > 1000:
            sample_nodes = random.sample(list(G.nodes()), min(1000, num_nodes))
            betweenness = nx.betweenness_centrality(G, k=len(sample_nodes))
            print(f"\n  Betweenness centrality (sampled):")
        else:
            betweenness = nx.betweenness_centrality(G)
            print(f"\n  Betweenness centrality:")
        
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        for node, cent in top_betweenness:
            print(f"    Node {node}: {cent:.4f}")
        
        # === ASSORTATIVITY (Homophily) ===
        print("\n🤝 ASSORTATIVITY (Homophily - like connects to like)")
        print("-" * 70)
        degree_assortativity = nx.degree_assortativity_coefficient(G)
        
        print(f"  Degree assortativity:       {degree_assortativity:.4f}")
        if degree_assortativity > 0.1:
            print(f"  Interpretation:             Assortative (hubs connect to hubs)")
        elif degree_assortativity < -0.1:
            print(f"  Interpretation:             Disassortative (hubs connect to low-degree)")
        else:
            print(f"  Interpretation:             Neutral mixing")
        
        # === SUMMARY FOR MODELING ===
        print("\n" + "=" * 70)
        print("🎯 KEY MODELING CHARACTERISTICS")
        print("=" * 70)
        
        is_scale_free = max_degree > 10 * avg_degree
        # is_small_world = avg_clustering > density and (avg_shortest_path < np.log(num_nodes) if num_nodes > 1 else True)
        
        print(f"  Scale-free (power-law):     {is_scale_free} (max/avg degree ratio: {max_degree/avg_degree:.1f})")
        # print(f"  Small-world property:       {is_small_world}")
        print(f"  High clustering:            {avg_clustering > 0.1}")
        print(f"  Sparse network:             {density < 0.01}")
        
        print("\n" + "=" * 70)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'avg_clustering': avg_clustering,
            'transitivity': transitivity,
            # 'avg_shortest_path': avg_shortest_path,
            # 'diameter': diameter,
            'num_components': num_components,
            'degree_assortativity': degree_assortativity
        }
    
    def visualize_plotly(self, node_size=10, edge_width=2, width=1200, height=800):
        # Build NetworkX graph
        G = nx.Graph()
        for u in self.adjacency:
            for v, weight in self.adjacency[u].items():
                G.add_edge(int(u), int(v))
 
        # Sample nodes
        nodes_sample = random.sample(list(G.nodes), 500)
        G = G.subgraph(nodes_sample)
 
        pos = nx.spring_layout(G)
        
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=edge_width, color='#888'),
            hoverinfo='none')
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        node_trace = go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=node_size, color='lightblue', line=dict(width=0.5, color='darkblue')),
            hoverinfo='text')
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height
        )
        
        # Convert to image
        img_bytes = fig.to_image(format="png", width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))
        
        return img