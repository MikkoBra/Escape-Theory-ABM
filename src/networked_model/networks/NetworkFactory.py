from networked_model.networks.FacebookNetwork import FacebookNetwork
from networked_model.networks.fb_sub_networks.FacebookArtistNetwork import ArtistNetwork
from networked_model.networks.fb_sub_networks.FacebookNewSitesNetwork import NewSitesNetwork
from networked_model.networks.fb_sub_networks.FacebookAthletesNetwork import AthletesNetwork
from networked_model.networks.HolmeKimNetwork import HolmeKimNetwork
from networked_model.networks.BarabasiAlbertNetwork import BarabasiAlbertNetwork


empirical_networks = {
    "facebook": FacebookNetwork,
    "artist": ArtistNetwork,
    "athlete": AthletesNetwork,
    "new_site": NewSitesNetwork,
}


class NetworkFactory():
    def __init__(self):
        pass

    def create_network(self, model, parameters):
        network_type = parameters.get("network")
        if network_type == "empirical":
            network = empirical_networks[parameters.get("subtype")]()
            network.generate_network()
            model.num_agents = len(network.nodes)
            model.network = network
            return
        else:
            model.m = parameters.get("m")
            model.initial_attractiveness = parameters.get("initial_attractiveness")
            model.node_removal_rate = parameters.get("node_removal_rate")
            model.edge_removal_prob = parameters.get("edge_removal_prob")
            model.cluster_prob = parameters.get("cluster_prob")
            model.hub_count = parameters.get("hub_count")
            model.hub_degree = parameters.get('hub_degree')
            network = HolmeKimNetwork(model=model)
            network.generate_network()
            model.network = network
