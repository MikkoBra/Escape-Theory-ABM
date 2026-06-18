from pathlib import Path
import numpy as np
from networked_model.models.NetworkedModel import NetworkedModel
from networked_model.networks.AbstractNetwork import Network
from Constants import Constants
MINUTE_LENGTH = Constants.MINUTE_LENGTH


def init_empty_model(net_type=None, sub_type=None, num_agents=1000, m=16, num_news_stations=0, news_intensity=0.5, sim_length=10, warmup=0):
    if net_type == 'hk' and num_agents == 20000:
        parameters = {
            'num_steps': int(10/MINUTE_LENGTH),
            'm': 5,
            'cluster_prob': 0.6,
            "initial_attractiveness": 0,
            "node_removal_rate": 0,
            "edge_removal_prob": 0,
        }
    else:
        parameters = {
            'num_steps': int(sim_length/MINUTE_LENGTH),
            'm': m,
            'cluster_prob': 0.6,
            "initial_attractiveness": 5,
            "node_removal_rate": 0.1,
            "edge_removal_prob": 0.5,
            "news_stations": num_news_stations,
            "news_intensity": news_intensity,
            "hub_count": 2,
            "hub_degree": 400,
        }
    if net_type:
        parameters['network'] = net_type
        if sub_type:
            parameters["subtype"] = sub_type
    else:
        parameters['network'] = 'empirical'
        parameters["subtype"] = 'facebook'
    
    # Create model
    print("Creating model...")
    return NetworkedModel(
        dt=MINUTE_LENGTH,
        seed=42,
        parameters=parameters,
        verbose=True,
        num_agents=num_agents,
        warmup=warmup
    )


def plot_network(cache=True, recache=False, model=None, net_type=None, sub_type=None, num_agents=1000, xmax=2000, plot_now=True, m=5):
    # Create cache key
    if sub_type:
        cache_key = f"{net_type}_{sub_type}_{num_agents}"
    else:
        cache_key = f"{net_type}_{num_agents}"

    if cache:
        cache_dir = Path("src/output/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}_degree_dist.npz"
    
        # Try to load from cache
        if cache_file.exists() and not plot_now and not recache:
            data = np.load(cache_file)
            return {
                'degrees': data['degrees'],
                'pdf': data['pdf'],
                'name': str(data['name']),
                'size': data['size']
            }
    
    # Generate new model if needed
    if model is None:
        print(f"Generating new model for {cache_key}")
        model = init_empty_model(net_type=net_type, sub_type=sub_type, num_agents=num_agents,m=m,sim_length=1)
    
    # Get degree distribution data
    if plot_now:
        model.network.plot_degree_distribution(xmax=xmax, plot_now=plot_now, bin_count=4000)
    else:
        data = model.network.plot_degree_distribution(xmax=xmax, plot_now=plot_now, bin_count=4000)
        if net_type == "hk" and num_agents == 20000:
            data['name'] = "Holme-Kim Network (base)"
        # Save to cache
        if cache and data is not None:
            np.savez(cache_file,
                    degrees=data['degrees'],
                    pdf=data['pdf'],
                    size=data['size'],
                    name=data['name'])
            print(f"Saved cache for {cache_key}")
        
        return data


def plot_all_dists():
    fb_data = plot_network(
        recache=False, net_type='empirical', sub_type='facebook', plot_now=False)
    artist_data = plot_network(
        recache=False, net_type='empirical', sub_type='artist', plot_now=False)
    new_site_data = plot_network(
        recache=False, net_type='empirical', sub_type='new_site', plot_now=False)
    athlete_data = plot_network(
        recache=False, net_type='empirical', sub_type='athlete', plot_now=False)
    # hk_data = plot_network(
    #     recache=False,
    #     net_type='hk', num_agents=25000, plot_now=False, m=16)
    Network.plot_combined_distributions(networks_data=[
        fb_data,
        artist_data,
        new_site_data,
        athlete_data,
        # hk_data,
    ],
    title="Degree Distribution",
    output_file="src/output/empirical_networks.png")


def parameter_sweep_hk(param):
    parameters = {
        'num_steps': int(10/MINUTE_LENGTH),
        'm': 10,
        'cluster_prob': 0.5,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.2,
        "edge_removal_prob": 0.2,
        "network": 'hk',
        "num_agents": 20000,
    }
    plot_data = []
    value_data = []
    if param == 'm':
        value_data = [2, 5, 10, 20]
    elif param == 'cluster_prob':
        value_data = [0.3, 0.5, 0.7, 0.9]
    elif param == 'initial_attractiveness':
        value_data = [1, 5, 10, 20]
    elif param == 'node_removal_rate' or param == 'edge_removal_prob':
        value_data = [0.05, 0.1, 0.2, 0.3]
    for val in value_data:
        print(f'Creating distribution for {param}={val}')
        parameters[param] = val
        model = NetworkedModel(
            dt=MINUTE_LENGTH,
            seed=42,
            parameters=parameters,
            verbose=True,
            num_agents=25000
        )
        single_data = plot_network(model=model, cache=False, plot_now=False)
        single_data['name'] = f'HK ({param}={val})'
        plot_data.append(single_data)
    return plot_data


def parameter_sweeps():
    hk_base_data = plot_network(recache=False, net_type='hk', num_agents=20000, plot_now=False, m=10)

    plot_data = parameter_sweep_hk('m')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        # title="Degree distributions for extended HK Network\n(variable local cluster size)",
        title="",
        output_file="src/output/hk_dists_m.png", include_size=False)

    plot_data = parameter_sweep_hk('cluster_prob')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        # title=f"Degree distributions for extended HK Network\n(variable local cluster probability)",
        title="",
        output_file="src/output/hk_dists_cluster_prob.png", include_size=False)
    
    plot_data = parameter_sweep_hk('initial_attractiveness')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        # title=f"Degree distributions for extended HK Network\n(variable initial attractiveness)",
        title="",
        output_file="src/output/hk_dists_initial_attractiveness.png", include_size=False)
    
    plot_data = parameter_sweep_hk('node_removal_rate')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        # title=f"Degree distributions for extended HK Network\n(variable node removal rate)",
        title="",
        output_file="src/output/hk_dists_node_removal_rate.png", include_size=False)
    
    plot_data = parameter_sweep_hk('edge_removal_prob')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        # title=f"Degree distributions for extended HK Network\n(variable edge removal probability)",
        title="",
        output_file="src/output/hk_dists_edge_removal_prob.png", include_size=False)


if __name__ == "__main__":

    # parameter_sweeps()
    plot_all_dists()