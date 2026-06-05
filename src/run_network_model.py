"""
Example usage of the optimized Numba-based networked model.

This demonstrates how to:
1. Set up and run the optimized model
2. Compare performance with original implementation
3. Validate results
4. Access and analyze output data
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.NetworkedModel import NetworkedModel
from networked_model.networks.AbstractNetwork import Network
from Constants import MINUTE_LENGTH
SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4


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
            "initial_attractiveness": 15,
            "node_removal_rate": 0.2,
            "edge_removal_prob": 0.5,
            "news_stations": num_news_stations,
            "news_intensity": news_intensity,
            "agent_types": {
                "weekend": int(num_agents/2),
            }
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

def run_basic_simulation(net_type=None, sub_type=None, num_agents=4039):
    """Run a basic simulation with the optimized model."""
    
    # Define parameters
    model = init_empty_model(net_type, sub_type, num_agents)
    
    print(f"Model initialized with {model.num_agents} agents")
    print(f"Network edges: {len(model.network.edges)}")
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    
    for step in range(model.num_steps):
        model.step()
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"Step {step + 1}/{model.num_steps} "
                  f"({rate:.1f} steps/sec)")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Simulation complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/model.num_steps*1000:.2f} ms/step")
    print(f"Throughput: {model.num_steps/elapsed_time:.1f} steps/sec")
    
    return model


def analyze_results(model):
    """Analyze and summarize simulation results."""
    
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)
    
    # Calculate statistics for each tracked variable
    for var in model.tracked_vars:
        data = model.data[var]
        
        # Overall statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Time-averaged statistics per agent
        agent_means = np.mean(data, axis=0)
        
        # Find agents with highest/lowest average values
        highest_agent = np.argmax(agent_means)
        lowest_agent = np.argmin(agent_means)
        
        print(f"\n{var}:")
        print(f"  Overall: mean={mean_val:.4f}, std={std_val:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Highest avg (agent {highest_agent}): {agent_means[highest_agent]:.4f}")
        print(f"  Lowest avg (agent {lowest_agent}): {agent_means[lowest_agent]:.4f}")
    
    # Check for concerning patterns
    print("\n" + "-" * 60)
    print("Pattern Detection:")
    
    # High suicidal thought prevalence
    thought_data = model.data['suicidal_thought']
    high_thought_threshold = 0.5
    high_thought_count = np.sum(thought_data > high_thought_threshold)
    total_observations = thought_data.size
    
    print(f"  High suicidal thought (>{high_thought_threshold}): "
          f"{high_thought_count}/{total_observations} "
          f"({100*high_thought_count/total_observations:.1f}%)")
    
    # Temporal trends
    time_avg = np.mean(thought_data, axis=1)
    if time_avg[-1] > time_avg[0]:
        print(f"  ⚠ Increasing trend in suicidal thought detected")
    else:
        print(f"  ✓ Decreasing or stable trend in suicidal thought")


def export_results(model, filename='simulation_results.npz'):
    """Export simulation results to a file."""
    
    # Prepare data dictionary
    data_dict = {}
    for var in model.tracked_vars:
        data_dict[var] = model.data[var]
    
    # Add metadata
    data_dict['num_agents'] = model.num_agents
    data_dict['num_steps'] = model.num_steps
    data_dict['dt'] = model.dt
    
    # Save
    if filename is not None:
        np.savez_compressed(filename, **data_dict)
        print(f"✓ Results saved!")
    else:
        return data_dict


def find_zero_timesteps(filename='simulation_results.npz'):
    """
    Identify timesteps where all variables for all agents are exactly 0,
    ignoring metadata keys like 'num_agents', 'num_steps', 'dt'.
    """
    import numpy as np

    print(f"\nLoading results from {filename}...")
    data = np.load(filename)

    # Only keep actual simulation arrays (skip scalars / metadata)
    variables = [k for k in data.keys() if isinstance(data[k], np.ndarray) and data[k].ndim == 2]

    if not variables:
        raise ValueError("No 2D simulation arrays found in the file.")

    num_steps = data[variables[0]].shape[0]
    zero_mask = np.ones(num_steps, dtype=bool)

    for var in variables:
        var_data = data[var]  # shape: (time_steps, num_agents)
        # No need for ndim check here anymore
        zero_mask &= np.all(var_data == 0, axis=1)

    zero_steps = np.where(zero_mask)[0]
    print(f"Timesteps where all values are 0: {zero_steps.tolist()}")
    return zero_steps.tolist()


def plot_single_agent_dynamics(data=None, filename='simulation_results.npz', agent_idx=0, img_name='dynamics'):
    """Plot all dynamics for a single agent with state backgrounds and x-axis in days."""

    import numpy as np
    import matplotlib.pyplot as plt
    from Constants import Constants

    if data is None:
        print(f"\nLoading results from {filename}...")
        data = np.load(filename)

    print(f"Plotting combined dynamics for agent {agent_idx}")

    variables = [
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat', 'burdensomeness', 'suicide_history'
    ]

    # State background colours (opaque enough to read, light enough not to obscure lines)
    STATE_COLORS = {
        SLEEP:   '#cce5ff',  # pale blue
        MORNING: '#fff3cd',  # pale yellow
        COMMUTE: '#fde2e2',  # pale red
        WORK:    '#d4edda',  # pale green
        HOME:    '#e2d9f3',  # pale purple
    }
    STATE_LABELS = {
        SLEEP: 'Sleep', MORNING: 'Morning', COMMUTE: 'Commute',
        WORK: 'Work', HOME: 'Home',
    }

    FONT_SIZE = 16

    line_colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(16, 6))

    # ── build time axis ───────────────────────────────────────────────────────
    # Use the first variable to determine series length
    ref_series = data[variables[0]][:, agent_idx]
    time_steps = np.arange(len(ref_series)) * MINUTE_LENGTH

    # ── state background bands ────────────────────────────────────────────────
    if 'state' in data:
        state_series = np.asarray(data['state'][:, agent_idx], dtype=int)
        n = len(state_series)

        # Walk through contiguous runs of the same state and shade each span
        seen_states = set()
        i = 0
        while i < n:
            current_state = state_series[i]
            j = i + 1
            while j < n and state_series[j] == current_state:
                j += 1
            # span covers [time_steps[i], time_steps[j-1] + MINUTE_LENGTH)
            t_start = time_steps[i]
            t_end   = time_steps[j - 1] + MINUTE_LENGTH
            color   = STATE_COLORS.get(current_state, '#eeeeee')
            label   = STATE_LABELS.get(current_state, str(current_state)) \
                      if current_state not in seen_states else None
            ax.axvspan(t_start, t_end, facecolor=color, alpha=1.0,
                       label=label, zorder=0)
            seen_states.add(current_state)
            i = j

    # ── dynamics lines ────────────────────────────────────────────────────────
    for i, var in enumerate(variables):
        var_data = data[var]
        if var_data.ndim != 2:
            raise ValueError(f"{var} is not 2D (time x agents)")
        if agent_idx >= var_data.shape[1]:
            raise IndexError(f"Agent index {agent_idx} out of bounds for {var}")

        agent_series = var_data[:, agent_idx]
        ax.plot(
            time_steps,
            agent_series,
            label=var.replace('_', ' ').title(),
            color=line_colors[i % len(line_colors)],
            linewidth=2,
            zorder=2,
        )

    ax.set_xlabel("Time (days)", fontsize=FONT_SIZE)
    ax.set_ylabel("Value", fontsize=FONT_SIZE)
    ax.set_title(f"Agent {agent_idx} — Dynamics Across All Variables",
                 fontsize=FONT_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(time_steps[0], time_steps[-1])

    # Two-column legend: state bands first, then variable lines
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, zorder=1)
    fig.tight_layout()

    output_path = Path("src/output") / f"{img_name}_single.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved to {output_path}")

    plt.show()


def load_and_visualize(filename='simulation_results.npz', data=None, img_name="dynamics"):
    """Load results and create basic visualizations."""
    if data is None:
        print(f"\nLoading results from {filename}...")
        
        data = np.load(filename)
    
    print(f"Loaded data:")
    print(f"  Agents: {data['num_agents']}")
    print(f"  Steps: {data['num_steps']}")
    print(f"  dt: {data['dt']}")
    
    # Calculate some statistics
    stress = data['stress']
    print(f"\nStress statistics:")
    print(f"  Mean: {np.mean(stress):.4f}")
    print(f"  Std: {np.std(stress):.4f}")
    
    # Optional: Create plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Time series of population averages
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        variables = [
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'burdensomeness', 'suicide_history'
        ]
        
        for ax, var in zip(axes.flat, variables):
            var_data = data[var]
            
            # Plot mean and std bands
            mean_vals = np.mean(var_data, axis=1)
            std_vals = np.std(var_data, axis=1)
            time_steps = np.arange(len(mean_vals))
            
            ax.plot(time_steps, mean_vals, 'b-', linewidth=2, label='Mean')
            ax.fill_between(time_steps, 
                           mean_vals - std_vals,
                           mean_vals + std_vals,
                           alpha=0.3, label='±1 SD')
            ax.set_title(var.replace('_', ' ').title(), fontsize=16, fontweight='bold')
            ax.set_xlabel('Time step', fontsize=16)
            ax.set_ylabel('Value', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'src/output/{img_name}.png', dpi=150)
        print(f"✓ Visualization saved to {img_name}.png")
        
    except ImportError:
        print("  (matplotlib not available for visualization)")


def run_sensitivity_analysis():
    """Run a basic sensitivity analysis on key parameters."""
    
    print("\n" + "=" * 60)
    print("Sensitivity Analysis Example")
    print("=" * 60)
    
    base_params = {
        'num_steps': 500,
        # ... use defaults
    }
    
    # Test different stress baseline values
    baseline_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    results = []
    
    print("\nTesting different stress baseline values...")
    for baseline in baseline_values:
        params = base_params.copy()
        params['stress_baseline'] = baseline
        
        model = NetworkedModel(dt=0.1, seed=42, parameters=params)
        
        # Run simulation
        for _ in range(params['num_steps']):
            model.step()
        
        # Record final average suicidal thought
        final_thought = np.mean(model.data['suicidal_thought'][-1, :])
        results.append(final_thought)
        
        print(f"  Baseline={baseline:.2f} → "
              f"Final avg suicidal thought={final_thought:.4f}")
    
    print("\n✓ Sensitivity analysis complete!")


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
                # 'freq': data['freq'],
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


def summarize_network(model=None, net_type=None, sub_type=None, num_agents=1000, m=5):
    if model is None:
        model = init_empty_model(net_type=net_type, sub_type=sub_type, num_agents=num_agents, m=m)
    model.network.network_summary()


def run_and_plot_dynamics(net_type, sub_type, img_name):
    model = run_basic_simulation(net_type=net_type, sub_type=sub_type)
    data = export_results(model, filename=None)
    plot_single_agent_dynamics(data=data, img_name=img_name)
    return model


def parameter_sweep_hk(param):
    parameters = {
        'num_steps': int(10/MINUTE_LENGTH),
        'm': 10,
        'cluster_prob': 0.5,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.2,
        "edge_removal_prob": 0.2,
        "network": 'hk',
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
    hk_base_data = plot_network(recache=True, net_type='hk', num_agents=20000, plot_now=False, m=10)

    plot_data = parameter_sweep_hk('m')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        title="Degree distributions for extended HK Network\n(variable local cluster size)",
        output_file="src/output/hk_dists_m.png", include_size=False)

    plot_data = parameter_sweep_hk('cluster_prob')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        title=f"Degree distributions for extended HK Network\n(variable local cluster probability)",
        output_file="src/output/hk_dists_cluster_prob.png", include_size=False)
    
    plot_data = parameter_sweep_hk('initial_attractiveness')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        title=f"Degree distributions for extended HK Network\n(variable initial attractiveness)",
        output_file="src/output/hk_dists_initial_attractiveness.png", include_size=False)
    
    plot_data = parameter_sweep_hk('node_removal_rate')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        title=f"Degree distributions for extended HK Network\n(variable node removal rate)",
        output_file="src/output/hk_dists_node_removal_rate.png", include_size=False)
    
    plot_data = parameter_sweep_hk('edge_removal_prob')
    plot_data.append(hk_base_data)
    Network.plot_combined_distributions(
        plot_data,
        title=f"Degree distributions for extended HK Network\n(variable edge removal probability)",
        output_file="src/output/hk_dists_edge_removal_prob.png", include_size=False)


def plot_all_dists():
    fb_data = plot_network(
        recache=False, net_type='empirical', sub_type='facebook', plot_now=False)
    artist_data = plot_network(
        recache=False, net_type='empirical', sub_type='artist', plot_now=False)
    new_site_data = plot_network(
        recache=False, net_type='empirical', sub_type='new_site', plot_now=False)
    athlete_data = plot_network(
        recache=False, net_type='empirical', sub_type='athlete', plot_now=False)
    hk_data = plot_network(
        recache=False,
        net_type='hk', num_agents=25000, plot_now=False, m=16)
    # hk_base_data = plot_network(recache=False,
    #                             net_type='hk', num_agents=20000, plot_now=False, m=16)
    Network.plot_combined_distributions([
        fb_data,
        artist_data,
        new_site_data,
        athlete_data,
        hk_data,
        # hk_base_data,
    ])
    # Network.plot_combined_distributions([
    #     hk_data,
    #     hk_base_data,
    #     athlete_data,
    #     ], title="CCDFs for extended HK Network\nand Athlete Network",
    #     output_file="src/output/hk_athlete_dists.png")


def run_simulation_with_news(num_agents=1000, num_news_stations=5, num_days=20, 
                              m=5, news_intensity=0.5, seed=42, warmup=0):
    """
    Run a simulation with news stations for a specified number of days.
    
    Parameters:
    -----------
    num_agents : int
        Number of agents in the simulation
    num_news_stations : int
        Number of news stations
    num_days : int
        Number of days to simulate
    m : int
        Number of connections per node in HK network
    news_intensity : float
        Intensity of news effects (0-1)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    NetworkedModel
        The completed model with data
    """
    # Calculate number of steps based on days
    num_steps = int((num_days+warmup) / MINUTE_LENGTH)
    
    print(f"Creating model with {num_agents} agents and {num_news_stations} news stations...")

    model = init_empty_model(sim_length=(num_days+warmup), net_type='hk', num_agents=num_agents, m=16, num_news_stations=num_news_stations, news_intensity=news_intensity,
                             warmup=warmup)
    
    print(f"Model initialized:")
    print(f"  Agents: {model.num_agents}")
    print(f"  News stations: {len(model.news_stations)}")
    print(f"  Network edges: {len(model.network.edges)}")
    print(f"  Simulation time: {num_days} days ({num_steps} steps)")
    
    if model.news_stations:
        print("\nNews station distribution:")
        for station in model.news_stations:
            print(f"  {station}")
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    
    for step in range(num_steps):
        model.step()
        
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"Step {step + 1}/{num_steps} ({rate:.1f} steps/sec)")
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Simulation complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    return model


def plot_single_agent_with_news_signal(model, agent_idx=None, output_file='single_agent_news.png'):
    """
    Plot the internal dynamics of a single agent over time, with a marker for the news signal.
    
    Parameters:
    -----------
    model : NetworkedModel
        Completed simulation model
    agent_idx : int, optional
        Index of agent to plot. If None, selects a random agent.
    output_file : str
        Path to save the plot
    """
    if agent_idx is None:
        agent_idx = np.random.randint(0, model.num_agents)
    
    print(f"\nPlotting dynamics for agent {agent_idx}")
    
    variables = [
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat', 'burdensomeness', 'suicide_history'
    ]
    
    colors = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, var in enumerate(variables):
        var_data = model.data[var]
        agent_series = var_data[:, agent_idx]
        time_steps = np.arange(len(agent_series)) * model.dt
        
        ax.plot(time_steps, agent_series, label=var.replace('_', ' ').title(),
                color=colors[i % len(colors)], linewidth=2)
    
    # Add vertical line at news signal time (timestep 10)
    news_signal_time = 10
    ax.axvline(x=news_signal_time, color='red', linestyle='--', linewidth=2, 
               label='News Signal', alpha=0.7)
    
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.set_ylabel('Value', fontsize=16)
    ax.set_title(f'Agent {agent_idx} Internal Dynamics with News Signal', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path("src/output") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def plot_suicidal_thought_prevalence(model, threshold=0.02, output_file='suicidal_thought_prevalence.png'):
    """
    Plot:
    1. Number of agents above suicidal thought threshold
    2. Number of those agents who consume news
    """
    print(f"\nPlotting suicidal thought prevalence (threshold: {threshold})")
    
    thought_data = model.data['suicidal_thought']
    total_agents = thought_data.shape[1]
    
    # Boolean masks
    above_threshold = thought_data > threshold  # shape: (time, agents)
    consumes_news = model.consumes_news         # shape: (agents,)
    
    # Count agents above threshold
    agents_above_threshold = np.sum(above_threshold, axis=1)
    
    # Count agents above threshold AND consuming news
    agents_above_and_news = np.sum(
        above_threshold & consumes_news,
        axis=1
    )
    
    news_agents = agents_above_and_news
    
    time_steps = np.arange(len(agents_above_threshold)) * model.dt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- Total prevalence ---
    ax.plot(
        time_steps,
        agents_above_threshold,
        linewidth=2,
        color='darkblue',
        label='Total above threshold'
    )
    
    ax.fill_between(
        time_steps,
        agents_above_threshold,
        alpha=0.3,
        color='blue'
    )
    
    # --- Scaled proportion ---
    ax.plot(
        time_steps,
        news_agents,
        linewidth=2,
        linestyle='--',
        color='green',
        label='News-consuming proportion (scaled)'
    )
    
    # --- News signal marker ---
    news_signal_time = model.warmup + 10
    
    ax.axvline(
        x=news_signal_time,
        color='red',
        linestyle='--',
        linewidth=2,
        label='News Signal',
        alpha=0.7
    )
    
    # --- Labels ---
    ax.set_xlabel('Time (days)', fontsize=16)
    
    ax.set_ylabel('Number of Agents', fontsize=16)
    
    ax.set_ylim(
        0,
        total_agents
    )

    ax.set_xlim(
        model.warmup,
        max(time_steps)
    )
    
    ax.set_title(
        'Suicidal Thought Prevalence and News Exposure',
        fontsize=16,
        fontweight='bold'
    )
    
    ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    output_path = Path("src/output") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Plot saved to {output_path}")
    
    plt.close()


def run_news_simulation_example():
    """
    Example function demonstrating both types of plots.
    """
    print("=" * 70)
    print("RUNNING NEWS STATION SIMULATION EXAMPLE")
    print("=" * 70)
    
    # Run simulation
    model = run_simulation_with_news(
        num_agents=5000,
        num_news_stations=4,
        num_days=40,
        news_intensity=0.7,
        seed=42,
        warmup=20
    )
    
    # Plot single agent dynamics
    plot_single_agent_with_news_signal(
        model,
        agent_idx=None,  # Random agent
        output_file='single_agent_news_example.png'
    )
    
    # Plot suicidal thought prevalence
    plot_suicidal_thought_prevalence(
        model,
        threshold=0.02,
        output_file='suicidal_thought_prevalence_levy_low_alpha.png'
    )
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)




if __name__ == "__main__":
    # Run basic simulation
    # run_and_plot_dynamics(net_type="empirical", sub_type="facebook", img_name="fb_dynamics")
    # run_and_plot_dynamics(net_type="empirical", sub_type="artist", img_name="artist_dynamics")
    # run_and_plot_dynamics(net_type="empirical", sub_type="new_site", img_name="new_site_dynamics")
    # run_and_plot_dynamics(net_type="empirical", sub_type="athlete", img_name="athlete_dynamics")
    # run_and_plot_dynamics(net_type="ba", img_name="ba_dynamics")
    run_and_plot_dynamics(net_type="hk", sub_type=None, img_name="hk_dynamics")
    
    # Analyze results
    # analyze_results(model)
    
    # Export results
    # export_results(model)
    
    # Performance comparison
    # compare_performance()
    
    # load_and_visualize()

    # plot_single_agent_dynamics()

    # plot_all_dists()

    # parameter_sweeps()


    # summarize_network(net_type='facebook') # n=4039
    # summarize_network(net_type='ba', num_agents=4039)
    # summarize_network(net_type='hk', num_agents=4039)
    # summarize_network(net_type='hk', num_agents=25000, m=10)
    # summarize_network(net_type='hk', num_agents=20000)
    # summarize_network(net_type='empirical', sub_type='athlete', num_agents=25000)
    # summarize_network(net_type='empirical', sub_type='athlete', num_agents=25000)

    # NEW: Run news station simulation example
    # run_news_simulation_example()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)