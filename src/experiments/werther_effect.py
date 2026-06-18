import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.NewsSpreadModel import NewsSpreadModel
from experiments.util import run_basic_simulation, export_results, plot_single_agent_dynamics, generate_stress_signal,\
      plot_single_dynamics_in_comparison, clean_zero_entries, load_simulation_data, auc_per_agent, plot_auc_distribution,\
      perform_t_test, batched, plot_comparison, suicidal_prevalence_timeseries, plot_suicidal_prevalence_comparison
from Constants import MINUTE_LENGTH
SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4
from matplotlib import rcParams
default_colors = rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib.lines import Line2D
import json
from tqdm import tqdm


var_dict = {
    "stress": "Stress",
    "suicidal_thought": "Suicidal Thought",
    "aversive_internal_state": "Aversive Internal State",
    "suicide_history": "Memory of Suicidal Thought",
    "urge_to_escape": "Urge to Escape",
    "escape_behavior": "Escape Behavior",
    "external_strat": "External Strategy",
    "internal_strat": "Internal Strategy",
}
ALL_TRACKED_VARS = [
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'burdensomeness', 'total_time', 'labels', 'state',
        ]
SCHEDULE_VARS = [
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'total_time', 'labels', 'state',
        ]
MEMORY_VARS = [
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'total_time', 'labels',
        ]


def run_single_aucs_model(simulation_id):

    sim_length = 50

    model = NewsSpreadModel(num_agents=3, sim_length=sim_length, dt=MINUTE_LENGTH, seed=simulation_id,
                       verbose=False, warmup=10, parameters={
                           "agent_types": {
                               "default": 1,
                               "good_sleep": 1,
                               "bad_sleep": 1,
                           },
                           "randomize": False
                       })

    while model.time < sim_length:
        model.step()

    agent_results = []
    auc_Ts = auc_per_agent(model.data['suicidal_thought'])
    auc_As = auc_per_agent(model.data['aversive_internal_state'])
    for idx, agent_type in enumerate(["default", "good_sleep", "bad_sleep"]):
        agent_results.append({
            "agent_type": agent_type,
            "auc_T": float(auc_Ts[idx]),
            "auc_A": float(auc_As[idx])
        })

    return {
        "simulation_id": int(simulation_id),
        "agents": agent_results
    }


from multiprocessing import Pool, cpu_count
BATCH_SIZE = 1

def parallel_batches_auc(num_runs=1000, filename='src/output/batch_run_results.json'):

    all_results = []

    with Pool(cpu_count()) as pool:
        sim_ids = range(num_runs)

        for batch in batched(pool.imap_unordered(run_single_aucs_model, sim_ids), BATCH_SIZE):

            all_results.extend(batch)

            # Incremental write
            with open(filename, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"Saved {len(all_results)} simulations")


def load_results(filename="src/output/batch_run_results.json"):
    with open(filename, "r") as f:
        sim_results = json.load(f)

    good_T, bad_T, var_T = [], [], []
    good_A, bad_A, var_A = [], [], []

    for sim in sim_results:
        for agent in sim["agents"]:
            agent_type = agent["agent_type"]

            if agent_type == "default":
                var_T.append(agent["auc_T"])
                var_A.append(agent["auc_A"])

            elif agent_type == "bad_sleep":
                bad_T.append(agent["auc_T"])
                bad_A.append(agent["auc_A"])

            elif agent_type == "good_sleep":
                good_T.append(agent["auc_T"])
                good_A.append(agent["auc_A"])
    return good_T, good_A, bad_T, bad_A, var_T, var_A


def run_many(sim_length, num_sims, starting_seed=0,
             filename="src/output/sleep_experiment_results.jsonl",
             batch_size=100):

    batch = []

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    for i in tqdm(
        range(starting_seed, starting_seed + num_sims),
        desc="Simulations",
        unit="sim"
    ):

        model = NewsSpreadModel(
            num_agents=3,
            sim_length=sim_length,
            dt=MINUTE_LENGTH,
            seed=i,
            verbose=False,
            warmup=10,
            parameters={
                "agent_types": {
                    "default": 1,
                    "good_sleep": 1,
                    "bad_sleep": 1,
                },
                "randomize": False
            }
        )

        while model.time < sim_length:
            model.step()

        As = auc_per_agent(
            model.data["aversive_internal_state"],
            MINUTE_LENGTH
        )

        Ts = auc_per_agent(
            model.data["suicidal_thought"],
            MINUTE_LENGTH
        )

        batch.append({
            "seed": i,

            "var_A": float(As[0]),
            "good_A": float(As[1]),
            "bad_A": float(As[2]),

            "var_T": float(Ts[0]),
            "good_T": float(Ts[1]),
            "bad_T": float(Ts[2]),
        })

        if len(batch) >= batch_size:
            with open(filename, "a") as f:
                for row in batch:
                    f.write(json.dumps(row) + "\n")

            print(f"Saved through seed {i}")
            batch.clear()

    # save final partial batch
    if batch:
        with open(filename, "a") as f:
            for row in batch:
                f.write(json.dumps(row) + "\n")


def load_auc_results(filename="src/output/auc_results.jsonl"):

    results = []

    with open(filename, "r") as f:
        for line in f:
            results.append(json.loads(line))

    return results


def load_auc_arrays(filename="src/output/auc_results.jsonl"):

    results = load_auc_results(filename)

    return {
        "var_As": np.array([r["var_A"] for r in results]),
        "good_As": np.array([r["good_A"] for r in results]),
        "bad_As": np.array([r["bad_A"] for r in results]),

        "var_Ts": np.array([r["var_T"] for r in results]),
        "good_Ts": np.array([r["good_T"] for r in results]),
        "bad_Ts": np.array([r["bad_T"] for r in results]),
    }


def run_werther_models(batch_size=5000, batch_save=True,
                 seed=None, warmup=10, num_agents=1000, sim_length=50):
    params={
        "agent_types": {
            "default": num_agents,
        },
        "read_stress": False,
        "randomize": True,
        "only_sleep": False,
        "network": "hk",
        "m": 16,
        'cluster_prob': 0.6,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.1,
        "edge_removal_prob": 0.5,
        "hub_count": 2,
        "hub_degree": 250,
        "social_events": True,
        "weighted_network": True,
        "spread_news": True,
        "low_degree_consumer": True,
        "hub_consumer": False,
        "news_stations": 4,
        "news_intensity": [0.8, 0.8, 0.8, 0.8],
        "proportion_consumers": 0.05,
    }
    # Define parameters
    low_degree = NewsSpreadModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    params={
        "agent_types": {
            "default": num_agents,
        },
        "read_stress": False,
        "randomize": True,
        "only_sleep": False,
        "network": "hk",
        "m": 16,
        'cluster_prob': 0.6,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.1,
        "edge_removal_prob": 0.5,
        "hub_count": 2,
        "hub_degree": 250,
        "social_events": True,
        "weighted_network": True,
        "spread_news": True,
        "low_degree_consumer": False,
        "hub_consumer": True,
        "news_stations": 4,
        "news_intensity": [0.5, 0.5, 0.5, 0.5],
        "proportion_consumers": 0.05,
    }
    # Define parameters
    hub = NewsSpreadModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    run_basic_simulation(low_degree, batch_save=batch_save, run_name="low_degree", batch_size=batch_size)
    run_basic_simulation(hub, batch_save=batch_save, run_name="hub", batch_size=batch_size)


def plot_werther_runs(warmup=10, num_agents=1000, sim_length=50):
    low_degree = load_simulation_data("low_degree", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    hub = load_simulation_data("hub", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    # plot_single_agent_dynamics(low_degree, agent_type='News Consumer\n(Low-Degree)', warmup=warmup, img_name="low_degree_short",
    #                            variables=[
    #                                 'stress', 'aversive_internal_state', 'urge_to_escape',
    #                                 'suicidal_thought', 'escape_behavior', 'external_strat',
    #                                 'internal_strat', 'burdensomeness'
    #                             ])
    # plot_single_agent_dynamics(hub, agent_type='News Consumer\n(Hubs)', warmup=warmup, img_name="hub_short",
    #                            variables=[
    #                                 'stress', 'aversive_internal_state', 'urge_to_escape',
    #                                 'suicidal_thought', 'escape_behavior', 'external_strat',
    #                                 'internal_strat', 'burdensomeness'
    #                             ])

    plot_suicidal_prevalence_comparison(low_degree, hub, label_1="News Consumer (Low-Degree)",
                                        label_2="News Consumer (Hubs)", warmup=warmup, sim_length=sim_length,
                                        save_name="news_scenario_single_comparison")

    # plot_comparison(
    #         event_data,
    #         antisocial_data,
    #         title_1="Social Agent",
    #         title_2="Anti-Social\nAgent",
    #         fig_name="social_v_antisocial",
    #         vars_1=[
    #             'stress', 'aversive_internal_state', 'urge_to_escape',
    #             'suicidal_thought', 'escape_behavior', 'external_strat',
    #             'internal_strat', 'burdensomeness',
    #         ],
    #         vars_2=[
    #             'stress', 'aversive_internal_state', 'urge_to_escape',
    #             'suicidal_thought', 'escape_behavior', 'external_strat',
    #             'internal_strat', 'burdensomeness',
    #         ],
    #         warmup=warmup,
    #         sim_length=sim_length
    #         )


def run_batched_distribution_analysis():
    run_many(sim_length=50, num_sims=1000, batch_size=10, starting_seed=0)


def load_aucs(num_agents=1000, sim_length=50):
    social_data = load_simulation_data("schedule", vars_list=SCHEDULE_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    antisocial_data = load_simulation_data("social_burden", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    auc_A_soc = auc_per_agent(social_data['aversive_internal_state'])
    auc_A_anti = auc_per_agent(antisocial_data['aversive_internal_state'])
    auc_T_soc = auc_per_agent(social_data['suicidal_thought'])
    auc_T_anti = auc_per_agent(antisocial_data['suicidal_thought'])
    return auc_A_soc, auc_A_anti, auc_T_soc, auc_T_anti


def plot_distributions(warmup=10, num_agents=1000, sim_length=50):

    auc_A_soc, auc_A_anti, auc_T_soc, auc_T_anti = load_aucs(num_agents=num_agents, sim_length=sim_length)
    plot_auc_distribution(
        datasets=[auc_A_soc, auc_A_anti],
        labels=["Social", "Anti-Social"],
        title="AUC(A) of Social vs Anti-Social",
        xlabel="AUC(A)"
        )
    plot_auc_distribution(
        datasets=[auc_T_soc, auc_T_anti],
        labels=["Social", "Anti-Social"],
        title="AUC(T) of Social vs Anti-Social",
        xlabel="AUC(T)"
        )


def run_werther_experiment_single(num_agents=5000, seed=42, sim_length=35, warmup=10, proportion_news=0.5, batch_save=False, batch_size=5000):
    
    params={
        "agent_types": {
            "default": num_agents,
        },
        "read_stress": False,
        "randomize": True,
        "only_sleep": False,
        "network": "hk",
        "m": 16,
        'cluster_prob': 0.6,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.1,
        "edge_removal_prob": 0.5,
        "hub_count": 2,
        "hub_degree": 250,
        "social_events": True,
        "weighted_network": True,
        "spread_news": True,
        "low_degree_consumer": True,
        "hub_consumer": False,
        "news_stations": 4,
        "news_intensity": [0.8, 0.8, 0.8, 0.8],
        "proportion_consumers": proportion_news,
    }
    # Define parameters
    low_degree = NewsSpreadModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    params={
        "agent_types": {
            "default": num_agents,
        },
        "read_stress": False,
        "randomize": True,
        "only_sleep": False,
        "network": "hk",
        "m": 16,
        'cluster_prob': 0.6,
        "initial_attractiveness": 5,
        "node_removal_rate": 0.1,
        "edge_removal_prob": 0.5,
        "hub_count": 2,
        "hub_degree": 250,
        "social_events": True,
        "weighted_network": True,
        "spread_news": True,
        "low_degree_consumer": False,
        "hub_consumer": True,
        "news_stations": 4,
        "news_intensity": [0.5, 0.5, 0.5, 0.5],
        "proportion_consumers": proportion_news,
    }
    # Define parameters
    hub = NewsSpreadModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    run_basic_simulation(low_degree, batch_save=batch_save, run_name="low_degree", batch_size=batch_size)
    run_basic_simulation(hub, batch_save=batch_save, run_name="hub", batch_size=batch_size)
    return low_degree.data, hub.data
    

def max_suicidal_thought_proportion(data_dict, T_threshold=0.05, signal_time=20):
    """
    Parameters
    ----------
    data_dict : dict
        Dictionary containing a 'suicidal_thought' entry with shape
        (num_steps, num_agents).
    T_threshold : float
        Threshold above which an agent is counted as experiencing suicidal thoughts.
    signal_time : int
        Number of initial timesteps to ignore.

    Returns
    -------
    float
        Maximum proportion of agents above the threshold at any timestep
        after warmup.
    """
    arr = data_dict["suicidal_thought"]  # shape (num_steps, num_agents)

    # Only consider timesteps after warmup
    arr_after = arr[signal_time:]

    # Boolean array of agents above threshold
    above_threshold = arr_after > T_threshold

    # Fraction of agents above threshold at each timestep
    proportions = above_threshold.mean(axis=1)

    # Maximum fraction over time
    return proportions.max()


def run_multiple_werther(proportion_news=0.05, filename="werther_peaks_low.npz"):
    
    warmup = 10
    sim_length = 35
    num_agents = 5000
    num_sims = 100
    try:
        data = np.load(f"src/output/{filename}")
        low_deg_peaks = data["low_deg_peaks"]
        hub_peaks = data["hub_peaks"]
        start = int(data["completed"])
    except FileNotFoundError:
        low_deg_peaks = np.zeros(num_sims, dtype=np.float32)
        hub_peaks = np.zeros(num_sims, dtype=np.float32)
        start = 0

    print(f"starting at {start}")
    save_every = 1

    for i in range(start, num_sims):
        low_deg, hub = run_werther_experiment_single(
            proportion_news=proportion_news,
            warmup=warmup,
            sim_length=sim_length,
            num_agents=num_agents,
            seed=i+2
        )

        low_deg_peaks[i] = max_suicidal_thought_proportion(low_deg)
        hub_peaks[i] = max_suicidal_thought_proportion(hub)

        # Save every 5 simulations (and after the last one)
        if (i + 1) % save_every == 0 or (i + 1) == num_sims:
            np.savez(
                "src/output/werther_peaks_low.npz",
                low_deg_peaks=low_deg_peaks,
                hub_peaks=hub_peaks,
                completed=i + 1
            )
            print(f"Saved after {i+1} simulations")


def plot_peak_T_dists():
    data = np.load("werther_peaks.npz")

    low_deg_peaks = data["low_deg_peaks"]
    hub_peaks = data["hub_peaks"]


def means_tests(num_agents, sim_length):

    auc_A_soc, auc_A_anti, auc_T_soc, auc_T_anti = load_aucs(num_agents=num_agents, sim_length=sim_length)
    perform_t_test(auc_A_soc, "AUC(A) Social Agent", auc_A_anti, "AUC(A) Anti-Social Agent", mann_whitney=False, ttest=True)
    perform_t_test(auc_T_soc, "AUC(T) Social Agent", auc_T_anti, "AUC(T) Anti-Social Agent", mann_whitney=True, ttest=False)


if __name__ == "__main__":
    # Conditions:
    # 4500 agents after node removal
    # 5 day warmup
    # 25 day run period (days 5 to 25)
    # default weight only: 0.3
    # weighted network dist: [w=0.99-1, p=0.1], [w=0.4-0.6, p=0.23], [w=0.05-0.1, p=0.66]

    # warmup = 10
    # sim_length = 35
    # num_agents = 5000

    # Batch writing slows down performance due to writing to file, bigger
    # batch size = bigger pauses every batch_size steps as flushes get queued.
    # run_werther_models(batch_size=5000, seed=42, batch_save=True,
    #              warmup=warmup, sim_length=sim_length, num_agents=num_agents)

    # num_agents = 4500
    # warmup = 10
    # sim_length = 35
    # plot_werther_runs(warmup=warmup, num_agents=num_agents, sim_length=sim_length)

    # num_agents = 1500
    # plot_distributions(warmup=warmup, num_agents=num_agents, sim_length=sim_length)
    # means_tests(num_agents=num_agents, sim_length=sim_length)
    run_multiple_werther(proportion_news=0.05)