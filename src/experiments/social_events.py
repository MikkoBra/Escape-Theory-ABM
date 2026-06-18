import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.EventModel import EventModel
from experiments.util import run_basic_simulation, export_results, plot_single_agent_dynamics, generate_stress_signal,\
      plot_single_dynamics_in_comparison, clean_zero_entries, load_simulation_data, auc_per_agent, plot_auc_distribution,\
      perform_t_test, batched, plot_comparison
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

    model = EventModel(num_agents=3, sim_length=sim_length, dt=MINUTE_LENGTH, seed=simulation_id,
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

        model = EventModel(
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


def run_social_models(batch_size=5000, batch_save=True,
                 seed=None, warmup=10, num_agents=1000, sim_length=50):
    params={
        "agent_types": {
            "weekend": 5000,
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
    }
    # Define parameters
    social_agent = EventModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    params={
        "agent_types": {
            "weekend": 5000,
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
        "social_events": False,
        "weighted_network": True,
    }
    # Define parameters
    antisocial_agent = EventModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    run_basic_simulation(social_agent, batch_save=batch_save, run_name="events", batch_size=batch_size)
    run_basic_simulation(antisocial_agent, batch_save=batch_save, run_name="no_events", batch_size=batch_size)


def plot_event_runs(warmup=10, num_agents=1000, sim_length=50):
    event_data = load_simulation_data("events", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    antisocial_data = load_simulation_data("no_events", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    # plot_single_agent_dynamics(event_data, agent_type='Social', warmup=warmup, img_name="event_short",
    #                            variables=[
    #                                 'stress', 'aversive_internal_state', 'urge_to_escape',
    #                                 'suicidal_thought', 'escape_behavior', 'external_strat',
    #                                 'internal_strat', 'burdensomeness'
    #                             ])

    plot_comparison(
            event_data,
            antisocial_data,
            title_1="Social Agent",
            title_2="Anti-Social\nAgent",
            fig_name="social_v_antisocial",
            vars_1=[
                'stress', 'aversive_internal_state', 'urge_to_escape',
                'suicidal_thought', 'escape_behavior', 'external_strat',
                'internal_strat', 'burdensomeness',
            ],
            vars_2=[
                'stress', 'aversive_internal_state', 'urge_to_escape',
                'suicidal_thought', 'escape_behavior', 'external_strat',
                'internal_strat', 'burdensomeness',
            ],
            warmup=warmup,
            sim_length=sim_length
            )


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

    # warmup = 5
    # sim_length = 30
    # num_agents = 5000

    # # Batch writing slows down performance due to writing to file, bigger
    # # batch size = bigger pauses every batch_size steps as flushes get queued.
    # run_social_models(batch_size=5000, seed=42,
    #              warmup=warmup, sim_length=sim_length, num_agents=num_agents)

    num_agents = 4500
    warmup = 5
    sim_length = 30
    # plot_event_runs(warmup=warmup, num_agents=num_agents, sim_length=sim_length)

    num_agents = 1500
    plot_distributions(warmup=warmup, num_agents=num_agents, sim_length=sim_length)
    means_tests(num_agents=num_agents, sim_length=sim_length)