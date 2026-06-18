import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.ScheduledModel import ScheduledModel
from experiments.util import run_basic_simulation, export_results, plot_single_agent_dynamics, generate_stress_signal,\
      plot_single_dynamics_in_comparison, clean_zero_entries, load_simulation_data, auc_per_agent, plot_auc_distribution,\
      perform_t_test, batched
from Constants import Constants
MINUTE_LENGTH = Constants.MINUTE_LENGTH
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
            'internal_strat', 'total_time', 'labels', 'state',
        ]
MEMORY_VARS = [
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'total_time', 'labels',
        ]


def generate_baseline_stress():
    seed = 42
    warmup = 0
    num_agents = 1
    sim_length = 200
    params = {
        "agent_types": {
            "default": num_agents,
        }
    }
    baseline_model = ScheduledModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    generate_stress_signal(baseline_model)


def run_single_aucs_model(simulation_id):

    sim_length = 50

    model = ScheduledModel(num_agents=3, sim_length=sim_length, dt=MINUTE_LENGTH, seed=simulation_id,
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

        model = ScheduledModel(
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


def run_schedule_models(batch_size=5000, batch_save=True,
                 seed=None, warmup=10, num_agents=1000, sim_length=50):
    params = {
        "agent_types": {
            "default": num_agents,
        },
        "only_sleep": False,
    }
    # Define parameters
    schedule_model = ScheduledModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    # params = {
    #     "agent_types": {
    #         "weekend": num_agents,
    #     },
    #     "only_sleep": False,
    # }
    # # Define parameters
    # weekend_model = ScheduledModel(
    #     num_agents=num_agents,
    #     sim_length=sim_length,
    #     warmup=warmup,
    #     verbose=True,
    #     parameters=params,
    #     seed=seed
    #     )
    run_basic_simulation(schedule_model, batch_save=batch_save, run_name="schedule_4500", batch_size=batch_size)
    # run_basic_simulation(weekend_model, batch_save=batch_save, run_name="wkd_yes", batch_size=batch_size)


def plot_schedule_runs(warmup=10, num_agents=1000, sim_length=50):
    weekend_data = load_simulation_data("wkd_yes", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # schedule_data = load_simulation_data("schedule", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    plot_single_agent_dynamics(weekend_data, filename=None, agent_type='Weekend', warmup=warmup, img_name="weekend_dynamics_short")
    # plot_single_agent_dynamics(schedule_data, filename=None, agent_type='Schedule', warmup=warmup, img_name="schedule_dynamics_short")

    # plot_comparison(
    #         gs_data,
    #         bs_data,
    #         title_1="Good Sleep Agent",
    #         title_2="Bad Sleep Agent",
    #         fig_name="good_v_bad_sleep",
    #         vars_2=[
    #             'stress', 'aversive_internal_state', 'urge_to_escape',
    #             'suicidal_thought', 'escape_behavior', 'external_strat',
    #             'internal_strat',
    #         ],
    #         warmup=warmup,
    #         sim_length=sim_length
    #         )


def run_batched_distribution_analysis():
    run_many(sim_length=50, num_sims=1000, batch_size=10, starting_seed=0)


def load_aucs(num_agents=1000, sim_length=50):
    weekend_data = load_simulation_data("wkd_yes", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    schedule_data = load_simulation_data("schedule", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    baseline_data  = load_simulation_data("mem_base_var_stress", vars_list=MEMORY_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    auc_A_base = auc_per_agent(baseline_data['aversive_internal_state'])
    auc_A_wkd = auc_per_agent(weekend_data['aversive_internal_state'])
    auc_A_sched = auc_per_agent(schedule_data['aversive_internal_state'])
    auc_T_base = auc_per_agent(baseline_data['suicidal_thought'])
    auc_T_wkd = auc_per_agent(weekend_data['suicidal_thought'])
    auc_T_sched = auc_per_agent(schedule_data['suicidal_thought'])
    return auc_A_base, auc_A_wkd, auc_A_sched, auc_T_base, auc_T_wkd, auc_T_sched


def plot_distributions(warmup=10, num_agents=1000, sim_length=50):

    auc_A_base, auc_A_wkd, auc_A_sched, auc_T_base, auc_T_wkd, auc_T_sched = load_aucs(num_agents=num_agents, sim_length=sim_length)
    plot_auc_distribution(
        datasets=[auc_A_base, auc_A_sched, auc_A_wkd],
        labels=["Baseline", "Schedule", "Weekend"],
        title="AUC(A) of Baseline vs Schedule vs Weekend",
        xlabel="AUC(A)"
        )
    plot_auc_distribution(
        datasets=[auc_T_base, auc_T_sched, auc_T_wkd],
        labels=["Baseline", "Schedule", "Weekend"],
        title="AUC(T) of Baseline vs Schedule vs Weekend",
        xlabel="AUC(T)"
        )


def means_tests(num_agents, sim_length):

    auc_A_base, auc_A_wkd, auc_A_sched, auc_T_base, auc_T_wkd, auc_T_sched = load_aucs(num_agents=num_agents, sim_length=sim_length)
    perform_t_test(auc_A_base, "AUC(A) Baseline Agent", auc_A_sched, "AUC(A) Schedule Agent", mann_whitney=False, ttest=True)
    perform_t_test(auc_A_base, "AUC(A) Baseline Agent", auc_A_wkd, "AUC(A) Weekend Agent", mann_whitney=False, ttest=True)
    perform_t_test(auc_A_sched, "AUC(A) Schedule Agent", auc_A_wkd, "AUC(A) Weekend Agent", mann_whitney=False, ttest=True)
    perform_t_test(auc_T_base, "AUC(T) Default Agent", auc_T_sched, "AUC(T) Schedule Agent", mann_whitney=True, ttest=False)
    perform_t_test(auc_T_base, "AUC(T) Default Agent", auc_T_wkd, "AUC(T) Weekend Agent", mann_whitney=True, ttest=False)
    perform_t_test(auc_T_sched, "AUC(T) Schedule Agent", auc_T_wkd, "AUC(T) Weekend Agent", mann_whitney=True, ttest=False)


if __name__ == "__main__":
    warmup = 10
    sim_length = 50
    num_agents = 4500

    # Batch writing slows down performance due to writing to file, bigger
    # batch size = bigger pauses every batch_size steps as flushes get queued.
    run_schedule_models(batch_save=True, batch_size=5000,
                 warmup=warmup, sim_length=sim_length, num_agents=num_agents)

    # sim_length = 20
    # plot_schedule_runs(warmup=warmup, num_agents=num_agents, sim_length=sim_length)

    # plot_distributions(warmup=warmup, num_agents=num_agents, sim_length=sim_length)
    # means_tests(num_agents=num_agents, sim_length=sim_length)

    # Run this to generate a stress signal that can be used instead
    # of a randomly generated signal (stored in output/stress_signal.npy)
    # generate_baseline_stress()