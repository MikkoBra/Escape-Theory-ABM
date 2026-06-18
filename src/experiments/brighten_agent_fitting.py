import numpy as np
import os
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


def run_brighten_model(batch_size=5000, batch_save=True,
                 seed=None, warmup=10, num_agents=1000, sim_length=50):
    # params={
    #     "agent_types": {
    #         "brighten": 5000,
    #     },
    #     "read_stress": False,
    #     "randomize": True,
    #     "only_sleep": True,
    #     "network": "hk",
    #     "m": 16,
    #     'cluster_prob': 0.6,
    #     "initial_attractiveness": 5,
    #     "node_removal_rate": 0.1,
    #     "edge_removal_prob": 0.5,
    #     "hub_count": 2,
    #     "hub_degree": 250,
    #     "social_events": True,
    #     "weighted_network": True,
    # }
    # # Define parameters
    # brighten_agent = EventModel(
    #     num_agents=num_agents,
    #     sim_length=sim_length,
    #     warmup=warmup,
    #     verbose=True,
    #     parameters=params,
    #     seed=seed
    #     )
    params={
        "agent_types": {
            "uninformed": 5000,
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
    uninformed_agent = EventModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    # run_basic_simulation(brighten_agent, batch_save=batch_save, run_name="brighten_2", batch_size=batch_size)
    run_basic_simulation(uninformed_agent, batch_save=batch_save, run_name="uninformed", batch_size=batch_size)


def plot_brighten_runs(warmup=10, num_agents=1000, sim_length=50):
    # brighten_data = load_simulation_data("brighten", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    fitted_data = load_simulation_data("brighten_2", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    uninformed_data = load_simulation_data("uninformed", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # antisocial_data = load_simulation_data("no_events", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    # plot_single_agent_dynamics(brighten_data, agent_type='Data', warmup=warmup, img_name="brighten_new",
    #                            variables=[
    #                                 'stress', 'aversive_internal_state', 'urge_to_escape',
    #                                 'suicidal_thought', 'escape_behavior', 'external_strat',
    #                                 'internal_strat', 'burdensomeness'
    #                             ])

    # plot_comparison(
    #         brighten_data,
    #         fitted_data,
    #         title_1="Data Agent",
    #         title_2="Data Agent (Fitted)",
    #         fig_name="brighten_v_fitted",
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
    #         sim_length=sim_length,
    #         secondary_var="escape_behavior"
    #         )


def load_aucs(num_agents=1000, sim_length=50):
    fitted_data = load_simulation_data("brighten_2", vars_list=SCHEDULE_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    uninformed_data = load_simulation_data("uninformed", vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    auc_A_fit = auc_per_agent(fitted_data['aversive_internal_state'])
    auc_A_unfit = auc_per_agent(uninformed_data['aversive_internal_state'])
    auc_T_fit = auc_per_agent(fitted_data['suicidal_thought'])
    auc_T_unfit = auc_per_agent(uninformed_data['suicidal_thought'])
    return auc_A_fit, auc_A_unfit, auc_T_fit, auc_T_unfit


def plot_distributions(warmup=10, num_agents=1000, sim_length=50):

    auc_A_fit, auc_A_unfit, auc_T_fit, auc_T_unfit = load_aucs(num_agents=num_agents, sim_length=sim_length)
    plot_auc_distribution(
        datasets=[auc_A_fit, auc_A_unfit],
        labels=["Data (Fitted)", "Uninformed"],
        title="AUC(A) of Data (Fitted) vs Uninformed",
        xlabel="AUC(A)"
        )
    plot_auc_distribution(
        datasets=[auc_T_fit, auc_T_unfit],
        labels=["Data (Fitted)", "Uninformed"],
        title="AUC(T) of Data (Fitted) vs Uninformed",
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

    warmup = 5
    sim_length = 30
    num_agents = 5000

    # Batch writing slows down performance due to writing to file, bigger
    # batch size = bigger pauses every batch_size steps as flushes get queued.
    run_brighten_model(batch_size=5000, seed=42,
                 warmup=warmup, sim_length=sim_length, num_agents=num_agents)

    # num_agents = 4500
    # warmup = 5
    # sim_length = 30
    # plot_brighten_runs(warmup=warmup, num_agents=num_agents, sim_length=sim_length)

    num_agents = 1500
    plot_distributions(warmup=warmup, num_agents=num_agents, sim_length=sim_length)
    means_tests(num_agents=num_agents, sim_length=sim_length)