import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.MemoryModel import MemoryModel
from experiments.util import run_basic_simulation, export_results, plot_single_agent_dynamics, generate_stress_signal,\
      plot_single_dynamics_in_comparison, clean_zero_entries, load_simulation_data, auc_per_agent, plot_auc_distribution,\
      perform_t_test, plot_comparison
from Constants import MINUTE_LENGTH
SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4
from matplotlib import rcParams
default_colors = rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib.lines import Line2D


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
            'internal_strat', 'total_time', 'labels', 'suicide_history'
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
            "baseline": num_agents,
        }
    }
    baseline_model = MemoryModel(
        num_agents=num_agents,
        sim_length=sim_length,
        warmup=warmup,
        verbose=True,
        parameters=params,
        seed=seed
        )
    generate_stress_signal(baseline_model)
    


def run_and_plot(run_new=True, variable_stress=True, batch_save=False):
    # 1) Define models for different agent types
    seed = 42
    warmup = 10
    num_agents = 1000
    sim_length = 50
    name_attachment = "_var_stress" if variable_stress else ""
    
    # 2) Run the models to gather data
    if run_new:
        params = {
            "agent_types": {
                "baseline": num_agents,
            },
            "read_stress": not variable_stress,
        }
        # Define parameters
        baseline_model = MemoryModel(
            num_agents=num_agents,
            sim_length=sim_length,
            warmup=warmup,
            verbose=True,
            parameters=params,
            seed=seed
            )
        params = {
            "agent_types": {
                "memory": num_agents,
            },
            "read_stress": not variable_stress,
        }
        memory_model = MemoryModel(
            num_agents=num_agents,
            sim_length=sim_length,
            warmup=warmup,
            verbose=True,
            parameters=params,
            seed=seed
            )
        run_basic_simulation(baseline_model, batch_save=batch_save, run_name="mem_base" + name_attachment, batch_size=5000)
        # baseline_data = export_results(baseline_model, filename=None)
        run_basic_simulation(memory_model, batch_save=batch_save, run_name="mem_mem" + name_attachment, batch_size=5000)
        # memory_data = export_results(memory_model, filename=None)
    else:
        baseline_data = load_simulation_data("mem_base" + name_attachment, vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
        memory_data = load_simulation_data("mem_mem" + name_attachment, vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    # plot_single_agent_dynamics(baseline_data, filename=None, agent_type='Baseline', warmup=warmup, img_name="baseline_dynamics")
    plot_comparison(
        baseline_data,
        memory_data,
        title_1="Baseline Agent",
        title_2="Memory Agent",
        fig_name="memory_v_baseline",
        vars_2=[
            'stress', 'aversive_internal_state', 'urge_to_escape',
            'suicidal_thought', 'escape_behavior', 'external_strat',
            'internal_strat', 'suicide_history',
        ],
        warmup=warmup,
        sim_length=sim_length
        )
    
    # # 4) Plot distributions of AUC(A) and AUC(T)
    if variable_stress:
        auc_A_base = auc_per_agent(baseline_data['aversive_internal_state'])
        auc_A_memory = auc_per_agent(memory_data['aversive_internal_state'])
        auc_T_base = auc_per_agent(baseline_data['suicidal_thought'])
        auc_T_memory = auc_per_agent(memory_data['suicidal_thought'])

        plot_auc_distribution(
            datasets=[auc_A_base, auc_A_memory],
            labels=["Baseline", "Memory"],
            title="AUC(A) of Baseline vs Memory",
            xlabel="AUC(A)"
            )
        plot_auc_distribution(
            datasets=[auc_T_base, auc_T_memory],
            labels=["Baseline", "Memory"],
            title="AUC(T) of Baseline vs Memory",
            xlabel="AUC(T)"
            )

        # 5) Run t-tests
        perform_t_test(auc_A_base, "AUC(A) Baseline Agent", auc_A_memory, "AUC(A) Memory Agent", mann_whitney=False, ttest=True)
        perform_t_test(auc_T_base, "AUC(T) Baseline Agent", auc_T_memory, "AUC(T) Memory Agent", mann_whitney=True, ttest=False)


if __name__ == "__main__":
    run_and_plot(run_new=False, variable_stress=False, batch_save=False)
    # generate_baseline_stress()