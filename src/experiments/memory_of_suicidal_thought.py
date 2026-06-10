import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.MemoryModel import MemoryModel
from experiments.util import run_basic_simulation, export_results, plot_single_agent_dynamics, generate_stress_signal,\
      plot_single_dynamics_in_comparison, clean_zero_entries, load_simulation_data, auc_per_agent, plot_auc_distribution,\
      perform_t_test
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


def plot_comparison(
    model_1_data,
    model_2_data,
    title_1="Agent 1",
    title_2="Agent 2",
    agent_idx=0,
    main_var="suicidal_thought",
    secondary_var="aversive_internal_state",
    fig_name="memory_v_baseline",
    vars_1=[
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat',
    ],
    vars_2=[
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat', 'suicide_history'
    ],
    warmup=10,
    sim_length=50,
):
    """
    Plot comparison using the same agent ID from two different simulation runs.
    This ensures identical stress trajectories when using the same random seed.
    
    Parameters:
    -----------
    df1 : DataFrame
        Data from first simulation (e.g., StandardAgent type)
    df2 : DataFrame  
        Data from second simulation (e.g., BaselineAgent type)
    agent_id : int
        The agent ID to compare from both simulations (usually 1)
    title1, title2 : str
        Titles for the two agent types
    main_var, secondary_var : str
        Variables to highlight in the comparison
    fig_name : str
        Output filename (without extension)
    """
    fontsize = 20
    lw = 2
 
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), constrained_layout=False)
    
    # ============================================================
    # PANEL 1 — First agent type (e.g., Baseline)
    # ============================================================
    model_1_data = clean_zero_entries(model_1_data)
    plot_single_dynamics_in_comparison(
        ax=axes[0],
        time_data=model_1_data["total_time"][:, agent_idx],
        model_data=model_1_data,
        warmup=warmup,
        agent_idx=agent_idx,
        lw=lw,
        title=title_1,
        fontsize=fontsize,
        vars=vars_1,
        sim_length=sim_length
        )
 
    # ============================================================
    # PANEL 2 — Second agent type (e.g., Memory)
    # ============================================================
    model_2_data = clean_zero_entries(model_2_data)
    plot_single_dynamics_in_comparison(
        ax=axes[1],
        time_data=model_2_data["total_time"][:, agent_idx],
        model_data=model_2_data,
        warmup=warmup,
        agent_idx=agent_idx,
        lw=lw,
        title=title_2,
        fontsize=fontsize,
        vars=vars_2,
        sim_length=sim_length
        )
 
    # ============================================================
    # PANEL 3 — Direct Comparison
    # ============================================================
    comparison_vars = [main_var, secondary_var]
    default_map = {
        main_var: default_colors[3],
        secondary_var: default_colors[4]
    }
    legend_handles = []
 
    for var in comparison_vars:
        # Calculate statistics
        agent_data_1 = model_1_data[var][:, agent_idx]
        m1, s1 = agent_data_1.mean(), agent_data_1.std()
        agent_data_2 = model_2_data[var][:, agent_idx]
        m2, s2 = agent_data_2.mean(), agent_data_2.std()
 
        # First simulation — solid line
        axes[2].plot(
            model_2_data["total_time"][:, agent_idx],
            agent_data_1,
            color=default_map[var],
            linestyle="-",
            linewidth=3,
            label=f"{var_dict[var]} ({title_1})"
        )
 
        # Second simulation — dashed line
        axes[2].plot(
            model_2_data["total_time"][:, agent_idx],
            agent_data_2,
            color=default_map[var],
            linestyle="--",
            linewidth=3,
            label=f"{var_dict[var]} ({title_2})"
        )
 
        legend_handles.extend([
            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="-",
                   linewidth=3,
                   label=f"{var_dict[var]} ({title_1}, μ={m1:.2f}, σ={s1:.2f})"),
            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="--",
                   linewidth=3,
                   label=f"{var_dict[var]} ({title_2}, μ={m2:.2f}, σ={s2:.2f})")
        ])
 
    axes[2].set_title(f"Target Variable Comparison between\n{title_1} and {title_2}", fontsize=fontsize,
                  fontweight="bold",
                  pad=15)
    axes[2].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[2].tick_params(labelsize=fontsize)
    axes[2].set_xlim(warmup, sim_length)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True)
 
    # ============================================================
    # LEGENDS
    # ============================================================
    bottom_handles = [
        Line2D([0], [0],
               color=default_colors[i % len(default_colors)],
               linestyle="-",
               linewidth=2,
               label=var_dict[var])
        # Ensure that vars_2 is the set with more variables
        for i, var in enumerate(vars_2)
    ]
 
    # Get axis positions
    bbox0 = axes[0].get_position()
    bbox1 = axes[1].get_position()
    bbox2 = axes[2].get_position()
 
    x_center_left = (bbox0.x0 + bbox1.x1) / 2
    x_center_right = (bbox2.x0 + bbox2.x1) / 2
    y_bottom = min(bbox0.y0, bbox1.y0, bbox2.y0)
 
    vertical_offset = -0.15
    left_legend_to_left = 0.08
    right_legend_to_right = -0.015
 
    # Legend for all parameters
    fig.legend(
        handles=bottom_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_left - left_legend_to_left - 0.02,
                        y_bottom - vertical_offset),
        ncol=min(len(vars_2), 2),
        fontsize=fontsize,
        frameon=False
    )
 
    # Legend for comparison statistics
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_right + right_legend_to_right,
                        y_bottom - vertical_offset),
        ncol=1,
        fontsize=fontsize,
        frameon=False
    )
 
    fig.subplots_adjust(bottom=0.32)
 
    fig.savefig(
        f"src/output/{fig_name}.svg",
        format="svg",
        bbox_inches="tight"
    )
 
    plt.show()
    print(f"✓ Saved plot to {fig_name}.svg\n")


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
        run_basic_simulation(baseline_model, batch_save=batch_save, run_name="mem_base" + name_attachment)
        baseline_data = export_results(baseline_model, filename=None)
        run_basic_simulation(memory_model, batch_save=batch_save, run_name="mem_mem" + name_attachment)
        memory_data = export_results(memory_model, filename=None)
    else:
        baseline_data = load_simulation_data("mem_base" + name_attachment, vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
        memory_data = load_simulation_data("mem_mem" + name_attachment, vars_list=ALL_TRACKED_VARS, num_steps=int(sim_length/MINUTE_LENGTH), num_agents=num_agents)
    # 3) Plot dynamics
    if not variable_stress:
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
    
    # 4) Plot distributions of AUC(A) and AUC(T)
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
    run_and_plot(run_new=False, variable_stress=True, batch_save=False)
    # generate_baseline_stress()