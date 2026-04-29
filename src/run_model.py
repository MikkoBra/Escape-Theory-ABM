from iccs_model.sleep_deprivation.models.DefaultModel import DefaultModel
from iccs_model.sleep_deprivation.models.SleepModel import SleepModel
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
from tqdm import trange
from pathlib import Path
import pandas as pd
import json
from matplotlib import rcParams
default_colors = rcParams['axes.prop_cycle'].by_key()['color']

np.random.seed(43)


def _prepare_agent_df(df, agent_id):
    df = df.reset_index()
    df = df[df["AgentID"] == agent_id]
    df = df.sort_values("Time")
    return df


def plot_three_panel_comparison(
    df,
    agent1=1,
    agent2=2,
    title1="Agent 1",
    title2="Agent 2",
    main_var="Suicidal Thought",
    secondary_var="Escape Behavior",
    fig_name="three_panel_comparison"
):
    fontsize = 22

    df1 = _prepare_agent_df(df, agent1)
    df2 = _prepare_agent_df(df, agent2)

    exclude = {"Step", "AgentID", "Time", "State"}
    y_cols = [c for c in df1.columns if c not in exclude]

    # ============================================================
    # COLORBLIND-SAFE OKABE–ITO PALETTE
    # ============================================================
    okabe_ito = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#CC79A7",  # reddish purple
    ]

    # Cycle styles too (helps if >8 parameters)
    linestyles = ["-", "--", "-.", ":"]

    color_map = {
        col: okabe_ito[i % len(okabe_ito)]
        for i, col in enumerate(y_cols)
    }

    style_map = {
        col: linestyles[(i // len(okabe_ito)) % len(linestyles)]
        for i, col in enumerate(y_cols)
    }

    # ============================================================
    # CREATE FIGURE
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=False)

    highlight = {main_var, secondary_var}
    # ============================================================
    # PANEL 1 — Agent 1 (DEFAULT COLORS)
    # ============================================================
    for col in y_cols:
        if col in highlight:
            lw = 2     # thicker for targets
        else:
            lw = 2     # thinner for non-targets

        axes[0].plot(
            df1["Time"],
            df1[col],
            linewidth=lw,
            alpha=1
        )

    axes[0].set_title(title1, fontsize=fontsize, fontweight="bold", pad=12)
    axes[0].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[0].set_ylabel("Value", fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)


    # ============================================================
    # PANEL 2 — Agent 2 (DEFAULT COLORS)
    # ============================================================
    for col in y_cols:
        if col in highlight:
            lw = 2
        else:
            lw = 2

        axes[1].plot(
            df2["Time"],
            df2[col],
            linewidth=lw,
            alpha=1
        )

    axes[1].set_title(title2, fontsize=fontsize, fontweight="bold", pad=12)
    axes[1].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[1].tick_params(labelsize=fontsize)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    # ============================================================
    # PANEL 3 — Comparison
    # ============================================================
    comparison_vars = [main_var, secondary_var]
    default_map = {
        main_var: default_colors[3],
        secondary_var: default_colors[4]
    }
    legend_handles = []

    for var in comparison_vars:

        # ---- Stats ----
        m1, s1 = df1[var].mean(), df1[var].std()
        m2, s2 = df2[var].mean(), df2[var].std()

        # Agent 1 — solid
        l1, = axes[2].plot(
            df1["Time"], df1[var],
            color=default_map[var],
            linestyle="-",
            linewidth=3
        )

        # Agent 2 — dotted
        l2, = axes[2].plot(
            df2["Time"], df2[var],
            color=default_map[var],
            linestyle="--",
            linewidth=3
        )

        legend_handles.extend([
            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="-",
                   label=f" {var} (Extended, μ={m1:.2f}, σ={s1:.2f})"),

            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="--",
                   label=f"{var} (Baseline, μ={m2:.2f}, σ={s2:.2f})")
        ])

    axes[2].set_title("Target Variable Comparison between\nBaseline and Extended Model", fontsize=fontsize,
                  fontweight="bold",
                  pad=15)
    axes[2].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[2].tick_params(labelsize=fontsize)
    axes[2].set_ylim(0, 0.3)
    axes[2].grid(True)

    # ============================================================
    # SHARED LEGEND (ALL PARAMETERS)
    # ============================================================
    bottom_handles = [
        Line2D([0], [0],
                color=default_colors[i % len(default_colors)],
               linestyle="-",
               label=col)
        for i, col in enumerate(y_cols)
    ]

    # ============================================================
    # PRECISE LEGEND PLACEMENT — FINAL
    # ============================================================

    # Get axis positions in figure coordinates
    bbox0 = axes[0].get_position()
    bbox1 = axes[1].get_position()
    bbox2 = axes[2].get_position()

    # Centers
    x_center_left = (bbox0.x0 + bbox1.x1) / 2
    x_center_right = (bbox2.x0 + bbox2.x1) / 2

    # Vertical reference
    y_bottom = min(bbox0.y0, bbox1.y0, bbox2.y0)

    # --- Reduced gap between plots and legends ---
    vertical_offset = -0.15

    # --- Horizontal padding between legends ---
    horizontal_padding = 0.02

    # ------------------------------------------------------------
    # Legend for ALL parameters (under axes 0–1)
    # ------------------------------------------------------------
    fig.legend(
        handles=bottom_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_left - horizontal_padding - 0.02,
                        y_bottom - vertical_offset),
        ncol=min(len(y_cols), 2),
        fontsize=fontsize,
        frameon=False
    )

    # ------------------------------------------------------------
    # Legend for μ / σ (under axis 2)
    # ------------------------------------------------------------
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_right + horizontal_padding,
                        y_bottom - vertical_offset),
        ncol=1,
        fontsize=fontsize,
        frameon=False
    )

    # Ensure space for legends (but not excessive)
    fig.subplots_adjust(bottom=0.32)

    fig.savefig(
        f"{fig_name}.svg",
        format="svg",
        bbox_inches="tight"
    )

    plt.show()


def run_sympathy_network():
    T = 0
    N_agents = 15
    # Timestep size
    dt = 1/(24*60)
    model = DefaultModel(dt=dt, n=N_agents, parameters={}, collect_all=True, verbose=True)
    # Days to model
    T = int(input("Enter number of days to model\n> "))
    N_steps = int(T/dt)
    for _ in trange(1, N_steps + 1, desc="Running simulation"):
        model.step()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    plot = input("Generate plot? (y/n)\n> ")
    if plot == "y":
        if not isinstance(agent_df.index, pd.MultiIndex):
            agent_df.set_index(["AgentID", "Step"], inplace=True)
        plot_three_panel_comparison(
            agent_df,
            agent1=1,
            agent2=16,
            title1="Dynamics of the Extended Model",
            title2="Dynamics of the Baseline Model",
            fig_name="base_v_extended"
        )
        plot_three_panel_comparison(
            agent_df,
            agent1=17,
            agent2=16,
            title1="Dynamics with Suicide Memory",
            title2="Dynamics of the Baseline Model",
            fig_name="base_v_memory"
        )


def run_good_friends_network():
    T = 0
    N_agents = 50
    # Timestep size
    dt = 1/(24*60)
    model = DefaultModel(dt=dt, n=N_agents)
    # Days to model
    T = int(input("Enter number of days to model\n> "))
    N_steps = int(T/dt)
    for _ in trange(1, N_steps + 1, desc="Running simulation"):
        model.step()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    plot = input("Generate plot? (y/n)\n> ")
    if plot == "y":
        if not isinstance(agent_df.index, pd.MultiIndex):
            agent_df.set_index(["AgentID", "Step"], inplace=True)
        plot_three_panel_comparison(
            agent_df,
            agent1=1,
            agent2=16,
            title1="Dynamics of the Extended Model",
            title2="Dynamics of the Baseline Model"
        )


def run_social_network():
    T = 0
    N_agents = 150
    # Timestep size
    dt = 1/(24*60)
    model = DefaultModel(dt=dt, n=N_agents, parameters={"type": "standard"}, stress_gen=True)
    # Days to model
    T = int(input("Enter number of days to model\n> "))
    N_steps = int(T/dt)
    for _ in trange(1, N_steps + 1, desc="Running simulation"):
        model.step()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    plot = input("Generate plot? (y/n)\n> ")
    if plot == "y":
        if not isinstance(agent_df.index, pd.MultiIndex):
            agent_df.set_index(["AgentID", "Step"], inplace=True)
        plot_three_panel_comparison(
            agent_df,
            agent1=1,
            agent2=16,
            title1="Dynamics of the Extended Model",
            title2="Dynamics of the Baseline Model"
        )


def run_custom():
    T = 0
    N_agents = int(input("Enter number of agents\n> "))
    # Timestep size
    dt = 1/(24*60)
    model = DefaultModel(dt=dt, n=N_agents)
    # Days to model
    T = int(input("Enter number of days to model\n> "))
    N_steps = int(T/dt)
    for _ in trange(1, N_steps + 1, desc="Running simulation"):
        model.step()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    plot = input("Generate plot? (y/n)\n> ")
    if plot == "y":
        if not isinstance(agent_df.index, pd.MultiIndex):
            agent_df.set_index(["AgentID", "Step"], inplace=True)
        plot_three_panel_comparison(
            agent_df,
            agent1=1,
            agent2=16,
            title1="Dynamics of the Extended Model",
            title2="Dynamics of the Baseline Model"
        )


def plot_comparison_from_two_sims(
    df1,
    df2,
    agent_id=1,
    title1="Standard Agent",
    title2="Baseline Agent",
    main_var="Suicidal Thought",
    secondary_var="Escape Behavior",
    fig_name="comparison_same_seed"
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
    fontsize = 22
 
    df1_agent = _prepare_agent_df(df1, agent_id)
    df2_agent = _prepare_agent_df(df2, agent_id)
 
    exclude = {"Step", "AgentID", "Time", "State"}
    y_cols = [c for c in df1_agent.columns if c not in exclude]
 
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=False)
 
    highlight = {main_var, secondary_var}
    
    # ============================================================
    # PANEL 1 — First agent type (e.g., Standard)
    # ============================================================
    for col in y_cols:
        lw = 2 if col in highlight else 2
        axes[0].plot(
            df1_agent["Time"],
            df1_agent[col],
            linewidth=lw,
            alpha=1
        )
 
    axes[0].set_title(title1, fontsize=fontsize, fontweight="bold", pad=12)
    axes[0].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[0].set_ylabel("Value", fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
 
    # ============================================================
    # PANEL 2 — Second agent type (e.g., Baseline)
    # ============================================================
    for col in y_cols:
        lw = 2 if col in highlight else 2
        axes[1].plot(
            df2_agent["Time"],
            df2_agent[col],
            linewidth=lw,
            alpha=1
        )
 
    axes[1].set_title(title2, fontsize=fontsize, fontweight="bold", pad=12)
    axes[1].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[1].tick_params(labelsize=fontsize)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
 
    # ============================================================
    # PANEL 3 — Direct Comparison with Stress Verification
    # ============================================================
    comparison_vars = [main_var, secondary_var]
    default_map = {
        main_var: default_colors[3],
        secondary_var: default_colors[4]
    }
    legend_handles = []
 
    # Verify stress trajectories match
    stress_diff = np.abs(df1_agent["Stress"].values - df2_agent["Stress"].values).max()
    if stress_diff > 1e-6:
        print(f"\n⚠ WARNING: Stress trajectories differ by max {stress_diff:.6f}")
        print("This suggests different random seeds were used between simulations.")
        print("Ensure both simulations use the same seed for fair comparison.\n")
    else:
        print(f"\n✓ Stress trajectories match perfectly (max diff: {stress_diff:.10f})")
        print("Fair comparison enabled: both agents experienced identical stress.\n")
 
    for var in comparison_vars:
        # Calculate statistics
        m1, s1 = df1_agent[var].mean(), df1_agent[var].std()
        m2, s2 = df2_agent[var].mean(), df2_agent[var].std()
 
        # First simulation — solid line
        axes[2].plot(
            df1_agent["Time"], df1_agent[var],
            color=default_map[var],
            linestyle="-",
            linewidth=3,
            label=f"{var} ({title1})"
        )
 
        # Second simulation — dashed line
        axes[2].plot(
            df2_agent["Time"], df2_agent[var],
            color=default_map[var],
            linestyle="--",
            linewidth=3,
            label=f"{var} ({title2})"
        )
 
        legend_handles.extend([
            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="-",
                   linewidth=3,
                   label=f"{var} ({title1}, μ={m1:.2f}, σ={s1:.2f})"),
            Line2D([0], [0],
                   color=default_map[var],
                   linestyle="--",
                   linewidth=3,
                   label=f"{var} ({title2}, μ={m2:.2f}, σ={s2:.2f})")
        ])
 
    axes[2].set_title(f"Target Variable Comparison between\n{title2} and {title1}", fontsize=fontsize,
                  fontweight="bold",
                  pad=15)
    axes[2].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[2].tick_params(labelsize=fontsize)
    axes[2].set_ylim(0, 0.3)
    axes[2].grid(True)
 
    # ============================================================
    # LEGENDS
    # ============================================================
    bottom_handles = [
        Line2D([0], [0],
               color=default_colors[i % len(default_colors)],
               linestyle="-",
               linewidth=2,
               label=col)
        for i, col in enumerate(y_cols)
    ]
 
    # Get axis positions
    bbox0 = axes[0].get_position()
    bbox1 = axes[1].get_position()
    bbox2 = axes[2].get_position()
 
    x_center_left = (bbox0.x0 + bbox1.x1) / 2
    x_center_right = (bbox2.x0 + bbox2.x1) / 2
    y_bottom = min(bbox0.y0, bbox1.y0, bbox2.y0)
 
    vertical_offset = -0.15
    horizontal_padding = 0.02
 
    # Legend for all parameters
    fig.legend(
        handles=bottom_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_left - horizontal_padding - 0.02,
                        y_bottom - vertical_offset),
        ncol=min(len(y_cols), 2),
        fontsize=fontsize,
        frameon=False
    )
 
    # Legend for comparison statistics
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_right + horizontal_padding,
                        y_bottom - vertical_offset),
        ncol=1,
        fontsize=fontsize,
        frameon=False
    )
 
    fig.subplots_adjust(bottom=0.32)
 
    fig.savefig(
        f"{fig_name}.svg",
        format="svg",
        bbox_inches="tight"
    )
 
    plt.show()
    print(f"✓ Saved plot to {fig_name}.svg\n")


def run_comparison():
    """
    Run separate simulations with identical seeds to fairly compare agent types.
    This is the RECOMMENDED way to compare different agent dynamics.
    """
    
    T = int(input("Enter number of days to model: "))
    N_agents = 15
    dt = 1/(24*60)
    N_steps = int(T/dt)
    seed = 43  # Fixed seed for reproducibility
    agent_id = 1  # Compare agent 1 from each simulation
    
    # Run simulation 1: Standard agents
    # print("\n" + "="*70)
    # print("Running Simulation 1: STANDARD agents")
    # print("="*70)
    # model1 = DefaultModel(
    #     dt=dt, 
    #     n=N_agents, 
    #     parameters={}, 
    #     collect_all=True, 
    #     verbose=False,
    #     seed=seed,
    #     agent_type="standard"
    # )
    # for _ in trange(1, N_steps + 1, desc="Standard agents"):
    #     model1.step()
    # df_standard = model1.datacollector.get_agent_vars_dataframe()
    
    # Run simulation 2: Baseline agents  
    print("\n" + "="*70)
    print("Running Simulation 2: BASELINE agents")
    print("="*70)
    model2 = DefaultModel(
        dt=dt,
        n=3,
        parameters={},
        collect_all=True,
        verbose=False,
        seed=seed,
        agent_type="baseline"
    )
    for _ in trange(1, N_steps + 1, desc="Baseline agents"):
        model2.step()
    df_baseline = model2.datacollector.get_agent_vars_dataframe()
    
    # Run simulation 3: Suicide history agents
    print("\n" + "="*70)
    print("Running Simulation 3: SUICIDE HISTORY agents")
    print("="*70)
    model3 = DefaultModel(
        dt=dt,
        n=3,
        parameters={},
        collect_all=True,
        verbose=False,
        seed=seed,
        agent_type="suicide_history"
    )
    for _ in trange(1, N_steps + 1, desc="Suicide history agents"):
        model3.step()
    df_suicide_history = model3.datacollector.get_agent_vars_dataframe()
    
    # Generate plots
    plot = input("\nGenerate comparison plots? (y/n): ")
    if plot.lower() == "y":
        print("\nPreparing data...")
        # if not isinstance(df_standard.index, pd.MultiIndex):
        #     df_standard.set_index(["AgentID", "Step"], inplace=True)
        if not isinstance(df_baseline.index, pd.MultiIndex):
            df_baseline.set_index(["AgentID", "Step"], inplace=True)
        if not isinstance(df_suicide_history.index, pd.MultiIndex):
            df_suicide_history.set_index(["AgentID", "Step"], inplace=True)
        
        # print("\n" + "="*70)
        # print("Generating Plot 1: Standard vs Baseline")
        # print("="*70)
        # plot_comparison_from_two_sims(
        #     df_standard,
        #     df_baseline,
        #     agent_id=agent_id,
        #     title1="Standard Agent (Extended Model)",
        #     title2="Baseline Agent",
        #     fig_name="standard_vs_baseline_fair"
        # )
        
        print("="*70)
        print("Generating Plot 2: Suicide History vs Baseline")
        print("="*70)
        plot_comparison_from_two_sims(
            df_suicide_history,
            df_baseline,
            agent_id=2,
            title1="Suicide History",
            title2="Baseline",
            fig_name="suicide_history_vs_baseline_fair"
        )
        
        print("="*70)
        print("✓ All comparisons complete!")
        print("="*70)


if __name__=="__main__":
    sim = input("> What kind of network? (sympathy=s, good friends=g, typical=t, custom=c)\n")
    if sim == "s":
        run_sympathy_network()
    elif sim == "g":
        run_good_friends_network()
    elif sim == "t":
        run_social_network()
    else:
        run_comparison()

