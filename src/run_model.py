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
    secondary_var="Escape Behavior"
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
        "three_panel_agent_comparison.svg",
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
            title2="Dynamics of the Baseline Model"
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


if __name__=="__main__":
    sim = input("> What kind of network? (sympathy=s, good friends=g, typical=t, custom=c)\n")
    if sim == "s":
        run_sympathy_network()
    elif sim == "g":
        run_good_friends_network()
    elif sim == "t":
        run_social_network()
    else:
        run_custom()

