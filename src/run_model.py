from model.SuicideModel import SuicideModel
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import numpy as np
from tqdm import trange
from pathlib import Path
import pandas as pd


def plot_combined(agent_df, agent_id, label=None):
    """
    Plot continuous parameters above a categorical state timeline.
    X-axis shows time of day (00:00–24:00) repeating for each day.
    """
    # --- Prepare state data ---
    state_df = agent_df[["Time", "State"]].reset_index(drop=True)
    state_df = state_df[state_df["State"].ne(state_df["State"].shift())].reset_index(drop=True)
    state_df["End"] = state_df["Time"].shift(-1)
    state_df.loc[state_df.index[-1], "End"] = agent_df["Time"].iloc[-1]

    state_palette = {
        "sleep": "navy",
        "morning": "orange",
        "commute": "green",
        "work": "brown",
        "home": "purple",
    }

    # --- Create figure with two stacked subplots ---
    fig, (ax_params, ax_state) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # --- Plot continuous parameters (absolute days) ---
    sns.lineplot(data=agent_df, x="Time", y="Stress", ax=ax_params, label="S")
    sns.lineplot(data=agent_df, x="Time", y="Aversive Internal State", ax=ax_params, label="A")
    sns.lineplot(data=agent_df, x="Time", y="Urge to Escape", ax=ax_params, label="U")
    sns.lineplot(data=agent_df, x="Time", y="Suicidal Thought", ax=ax_params, label="T")
    sns.lineplot(data=agent_df, x="Time", y="Escape Behavior", ax=ax_params, label="X")
    sns.lineplot(data=agent_df, x="Time", y="External-Focused Change", ax=ax_params, label="E")
    sns.lineplot(data=agent_df, x="Time", y="Internal-Focused Change", ax=ax_params, label="I")

    ax_params.set_ylim(0, 1)
    ax_params.set_ylabel("Value")
    ax_params.set_title(f"{label} Agent {agent_id} - Parameters and State Timeline")
    ax_params.legend(loc="upper right")

    # --- Plot state timeline ---
    y_center = 0.5
    bar_height = 0.2

    for state, color in state_palette.items():
        subset = state_df[state_df["State"] == state]
        if subset.empty:
            continue
        bars = [(row.Time, row.End - row.Time) for row in subset.itertuples()]
        ax_state.broken_barh(bars, (y_center - bar_height / 2, bar_height), facecolors=color)

    # --- Format X-axis as time-of-day ---
    def time_of_day_formatter(x, pos):
        hour = int((x % 1) * 24)
        return f"{hour:02d}:00"

    ax_params.xaxis.set_major_formatter(mticker.FuncFormatter(time_of_day_formatter))
    ax_state.xaxis.set_major_formatter(mticker.FuncFormatter(time_of_day_formatter))
    ax_state.set_yticks([])
    ax_state.set_xlabel("Time of Day")
    ax_state.set_xlim(agent_df["Time"].min(), agent_df["Time"].max())

    # --- Optional: vertical lines to mark day boundaries ---
    max_day = int(agent_df["Time"].max()) + 1
    for d in range(max_day):
        ax_state.axvline(x=d, color="gray", alpha=0.3, linestyle="--")

    # --- Legend for states ---
    legend_handles = [Patch(facecolor=color, label=state) for state, color in state_palette.items()]
    ax_state.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=len(state_palette), frameon=False)

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    run = input("Run simulation? (y/n)\n> ")
    if run == "y":
        N_agents = int(input("Enter number of agents\n> "))
        model = SuicideModel(N_agents)
        # Timestep size
        dt = 1/(24*60)
        # Days to model
        T = int(input("Enter number of days to model\n> "))
        N_steps = int(T/dt)
        t = np.linspace(0, T, N_steps+1)
        for _ in trange(1, N_steps + 1, desc="Running simulation"):
            model.step(dt)

        # Ensure folder exists
        data_folder = Path("data")
        data_folder.mkdir(parents=True, exist_ok=True)

        # Save full agent DataFrame
        agent_df = model.datacollector.get_agent_vars_dataframe()
        csv_path = data_folder / f"{T}_days_{N_agents}_agents.csv"
        agent_df.to_csv(csv_path, index=True)
    else:
        agent_df = pd.read_csv("data/14_days_100_agents.csv")
    plot = input("Generate plot? (y/n)\n> ")
    if plot == "y":
        if not isinstance(agent_df.index, pd.MultiIndex):
            agent_df.set_index(["AgentID", "Step"], inplace=True)
        agent_to_observe = 1
        single_agent_df = agent_df.xs(agent_to_observe, level="AgentID")
        plot_combined(single_agent_df, agent_to_observe, label="Standard")
        agent_to_observe = 47
        single_agent_df = agent_df.xs(agent_to_observe, level="AgentID")
        plot_combined(single_agent_df, agent_to_observe, label="Volatile")
        agent_to_observe = 59
        single_agent_df = agent_df.xs(agent_to_observe, level="AgentID")
        plot_combined(single_agent_df, agent_to_observe, label="Popular")
        agent_to_observe = 80
        single_agent_df = agent_df.xs(agent_to_observe, level="AgentID")
        plot_combined(single_agent_df, agent_to_observe, label="Bullied")
