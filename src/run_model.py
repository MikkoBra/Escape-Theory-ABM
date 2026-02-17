from model.SuicideModel import SuicideModel
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import numpy as np
from tqdm import trange
from pathlib import Path
import pandas as pd
import json

np.random.seed(43)


def plot_vars(df, agent_id=1):
    """
    Function for plotting the model variables over the simulated time
    """
    df = df.reset_index()
    df = df[df["AgentID"] == agent_id]
    exclude = {"Step", "AgentID", "Time", "State"}
    y_cols = [c for c in df.columns if c not in exclude]
    df = df.sort_values("Time")

    plt.figure(figsize=(16, 6))
    for col in y_cols:
        plt.plot(df["Time"], df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.title(f"Agent {agent_id} — State Variables Over Time")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_sympathy_network():
    T = 0
    N_agents = 15
    # Timestep size
    dt = 1/(24*60)
    model = SuicideModel(dt=dt, n=N_agents)
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
        plot_vars(agent_df)


def run_good_friends_network():
    T = 0
    N_agents = 50
    # Timestep size
    dt = 1/(24*60)
    model = SuicideModel(dt=dt, n=N_agents)
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
        plot_vars(agent_df)


def run_social_network():
    T = 0
    N_agents = 150
    # Timestep size
    dt = 1/(24*60)
    model = SuicideModel(dt=dt, n=N_agents)
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
        plot_vars(agent_df)


def run_custom():
    T = 0
    N_agents = int(input("Enter number of agents\n> "))
    # Timestep size
    dt = 1/(24*60)
    model = SuicideModel(dt=dt, n=N_agents)
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
        plot_vars(agent_df)


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

