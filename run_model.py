from model.SuicideModel import SuicideModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":
    model = SuicideModel(5)
    # Timestep size
    dt = 0.02
    # Days to model
    T = 14
    N = int(T/dt)
    t = np.linspace(0, T, N+1)
    for i in range(1, N+1):
        model.step(dt)
    
    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_to_observe = 1
    agent_df = agent_df.xs(agent_to_observe, level="AgentID")
    g = sns.lineplot(data=agent_df, x="Days", y="Stress", label="S")
    g = sns.lineplot(data=agent_df, x="Days", y="Aversive Internal State", label="A")
    g = sns.lineplot(data=agent_df, x="Days", y="Urge to Escape", label="U")
    g = sns.lineplot(data=agent_df, x="Days", y="Suicidal Thought", label="T")
    g = sns.lineplot(data=agent_df, x="Days", y="Escape Behavior", label="X")
    g = sns.lineplot(data=agent_df, x="Days", y="External-Focused Change", label="E")
    g = sns.lineplot(data=agent_df, x="Days", y="Internal-Focused Change", label="I")
    g.set(title=f"State of agent {agent_to_observe} over time")
    g.set_ylim(0, 1)
    plt.show()
