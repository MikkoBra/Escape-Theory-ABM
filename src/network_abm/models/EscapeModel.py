import mesa
import random
import numpy as np
from network_abm.agents.EscapeAgentUpdater import EscapeAgentUpdater
from network_abm.agents.agent_types.Agent import Agent


class EscapeModel(mesa.Model):
    """
    Mesa-based ABM of social contagion of suicidality using the escape
    theorem of suicide.
    """
    def __init__(self, dt, end_time=30, n_agents=15, seed=None, debug=False):
        """
        Initializes random seed for the simulation, time data, parameter
        arrays, data collector for analysis, and agents.
        """
        # Random seed
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.debug = debug
        
        # Time data: timestep size and simulation runtime
        self.dt = dt
        self.time = dt
        self.end_time = end_time
        self.end_steps = int(end_time/dt)

        # Agent parameters
        self.S = np.zeros(n_agents, dtype=np.float64)
        self.alive_agents = np.ones(n_agents, dtype=np.int32)

        # Datacollector for plotting/analysis
        self.datacollector = {
            "Stress": np.zeros((self.end_steps + 1, n_agents)),
        }

        # Agents and agent updater
        Agent.create_agents(model=self, n=n_agents)
        self.agent_updater = EscapeAgentUpdater(dt=dt, agents=self.agents)

    def step(self):
        params = {"S": self.S}
        self.agent_updater.update_all(
            params=params,
            alive_agents=self.alive_agents,
            network=None
        )
        
        self.datacollector["Stress"][self.steps] = self.S.copy()
        
        self.time  += self.dt

        if self.debug:
            print(f"[step={self.steps}] S={self.S}", flush=True)
