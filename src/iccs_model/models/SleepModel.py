import mesa
from iccs_model.agents.AgentFactory import AgentFactory
from iccs_model.dynamics.state_registry import register_all_states
import numpy as np
import random


class SleepModel(mesa.Model):
    """
    Agent-based model of suicidality in a small world network.
    """
    def __init__(self, dt, n=2, seed=None, parameters={}, verbose=False, collect_all=False, warmup=20):
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.verbose = verbose
        self.warmup = warmup
        self.dt = dt
        self.num_agents = n
        self.time = 0
        if collect_all:
            self.datacollector = mesa.DataCollector(
                agent_reporters={
                    "Stress": "stress",
                    "Aversive Internal State": "aversive_internal_state",
                    "Urge to Escape": "urge_to_escape",
                    "Suicidal Thought": "suicidal_thought",
                    "Escape Behavior": "escape_behavior",
                    "External-Focused Change": "external_strat",
                    "Internal-Focused Change": "internal_strat",
                    "Time": "total_time",
                    "State": lambda agent: agent.state_manager.state.to_string(),
                }
            )
        else:
            self.datacollector = mesa.DataCollector(
                agent_reporters={
                    "Suicidal Thought": "suicidal_thought",
                    "Aversion": "aversive_internal_state",
                    "Time": "total_time",
                }
            )
        register_all_states()
        
        num_sleep_agents = parameters.get("n_sleep", 1)
        num_bad_sleep_agents = parameters.get("n_sleep", 1)
        num_baseline_agents = parameters.get("n_baseline", 1)
        AgentFactory.create_agents(type="sleep", model=self, n=num_sleep_agents, stress_gen=True, default_params=parameters)
        AgentFactory.create_agents(type="bad_sleep", model=self, n=num_bad_sleep_agents, stress_gen=True, default_params=parameters)
        AgentFactory.create_agents(type="baseline", model=self, n=num_baseline_agents, stress_gen=True, default_params=parameters)

    def step(self):
        """
        Performs one timestep of the model.
        """
        if self.time >= self.warmup:
            self.datacollector.collect(self)
        self.agents.do(lambda agent: agent.update_agent(self.dt))
        self.time += self.dt
