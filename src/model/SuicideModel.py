import mesa
from model.agents.StandardAgent import StandardAgent
from model.agents.VolatileAgent import VolatileAgent
from model.agents.PopularAgent import PopularAgent
from model.agents.BulliedAgent import BulliedAgent
from model.system_updates.state_registry import register_all_states
import numpy as np


class SuicideModel(mesa.Model):
    """
    Agent-based model of suicidality in a small community.
    """

    def __init__(self, n=10, seed=None):
        """
        Initializes the model with a number of agents.
        """
        super().__init__(seed=seed)
        self.num_agents = n
        self.time = 0
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Type": "type",
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
        register_all_states()

        type_probs = [0.5, 0.1, 0.2, 0.2]
        counts = np.random.multinomial(n, type_probs)
        StandardAgent.create_agents(model=self, n=counts[0])
        VolatileAgent.create_agents(model=self, n=counts[1])
        PopularAgent.create_agents(model=self, n=counts[2])
        BulliedAgent.create_agents(model=self, n=counts[3])
        for agent in self.agents:
            agent.set_friends()
            agent.set_bullies()
    

    def step(self, dt):
        """
        Performs one timestep of the model.
        """
        self.datacollector.collect(self)
        self.agents.do(lambda agent: agent.update_agent(dt))
        self.time += dt
