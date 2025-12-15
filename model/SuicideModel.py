import mesa
from model.agents.StandardAgent import StandardAgent


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
            agent_reporters={"Stress": "stress",
                             "Aversive Internal State": "aversive_internal_state",
                             "Urge to Escape": "urge_to_escape",
                             "Suicidal Thought": "suicidal_thought",
                             "Escape Behavior": "escape_behavior",
                             "External-Focused Change": "external_strat",
                             "Internal-Focused Change": "internal_strat",
                             "Days": "total_time"}
        )
        StandardAgent.create_agents(model=self, n=n)
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
