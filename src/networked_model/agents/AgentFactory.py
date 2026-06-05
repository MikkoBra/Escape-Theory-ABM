from networked_model.agents.DefaultAgent import DefaultAgent
from networked_model.agents.BaselineAgent import BaselineAgent
from networked_model.agents.WeekendAgent import WeekendAgent


TYPE_DICT = {
    "default": DefaultAgent,
    "baseline": BaselineAgent,
    'weekend': WeekendAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def initialize_coefficients(self, agent_type):
        return TYPE_DICT[agent_type]().initialize_coefficients()
        
    def initialize_characteristics(self, agent_type):
        return TYPE_DICT[agent_type]().initialize_characteristics()
