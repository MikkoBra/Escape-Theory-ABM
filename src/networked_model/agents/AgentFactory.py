from networked_model.agents.DefaultAgent import DefaultAgent
from networked_model.agents.BaselineAgent import BaselineAgent
from networked_model.agents.WeekendAgent import WeekendAgent
from networked_model.agents.MemoryAgent import MemoryAgent


TYPE_DICT = {
    "default": DefaultAgent,
    "baseline": BaselineAgent,
    'weekend': WeekendAgent,
    "memory": MemoryAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def initialize_coefficients(self, agent_type):
        coeffs =  TYPE_DICT[agent_type]().initialize_coefficients()
        return coeffs
        
    def initialize_characteristics(self, agent_type):
        return TYPE_DICT[agent_type]().initialize_characteristics()
