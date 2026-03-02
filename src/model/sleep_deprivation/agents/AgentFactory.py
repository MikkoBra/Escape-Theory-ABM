from model.sleep_deprivation.agents.StandardAgent import StandardAgent
from model.sleep_deprivation.agents.SleepAgent import SleepAgent
from model.sleep_deprivation.agents.BadSleepAgent import BadSleepAgent
from model.sleep_deprivation.agents.BaselineAgent import BaselineAgent


TYPE_DICT = {
    "standard": StandardAgent,
    "sleep": SleepAgent,
    "bad_sleep": BadSleepAgent,
    "baseline": BaselineAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def create_agents(type, model, n, default_params, stress_gen=False):
        TYPE_DICT[type].create_agents(model=model, n=n, default_params=default_params, stress_gen=stress_gen)
