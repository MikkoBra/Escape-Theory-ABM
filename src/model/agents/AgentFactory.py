from model.agents.StandardAgent import StandardAgent
from model.agents.SleepAgent import SleepAgent
from model.agents.BadSleepAgent import BadSleepAgent
from model.agents.WangAgent import WangAgent


TYPE_DICT = {
    "standard": StandardAgent,
    "sleep": SleepAgent,
    "bad_sleep": BadSleepAgent,
    "wang": WangAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def create_agents(type, model, n, default_params, stress_gen=False):
        TYPE_DICT[type].create_agents(model=model, n=n, default_params=default_params, stress_gen=stress_gen)
