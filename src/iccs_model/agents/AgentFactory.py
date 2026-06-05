from iccs_model.agents.StandardAgent import StandardAgent
from iccs_model.agents.SleepAgent import SleepAgent
from iccs_model.agents.BadSleepAgent import BadSleepAgent
from iccs_model.agents.BaselineAgent import BaselineAgent
from iccs_model.agents.SuicideHistoryAgent import SuicideHistoryAgent


TYPE_DICT = {
    "standard": StandardAgent,
    "sleep": SleepAgent,
    "bad_sleep": BadSleepAgent,
    "baseline": BaselineAgent,
    "suicide_history": SuicideHistoryAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def create_agents(type, model, n, default_params, stress_gen=False):
        TYPE_DICT[type].create_agents(model=model, n=n, default_params=default_params, stress_gen=stress_gen)
