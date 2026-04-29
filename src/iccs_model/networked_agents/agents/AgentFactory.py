from iccs_model.networked_agents.agents.StandardAgent import StandardAgent


TYPE_DICT = {
    "standard": StandardAgent,
}

class AgentFactory():
    def __init__(self):
        pass

    def create_agents(type, model, n, default_params):
        TYPE_DICT[type].create_agents(model=model, n=n, default_params=default_params)
