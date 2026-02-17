from model.agents.StandardAgent import StandardAgent


TYPE_DICT = {
    "standard": StandardAgent
}

class AgentFactory():

    def __init__(self):
        pass

    def create_agents(type, model, n):
        TYPE_DICT[type].create_agents(model=model, n=n)
        TYPE_DICT[type].init_stress()
