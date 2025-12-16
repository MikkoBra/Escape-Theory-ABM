from model.agents.StandardAgent import StandardAgent
from model.parameters.VolatileParameters import VolatileParameters

class VolatileAgent(StandardAgent):

    def __init__(self, model):
        super().__init__(model)
        self.type = "volatile"
        self.parameters = VolatileParameters()

