import mesa


class Agent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.network_id = self.unique_id - 1
        self.diathesis = 0
        self.impulsivity = 0.5
        self.stress_magnitude = 0.1
