from model.agents.StandardAgent import StandardAgent

class PopularAgent(StandardAgent):

    def __init__(self, model):
        super().__init__(model)
        self.type = "popular"
        
    def set_friends(self, n=10):
        super().set_friends(n)

    def set_bullies(self, n=0):
        super().set_bullies(n)
