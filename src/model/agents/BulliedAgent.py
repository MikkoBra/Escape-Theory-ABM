from model.agents.StandardAgent import StandardAgent

class BulliedAgent(StandardAgent):

    def __init__(self, model):
        super().__init__(model)
        self.type = "bullied"
        
    def set_friends(self, n=1):
        super().set_friends(n)

    def set_bullies(self, n=2):
        super().set_bullies(n)
