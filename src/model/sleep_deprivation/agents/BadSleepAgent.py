from model.sleep_deprivation.agents.SleepAgent import SleepAgent

class BadSleepAgent(SleepAgent):
    """
    Agent type used to analyze the effects of poor sleep on the agent's
    suicidal thoughts, as well as its social network.
    """

    def __init__(self, model, stress_gen=False, default_params={}):
        super().__init__(model)
        self.type = "bad_sleep"
        self.state_params.consistent_sleep = 5.5

