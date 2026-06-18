from networked_model.agents.DefaultAgent import DefaultAgent

class GoodSleepAgent(DefaultAgent):
    """
    Default agent (with daily schedule) but with
    standard sleep duration of 8 hours instead
    of variable sleep.
    """
    def initialize_characteristics(
            self,
            variable_sleep=False,
            variable_commute=False
            ):
        chars = super().initialize_characteristics(variable_sleep, variable_commute)
        return chars