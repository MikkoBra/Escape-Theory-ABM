from networked_model.agents.BaselineAgent import BaselineAgent

class BadSleepAgent(BaselineAgent):
    """
    Default agent (with daily schedule) but with
    standard sleep duration of 6 hours instead
    of variable sleep.
    """

    def initialize_characteristics(
            self,
            variable_sleep=False,
            variable_commute=False
            ):
        chars = super().initialize_characteristics(variable_sleep, variable_commute)
        chars['sleep'] = 6
        return chars