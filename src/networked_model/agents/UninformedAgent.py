from networked_model.agents.DefaultAgent import DefaultAgent

class UninformedAgent(DefaultAgent):

    """
    Agent to compare with fitted Brighten agent,
    basically a BadSleepAgent but with schedule
    and social effects.
    """

    def initialize_characteristics(
            self,
            variable_sleep=False,
            variable_commute=False
            ):
        chars = super().initialize_characteristics(variable_sleep, variable_commute)
        chars['sleep'] = 6
        chars['weekends'] = True
        return chars