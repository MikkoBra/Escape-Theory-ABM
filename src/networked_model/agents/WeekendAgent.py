from networked_model.agents.DefaultAgent import DefaultAgent

class WeekendAgent(DefaultAgent):
    def initialize_characteristics(
            self,
            variable_sleep=True,
            variable_commute=False
            ):
        chars = super().initialize_characteristics(variable_sleep, variable_commute)
        chars['weekends'] = True
        return chars