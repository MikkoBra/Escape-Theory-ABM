from networked_model.agents.BaselineAgent import BaselineAgent


class MemoryAgent(BaselineAgent):
    
    def initialize_coefficients(self):
        
        coeffs = super().initialize_coefficients()
        coeffs['urge_to_escape']['M_weight'] = 0.3
        coeffs['suicide_history']['decay'] = 0.5
        return coeffs
    
    def initialize_characteristics(variable_sleep=False, variable_commute=False):
        return super().initialize_characteristics(variable_sleep=variable_sleep, variable_commute=variable_commute)
