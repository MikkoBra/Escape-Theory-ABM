from model.parameters.DefaultParameters import DefaultParameters

class VolatileParameters(DefaultParameters):
    """
    Parameters class defining parameters for an agent whose stress
    is more volatile than default.
    """

    def set_stress_params(
            self,
            mean=0.5,
            sigma=0.3,
            reversion=0.5,
            E_weight=3.0,
        ):
        return super().set_stress_params(mean, sigma, reversion, E_weight)

    def set_suicidal_thought_params(
            self,
            weight_new=0.9,
            sig_middle=0.35,
            sig_steepness=100
        ):
        return super().set_suicidal_thought_params(weight_new, sig_middle, sig_steepness)
    
    def set_escape_behavior_params(
            self,
            weight_new=0.9,
            sig_middle=0.32,
            sig_steepness=50
        ):
        return super().set_escape_behavior_params(weight_new, sig_middle, sig_steepness)
