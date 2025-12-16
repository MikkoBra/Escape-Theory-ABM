from model.parameters.AbstractParameters import Parameters

class DefaultParameters(Parameters):
    """
    Parameters class defining default agent parameters.
    """
    def __init__(self):
        super().__init__()
        # Stress
        self.set_stress_params()
        # Aversive internal state
        self.set_aversion_params()
        # Urge to escape
        self.set_urge_to_escape_params()
        # Suicidal thought
        self.set_suicidal_thought_params()
        # Escape behavior
        self.set_escape_behavior_params()
        # External strategy parameters
        self.set_external_strategy_params()
        # Internal strategy parameters
        self.set_internal_strategy_params()

    def set_defaults(self, S=True, A=True, U=True, T=True, X=True, E=True, I=True):
        # Stress
        if S:
            self.set_stress_params()
        # Aversive internal state
        if A:
            self.set_aversion_params()
        # Urge to escape
        if U:
            self.set_urge_to_escape_params()
        # Suicidal thought
        if T:
            self.set_suicidal_thought_params()
        # Escape behavior
        if X:
            self.set_escape_behavior_params()
        # External strategy parameters
        if E:
            self.set_external_strategy_params()
        # Internal strategy parameters
        if I:
            self.set_internal_strategy_params()

    
    def set_stress_params(
            self,
            mean=0.5,
            sigma=0.15,
            reversion=1.0,
            E_weight=3.0,
        ):
        return super().set_stress_params(mean, sigma, reversion, E_weight)
    
    def set_aversion_params(
            self,
            feedback=6,
            carrying_capacity=0.2,
            S_weight=3,
            T_weight=0.1,
            X_weight=2,
            I_weight=0.5,
            F_weight=3,
            B_weight=4,
        ):
        return super().set_aversion_params(feedback, carrying_capacity, S_weight, T_weight, X_weight, I_weight, F_weight, B_weight)
    
    def set_urge_to_escape_params(
            self,
            feedback=5,
            A_weight=3
        ):
        return super().set_urge_to_escape_params(feedback, A_weight)
    
    def set_suicidal_thought_params(
            self,
            weight_new=0.8,
            sig_middle=0.4,
            sig_steepness=100
        ):
        return super().set_suicidal_thought_params(weight_new, sig_middle, sig_steepness)
    
    def set_escape_behavior_params(
            self,
            weight_new=0.8,
            sig_middle=0.35,
            sig_steepness=50
        ):
        return super().set_escape_behavior_params(weight_new, sig_middle, sig_steepness)
    
    def set_external_strategy_params(
            self,
            feedback=3,
            carrying_capacity=0.1,
            A_weight=0.41,
            U_weight=0.6
        ):
        return super().set_external_strategy_params(feedback, carrying_capacity, A_weight, U_weight)
    
    def set_internal_strategy_params(
            self,
            feedback=3,
            carrying_capacity=0.05,
            A_weight=0.65,
            U_weight=1.05
        ):
        return super().set_internal_strategy_params(feedback, carrying_capacity, A_weight, U_weight)
