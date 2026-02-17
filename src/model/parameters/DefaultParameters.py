from model.parameters.AbstractParameters import Parameters

class DefaultParameters(Parameters):
    """
    Parameters class defining default agent parameters.
    """
    def __init__(self):
        super().__init__()
        self.set_defaults()

    def set_defaults(self, S=True, A=True, U=True, M=True, T=True, X=True, E=True, I=True, B=True):
        # Stress
        if S:
            self.set_default_stress_params()
        # Aversive internal state
        if A:
            self.set_default_aversion_params()
        # Urge to escape
        if U:
            self.set_default_urge_to_escape_params()
        # Suicide history
        if M:
            self.set_default_suicide_history_params()
        # Suicidal thought
        if T:
            self.set_default_suicidal_thought_params()
        # Escape behavior
        if X:
            self.set_default_escape_behavior_params()
        # External strategy parameters
        if E:
            self.set_default_external_strategy_params()
        # Internal strategy parameters
        if I:
            self.set_default_internal_strategy_params()
        if B:
            self.set_default_burdensomeness_params()

    
    def set_default_stress_params(
            self,
            baseline=0.2,
            decay=3,
            impulse_rate=3,
            impulse_strength=0.1,
            morning_impulse=0,
            alpha=0.1,
            beta=0.1,
            gamma=0.1,
        ):
        return super().set_stress_params(
            baseline=baseline,
            decay=decay,
            impulse_rate=impulse_rate,
            impulse_strength=impulse_strength,
            morning_impulse=morning_impulse,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
    
    def set_default_aversion_params(
            self,
            feedback=4,
            carrying_capacity=0.1,
            S_weight=2,
            T_weight=0.1,
            X_weight=1,
            I_weight=0.5,
            B_weight=2.5,
            c_weight=0.7,
        ):
        return super().set_aversion_params(feedback, carrying_capacity, S_weight, T_weight, X_weight, I_weight, B_weight, c_weight=c_weight)
    
    def set_default_urge_to_escape_params(
            self,
            feedback=2.5,
            A_weight=1.75,
            M_weight=0.1,
            C_weight=1,
        ):
        return super().set_urge_to_escape_params(feedback, A_weight, M_weight, C_weight=C_weight)
    
    def set_default_suicide_history_params(
            self,
            decay=0.5,
        ):
        return super().set_suicide_history_params(decay)
    
    def set_default_suicidal_thought_params(
            self,
            feedback=3,
            sig_middle=0.35,
            sig_steepness=100
        ):
        return super().set_suicidal_thought_params(feedback, sig_middle, sig_steepness)
    
    def set_default_escape_behavior_params(
            self,
            feedback=3,
            sig_middle=0.3,
            sig_steepness=50
        ):
        return super().set_escape_behavior_params(feedback, sig_middle, sig_steepness)
    
    def set_default_external_strategy_params(
            self,
            feedback=0.5,
            carrying_capacity=0.1,
            A_weight=0.41,
            U_weight=0.45
        ):
        return super().set_external_strategy_params(feedback, carrying_capacity, A_weight, U_weight)
    
    def set_default_internal_strategy_params(
            self,
            feedback=3,
            carrying_capacity=0.05,
            A_weight=0.65,
            U_weight=0.75
        ):
        return super().set_internal_strategy_params(feedback, carrying_capacity, A_weight, U_weight)

    def set_default_burdensomeness_params(
            self,
            neighbors = [],
            neighbor_ws = [],
            feedback=0.35,
            A_weight=0.25,
            I_weight=0.1,
            B_lonely=1,
    ):
        return super().set_burdensomeness_params(neighbors=neighbors, neighbor_ws=neighbor_ws,
                                                 feedback=feedback, A_weight=A_weight,
                                                 I_weight=I_weight, B_lonely=B_lonely)
