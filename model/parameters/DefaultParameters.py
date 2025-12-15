from model.parameters.Parameters import Parameters

class DefaultParameters(Parameters):
    """
    Parameters class extension defining default agent parameters.
    """
    def __init__(self):
        super().__init__()
        # Stress
        self.set_stress_params(
            mean=0.5,
            sigma=0.15,
            reversion=1.0,
            E_weight=3.0,
        )
        # Aversive internal state
        self.set_aversion_params(
            feedback=6,
            carrying_capacity=0.2,
            S_weight=3,
            T_weight=0.1,
            X_weight=2,
            I_weight=0.5,
            F_weight=7,
            B_weight=5,
        )
        # Urge to escape
        self.set_urge_to_escape_params(
            feedback=5,
            A_weight=3
        )
        # Suicidal thought
        self.set_suicidal_thought_parameters(
            weight_new=0.8,
            sig_middle=0.4,
            sig_steepness=100
        )
        # Escape behavior
        self.set_escape_behavior_parameters(
            weight_new=0.8,
            sig_middle=0.35,
            sig_steepness=50
        )
        # External strategy parameters
        self.set_external_strategy_params(
            feedback=3,
            carrying_capacity=0.1,
            A_weight=0.41,
            U_weight=0.6
        )
        # Internal strategy parameters
        self.set_internal_strategy_params(
            feedback=3,
            carrying_capacity=0.05,
            A_weight=0.65,
            U_weight=1.05
        )
