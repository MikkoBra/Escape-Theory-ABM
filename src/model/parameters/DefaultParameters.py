from model.parameters.AbstractParameters import Parameters
from model.parameters.sets.StressParameterSet import StressParameterSet
from model.parameters.sets.AversionParameterSet import AversionParameterSet
from model.parameters.sets.UrgeToEscapeParameterSet import UrgeToEscapeParameterSet
from model.parameters.sets.SuicidalParameterSet import SuicidalParameterSet
from model.parameters.sets.EscapeBehaviorParameterSet import EscapeBehaviorParameterSet
from model.parameters.sets.ExternalParameterSet import ExternalParameterSet
from model.parameters.sets.InternalParameterSet import InternalParameterSet
from model.parameters.sets.SuicideHistoryParameterSet import SuicideHistoryParameterSet
from model.parameters.sets.BurdenParameterSet import BurdenParameterSet

class DefaultParameters(Parameters):
    """
    Parameters class defining default agent parameters.
    """
    def __init__(self):
        super().__init__()
        self.default_stress = StressParameterSet(
            baseline=0.2,
            decay=3,
            impulse_rate=4,
            impulse_strength=0.1,
            morning_impulse=0,
            alpha=0.1,
            beta=0.1,
            gamma=0.1,
            sigma=0.12
        )
        self.default_aversion = AversionParameterSet(
            feedback=4,
            carrying_capacity=0.1,
            S_weight=2,
            T_weight=0.1,
            X_weight=1,
            I_weight=0.5,
            B_weight=2.5,
            c_weight=0.7,
        )
        self.default_urge_to_escape = UrgeToEscapeParameterSet(
            feedback=2.5,
            A_weight=1.75,
            M_weight=0.1,
            C_weight=0.1,
        )
        self.default_suicide_history = SuicideHistoryParameterSet(
            decay=0.7
        )
        self.default_suicidal_thought = SuicidalParameterSet(
            feedback=3,
            sig_middle=0.35,
            sig_steepness=100
        )
        self.default_escape_behavior = EscapeBehaviorParameterSet(
            feedback=3,
            sig_middle=0.3,
            sig_steepness=50
        )
        self.default_external_strategy = ExternalParameterSet(
            feedback=0.5,
            carrying_capacity=0.1,
            A_weight=0.41,
            U_weight=0.45
        )
        self.default_internal_strategy = InternalParameterSet(
            feedback=3,
            carrying_capacity=0.05,
            A_weight=0.65,
            U_weight=0.75
        )
        self.default_burdensomeness = BurdenParameterSet(
            neighbors=[],
            neighbor_ws=[],
            feedback=0.35,
            A_weight=0.25,
            I_weight=0.1,
            B_lonely=1
        )
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
        ):
        return super().set_stress_params(
            baseline=self.default_stress.baseline,
            decay=self.default_stress.decay,
            impulse_rate=self.default_stress.impulse_rate,
            impulse_strength=self.default_stress.impulse_strength,
            morning_impulse=self.default_stress.morning_impulse,
            alpha=self.default_stress.alpha,
            beta=self.default_stress.beta,
            gamma=self.default_stress.gamma,
            sigma=self.default_stress.sigma,
        )
    
    def set_default_aversion_params(
            self,
        ):
        return super().set_aversion_params(
            feedback=self.default_aversion.feedback,
            carrying_capacity=self.default_aversion.carrying_capacity,
            S_weight=self.default_aversion.S_weight,
            T_weight=self.default_aversion.T_weight,
            X_weight=self.default_aversion.X_weight,
            I_weight=self.default_aversion.I_weight,
            B_weight=self.default_aversion.B_weight,
            c_weight=self.default_aversion.c_weight)
    
    def set_default_urge_to_escape_params(self):
        return super().set_urge_to_escape_params(
            feedback=self.default_urge_to_escape.feedback,
            A_weight=self.default_urge_to_escape.A_weight,
            M_weight=self.default_urge_to_escape.M_weight,
            C_weight=self.default_urge_to_escape.C_weight
        )

    def set_default_suicide_history_params(self):
        return super().set_suicide_history_params(
            decay=self.default_suicide_history.decay
        )

    def set_default_suicidal_thought_params(self):
        return super().set_suicidal_thought_params(
            feedback=self.default_suicidal_thought.feedback,
            sig_middle=self.default_suicidal_thought.sig_middle,
            sig_steepness=self.default_suicidal_thought.sig_steepness
        )

    def set_default_escape_behavior_params(self):
        return super().set_escape_behavior_params(
            feedback=self.default_escape_behavior.feedback,
            sig_middle=self.default_escape_behavior.sig_middle,
            sig_steepness=self.default_escape_behavior.sig_steepness
        )

    def set_default_external_strategy_params(self):
        return super().set_external_strategy_params(
            feedback=self.default_external_strategy.feedback,
            carrying_capacity=self.default_external_strategy.carrying_capacity,
            A_weight=self.default_external_strategy.A_weight,
            U_weight=self.default_external_strategy.U_weight
        )

    def set_default_internal_strategy_params(self):
        return super().set_internal_strategy_params(
            feedback=self.default_internal_strategy.feedback,
            carrying_capacity=self.default_internal_strategy.carrying_capacity,
            A_weight=self.default_internal_strategy.A_weight,
            U_weight=self.default_internal_strategy.U_weight
        )

    def set_default_burdensomeness_params(self):
        return super().set_burdensomeness_params(
            neighbors=self.default_burdensomeness.neighbors,
            neighbor_ws=self.default_burdensomeness.neighbor_ws,
            feedback=self.default_burdensomeness.feedback,
            A_weight=self.default_burdensomeness.A_weight,
            I_weight=self.default_burdensomeness.I_weight,
            B_lonely=self.default_burdensomeness.B_lonely
        )

