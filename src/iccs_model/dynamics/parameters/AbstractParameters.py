from iccs_model.dynamics.parameters.sets.StressParameterSet import StressParameterSet
from iccs_model.dynamics.parameters.sets.AversionParameterSet import AversionParameterSet
from iccs_model.dynamics.parameters.sets.UrgeToEscapeParameterSet import UrgeToEscapeParameterSet
from iccs_model.dynamics.parameters.sets.SuicidalParameterSet import SuicidalParameterSet
from iccs_model.dynamics.parameters.sets.EscapeBehaviorParameterSet import EscapeBehaviorParameterSet
from iccs_model.dynamics.parameters.sets.ExternalParameterSet import ExternalParameterSet
from iccs_model.dynamics.parameters.sets.InternalParameterSet import InternalParameterSet
from iccs_model.dynamics.parameters.sets.SuicideHistoryParameterSet import SuicideHistoryParameterSet
from iccs_model.dynamics.parameters.sets.BurdenParameterSet import BurdenParameterSet
from abc import ABC

class Parameters(ABC):
    """
    Abstract class to extend for containing and modifying coefficients
    to use in update equations.
    """
    def __init__(self):
        self.stress = StressParameterSet()
        self.aversion = AversionParameterSet()
        self.urge_to_escape = UrgeToEscapeParameterSet()
        self.suicide_history = SuicideHistoryParameterSet()
        self.suicidal_thought = SuicidalParameterSet()
        self.escape_behavior = EscapeBehaviorParameterSet()
        self.external_strategy = ExternalParameterSet()
        self.internal_strategy = InternalParameterSet()
        self.burdensomeness = BurdenParameterSet()
    
    def set_stress_coefficients(
            self,
            baseline=None,
            decay=None,
            impulse_rate=None,
            impulse_strength=None,
            morning_impulse=None,
            alpha=None,
            beta=None,
            gamma=None,
            sigma=None,
        ):
        """
        Initializes or modifies stress coefficients.

        Parameters
        ----------
        baseline: float
            Baseline stress value.
        decay: float
            Decay rate of experienced stress.
        impulse_rate: float
            Rate of impulses in impulse per time unit.
        impulse_strength: float
            Strength of a single impulse
        morning_impulse: None
            Strength of the stress impulse after sleeping
        alpha: float
            Effect of external strategy on baseline stress.
        beta: float
            Effect of external strategy on decay rate.
        gamma: float
            Effect of external strategy on impulse strength.
        """
        coefficients = {
            "baseline": baseline,
            "decay": decay,
            "impulse_rate": impulse_rate,
            "impulse_strength": impulse_strength,
            "morning_impulse": morning_impulse,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "sigma": sigma,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.stress, name, value)
    
    def set_aversion_coefficients(
            self,
            feedback=None,
            carrying_capacity=None,
            S_weight=None,
            T_weight=None,
            X_weight=None,
            I_weight=None,
            B_weight=None,
            c_weight=None,
        ):
        """
        Initializes or modifies aversive internal state
        coefficients.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            aversive internal state
        carrying_capacity: float
            Carrying capacity of the logistic growth equation
            part of the update equation for aversive internal
            state
        S_weight: float
            Weight of stress from previous timestep on aversive
            internal state
        T_weight: float
            Weight of suicidal thoughts from previous timestep
            on aversive internal state
        X_weight: float
            Weight of escape behavior from previous timestep on
            aversive internal state
        I_weight: float
            Weight of internal escape strategies from previous
            timestep on aversive internal state
        B_weight: float
            Weight of social burden from previous timestep
            on aversive internal state
        c_weight: float
            Weight of clustering coefficient on aversive internal state
        """
        coefficients = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "S_weight": S_weight,
            "T_weight": T_weight,
            "X_weight": X_weight,
            "I_weight": I_weight,
            "B_weight": B_weight,
            "c_weight": c_weight,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.aversion, name, value)
    
    def set_urge_to_escape_coefficients(
            self,
            feedback=None,
            A_weight=None,
            M_weight=None,
            C_weight=None,
        ):
        """
        Initializes or modifies urge to escape coefficients.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            urge to escape
        A_weight: float
            Weight of aversive internal state from previous
            timestep on urge to escape
        M_weight: float
            Weight of memory of suicidal thought from previous
            timestep on urge to escape
        C_weight: float
            Weight of connectedness from previous
            timestep on urge to escape
        
        """
        coefficients = {
            "feedback": feedback,
            "A_weight": A_weight,
            "M_weight": M_weight,
            "C_weight": C_weight,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.urge_to_escape, name, value)
    
    def set_suicide_history_coefficients(
            self,
            decay=None,
        ):
        """
        Initializes or modifies suicide history coefficients.

        Parameters
        ----------
        decay: float
            Rate at which memory of suicidal thought decays
        """
        coefficients = {
            "decay": decay,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.suicide_history, name, value)
    
    def set_suicidal_thought_coefficients(
            self,
            feedback=None,
            sig_middle=None,
            sig_steepness=None,
        ):
        """
        Initializes or modifies suicidal thought coefficients.

        Parameters
        ----------
        feedback: float
            Weight of suicidal thought based on current
            parameters compared to feedback of suicidal
            thoughts from previous timestep
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of suicidal thoughts
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of suicidal thoughts
        """
        coefficients = {
            "feedback": feedback,
            "sig_middle": sig_middle,
            "sig_steepness": sig_steepness,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.suicidal_thought, name, value)
    
    def set_escape_behavior_coefficients(
            self,
            feedback=None,
            sig_middle=None,
            sig_steepness=None,
        ):
        """
        Initializes escape behavior coefficients.

        Parameters
        ----------
        feedback: float
            Weight of escape behavior based on current
            parameters compared to feedback of escape 
            behavior from previous timestep
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of escape behavior
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of escape behavior
        """
        coefficients = {
            "feedback": feedback,
            "sig_middle": sig_middle,
            "sig_steepness": sig_steepness,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.escape_behavior, name, value)
    
    def set_external_strategy_coefficients(
            self,
            feedback=None,
            carrying_capacity=None,
            A_weight=None,
            U_weight=None,
        ):
        """
        Initializes external escape strategy coefficients.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            external escape strategy
        carrying_capacity: float
            Carrying capacity of the logistic growth equation
            part of the update equation for external escape
            strategy
        A_weight: float
            Weight of aversive internal state from previous 
            timestep on external escape strategy
        U_weight: float
            Weight of urge to escape from previous timestep
            on external escape strategy
        """
        coefficients = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": A_weight,
            "U_weight": U_weight,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.external_strategy, name, value)
    
    def set_internal_strategy_coefficients(
            self,
            feedback=None,
            carrying_capacity=None,
            A_weight=None,
            U_weight=None,
        ):
        """
        Initializes internal escape strategy coefficients.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            internal escape strategy
        carrying_capacity: float
            Carrying capacity of the logistic growth equation
            part of the update equation for internal escape
            strategy
        A_weight: float
            Weight of aversive internal state from previous 
            timestep on internal escape strategy
        U_weight: float
            Weight of urge to escape from previous timestep
            on internal escape strategy
        """
        coefficients = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": A_weight,
            "U_weight": U_weight,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.internal_strategy, name, value)
    
    def set_burdensomeness_coefficients(
        self,
        neighbors=None,
        neighbor_ws=None,
        feedback=None,
        A_weight=None,
        I_weight=None,
        B_lonely=None,
        ):
        """
        Initializes social burden coefficients.

        Parameters
        ----------
        neighbors: List<Agent>
            List of connected agents
        neighbor_ws: List<float>
            List of weights associated with agent connections
        feedback: float
            Strength of feedback from previous state of
            social burden
        A_weight: float
            Weight of aversive internal state from previous 
            timestep on internal escape strategy
        I_weight: float
            Weight of internal escape strategy from previous timestep
            on social burden
        B_lonely: float
            Value of social burden when the agent has no connections
        """
        coefficients = {
            "neighbors": neighbors,
            "neighbor_ws": neighbor_ws,
            "feedback": feedback,
            "A_weight": A_weight,
            "I_weight": I_weight,
            "B_lonely": B_lonely,
        }

        for name, value in coefficients.items():
            if value is not None:
                setattr(self.burdensomeness, name, value)
    
    def get_S_params(
            self,
            external_strat,
    ):
        """
        Gathers all stress coefficients and parameters into
        a dictionary.
        """
        return {
            "E": external_strat,
            "baseline": self.stress.baseline,
            "decay": self.stress.decay,
            "impulse_rate": self.stress.impulse_rate,
            "impulse_strength": self.stress.impulse_strength,
            "morning_impulse": self.stress.morning_impulse,
            "alpha": self.stress.alpha,
            "beta": self.stress.beta,
            "gamma": self.stress.gamma,
            "sigma": self.stress.sigma,
        }
    
    def get_A_params(
            self,
            stress,
            suicidal_thought,
            escape_behavior,
            internal_strat,
            burdensomeness,
            clustering_coefficient,
    ):
        """
        Gathers all aversive internal state coefficients and parameters into
        a dictionary.
        """
        return {
            "S": stress,
            "T": suicidal_thought,
            "X": escape_behavior,
            "I": internal_strat,
            "B": burdensomeness,
            "clustering_coefficient": clustering_coefficient,
            "feedback": self.aversion.feedback,
            "carrying_capacity": self.aversion.carrying_capacity,
            "S_weight": self.aversion.S_weight,
            "T_weight": self.aversion.T_weight,
            "X_weight": self.aversion.X_weight,
            "I_weight": self.aversion.I_weight,
            "B_weight": self.aversion.B_weight,
            "c_weight": self.aversion.c_weight,
        }
    
    def get_U_params(
            self,
            aversive_internal_state,
            suicide_history,
            connectedness,
    ):
        """
        Gathers all urge to escape coefficients and parameters into a dictionary.
        """
        return {
            "A": aversive_internal_state,
            "M": suicide_history,
            "C": connectedness,
            "feedback": self.urge_to_escape.feedback,
            "A_weight": self.urge_to_escape.A_weight,
            "M_weight": self.urge_to_escape.M_weight,
            "C_weight": self.urge_to_escape.C_weight,
        }
    
    def get_M_params(
            self,
            suicidal_thought,
    ):
        """
        Gathers all suicide histroy coefficients and parameters into a dictionary.
        """
        return {
            "T": suicidal_thought,
            "decay": self.suicide_history.decay,
        }
    
    def get_T_params(
            self,
            urge_to_escape,
    ):
        """
        Gathers all suicidal thought coefficients and parameters into a dictionary.
        """
        return {
            "U": urge_to_escape,
            "feedback": self.suicidal_thought.feedback,
            "sig_middle": self.suicidal_thought.sig_middle,
            "sig_steepness": self.suicidal_thought.sig_steepness,
        }
    
    def get_X_params(
            self,
            urge_to_escape,
    ):
        """
        Gathers all escape behavior coefficients and parameters into a dictionary.
        """
        return {
            "U": urge_to_escape,
            "feedback": self.escape_behavior.feedback,
            "sig_middle": self.escape_behavior.sig_middle,
            "sig_steepness": self.escape_behavior.sig_steepness,
        }
    
    def get_E_params(
            self,
            aversive_internal_state,
            urge_to_escape,
    ):
        """
        Gathers all external strategy coefficients and parameters into a dictionary.
        """
        return {
            "A": aversive_internal_state,
            "U": urge_to_escape,
            "feedback": self.external_strategy.feedback,
            "carrying_capacity": self.external_strategy.carrying_capacity,
            "A_weight": self.external_strategy.A_weight,
            "U_weight": self.external_strategy.U_weight,
        }
    
    def get_I_params(
            self,
            aversive_internal_state,
            urge_to_escape,
    ):
        """
        Gathers all internal strategy coefficients and parameters into a dictionary.
        """
        return {
            "A": aversive_internal_state,
            "U": urge_to_escape,
            "feedback": self.internal_strategy.feedback,
            "carrying_capacity": self.internal_strategy.carrying_capacity,
            "A_weight": self.internal_strategy.A_weight,
            "U_weight": self.internal_strategy.U_weight,
        }
    
    def get_B_params(
        self,
        internal_strategy,
        ):
        neighbor_As = []
        for agent in self.burdensomeness.neighbors:
            neighbor_As.append(agent.aversive_internal_state)
        return {
            "neighbor_As": neighbor_As,
            "neighbor_ws": self.burdensomeness.neighbor_ws,
            "I": internal_strategy,
            "feedback": self.burdensomeness.feedback,
            "A_weight": self.burdensomeness.A_weight,
            "I_weight": self.burdensomeness.I_weight,
            "B_lonely": self.burdensomeness.B_lonely,
        }