from model.parameters.sets.StressParameterSet import StressParameterSet
from model.parameters.sets.AversionParameterSet import AversionParameterSet
from model.parameters.sets.UrgeToEscapeParameterSet import UrgeToEscapeParameterSet
from model.parameters.sets.SuicidalParameterSet import SuicidalParameterSet
from model.parameters.sets.EscapeBehaviorParameterSet import EscapeBehaviorParameterSet
from model.parameters.sets.ExternalParameterSet import ExternalParameterSet
from model.parameters.sets.InternalParameterSet import InternalParameterSet

class Parameters():
    """
    Class to contain and modify parameters to use for update equations.
    """
    def __init__(self):
        self.stress = StressParameterSet()
        self.aversion = AversionParameterSet()
        self.urge_to_escape = UrgeToEscapeParameterSet()
        self.suicidal_thought = SuicidalParameterSet()
        self.escape_behavior = EscapeBehaviorParameterSet()
        self.external_strategy = ExternalParameterSet()
        self.internal_strategy = InternalParameterSet()
    
    def set_stress_params(
            self,
            mean=None,
            sigma=None,
            reversion=None,
            E_weight=None):
        """
        Initializes or modifies stress parameters.

        Parameters
        ----------
        mean: float
            Long-term mean of the stress movement.
        sigma: float
            Standard deviation of the stochastic jitter of the
            stress movement.
        reversion: float
            Strength of pull towards the mean of the stress
            movement.
        E_weight: float
            Weight of external escape strategies on stress.
        """
        params = {
            "mean": mean,
            "sigma": sigma,
            "reversion": reversion,
            "E_weight": E_weight,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.stress, name, value)
    
    def set_aversion_params(
            self,
            feedback=None,
            carrying_capacity=None,
            S_weight=None,
            T_weight=None,
            X_weight=None,
            I_weight=None,
            F_weight=None,
            B_weight=None,
        ):
        """
        Initializes or modifies aversive internal state
        parameters.

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
        F_weight: float
            Weight of social influence from previous timestep
            on aversive internal state
        """
        params = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "S_weight": S_weight,
            "T_weight": T_weight,
            "X_weight": X_weight,
            "I_weight": I_weight,
            "F_weight": F_weight,
            "B_weight": B_weight,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.aversion, name, value)
    
    def set_urge_to_escape_params(
            self,
            feedback=None,
            A_weight=None,
        ):
        """
        Initializes or modifies urge to escape parameters.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            urge to escape
        A_weight: float
            Weight of aversive internal state from previous
            timestep on urge to escape
        """
        params = {
            "feedback": feedback,
            "A_weight": A_weight,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.urge_to_escape, name, value)
    
    def set_suicidal_thought_parameters(
            self,
            weight_new=None,
            sig_middle=None,
            sig_steepness=None,
        ):
        """
        Initializes or modifies suicidal thought parameters.

        Parameters
        ----------
        weight_new: float
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
        params = {
            "weight_new": weight_new,
            "sig_middle": sig_middle,
            "sig_steepness": sig_steepness,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.suicidal_thought, name, value)
    
    def set_escape_behavior_parameters(
            self,
            weight_new=None,
            sig_middle=None,
            sig_steepness=None,
        ):
        """
        Initializes escape behavior parameters.

        Parameters
        ----------
        weight_new: float
            Weight of suicidal thought based on current
            parameters compared to feedback of escape 
            behavior from previous timestep
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of escape behavior
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of escape behavior
        """
        params = {
            "weight_new": weight_new,
            "sig_middle": sig_middle,
            "sig_steepness": sig_steepness,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.escape_behavior, name, value)
    
    def set_external_strategy_params(
            self,
            feedback=None,
            carrying_capacity=None,
            A_weight=None,
            U_weight=None,
        ):
        """
        Initializes external escape strategy parameters.

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
        params = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": A_weight,
            "U_weight": U_weight,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.external_strategy, name, value)
    
    def set_internal_strategy_params(
            self,
            feedback=None,
            carrying_capacity=None,
            A_weight=None,
            U_weight=None,
        ):
        """
        Initializes internal escape strategy parameters.

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
        params = {
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": A_weight,
            "U_weight": U_weight,
        }

        for name, value in params.items():
            if value is not None:
                setattr(self.internal_strategy, name, value)
    
    def get_A_params(
            self,
            stress,
            suicidal_thought,
            escape_behavior,
            internal_strat,
            friend_influence,
            bully_influence,
    ):
        """
        Gathers all aversive internal state parameters into
        a dictionary.
        """
        return {
            "S": stress,
            "T": suicidal_thought,
            "X": escape_behavior,
            "I": internal_strat,
            "F": friend_influence,
            "B": bully_influence,
            "feedback": self.aversion.feedback,
            "carrying_capacity": self.aversion.carrying_capacity,
            "S_weight": self.aversion.S_weight,
            "T_weight": self.aversion.T_weight,
            "X_weight": self.aversion.X_weight,
            "I_weight": self.aversion.I_weight,
            "F_weight": self.aversion.F_weight,
            "B_weight": self.aversion.B_weight,
        }
    
    def get_U_params(
            self,
            aversive_internal_state
    ):
        """
        Gathers all urge to escape parameters into a dictionary.
        """
        return {
            "A": aversive_internal_state,
            "feedback": self.urge_to_escape.feedback,
            "A_weight": self.urge_to_escape.A_weight,
        }
    
    def get_T_params(
            self,
            urge_to_escape,
    ):
        """
        Gathers all suicidal thought parameters into a dictionary.
        """
        return {
            "U": urge_to_escape,
            "weight_new": self.suicidal_thought.weight_new,
            "sig_middle": self.suicidal_thought.sig_middle,
            "sig_steepness": self.suicidal_thought.sig_steepness,
        }
    
    def get_X_params(
            self,
            urge_to_escape,
    ):
        """
        Gathers all escape behavior parameters into a dictionary.
        """
        return {
            "U": urge_to_escape,
            "weight_new": self.escape_behavior.weight_new,
            "sig_middle": self.escape_behavior.sig_middle,
            "sig_steepness": self.escape_behavior.sig_steepness,
        }
    
    def get_E_params(
            self,
            aversive_internal_state,
            urge_to_escape,
    ):
        """
        Gathers all external strategy parameters into a dictionary.
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
        Gathers all internal strategy parameters into a dictionary.
        """
        return {
            "A": aversive_internal_state,
            "U": urge_to_escape,
            "feedback": self.external_strategy.feedback,
            "carrying_capacity": self.external_strategy.carrying_capacity,
            "A_weight": self.external_strategy.A_weight,
            "U_weight": self.external_strategy.U_weight,
        }