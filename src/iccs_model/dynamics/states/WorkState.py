from iccs_model.dynamics.states.AbstractState import State
from errors.StateError import PreviousStateError
from Constants import Constants
import numpy as np

class WorkState(State):
    """
    State implementation representing a workday.
    """
    STATE_NAME = "work"

    def __init__(self):
        super().__init__()
    
    def generate_time(self, time, prev_state, state_params, is_weekend=False):
        # Check if last state was the correct one
        if prev_state is not None \
          and prev_state.to_string() not in self.preceding_states():
            raise PreviousStateError(self, prev_state)
        
        self._start_time = time
        self.state_length = Constants.WORKDAY_LENGTH
        self.end_time = time + Constants.WORKDAY_LENGTH
        self.time_left = Constants.WORKDAY_LENGTH
    
    def modify_parameters(self, params):
        # Reset from last state
        params.set_default_stress_coefficients()
        params.set_default_escape_behavior_coefficients()

        # # Increase urge to escape feedback
        new_escape_weight_E = params.external_strategy.U_weight + 0.3
        params.set_external_strategy_coefficients(U_weight=new_escape_weight_E)
        new_escape_weight_I = params.internal_strategy.U_weight + 0.3
        params.set_internal_strategy_coefficients(U_weight=new_escape_weight_I)
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["commute"])
    
    def following_state(self, is_weekend=False):
        return "commute"