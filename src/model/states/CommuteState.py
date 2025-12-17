from model.states.AbstractState import State
from errors.StateError import PreviousStateError, NextStateError
from Constants import Constants
import numpy as np


class CommuteState(State):
    """
    State implementation representing a commute.
    """
    STATE_NAME = "commute"

    def __init__(self):
        super().__init__()
    
    def generate_time(self, time, prev_state, state_params):
        # Check if last state was the correct one
        if prev_state is not None \
          and prev_state.to_string() not in self.preceding_states():
            raise PreviousStateError(self, prev_state)
        # Save previous state for determining next state later
        self._prev_state = prev_state
        
        # Commute is constant throughout simulation
        self._start_time = time
        self.state_length = state_params.commute
        self.end_time = time + state_params.commute
        self.time_left = state_params.commute
    
    def modify_parameters(self, params):
        # Reset from last state
        params.set_defaults()

        # Increase mean stress
        new_s_mean = max(params.stress.mean + 0.2, 1)
        params.set_stress_params(mean=new_s_mean)
        
        # Escape behavior is impossible
        # params.set_escape_behavior_params(sig_middle=1.1)
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["morning", "work"])
    
    def following_state(self):
        if self._prev_state.to_string() == "morning":
            return "work"
        elif self._prev_state.to_string() == "work":
            return "home"
        else: raise NextStateError(self)
