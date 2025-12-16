from model.system_updates.states.AbstractState import State
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
    
    def generate_time(self, time, prev_state, state_params):
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
        params.set_defaults()

        # Increase social influence
        new_F_w = params.aversion.F_weight + 2
        new_B_w = params.aversion.B_weight + 2
        params.set_aversion_params(F_weight=new_F_w, B_weight=new_B_w)

        # Increase urge to escape feedback
        new_feedback = params.urge_to_escape.feedback + 2
        params.set_urge_to_escape_params(feedback=new_feedback)
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["commute"])
    
    def following_state(self):
        return "commute"