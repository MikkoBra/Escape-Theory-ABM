from model.system_updates.states.AbstractState import State
from model.system_updates.states.MorningState import MorningState
from errors.StateError import PreviousStateError
from Constants import Constants
import numpy as np


class SleepState(State):
    """
    State implementation representing sleep.
    """
    STATE_NAME = "sleep"
    
    def generate_time(self, time, prev_state, state_params):
        # Check if last state was the correct one
        if prev_state is not None \
          and prev_state.to_string() not in self.preceding_states():
            raise PreviousStateError(self, prev_state)
        
        self._start_time = time
        if time != 0:
            # Compute end time
            time_of_day = time % Constants.DAY_LENGTH
            # Start after midnight, its >= 00:00
            if time_of_day < Constants.WAKE_TIME:
                wake_time = (time - time_of_day) + Constants.WAKE_TIME
            # Start before midnight, its <= 23:59
            else:
                wake_time = (time + (Constants.DAY_LENGTH - time_of_day)) + Constants.WAKE_TIME
            self.end_time = wake_time
        else:
            self.end_time = Constants.WAKE_TIME
        self.state_length = self.end_time - time
        self.time_left = self.state_length
    
    def modify_parameters(self, params):
        # Reset from last state
        params.set_defaults()
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["home"])
    
    def following_state(self):
        return "morning"
