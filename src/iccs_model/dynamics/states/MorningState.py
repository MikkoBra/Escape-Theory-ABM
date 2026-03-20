from iccs_model.dynamics.states.AbstractState import State
from errors.StateError import PreviousStateError
from Constants import Constants
import numpy as np



class MorningState(State):
    """
    State implementation representing a morning ritual.
    """
    STATE_NAME = "morning"

    def __init__(self):
        super().__init__()
        self.sleep = 0
    
    def generate_time(self, time, prev_state, state_params):
        # Check if last state was the correct one
        if prev_state is not None \
          and prev_state.to_string() not in self.preceding_states():
            raise PreviousStateError(self, prev_state)
        self.sleep = prev_state.state_length
        
        # Compute end time in minutes
        self._start_time = time
        last_midnight = time - (time % Constants.DAY_LENGTH)
        self.end_time = last_midnight + (Constants.WORK_TIME - state_params.commute)
        self.state_length = self.end_time - time
        self.time_left = self.state_length
    
    def modify_parameters(self, params):
        # Modify stress and suicidal thought
        # based on shortage of sleep
        sleep_deficit = max(0.0, (Constants.HEALTHY_SLEEP - self.sleep) / Constants.HEALTHY_SLEEP)
        morning_impulse = params.stress.impulse_strength * (1 + np.exp(sleep_deficit/Constants.HEALTHY_SLEEP))
        params.set_stress_params(morning_impulse=morning_impulse, impulse_rate=0)
        new_T_threshold = params.suicidal_thought.sig_middle - 0.1 * sleep_deficit/Constants.HEALTHY_SLEEP
        params.set_suicidal_thought_params(sig_middle=new_T_threshold)

    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["sleep"])
    
    def following_state(self):
        return "commute"