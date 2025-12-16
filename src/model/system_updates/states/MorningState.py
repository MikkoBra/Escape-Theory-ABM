from model.system_updates.states.AbstractState import State
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
        # Reset from last state
        params.set_defaults()

        # Modify stress, suicidal thought, and their weights on aversion
        # based on shortage of sleep
        sleep_deficit = max(0.0, (Constants.HEALTHY_SLEEP - self.sleep) / Constants.HEALTHY_SLEEP)
        new_s_mean = min(1.0, params.stress.mean + 0.3 * sleep_deficit)
        params.set_stress_params(mean=new_s_mean)
        new_t_mid = max(0.0, params.suicidal_thought.sig_middle - 0.2 * sleep_deficit)
        params.set_suicidal_thought_params(sig_middle=new_t_mid)
        new_s_weight = params.aversion.S_weight + 3 * sleep_deficit
        new_t_weight = params.aversion.T_weight + 0.5 * sleep_deficit
        params.set_aversion_params(S_weight=new_s_weight, T_weight=new_t_weight)
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["sleep"])
    
    def following_state(self):
        return "commute"