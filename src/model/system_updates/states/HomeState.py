from model.system_updates.states.AbstractState import State
from errors.StateError import PreviousStateError
from Constants import Constants
import numpy as np

class HomeState(State):
    """
    State implementation representing being at home doing nothing.
    """
    STATE_NAME = "home"

    def __init__(self):
        super().__init__()
    
    def generate_time(self, time, prev_state, state_params):
        # Check if last state was the correct one
        if prev_state is not None \
          and prev_state.to_string() not in self.preceding_states():
            raise PreviousStateError(self, prev_state)
        
        # Compute end time in minutes
        self._start_time = time
        mean_sleep = state_params.mean_sleep
        sigma_sleep = state_params.sigma_sleep
        sleep_hours = max(0, np.random.normal(mean_sleep, sigma_sleep))
        sleep_length = sleep_hours * Constants.DAY_LENGTH * (1/24)
        
        time_of_day = time % Constants.DAY_LENGTH
        wake_time = (time + (Constants.DAY_LENGTH - time_of_day)) + Constants.WAKE_TIME
        self.end_time = wake_time - sleep_length
        self.state_length = self.end_time - time
        self.time_left = self.state_length
    
    def modify_parameters(self, params):
        # Reset from last state
        params.set_defaults()

        # Escape behavior is easier
        new_middle = max(params.escape_behavior.sig_middle - 0.05, 0)
        params.set_suicidal_thought_params(sig_middle=new_middle)

        # New suicidal thoughts are weighted heavier
        updated_weight = min(params.suicidal_thought.weight_new + 0.1, 1)
        params.set_suicidal_thought_params(weight_new=updated_weight)
    
    def to_string(self):
        return self.STATE_NAME
    
    def preceding_states(self):
        return np.array(["commute"])
    
    def following_state(self):
        return "sleep"