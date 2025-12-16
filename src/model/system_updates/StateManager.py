from model.system_updates.state_registry import get_state


class StateManager():
    """
    Manages the transition from one state to another.
    """
    def __init__(self, state_params):
        self._state = None
        self._state_params = state_params
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, state):
        self._state = state
    
    @property
    def state_params(self):
        return self._state_params
    
    @state.setter
    def state_params(self, state_params):
        self._state_params = state_params
    
    def update_state(self, dt, time, agent_params):
        self._state.pass_time(dt)
        if self._state.time_left <= 0:
            next_state_name = self._state.following_state()
            next_state_class = get_state(next_state_name)
            next_state = next_state_class()
            next_state.generate_time(time, self._state, self._state_params)
            next_state.modify_parameters(agent_params)
            next_state.last_state = self._state
            self._state = next_state
