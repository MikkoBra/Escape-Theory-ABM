from abc import ABC, abstractmethod
from model.parameters.AbstractParameters import Parameters
from model.system_updates.StateParameters import StateParameters

class State(ABC):
    """
    Abstract class containing to-be-implemented functionality
    of State objects.
    """

    def __init__(self):
        self._start_time = 0
        self._end_time = 0
        self._time_left = 0
        self._state_length = 0
        self._last_state = None

    @property
    def start_time(self):
        return self._start_time
    
    @start_time.setter
    def start_time(self, time):
        self._start_time = time

    @property
    def end_time(self):
        return self._end_time
    
    @end_time.setter
    def end_time(self, time):
        self._end_time = time

    @property
    def time_left(self):
        return self._time_left
    
    @time_left.setter
    def time_left(self, time):
        self._time_left = time

    @property
    def state_length(self):
        return self._state_length
    
    @state_length.setter
    def state_length(self, time):
        self._state_length = time

    @property
    def last_state(self):
        return self._last_state
    
    @last_state.setter
    def last_state(self, state):
        self._last_state = state
    
    def pass_time(self, dt):
        self._time_left -= dt

    @abstractmethod
    def generate_time(self, time, prev_state, state_params):
        """
        Generates the time length of the state.

        Parameters
        ----------
        time: int
            Current time in steps.
        prev_state: State
            Previous state, used for knowledge of when the last
            state ended.
        state_params: StateParameters
            Parameters to use in time calculations.
        """
        pass

    @abstractmethod
    def modify_parameters(self, params: Parameters):
        """
        Modifies a Parameters object according to the current
        state's rules.

        Parameters
        ----------
        params: Parameters
            Parameters object containing the agent's current
            parameters for each evolution equation.
        """
        pass

    @abstractmethod
    def to_string(self):
        """
        Gives a string representation of the current state.
        """
        pass

    @abstractmethod
    def preceding_states(self):
        """
        Returns the allowed classes of the preceding state.
        """
        pass

    @abstractmethod
    def following_state(self):
        """
        Returns the class of the following state.
        """
        pass
