class PreviousStateError(TypeError):
    def __init__(self, curr_state, prev_state):
        message = f"Tried starting {curr_state.to_string()}, previous state was of wrong type: {type(prev_state)}"
        super().__init__(message)

class NextStateError(TypeError):
    def __init__(self, curr_state):
        message = f"Tried finding next state after {curr_state.to_string()}, but previous state was of wrong type."
        super().__init__(message)