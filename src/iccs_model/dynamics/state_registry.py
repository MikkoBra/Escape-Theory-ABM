from iccs_model.dynamics.states.SleepState import SleepState
from iccs_model.dynamics.states.MorningState import MorningState
from iccs_model.dynamics.states.CommuteState import CommuteState
from iccs_model.dynamics.states.WorkState import WorkState
from iccs_model.dynamics.states.HomeState import HomeState

STATE_REGISTRY = {}

def get_state(name):
    return STATE_REGISTRY[name]

def register_all_states():
    STATE_REGISTRY.update({
        "sleep": SleepState,
        "morning": MorningState,
        "commute": CommuteState,
        "work": WorkState,
        "home": HomeState
    })
