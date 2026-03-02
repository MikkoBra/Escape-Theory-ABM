from model.dynamics.states.SleepState import SleepState
from model.dynamics.states.MorningState import MorningState
from model.dynamics.states.CommuteState import CommuteState
from model.dynamics.states.WorkState import WorkState
from model.dynamics.states.HomeState import HomeState

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
