from model.system_updates.states.SleepState import SleepState
from model.system_updates.states.MorningState import MorningState
from model.system_updates.states.CommuteState import CommuteState
from model.system_updates.states.WorkState import WorkState
from model.system_updates.states.HomeState import HomeState

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
