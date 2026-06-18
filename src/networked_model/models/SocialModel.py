from networked_model.models.AbstractModel import Model
from Constants import Constants
import numpy as np
from networked_model.dynamics.AgentUpdater import AgentUpdater
from networked_model.dynamics.ScheduleManager import ScheduleManager


class SocialModel(Model):
    """
    Model with settings suited for analysis of the
    differences between a baseline Wang et al. agent,
    an agent with a default schedule, and agents that
    sleep consistent durations.
    """
    def __init__(self, num_agents=4000, sim_length=50, dt=Constants.MINUTE_LENGTH, seed=None, verbose=False, warmup=10,
                 parameters={
                     "agent_types": {
                         "baseline": 1000,
                         "default": 1000,
                         "bad_sleep": 1000,
                         "good_sleep": 1000,
                     },
                     "read_stress": False,
                     "randomize": True,
                 }):
        super().__init__(num_agents=num_agents, sim_length=sim_length, dt=dt, seed=seed, verbose=verbose, warmup=warmup)
        if "randomize" not in parameters:
            randomize = True
        else: randomize = parameters["randomize"]
        
        self.set_agent_type_labels(parameters, randomize=randomize)
        self.set_coeffs_and_characteristics(parameters)

        self.constants['connectedness'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['clustering_coefficient'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['opinion_scalar'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['commute'] = self.compute_commute_durations()

        # Set up state representation dictionary (index = agent id)
        if "read_stress" in parameters and parameters["read_stress"]:
            self.read_stress = True
            stress = np.load("src/output/stress_signal.npy")
            self.stress_signal = stress[:self.num_steps]
            temp_stress = stress[:self.num_steps][0]
        else:
            self.read_stress = False
            temp_stress = np.zeros(self.num_agents, dtype=np.float32)
        self.states = {
            'stress': np.full(self.num_agents, temp_stress, dtype=np.float32),
            'aversive_internal_state': np.zeros(self.num_agents, dtype=np.float32),
            'urge_to_escape': np.zeros(self.num_agents, dtype=np.float32),
            'suicidal_thought': np.zeros(self.num_agents, dtype=np.float32),
            'escape_behavior': np.zeros(self.num_agents, dtype=np.float32),
            'external_strat': np.zeros(self.num_agents, dtype=np.float32),
            'internal_strat': np.zeros(self.num_agents, dtype=np.float32),
            "suicide_history": np.zeros(self.num_agents, dtype=np.float32),
            "morning_impulse": np.zeros(self.num_agents, dtype=np.float32),
            'prev_sleep': np.zeros(self.num_agents, dtype=np.float32),
            'total_time': np.zeros(self.num_agents, dtype=np.float32),
        }

        self.updater = AgentUpdater(seed=seed)
        
        # Track agents' schedules
        self.schedule = {
            'state': np.zeros(self.num_agents, dtype=np.int32),
            'time_left': np.zeros(self.num_agents, dtype=np.float32)
        }

        self.schedule_manager = ScheduleManager()
        self.schedule = self.schedule_manager.init_schedule(self.num_agents, Constants.WAKE_TIME)

        # Tracked variables
        self.tracked_vars = [
            "stress",
            "aversive_internal_state",
            "urge_to_escape",
            "suicidal_thought",
            "escape_behavior",
            "external_strat",
            "internal_strat",
            "suicide_history",
            "total_time",
            "labels",
            "state",
        ]
        
        # Preallocate data arrays
        self.data = {}
        for var in self.tracked_vars:
            if var == "labels":
                self.data["labels"] = self.constants['labels']
            else:
                self.data[var] = np.zeros((self.num_steps, self.num_agents), dtype=np.float32)
