from networked_model.models.AbstractModel import Model
from networked_model.dynamics.AgentUpdater import AgentUpdater
from Constants import Constants
import numpy as np


class MemoryModel(Model):
    """
    Model with settings suited for analysis of the
    differences between a baseline Wang et al. agent
    and an agent with memory of suicidal thought added.
    """
    def __init__(self, num_agents=2000, sim_length=50, dt=Constants.MINUTE_LENGTH, seed=None, verbose=False, warmup=10,
                 parameters={
                     "agent_types": {
                         "memory": 1000,
                         "baseline": 1000,
                     },
                     "read_stress": False,
                 }):
        super().__init__(num_agents=num_agents, sim_length=sim_length, dt=dt, seed=seed, verbose=verbose, warmup=warmup)
        self.set_agent_type_labels(parameters)
        self.set_coeffs_and_characteristics(parameters)

        self.constants['connectedness'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['clustering_coefficient'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['opinion_scalar'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['commute'] = np.zeros(num_agents, dtype=np.float32)

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
            'total_time': np.zeros(self.num_agents, dtype=np.float32),
        }
        
        
        # Initialize updater
        self.updater = AgentUpdater(seed=seed)
        
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
        ]
        
        # Preallocate data arrays
        self.data = {}
        for var in self.tracked_vars:
            if var == "labels":
                self.data["labels"] = self.constants['labels']
            else:
                self.data[var] = np.zeros((self.num_steps, self.num_agents), dtype=np.float32)
    

    def collect_data(self):
        """
        Copy current agent states directly into preallocated arrays.
        Uses vectorized assignment for speed.
        """
        t = int(self.time / self.dt)
        
        for var in self.tracked_vars:
            if var == "state":
                self.data["state"][t, :] = self.schedule["state"]
            elif var != "labels":
                self.data[var][t, :] = self.states[var]


    def step(self):
        """
        Performs one timestep of the model using vectorized operations.
        """
        if self.time >= self.warmup:
            self.collect_data()
        
        # Update all agents simultaneously
        if self.read_stress:
            t = int(self.time / self.dt)
            self.states['stress'][:] = self.stress_signal[t]
        self.states = self.updater.update_all_agents(
            self.states,
            self.constants,
            self.dt,
            neighbor_data=None,
            neighbor_counts=None,
            neighbor_offsets=None,
            read_stress=self.read_stress,
        )
        
        # Update time
        self.states['total_time'] += self.dt
        self.time += self.dt