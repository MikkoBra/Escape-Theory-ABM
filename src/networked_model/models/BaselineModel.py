import mesa
import numpy as np
import random
from networked_model.dynamics.AgentUpdater import AgentUpdater
from networked_model.agents.AgentFactory import AgentFactory


class BaselineModel(mesa.Model):
    """
    Optimized agent-based model using Numba for vectorized operations.
    Processes all agents simultaneously instead of iterating.
    """
    
    def __init__(self, dt, seed=42, parameters={}, verbose=False, collect_all=False, warmup=0, num_agents=0):
        """
        Parameters
        ----------
        parameters: dict
            Contains following parameters:
            - News stations
                news_stations: int
                news_intensity: array with floats, length = news_stations
                proportion_consumers: float
            - Networks
                network: string
                m: int
                initial_attractiveness: float
                node_removal_rate: float
                edge_removal_prob: float
                cluster_prob: float
            - Agents and states
                agent_types: dict with type as key, amount as value
                
        """
        super().__init__(seed=seed)
        # Set seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize config values
        self.warmup = warmup
        self.verbose = verbose
        self.dt = dt
        self.time = 0
        self.num_agents = num_agents
        self.num_steps = parameters.get("num_steps")
        if self.num_steps is None:
            raise ValueError("parameters must include 'num_steps'")


        self.constants = {}

        # Define coefficient values and agent attributes
        agent_factory = AgentFactory()
        self.constants['labels'] = np.full(num_agents, 'baseline')
        self.constants['baseline'] = {}
        self.constants['baseline']["coefficients"] = agent_factory.initialize_coefficients('baseline')
        self.constants['baseline']["agent_characteristics"] = agent_factory.initialize_characteristics('baseline')
        self._build_coefficient_arrays(['baseline'])
        self._build_agent_characteristic_arrays(['baseline'])
        
        # Set all unused constants to 0
        self.constants['connectedness'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['clustering_coefficient'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['opinion_scalar'] = np.zeros(num_agents, dtype=np.float32)
        self.constants['commute'] = np.zeros(num_agents, dtype=np.float32)
        
        # Set up state representation dictionary (index = agent id)
        self.states = {
            'stress': np.zeros(self.num_agents, dtype=np.float32),
            'aversive_internal_state': np.zeros(self.num_agents, dtype=np.float32),
            'urge_to_escape': np.zeros(self.num_agents, dtype=np.float32),
            'suicidal_thought': np.zeros(self.num_agents, dtype=np.float32),
            'escape_behavior': np.zeros(self.num_agents, dtype=np.float32),
            'external_strat': np.zeros(self.num_agents, dtype=np.float32),
            'internal_strat': np.zeros(self.num_agents, dtype=np.float32),
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
        ]
        
        # Preallocate data arrays
        self.data = {}
        for var in self.tracked_vars:
            self.data[var] = np.zeros((self.num_steps, self.num_agents), dtype=np.float32)


    def _build_coefficient_arrays(self, agent_types):
        """
        Pre-expand all per-agent-type coefficients into flat numpy arrays
        of length num_agents, so Numba functions can index directly by agent id.
        """
        # Collect all coefficient keys across all agent types
        coeff_arrays = {}
        
        for agent_type in agent_types:
            coeffs = self.constants[agent_type]['coefficients']
            for state_key, state_coeffs in coeffs.items():
                for coeff_key, value in state_coeffs.items():
                    arr_key = f"{state_key}_{coeff_key}"  # e.g. "stress__decay"
                    if arr_key not in coeff_arrays:
                        coeff_arrays[arr_key] = np.zeros(self.num_agents, dtype=np.float32)

        # Fill arrays by iterating over agents
        for i in range(self.num_agents):
            label = self.constants['labels'][i]
            coeffs = self.constants[label]['coefficients']
            for state_key, state_coeffs in coeffs.items():
                for coeff_key, value in state_coeffs.items():
                    arr_key = f"{state_key}_{coeff_key}"
                    coeff_arrays[arr_key][i] = value

        self.constants['coeff_arrays'] = coeff_arrays


    def _build_agent_characteristic_arrays(self, agent_types):
        """
        Pre-expand all per-agent-type coefficients into flat numpy arrays
        of length num_agents, so Numba functions can index directly by agent id.
        """
        # Collect all coefficient keys across all agent types
        char_arrays = {}
        
        for agent_type in agent_types:
            chars = self.constants[agent_type]['agent_characteristics']
            for attribute_key in chars.keys():
                if attribute_key not in char_arrays:
                    char_arrays[attribute_key] = np.zeros(self.num_agents, dtype=np.float32)

        # Fill arrays by iterating over agents
        for i in range(self.num_agents):
            label = self.constants['labels'][i]
            chars = self.constants[label]['agent_characteristics']
            for attribute_key, attribute_val in chars.items():
                char_arrays[attribute_key][i] = attribute_val

        self.constants['char_arrays'] = char_arrays


    def collect_data(self):
        """
        Copy current agent states directly into preallocated arrays.
        Uses vectorized assignment for speed.
        """
        t = int(self.time / self.dt)
        
        for var in self.tracked_vars:
            if var == "state":
                self.data["state"][t, :] = self.schedule["state"]
            else:
                self.data[var][t, :] = self.states[var]
    
    def step(self):
        """
        Performs one timestep of the model using vectorized operations.
        """
        if self.time >= self.warmup:
            self.collect_data()
        
        # Update all agents simultaneously
        self.states = self.updater.update_all_agents(
            self.states,
            self.constants,
            self.dt,
            neighbor_data=None,
            neighbor_counts=None,
            neighbor_offsets=None,
        )
        
        # Update time
        self.states['total_time'] += self.dt
        self.time += self.dt
                