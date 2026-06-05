import mesa
import numpy as np
import random
from numba import njit, prange
from networked_model.networks.NetworkFactory import NetworkFactory
from networked_model.dynamics.AgentUpdater import AgentUpdater
from networked_model.agents.NewsStation import NewsStation
from networked_model.agents.NewsSignalManager import NewsSignalManager
from networked_model.dynamics.ScheduleManager import ScheduleManager
from networked_model.agents.AgentFactory import AgentFactory
from Constants import Constants
from networked_model.networks.util import compute_connectedness_vectorized


SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4


STATE_DICT = {
    SLEEP:   "sleep",
    MORNING: "morning",
    COMMUTE: "commute",
    WORK:    "work",
    HOME:    "home",
}


class NetworkedModel(mesa.Model):
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
        self.num_agents = num_agents if num_agents != 0 else sum(parameters['agent_types'].values())
        self.num_steps = parameters.get("num_steps")
        if self.num_steps is None:
            raise ValueError("parameters must include 'num_steps'")
        
        # Generate network
        NetworkFactory().create_network(model=self, parameters=parameters)
        # Ensure that number of agents corresponds with number of network nodes
        if self.num_agents != len(self.network.nodes):
            print(f"Passed num_agents: {num_agents}, number of nodes in network: {len(self.network.nodes)}, setting num_agents to number of nodes.")
            self.num_agents = len(self.network.nodes)


        self.constants = {}

        # Randomly assign agent types to indices (= network ids of nodes)
        # If number of defined agents is less than the size of the network,
        # default agents are created until the number of nodes is satisfied.
        if "agent_types" not in parameters:
            parameters["agent_types"] = {"default": self.num_agents}
        elif sum(parameters["agent_types"].values()) < self.num_agents:
            print(f'Passed number of agents in agent_types: {sum(parameters["agent_types"].values())}, num_agents: {self.num_agents}, adding more default agents.')
            if "default" not in parameters["agent_types"]:
                parameters["agent_types"]["default"] = 0
            parameters["agent_types"]["default"] += self.num_agents - sum(parameters["agent_types"].values())
        labels = [agent_type
                  for agent_type, amount in parameters["agent_types"].items()
                  for _ in range(amount)]
        random.shuffle(labels)
        self.constants["labels"] = labels

        # Define coefficient values and agent attributes for each agent type
        agent_factory = AgentFactory()
        for agent_type in parameters["agent_types"].keys():
            self.constants[agent_type] = {}
            self.constants[agent_type]["coefficients"] = agent_factory.initialize_coefficients(agent_type)
            self.constants[agent_type]["agent_characteristics"] = agent_factory.initialize_characteristics(agent_type)
        self._build_coefficient_arrays(parameters["agent_types"].keys())
        self._build_agent_characteristic_arrays(parameters["agent_types"].keys())

        # Build neighbor data structures
        self._build_neighbor_structures()
        
        # Compute connectedness and clustering coefficient once at initialization
        self.constants['connectedness'] = compute_connectedness_vectorized(
            self.neighbor_data,
            self.neighbor_counts,
            self.neighbor_offsets
        )
        self.constants['clustering_coefficient'] = self.network.compute_clustering_coefficients()

        # Set opinion scalars for preferential attachment
        self.constants['opinion_scalar'] = np.random.rand(num_agents)


        # Commute can be made variable within agent types but is unchanging
        self.constants['commute'] = self.compute_commute_duration(self.constants)
        
        # Set up state representation dictionary (index = agent id)
        self.states = {
            'stress': np.zeros(self.num_agents, dtype=np.float32),
            'aversive_internal_state': np.zeros(self.num_agents, dtype=np.float32),
            'urge_to_escape': np.zeros(self.num_agents, dtype=np.float32),
            'suicide_history': np.zeros(self.num_agents, dtype=np.float32),
            'suicidal_thought': np.zeros(self.num_agents, dtype=np.float32),
            'escape_behavior': np.zeros(self.num_agents, dtype=np.float32),
            'external_strat': np.zeros(self.num_agents, dtype=np.float32),
            'internal_strat': np.zeros(self.num_agents, dtype=np.float32),
            'burdensomeness': np.zeros(self.num_agents, dtype=np.float32),
            'news_signal': np.zeros(self.num_agents, dtype=np.float32),
            'total_time': np.zeros(self.num_agents, dtype=np.float32),
        }


        # Track agents' schedules
        self.schedule = {
            'state': np.zeros(self.num_agents, dtype=np.int32),
            'time_left': np.zeros(self.num_agents, dtype=np.float32)
        }

        self.schedule_manager = ScheduleManager()
        self.schedule = self.schedule_manager.init_schedule(self.num_agents, Constants.WAKE_TIME)


        # Initialize news signal manager
        self.news_signal_manager = NewsSignalManager()
        
        # Initialize news stations if specified
        num_news_stations = parameters.get("news_stations", 0)
        self.news_stations = []
        if num_news_stations > 0:
            intensities = parameters.get("news_intensity", np.full(num_news_stations, 0.5))
            for i in range(0, num_news_stations):
                intensity = intensities[i]
                station = NewsStation(
                    station_id=i,
                    news_intensity=intensity
                )
                station.signal_manager = self.news_signal_manager
                self.news_stations.append(station)
        
        # Assign agents to news stations based on opinion similarity
        if self.news_stations:
            proportion_consumers = parameters.get("proportion_consumers", 0.5)
            self._assign_agents_to_news_stations(proportion_consumers)
        
        
        # Initialize updater
        self.updater = AgentUpdater()
        
        # Tracked variables
        self.tracked_vars = [
            "stress",
            "aversive_internal_state",
            "urge_to_escape",
            "suicidal_thought",
            "escape_behavior",
            "external_strat",
            "internal_strat",
            "burdensomeness",
            "suicide_history",
            "news_signal",
            "state",
        ]
        
        # Preallocate data arrays
        self.data = {}
        for var in self.tracked_vars:
            if var == "state":
                self.data[var] = np.zeros((self.num_steps, self.num_agents), dtype=np.int32)
            else:
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

    
    def _build_neighbor_structures(self):
        """
        Build flattened neighbor data structures for efficient Numba processing.
        """
        net = self.network
        
        # Count neighbors for each node
        self.neighbor_counts = np.zeros(self.num_agents, dtype=np.int32)
        for i, node in enumerate(self.network.nodes):
            self.neighbor_counts[i] = len(net.adjacency[node])
        
        # Calculate offsets
        self.neighbor_offsets = np.zeros(self.num_agents, dtype=np.int32)
        cumsum = 0
        for i in range(self.num_agents):
            self.neighbor_offsets[i] = cumsum
            cumsum += self.neighbor_counts[i]
        
        # Flatten neighbor data: [neighbor_id, weight] pairs
        total_neighbors = cumsum
        self.neighbor_data = np.zeros((total_neighbors, 2), dtype=np.float32)
        
        for i, node in enumerate(self.network.nodes):
            neighbors = net.adjacency[node]
            offset = self.neighbor_offsets[i]
            
            for j, (neighbor, weight) in enumerate(neighbors.items()):
                # Find the index of this neighbor in the nodes array
                neighbor_idx = np.where(self.network.nodes == neighbor)[0][0]
                self.neighbor_data[offset + j, 0] = neighbor_idx
                self.neighbor_data[offset + j, 1] = weight

    
    def compute_commute_duration(self, constants):
        """
        Define duration of commute for each agent separately.
        Set to a default duration if 'commute' exists in constants[agent_type],
        otherwise drawn from a lognormal distribution using 'mean_commute'
        and 'sigma_commute'
        """
        commutes = np.zeros_like(constants["labels"])
        for i, agent_type in enumerate(constants["labels"]):
            if 'commute' in constants[agent_type]["agent_characteristics"]:
                commutes[i] = constants[agent_type]["agent_characteristics"]['commute']
            else:
                mean = constants[agent_type]["agent_characteristics"]['mean_commute']
                sigma = constants[agent_type]["agent_characteristics"]['sigma_commute']
                commute_len = np.random.lognormal(mean=mean, sigma=sigma)
                commutes[i] = min(commute_len * Constants.DAY_LENGTH * (1/24),
                            1.5 * Constants.DAY_LENGTH * (1/24))
        return commutes

    
    def _assign_agents_to_news_stations(self, proportion_consumers):
        """
        Assign agents to news stations based on opinion similarity.
        Each agent is assigned to the news station with the closest opinion_scalar.
        """
        n_consumers = int(self.num_agents * proportion_consumers)

        # Randomly select consumers (no replacement)
        consumer_ids = np.random.choice(self.num_agents, size=n_consumers, replace=False)

        # Initialize full mapping with -1 (means: no station / non-consumer)
        self.constants['news_station_mapping'] = np.full(self.num_agents, -1, dtype=np.int32)

        # Get station opinions
        station_opinions = np.array([station.opinion_scalar for station in self.news_stations])

        # Assign ONLY selected consumers
        for agent_id in consumer_ids:
            agent_opinion = self.constants['opinion_scalar'][agent_id]

            distances = np.abs(station_opinions - agent_opinion)
            closest_station_idx = np.argmin(distances)

            self.constants['news_station_mapping'][agent_id] = self.news_stations[closest_station_idx].station_id
            # self.news_stations[closest_station_idx].add_agent(agent_id)


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
        
        self.schedule['state'], self.schedule['time_left'] = self.schedule_manager.update_states(
            dt=self.dt,
            time=self.time,
            states=self.schedule['state'],
            time_left=self.schedule['time_left'],
            constants=self.constants,
            )
        
        # Check for news signal at t = 10
        if self.news_stations and self.news_signal_manager.send_signal(current_time=self.time, warmup=self.warmup):
            self._apply_news_effects()
        
        # Update all agents simultaneously
        self.states = self.updater.update_all_agents(
            self.states,
            self.constants,
            self.dt,
            self.neighbor_data,
            self.neighbor_counts,
            self.neighbor_offsets
        )
        
        # Update time
        self.states['total_time'] += self.dt
        self.time += self.dt
    
    def _apply_news_effects(self):
        """
        Apply news station effects to their assigned agents.
        This modifies the parameter arrays based on agent characteristics.
        """
        for i, label in enumerate(self.constants['label']):
            station_idx = self.constants['news_station_mapping'][i]
            if station_idx != -1 and self.constants[label]['agent_characteristics']['consumes_news']:
                station = self.news_stations[station_idx]
                impulsivity = self.constants[label]['agent_characteristics']['impulsivity']
                base_effect = station.news_intensity
                news_signal = base_effect * (1 + impulsivity)
                self.agent_states['news_signal'][i] = news_signal
                