import mesa
import numpy as np
import random
from iccs_model.networked_agents.networks.NetworkFactory import NetworkFactory
from iccs_model.dynamics.state_registry import register_all_states
from iccs_model.dynamics.AgentUpdaterNumba import NumbaAgentUpdater, compute_connectedness_vectorized
from iccs_model.networked_agents.agents.NewsStation import NewsStation
from iccs_model.networked_agents.agents.NewsSignalManager import NewsSignalManager


class NetworkedModel(mesa.Model):
    """
    Optimized agent-based model using Numba for vectorized operations.
    Processes all agents simultaneously instead of iterating.
    """
    
    def __init__(self, dt, seed=42, parameters={}, verbose=False, collect_all=False, warmup=0, num_agents=0):
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.warmup = warmup
        self.verbose = verbose
        self.dt = dt
        self.time = 0
        self.num_agents = num_agents
        self.num_steps = parameters.get("num_steps")
        if self.num_steps is None:
            raise ValueError("parameters must include 'num_steps'")
        
        register_all_states()
        
        # Initialize news signal manager
        self.news_signal_manager = NewsSignalManager()
        
        # Initialize news stations if specified
        num_news_stations = parameters.get("news_stations", 0)
        self.news_stations = []
        if num_news_stations > 0:
            for i in range(1, num_news_stations + 1):
                station = NewsStation(
                    station_id=i,
                    news_intensity=parameters.get("news_intensity", 0.5)
                )
                station.signal_manager = self.news_signal_manager
                self.news_stations.append(station)
        
        # Generate network
        NetworkFactory().create_network(model=self, parameters=parameters)
        
        # Assign agents to news stations based on opinion similarity
        if self.news_stations:
            proportion_consumers = parameters.get("proportion_consumers", 0.5)
            self._assign_agents_to_news_stations(proportion_consumers)
        
        # Initialize updater
        self.updater = NumbaAgentUpdater()
        
        # Build neighbor data structures for vectorized operations
        self._build_neighbor_structures()
        
        # Initialize all agent states as numpy arrays
        self._initialize_agent_states()
        
        # Initialize parameters
        self._initialize_parameters(parameters)
        
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
        ]
        
        # Preallocate data arrays
        self.data = {}
        for var in self.tracked_vars:
            self.data[var] = np.zeros((self.num_steps, self.num_agents), dtype=np.float32)
    
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
    
    def _assign_agents_to_news_stations(self, proportion_consumers):
        """
        Assign agents to news stations based on opinion similarity.
        Each agent is assigned to the news station with the closest opinion_scalar.
        
        Note: This method assumes agents have been created and have opinion_scalar values.
        Since this is a vectorized model, we need to work with agent arrays.
        """
        n_consumers = int(self.num_agents * proportion_consumers)

        # Randomly select consumers (no replacement)
        self.consumes_news = np.zeros(self.num_agents, dtype=np.bool_)
        consumer_ids = np.random.choice(self.num_agents, size=n_consumers, replace=False)
        self.consumes_news[consumer_ids] = True

        # Initialize full mapping with -1 (means: no station / non-consumer)
        self.agent_news_stations = np.full(self.num_agents, -1, dtype=np.int32)

        # Get station opinions
        station_opinions = np.array([station.opinion_scalar for station in self.news_stations])

        # Ensure agent opinions exist
        if not hasattr(self, 'agent_opinion_scalars'):
            self.agent_opinion_scalars = np.random.uniform(0, 1, self.num_agents)

        # Assign ONLY selected consumers
        for agent_id in consumer_ids:
            agent_opinion = self.agent_opinion_scalars[agent_id]

            distances = np.abs(station_opinions - agent_opinion)
            closest_station_idx = np.argmin(distances)

            self.agent_news_stations[agent_id] = self.news_stations[closest_station_idx].station_id
            self.news_stations[closest_station_idx].add_agent(agent_id)
    
    
    def _initialize_agent_states(self):
        """
        Initialize all agent state variables as contiguous numpy arrays.
        """
        self.agent_states = {
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
            'clustering_coefficient': np.zeros(self.num_agents, dtype=np.float32),
            'total_time': np.zeros(self.num_agents, dtype=np.float32),
        }
        
        # Initialize agent characteristics for news influence
        self.agent_ages = np.random.uniform(18, 80, self.num_agents)
        self.agent_celebrity_agreement = np.random.uniform(0, 1, self.num_agents)
        self.agent_vulnerability = np.random.uniform(0, 1, self.num_agents)
        self.agent_opinion_scalars = np.random.uniform(0, 1, self.num_agents)
        
        # Compute connectedness once at initialization
        self.agent_states['connectedness'] = compute_connectedness_vectorized(
            self.neighbor_data,
            self.neighbor_counts,
            self.neighbor_offsets
        )
    
    def _initialize_parameters(self, parameters):
        """
        Initialize parameter arrays. Each parameter is stored as a scalar or array.
        For per-agent variation, use arrays. For global params, use scalars.
        """
        # Extract default parameter values
        self.params = {
            'news_signal': {
                'decay': parameters.get('news_decay', 3),
                'diffusion_rate': parameters.get('news_diffusion', 0.05),
            },

            'stress': {
                'baseline': parameters.get('stress_baseline', 0.2),
                'decay': parameters.get('stress_decay', 3),
                'impulse_rate': parameters.get('impulse_rate', 4),
                'impulse_strength': parameters.get('impulse_strength', 0.1),
                'morning_impulse': np.full(self.num_agents, 0, dtype=np.float32),
                'alpha': parameters.get('stress_alpha', 0.1),
                'beta': parameters.get('stress_beta', 0.1),
                'gamma': parameters.get('stress_gamma', 0.1),
                'sigma': parameters.get('stress_sigma', 0.12),
            },

            'aversion': {
                'feedback': parameters.get('aversion_feedback', 4),
                'carrying_capacity': parameters.get('aversion_capacity', 0.1),
                'S_weight': parameters.get('S_weight', 2),
                'T_weight': parameters.get('T_weight', 0.1),
                'X_weight': parameters.get('X_weight', 1),
                'I_weight': parameters.get('I_weight', 0.5),
                'B_weight': parameters.get('B_weight', 2.5),
                'c_weight': parameters.get('c_weight', 0.7),
            },

            'urge': {
                'feedback': parameters.get('urge_feedback', 2.5),
                'A_weight': parameters.get('urge_A_weight', 1.75),
                'M_weight': parameters.get('urge_M_weight', 0.1),
                'C_weight': parameters.get('urge_C_weight', 0.1),
            },

            'suicide_history': {
                'decay': parameters.get('history_decay', 0.7),
            },

            'suicidal_thought': {
                'feedback': parameters.get('thought_feedback', 3),
                'sig_middle': parameters.get('thought_sig_middle', 0.35),
                'sig_steepness': parameters.get('thought_sig_steepness', 100),
            },

            'escape_behavior': {
                'feedback': parameters.get('behavior_feedback', 3),
                'sig_middle': parameters.get('behavior_sig_middle', 0.3),
                'sig_steepness': parameters.get('behavior_sig_steepness', 50),
            },

            'external': {
                'feedback': parameters.get('external_feedback', 0.5),
                'carrying_capacity': parameters.get('external_capacity', 0.1),
                'A_weight': parameters.get('external_A_weight', 0.41),
                'U_weight': parameters.get('external_U_weight', 0.45),
            },

            'internal': {
                'feedback': parameters.get('internal_feedback', 3),
                'carrying_capacity': parameters.get('internal_capacity', 0.05),
                'A_weight': parameters.get('internal_A_weight', 0.65),
                'U_weight': parameters.get('internal_U_weight', 0.75),
            },

            'burden': {
                'feedback': parameters.get('burden_feedback', 0.35),
                'A_weight': parameters.get('burden_A_weight', 0.25),
                'I_weight': parameters.get('burden_I_weight', 0.1),
                'B_lonely': parameters.get('B_lonely', 1),
            }
        }
    
    def collect_data(self):
        """
        Copy current agent states directly into preallocated arrays.
        Uses vectorized assignment for speed.
        """
        t = int(self.time / self.dt)
        
        for var in self.tracked_vars:
            self.data[var][t, :] = self.agent_states[var]
    
    def step(self):
        """
        Performs one timestep of the model using vectorized operations.
        """
        if self.time >= self.warmup:
            self.collect_data()
        
        # Check for news signal at t = 10
        if self.news_stations and self.news_signal_manager.send_signal(current_timestep=self.time):
            self._apply_news_effects()
        
        # Update all agents simultaneously
        self.agent_states = self.updater.update_all_agents(
            self.agent_states,
            self.params,
            self.dt,
            self.neighbor_data,
            self.neighbor_counts,
            self.neighbor_offsets
        )
        
        # Update time
        self.agent_states['total_time'] += self.dt
        self.time += self.dt
        
        # Reset morning impulse after it's been used
        # (This would normally be handled by state manager)
        if np.any(self.params['stress']['morning_impulse'] > 0):
            self.params['stress']['morning_impulse'][:] = 0
    
    def _apply_news_effects(self):
        """
        Apply news station effects to their assigned agents.
        This modifies the parameter arrays based on agent characteristics.
        """
        for station in self.news_stations:
            for agent_id in station.agents:
                # Get agent characteristics
                age = self.agent_ages[agent_id]
                celebrity_agreement = self.agent_celebrity_agreement[agent_id]
                vulnerability = self.agent_vulnerability[agent_id]
                
                # Calculate effect multiplier
                age_factor = 1.0 - (age / 100.0) if age <= 100 else 0.5
                agreement_factor = celebrity_agreement
                vulnerability_factor = vulnerability
                
                base_effect = station.news_intensity
                news_signal = base_effect * (1 + age_factor + agreement_factor + vulnerability_factor) / 4.0
                self.agent_states['news_signal'][agent_id] = news_signal
                