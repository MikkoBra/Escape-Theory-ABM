from networked_model.models.AbstractModel import Model
import numpy as np
from networked_model.dynamics.AgentUpdater import AgentUpdater
from networked_model.networks.NetworkFactory import NetworkFactory
from networked_model.dynamics.ScheduleManager import ScheduleManager
from Constants import Constants
from networked_model.networks.util import compute_connectedness_vectorized
from networked_model.agents.NewsStation import NewsStation
from networked_model.dynamics.NewsSignalManager import NewsSignalManager
from numba import njit, prange


class NewsSpreadModel(Model):
    """
    Model with settings suited for analysis of the
    differences between agents with network effects.
    """
    def __init__(self, num_agents=4000, sim_length=10, dt=Constants.MINUTE_LENGTH, seed=None, verbose=False, warmup=0,
                 parameters={
                     "read_stress": False,
                     "randomize": True,
                     "only_sleep": False,
                     "network": "hk",
                     "m": 16,
                    'cluster_prob': 0.6,
                    "initial_attractiveness": 5,
                    "node_removal_rate": 0.1,
                    "edge_removal_prob": 0.5,
                    "hub_count": 2,
                    "hub_degree": 400,
                    "weighted_network": True,
                    "social_events": False,
                    "spread_news": True,
                    "low_degree_consumer": False,
                    "hub_consumer": False,
                 }):
        super().__init__(num_agents=num_agents, sim_length=sim_length, dt=dt, seed=seed, verbose=verbose, warmup=warmup)
        if "randomize" not in parameters:
            randomize = True
        else: randomize = parameters["randomize"]
        
        # Generate network
        self.weighted_network = parameters.get("weighted_network", False)
        NetworkFactory().create_network(model=self, parameters=parameters)
        # Ensure that number of agents corresponds with number of network nodes
        if self.num_agents != len(self.network.nodes):
            print(f"Passed num_agents: {num_agents}, number of nodes in network: {len(self.network.nodes)}, setting num_agents to number of nodes.")
            self.num_agents = len(self.network.nodes)

        self.set_agent_type_labels(parameters, randomize=randomize)
        self.set_coeffs_and_characteristics(parameters)
        self._build_neighbor_structures()

        self.constants['connectedness'] = compute_connectedness_vectorized(
            self.neighbor_data,
            self.neighbor_counts,
            self.neighbor_offsets
        )
        self.constants['clustering_coefficient'] = self.network.compute_clustering_coefficients()
        self.constants['commute'] = self.compute_commute_durations()

        self.social_events = parameters.get("social_events", False)
        if self.social_events:
            # Default: all agents participate
            self.constants['social_events_mask'] = np.ones(self.num_agents, dtype=bool)

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
            "burdensomeness": np.zeros(self.num_agents, dtype=np.float32),
            'prev_sleep': np.zeros(self.num_agents, dtype=np.float32),
            'total_time': np.zeros(self.num_agents, dtype=np.float32),
            'social_event_today': np.zeros(self.num_agents, dtype=np.float32),
            'home_time_remaining': np.zeros(self.num_agents, dtype=np.float32),
            'news_signal': np.zeros(self.num_agents, dtype=np.float32),
        }

        
        # Initialize news signal manager
        self.spread_news=parameters.get("spread_news", False)
        if self.spread_news:
            self.news_signal_manager = NewsSignalManager()
            self.constants['opinion_scalar'] = np.random.rand(num_agents)
            self.low_degree_consumer = parameters.get("low_degree_consumer", False)
            self.hub_consumer = parameters.get("hub_consumer", False)
            
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
        

        self.updater = AgentUpdater(seed=seed)
        
        # Track agents' schedules
        self.schedule = {
            'state': np.zeros(self.num_agents, dtype=np.int32),
            'time_left': np.zeros(self.num_agents, dtype=np.float32)
        }

        self.schedule_manager = ScheduleManager(constants=self.constants, social_events=self.social_events)
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
            "burdensomeness",
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

    
    def _assign_agents_to_news_stations(self, proportion_consumers):
        """
        Assign agents to news stations based on:
        - default: random selection
        - low_degree_consumer: lowest degree nodes selected
        - hub_consumer: highest degree nodes selected
        """
        n_consumers = int(self.num_agents * proportion_consumers)

        self.constants['news_station_mapping'] = np.full(
            self.num_agents, -1, dtype=np.int32
        )

        # --- compute degrees ---
        degree_values = np.array([len(self.network.adjacency[int(n)]) for n in self.network.nodes])

        # --- selection mode flags ---
        low_degree = self.low_degree_consumer
        hub_degree_flag = self.hub_consumer

        nodes = np.array(self.network.nodes)

        # --- choose consumer set ---
        if low_degree:
            consumer_ids = nodes[np.argsort(degree_values)[:n_consumers]]

        elif hub_degree_flag:
            consumer_ids = nodes[np.argsort(degree_values)[-n_consumers:]]

        else:
            consumer_ids = np.random.choice(
                self.num_agents,
                size=n_consumers,
                replace=False
            )

        # --- assign stations ---
        station_opinions = np.array([s.opinion_scalar for s in self.news_stations])

        for agent_id in consumer_ids:
            agent_id = int(agent_id)

            agent_opinion = self.constants['opinion_scalar'][agent_id]
            distances = np.abs(station_opinions - agent_opinion)

            closest_station_idx = np.argmin(distances)

            self.constants['news_station_mapping'][agent_id] = \
                self.news_stations[closest_station_idx].station_id
    
    def _apply_news_effects(self):
        """
        Apply news station effects to their assigned agents.
        This modifies the parameter arrays based on agent characteristics.
        """
        for i, label in enumerate(self.constants['labels']):
            station_idx = self.constants['news_station_mapping'][i]
            if station_idx != -1 and self.constants[label]['agent_characteristics']['consumes_news']:
                station = self.news_stations[station_idx]
                impulsivity = self.constants[label]['agent_characteristics']['impulsivity']
                age = self.constants[label]['agent_characteristics']['age']
                base_effect = station.news_intensity
                news_signal = base_effect * (1 + impulsivity + (100-age)/82)
                self.states['news_signal'][i] = news_signal

        
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
        self.schedule_manager.apply_transition_effects(
            self.schedule['state'],
            self.schedule['time_left'],
            self.states,
            self.constants
            )
        
        if self.news_stations and self.news_signal_manager.send_signal(current_time=self.time, warmup=self.warmup):
            self._apply_news_effects()
        
        # Update all agents simultaneously
        self.states = self.updater.update_all_agents(
            self.states,
            self.constants,
            self.dt,
            neighbor_data=self.neighbor_data,
            neighbor_counts=self.neighbor_counts,
            neighbor_offsets=self.neighbor_offsets,
            read_stress=self.read_stress,
        )
        
        # Update time
        self.states['total_time'] += self.dt
        self.time += self.dt