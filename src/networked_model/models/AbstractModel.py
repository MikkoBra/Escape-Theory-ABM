import mesa
import random
import numpy as np
from networked_model.networks.NetworkFactory import NetworkFactory
from networked_model.agents.AgentFactory import AgentFactory
from networked_model.networks.util import compute_connectedness_vectorized
from Constants import Constants


class Model(mesa.Model):
    def __init__(self, num_agents, sim_length, dt, seed=None, verbose=False, warmup=0):
        super().__init__(seed=seed)
        # Set seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.warmup = warmup
        self.verbose = verbose
        self.dt = dt
        self.num_agents = num_agents
        self.time = 0
        self.num_steps = int(sim_length/dt)
        self.constants = {}

    
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

    
    def generate_network(
        self,
        parameters={
        "network": "hk",
        'm': 16,
        'cluster_prob': 0.6,
        "initial_attractiveness": 15,
        "node_removal_rate": 0.2,
        "edge_removal_prob": 0.5,
        },
    ):
        # Generate network
        parameters['num_nodes'] = self.num_agents
        NetworkFactory().create_network(model=self, parameters=parameters)
        # If network was empirical ensure that num_agents corresponds with
        # number of nodes
        if self.num_agents != len(self.network.nodes):
            if self.verbose:
                print(f"Setting number of agents to number of nodes in empirical network: {len(self.network.nodes)}.")
            self.num_agents = len(self.network.nodes)
        self._build_neighbor_structures()
    
    
    def _check_agent_type_amounts(self, parameters={}):
        if "agent_types" not in parameters:
            parameters["agent_types"] = {"default": self.num_agents}
        
        # If number of agents in agent_types is less than the size of the network,
        # default agents are created until the number of nodes is satisfied.
        elif sum(parameters["agent_types"].values()) < self.num_agents:
            if self.verbose:
                print(f'Passed number of agents in agent_types: {sum(parameters["agent_types"].values())}, num_agents: {self.num_agents}, adding more default agents.')
            if "default" not in parameters["agent_types"]:
                parameters["agent_types"]["default"] = 0
            parameters["agent_types"]["default"] += self.num_agents - sum(parameters["agent_types"].values())
        
        elif sum(parameters["agent_types"].values()) > self.num_agents:
            if len(parameters["agent_types"].keys()) > 1:
                raise Exception(f"Too many agent types and too many of each agent type. Make sure the total does not exceed {self.num_agents}.")
            if self.verbose:
                print(f'Passed number of agents in agent_types: {sum(parameters["agent_types"].values())}, num_agents: {self.num_agents}, removing agents.')
            for key in parameters["agent_types"].keys():
                parameters["agent_types"][key] = self.num_agents

    

    def set_agent_type_labels(self, parameters={}, randomize=True):
        """
        Distributes defined agent types over available agent ids, and
        fills the rest with default agents.

        Parameters
        ----------
        parameters: dict
            Dictionary with settings related to agent types. Possible entries:
            - key: "agent_types",
              value: dictionary with the following information:
                - key: string representation of the agent (see AgentFactory),
                  value: amount of this agent to have in the simulation
        """
        self._check_agent_type_amounts(parameters)
        
        labels = [agent_type
                  for agent_type, amount in parameters["agent_types"].items()
                  for _ in range(amount)]
        # Randomly assign agent types to agent ids
        if len(parameters["agent_types"].keys()) > 1 and randomize:
            random.shuffle(labels)
        self.constants["labels"] = labels
    

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
    

    def set_coeffs_and_characteristics(self, parameters={}):
        self._check_agent_type_amounts(parameters)
            
        # Define coefficient values and agent attributes for each agent type
        agent_factory = AgentFactory()
        for agent_type in parameters["agent_types"].keys():
            self.constants[agent_type] = {}
            self.constants[agent_type]["coefficients"] = agent_factory.initialize_coefficients(agent_type)
            self.constants[agent_type]["agent_characteristics"] = agent_factory.initialize_characteristics(agent_type)
        self._build_coefficient_arrays(parameters["agent_types"].keys())
        self._build_agent_characteristic_arrays(parameters["agent_types"].keys())


    def set_network_and_schedule_constants(self):

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
        self.constants['opinion_scalar'] = np.random.rand(self.num_agents)

    
    def compute_commute_durations(self):
        """
        Define duration of commute for each agent separately.
        Set to a default duration if 'commute' exists in constants[agent_type],
        otherwise drawn from a lognormal distribution using 'mean_commute'
        and 'sigma_commute'
        """
        if "labels" not in self.constants:
            raise KeyError("Agent characteristics were not set before commute duration generation.")
        commutes = np.zeros_like(self.constants["labels"])
        for i, agent_type in enumerate(self.constants["labels"]):
            if agent_type not in self.constants:
                raise KeyError(f"Characteristics for agent type {agent_type} were not set.")
            elif 'commute' in self.constants[agent_type]["agent_characteristics"]:
                commutes[i] = self.constants[agent_type]["agent_characteristics"]['commute']
            else:
                mean = self.constants[agent_type]["agent_characteristics"]['mean_commute']
                sigma = self.constants[agent_type]["agent_characteristics"]['sigma_commute']
                commute_len = np.random.lognormal(mean=mean, sigma=sigma)
                commutes[i] = min(commute_len * Constants.DAY_LENGTH * (1/24),
                            1.5 * Constants.DAY_LENGTH * (1/24))

        self.constants['commute'] = commutes