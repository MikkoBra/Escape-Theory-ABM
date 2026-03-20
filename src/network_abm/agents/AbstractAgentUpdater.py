from abc import ABC, abstractmethod



class AbstractAgentUpdater(ABC):
    """
    Abstract class containing required functionality of an agent
    updater class
    """
    def __init__(self, agents, dt):
        """
        agents: Array<Agent>
            Contains the model's agents as Agent objects.
        dt: float
            Timedelta to use in forward Euler update equations
        """
        self.agents = agents
        self.dt = dt
    
    @abstractmethod
    def update_all(self, parameters, alive_agents, network):
        """
        Updates all agent states. AgentUpdater subclasses should implement
        their own numba-parallelizable function that uses agent state
        arrays extracted from parameters.

        params: Dict<string, np.array>
            Dictionary containing arrays with each agent's current value
            for a specific parameter. Example: {"stress": [0.1, 0.3, 0]}
            for three agents.
        alive_agents: np.array
            Array where each element indicates whether the agent of the
            corresponding network_id is alive. 1 = alive, 0 = deceased.
        network: Network
            Network object containing information about agents'
            social connections.
        """
        pass
