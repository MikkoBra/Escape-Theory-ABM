from network_abm.agents.AbstractAgentUpdater import AbstractAgentUpdater
from numba import njit, prange
import numpy as np


class EscapeAgentUpdater(AbstractAgentUpdater):
    """
    Class containing update logic for an ABM of social
    contagion based on the escape theory of suicide
    """
    def __init__(self, agents, dt):
        super().__init__(agents, dt)
        self._agents_to_array()
    
    def _agents_to_array(self):
        """
        Converts agent information to arrays for use in numba-compiled
        code.
        """
        num_agents = len(self.agents)
        self.diathesis = np.zeros(num_agents, dtype=np.float64)
        self.impulsivity = np.zeros(num_agents, dtype=np.float64)
        self.stress_variance = np.zeros(num_agents, dtype=np.float64)
        self.stress_rate = np.zeros(num_agents, dtype=np.float64)
        self.stress_magnitude = np.zeros(num_agents, dtype=np.float64)
        self.stress_baseline = np.full(num_agents, 0.1, dtype=np.float64)
        self.stress_decay = np.full(num_agents, 1.5, dtype=np.float64)
        for agent in self.agents:
            self.impulsivity[agent.network_id] = agent.impulsivity
            self.stress_variance[agent.network_id] = (
                0.1 + agent.diathesis * 0.5
            )
            self.stress_rate[agent.network_id] = 4
            self.stress_magnitude[agent.network_id] = agent.stress_magnitude
    
    def update_all(self, params, alive_agents, network):
        """
        Updates all agent states.

        params: Dict<string, np.array>
            Dictionary containing arrays with each agent's current value
            for a parameter. Should include:
            {
                "S": [...],
                "A": [...],
                "U": [...],
                "T": [...],
                "M": [...],
            }
        alive_agents: np.array
            Array where each element indicates whether the agent of the
            corresponding network_id is alive. 1 = alive, 0 = deceased.
        network: Network
            Network object containing information about agents'
            social connections.
        """
        S = params["S"]
        update_alive_agents(
            alive_agents, self.dt,
            S, self.stress_baseline,
            self.stress_decay,
            self.stress_rate,
            self.stress_variance,
            self.stress_magnitude,
        )


@njit
def update_alive_agents(
    alive_agents, dt,
    S, stress_baseline, stress_decay,
        stress_rate, stress_variance,
        stress_magnitude,
):
    """
    Updates all agent states.

    alive_agents: np.array
        Array where each element indicates whether the agent of the
        corresponding network_id is alive. 1 = alive, 0 = deceased.
    S: np.array<float>
        Array where each element indicates the stress level of the
        agent of the corresponding network_id.
    """
    for idx in prange(len(alive_agents)):
        if alive_agents[idx] == 1:
            update_stress(S, idx, dt,
                          stress_baseline[idx],
                          stress_decay[idx],
                          stress_rate[idx],
                          stress_variance[idx],
                          stress_magnitude[idx],
                          )

@njit
def poisson_event(rate, dt):
    """
    Generates a number of events taking place within a timestep of
    size dt using a Poisson distribution with a given event rate.
    Returns an integer amount of events that have taken place.
    """
    return np.random.poisson(rate * dt)

@njit
def update_stress(
        S,
        id,
        dt,
        baseline,
        decay,
        rate,
        variance,
        magnitude
    ):
    """
    Update function for stress. Consists of impulses governed by
    a poisson process that decay to a baseline with some noise
    modeled as a Wiener process.
    """
    S_old = S[id]
    impulse = poisson_event(rate, dt) * magnitude
    S_new = baseline + np.exp(-decay*dt) * (S_old - baseline) + impulse
    dW = np.random.normal(0, np.sqrt(dt))
    S_new += dW * variance
    S[id] = min(1, max(0, S_new))