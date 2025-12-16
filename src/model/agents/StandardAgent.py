import mesa
import numpy as np
from model.system_updates.AgentUpdater import (
    AgentUpdater
)
from model.parameters.DefaultParameters import DefaultParameters
from model.system_updates.StateParameters import StateParameters
from model.system_updates.StateManager import StateManager
from model.system_updates.states.SleepState import SleepState
SOCIAL_WEIGHT_IDX = 1


class StandardAgent(mesa.Agent):
    """
    Default agent in the suicide model.
    """

    def __init__(self, model):
        """
        Initializes the agent with a default stress value.
        """
        super().__init__(model)
        self.type = "standard"
        self.updater = AgentUpdater()
        self.parameters = DefaultParameters()

        # Initialize state-specific values
        self.state_params = StateParameters()
        self.state_params.set_commute()         # should be constant
        self.state_params.set_sleep_params()

        self.state_manager = StateManager(self.state_params)
        
        # Initial values
        self.stress = 0.5
        self.aversive_internal_state = 0.39
        self.urge_to_escape = 0
        self.suicidal_thought = 0
        self.escape_behavior = 0
        self.external_strat = 0
        self.internal_strat = 0
        self.total_time = 0
        self.state_manager.state = SleepState()
        self.state_manager.state.generate_time(0, None, self.state_params)

    def set_friends(self, n=5):
        n = min(n, self.model.num_agents)
        self.friends = self.set_social_connections(n)
        self.num_friends = n

    def set_bullies(self, n=0):
        n = min(n, self.model.num_agents)
        self.bullies = self.set_social_connections(n)
        self.num_bullies = n
    
    def set_social_connections(
            self,
            n
    ):
        if n == 0:
            return np.array([])
        other_agents = [agent.unique_id 
                        for agent in self.model.agents 
                        if agent.unique_id != self.unique_id]
        n = min(n, len(other_agents))
        agent_IDs = np.random.choice(other_agents, size=n)

        # Take n random samples from N(0.5, 0.15)
        # (roughly between 0 and 1), then clip
        weights = np.random.normal(loc=0.5, scale=0.15, size=n)
        weights = np.clip(weights, 0, 1)

        # Make 2D array
        connections = np.column_stack((agent_IDs, weights))
        return connections

    def saturated_mean_social_influence(self, connections, k=5):
        """
        Calculates social influence using the mean weight
        of the agent's social connections. Saturates by
        multiplying with f(n) = n/(k+n), so that f(n) =
        1 as n -> inf, and f(n) = 1/2 at k = n.
        """
        total = 0
        for agent_info in connections:
            total += agent_info[SOCIAL_WEIGHT_IDX]
        n = len(connections)
        if n == 0:
            return 0
        return (total/n) * (n/(k+n))
    
    def update_agent(self, dt):
        """
        Updates the agent over timestep dt.
        """

        # Update stress
        new_S = self.updater.stress(
            dt=dt,
            prev_stress=self.stress,
            prev_E=self.external_strat,
            mean=self.parameters.stress.mean,
            sigma=self.parameters.stress.sigma,
            reversion=self.parameters.stress.reversion,
            prev_E_weight=self.parameters.stress.E_weight,
            )

        # Update aversive internal state
        params = self.parameters.get_A_params(
            stress=self.stress,
            suicidal_thought=self.suicidal_thought,
            escape_behavior=self.escape_behavior,
            internal_strat=self.internal_strat,
            friend_influence=self.saturated_mean_social_influence(
                self.friends),
            bully_influence=self.saturated_mean_social_influence(
                self.bullies),
        )
        new_A = self.updater.rk4_step(
            self.aversive_internal_state,
            self.total_time,
            dt,
            self.updater.aversive_internal_state,
            params,
        )

        # Update urge to escape
        params = self.parameters.get_U_params(
            aversive_internal_state=self.aversive_internal_state,
        )
        new_U = self.updater.rk4_step(
            self.urge_to_escape,
            self.total_time,
            dt,
            self.updater.urge_to_escape,
            params,
        )

        # Update suicidal thought
        params = self.parameters.get_T_params(
            urge_to_escape=self.urge_to_escape,
        )
        new_T = self.updater.sigmoid(
            self.suicidal_thought, self.total_time, params
        )

        # Update escape behavior
        params = self.parameters.get_X_params(
            self.urge_to_escape,
        )
        new_X = self.updater.sigmoid(self.escape_behavior, self.total_time, params)

        # Update external strategy
        params = self.parameters.get_E_params(
            self.aversive_internal_state,
            self.urge_to_escape
        )
        new_E = self.updater.rk4_step(
            self.external_strat,
            self.total_time,
            dt,
            self.updater.strategy_for_escape,
            params,
        )

        # Update internal strategy
        params = self.parameters.get_I_params(
            self.aversive_internal_state,
            self.urge_to_escape
        )
        new_I = self.updater.rk4_step(
            self.internal_strat,
            self.total_time,
            dt,
            self.updater.strategy_for_escape,
            params,
        )

        self.stress = new_S
        self.aversive_internal_state = new_A
        self.urge_to_escape = new_U
        self.suicidal_thought = new_T
        self.escape_behavior = new_X
        self.external_strat = new_E
        self.internal_strat = new_I
        self.total_time += dt
        self.state_manager.update_state(dt, self.total_time, self.parameters)
