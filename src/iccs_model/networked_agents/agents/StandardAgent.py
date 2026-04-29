import mesa
import numpy as np
import pandas as pd
from pathlib import Path
from iccs_model.dynamics.AgentUpdaterNumba import (
    NumbaAgentUpdater
)
from iccs_model.dynamics.parameters.DefaultParameterFactory import DefaultParameterFactory
from iccs_model.dynamics.parameters.StateParameters import StateParameters
from iccs_model.dynamics.states.SleepState import SleepState
from iccs_model.dynamics.StateManager import StateManager

SOCIAL_WEIGHT_IDX = 1


class StandardAgent(mesa.Agent):
    """
    Agent with all extended dynamics enabled.
    """
    time_values = None
    stress_values = None

    def __init__(self, model, default_params={}, state_params={}):
        """
        Initializes the agent with a default stress value.
        """
        super().__init__(model)
        self.type = "standard"
        self.updater = NumbaAgentUpdater()
        self.parameters = DefaultParameterFactory().create_default_parameters(parameters=default_params)

        # Initialize state-specific values
        self.state_params = StateParameters(state_params)
        self.state_params.set_commute()
        self.state_params.set_sleep_params()

        self.state_manager = StateManager(self.state_params)
        
        # Initial values
        self.close_connections = 15
        self.medium_connections = 35
        self.clustering_coefficient = 0
        self.temperature = 0.5
        self.network_id = self.unique_id - 1

        # News-related attributes
        self.age = 18
        self.celebrity_agreement = np.random.uniform(0, 1)
        self.vulnerability = 0
        self.opinion_scalar = np.random.uniform(0, 1)

        self.stress = 0
        self.aversive_internal_state = 0
        self.urge_to_escape = 0
        self.suicide_history = 0
        self.suicidal_thought = 0
        self.escape_behavior = 0
        self.external_strat = 0
        self.internal_strat = 0
        self.burdensomeness = 0
        self.total_time = 0
        self.state_manager.state = SleepState()
        self.state_manager.state.generate_time(0, None, self.state_params)


    def set_connectedness(self, weights):
        if len(weights) == 0:
            self.connectedness = 0.0
            return

        weights = np.asarray(weights, dtype=np.float32)

        # Proper probability distribution
        p = weights / weights.sum()

        entropy = -np.sum(p * np.log(p + 1e-8))
        max_entropy = np.log(len(p))

        self.connectedness = 1.0 - (entropy / max_entropy)

    
    def update_agent(self, dt):
        """
        Updates the agent over timestep dt.
        """

        params = self.parameters.get_S_params(
            external_strat=self.external_strat
        )
        new_S = self.updater.stress(
            prev_state=self.stress,
            dt=dt,
            params=params
        )

        # Update aversive internal state
        params = self.parameters.get_A_params(
            stress=self.stress,
            suicidal_thought=self.suicidal_thought,
            escape_behavior=self.escape_behavior,
            internal_strat=self.internal_strat,
            burdensomeness=self.burdensomeness,
            clustering_coefficient = self.clustering_coefficient
        )
        new_A = self.updater.aversive_internal_state(
            prev_state=self.aversive_internal_state,
            dt=dt,
            params=params
        )

        # Update urge to escape
        params = self.parameters.get_U_params(
            aversive_internal_state=self.aversive_internal_state,
            suicide_history=self.suicide_history,
            connectedness=self.connectedness,
        )
        new_U = self.updater.urge_to_escape(
            prev_state=self.urge_to_escape,
            dt=dt,
            params=params
        )

        # Update suicide history
        params = self.parameters.get_M_params(
            suicidal_thought=self.suicidal_thought
        )
        new_M = self.updater.suicide_history(
            prev_state=self.suicide_history,
            dt=dt,
            params=params
        )

        # Update suicidal thought
        params = self.parameters.get_T_params(
            urge_to_escape=self.urge_to_escape,
        )
        new_T = self.updater.sigmoid(
            prev_state=self.suicidal_thought, dt=dt, params=params
        )

        # Update escape behavior
        params = self.parameters.get_X_params(
            self.urge_to_escape,
        )
        new_X = self.updater.sigmoid(self.escape_behavior, dt, params)

        # Update external strategy
        params = self.parameters.get_E_params(
            self.aversive_internal_state,
            self.urge_to_escape
        )
        new_E = self.updater.strategy_for_escape(
            prev_state=self.external_strat,
            dt=dt,
            params=params
        )

        # Update internal strategy
        params = self.parameters.get_I_params(
            self.aversive_internal_state,
            self.urge_to_escape
        )
        new_I = self.updater.strategy_for_escape(
            prev_state=self.internal_strat,
            dt=dt,
            params=params
        )
        
        t = int(self.model.time / self.model.dt) - 1
        params = self.parameters.get_B_params(
            self.internal_strat,
            self.model,
            t
        )
        new_B = self.updater.burdensomeness(
            prev_state=self.burdensomeness,
            dt=dt,
            params=params
        )
        
        self.stress = new_S
        self.aversive_internal_state = new_A
        self.urge_to_escape = new_U
        self.suicide_history = new_M
        self.suicidal_thought = new_T
        self.escape_behavior = new_X
        self.external_strat = new_E
        self.internal_strat = new_I
        self.burdensomeness = new_B
        self.total_time += dt

        if self.state_manager.state.STATE_NAME == "morning":
            self.parameters.set_stress_coefficients(morning_impulse=0)
        self.state_manager.update_state(dt, self.total_time, self.parameters)