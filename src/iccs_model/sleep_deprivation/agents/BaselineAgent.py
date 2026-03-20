from iccs_model.sleep_deprivation.agents.SleepAgent import SleepAgent
from Constants import Constants

class BaselineAgent(SleepAgent):
    """
    Agent with only the dynamics as described in Wang et al.
    """

    def __init__(self, model, stress_gen=False, default_params={}):
        super().__init__(model)
        self.type = "baseline"


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
            connectedness=0,
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
        
        self.stress = new_S
        self.aversive_internal_state = new_A
        self.urge_to_escape = new_U
        self.suicide_history = new_M
        self.suicidal_thought = new_T
        self.escape_behavior = new_X
        self.external_strat = new_E
        self.internal_strat = new_I
        self.total_time += dt