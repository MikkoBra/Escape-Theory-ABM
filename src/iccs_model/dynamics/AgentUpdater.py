import numpy as np


class AgentUpdater():
    """
    Contains parameter update logic for the agents.
    """
    def __init__(self):
        pass
    
    def stress(
            self,
            prev_state,
            dt,
            params,
    ):
        """
        Models stress evolution.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            E = params["E"]
            baseline = params["baseline"]
            decay = params["decay"]
            impulse_rate = params["impulse_rate"]
            impulse_strength = params["impulse_strength"]
            morning_impulse = params["morning_impulse"]
            alpha = params["alpha"]
            beta = params["beta"]
            gamma = params["gamma"]
            sigma = params["sigma"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for aversive internal state evolution")
        def poisson_event(rate, dt):
            return np.random.poisson(rate * dt)
        B_t = baseline + alpha * E
        lambda_t = decay + beta * E
        if morning_impulse > 0:
            I_t = morning_impulse
        else:
            I_t = poisson_event(impulse_rate, dt) * impulse_strength * (1 - gamma * E) 
        stress = B_t + np.exp(-lambda_t*dt) * (prev_state - B_t) + I_t
        dW = np.random.normal(0, np.sqrt(dt))
        stress += dW * sigma
        if stress < 0:
            stress = 0
        elif stress > 1:
            stress = 1
        return stress

    def aversive_internal_state(self, prev_state, dt, params):
        """
        Evolution equation of aversive internal state.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            S = params["S"]
            T = params["T"]
            X = params["X"]
            I = params["I"]
            B = params["B"]
            clustering_coefficient = params["clustering_coefficient"]
            feedback = params["feedback"]
            carrying_capacity = params["carrying_capacity"]
            S_weight = params["S_weight"]
            T_weight = params["T_weight"]
            X_weight = params["X_weight"]
            I_weight = params["I_weight"]
            B_weight = params["B_weight"]
            c_weight = params["c_weight"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for aversive internal state evolution")
        new_state = prev_state + dt * (feedback*prev_state * (carrying_capacity - prev_state)\
            + S_weight * S - T_weight * T - X_weight * X - I_weight * I\
                  + B_weight * B - c_weight * clustering_coefficient)
        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1
        return new_state
    
    def suicide_history(self, prev_state, dt, params):
        try:
            T = params["T"]
            decay = params["decay"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for suicide history evolution")
        if T < prev_state: # Decay memory slowly when T drops
            new_state = prev_state + dt*(-decay * prev_state)
        else:
            new_state = T

        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1
        
        return new_state

    def urge_to_escape(self, prev_state, dt, params):
        """
        Evolution equation of urge to escape.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            A = params["A"]
            M = params["M"]
            C = params["C"]
            feedback = params["feedback"]
            A_weight = params["A_weight"]
            M_weight = params["M_weight"]
            C_weight = params["C_weight"]
        except KeyError as e:
            print(f"Missing parameter {e.args[0]}"+
                            " for urge to escape evolution")
            raise Exception("Terminating program")
        new_state = prev_state + dt * (-feedback * prev_state  + A_weight * A + M_weight * M - C_weight * C)

        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1

        return new_state


    def sigmoid(self, prev_state, dt, params):
        """
        Discretized evolution equation of suicidal thoughts
        and escape behaviors.
        Uses a simple feedback model with given weight of new
        state vs old state.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            U = params["U"]
            feedback = params["feedback"]
            sig_middle = params["sig_middle"]
            sig_steepness = params["sig_steepness"]
        except KeyError as e:
            print(f"Missing parameter {e.args[0]}"+
                            " for suicidal thought evolution")
            raise Exception("Terminating program")
        sigmoid = (1 / (1 + np.exp(-sig_steepness * (U - sig_middle))))
        new_state = prev_state + dt * (-feedback * prev_state + sigmoid)
        
        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1
        
        return new_state


    def strategy_for_escape(self, prev_state, dt, params):
        """
        Evolution equation of external or internal escape
        strategy.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            A = params["A"]
            U = params["U"]
            feedback = params["feedback"]
            carrying_capacity = params["carrying_capacity"]
            A_weight = params["A_weight"]
            U_weight = params["U_weight"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for escape strategy evolution")
        new_state = prev_state + dt * (feedback*prev_state*(carrying_capacity - prev_state)\
            + A_weight * A - U_weight * U)

        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1

        return new_state


    def burdensomeness(self, prev_state, dt, params):
        """
        Evolution equation of social burden.

        prev_state: float
            Value of stress in the previous timestep
        dt: float
            Time delta (amount of time to progress)
        params: dict
            Dictionary containing the required parameters
        """
        try:
            neighbor_As = params["neighbor_As"]
            neighbor_ws = params["neighbor_ws"]
            I = params["I"]
            feedback = params["feedback"]
            A_weight = params["A_weight"]
            I_weight = params["I_weight"]
            B_lonely = params["B_lonely"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for social burden evolution")
        if len(neighbor_As) == 0:
            new_state = B_lonely
        else:
            social_effects = np.array(np.array(neighbor_As) * np.array(neighbor_ws))
            new_state = prev_state + dt * (A_weight * np.mean(social_effects) - I_weight * I - feedback * prev_state)
        if new_state < 0:
            new_state = 0
        elif new_state > 1:
            new_state = 1

        return new_state