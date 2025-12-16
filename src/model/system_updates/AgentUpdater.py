import numpy as np


class AgentUpdater():
    """
    Contains parameter update logic for the agents.
    """
    def __init__(self):
        pass
    
    def stress(
            self,
            dt,
            prev_stress,
            prev_E,
            mean=0.2,
            sigma=0.12,
            reversion=1.2,
            prev_E_weight=1.0
    ):
        """
        Models stress evolution using discrete-time (Euler-Maruyama)
        approximation of an Ornstein-Uhlenbeck process.
        """
        drift = reversion * (mean - prev_stress)
        dW = np.random.normal(0, np.sqrt(dt))
        stress = prev_stress + drift * dt + sigma * dW
        damping = np.exp(-prev_E_weight * prev_E * dt)
        stress *= damping
        if stress < 0:
            stress = -stress
        elif stress > 1:
            stress = 2 - stress
        return stress

    def aversive_internal_state(self, prev_state, t, params):
        """
        Evolution equation of aversive internal state.

        params: dict
            Dictionary containing the required parameters
        """
        try:
            S = params["S"]
            T = params["T"]
            X = params["X"]
            I = params["I"]
            F = params["F"]
            B = params["B"]
            feedback = params["feedback"]
            carrying_capacity = params["carrying_capacity"]
            S_weight = params["S_weight"]
            T_weight = params["T_weight"]
            X_weight = params["X_weight"]
            I_weight = params["I_weight"]
            F_weight = params["F_weight"]
            B_weight = params["B_weight"]
        except KeyError as e:
            raise Exception(f"Missing parameter {e.args[0]}"+
                            " for aversive internal state evolution")
        new_state = feedback * prev_state * (carrying_capacity - prev_state)\
            + S_weight * S - T_weight * T - X_weight * X - I_weight * I\
                - F_weight * F + B_weight * B
        return new_state

    def urge_to_escape(self, prev_state, t, params):
        """
        Evolution equation of urge to escape.

        params: dict
            Dictionary containing the required parameters
        """
        try:
            A = params["A"]
            feedback = params["feedback"]
            A_weight = params["A_weight"]
        except KeyError as e:
            print(f"Missing parameter {e.args[0]}"+
                            " for urge to escape evolution")
            raise Exception("Terminating program")
        new_state = -feedback * prev_state  + A_weight * A
        return new_state


    def sigmoid(self, prev_state, t, params):
        """
        Discretized evolution equation of suicidal thoughts
        and escape behaviors.
        Uses a simple feedback model with given weight of new
        state vs old state.

        params: dict
            Dictionary containing the required parameters
        """
        try:
            U = params["U"]
            weight_new = params["weight_new"]
            sig_middle = params["sig_middle"]
            sig_steepness = params["sig_steepness"]
        except KeyError as e:
            print(f"Missing parameter {e.args[0]}"+
                            " for suicidal thought evolution")
            raise Exception("Terminating program")
        sigmoid = (1 / (1 + np.exp(-sig_steepness * (U - sig_middle))))
        new_state = (1 - weight_new) * prev_state  + weight_new * sigmoid
        return new_state


    def strategy_for_escape(self, prev_state, t, params):
        """
        Evolution equation of external or internal escape
        strategy.

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
        new_state = feedback * prev_state * (carrying_capacity - prev_state)\
            + A_weight * A - U_weight * U
        return new_state


    def rk4_step(self, prev_state, t, dt, f, params):
        """
        Runge-Kutta 4 implementation that estimates the solution of
        a differential equation in time dt.

        Parameters
        ----------
        prev_state: float
            Previous value of the to-be-approximated parameter
        t: float
            Current time
        dt: float
            Timestep size
        f: function
            Evolution equation of the to-be-approximated parameter
        """
        k1 = f(prev_state, t, params)
        k2 = f(prev_state + 0.5*dt*k1, t + 0.5*dt, params)
        k3 = f(prev_state + 0.5*dt*k2, t + 0.5*dt, params)
        k4 = f(prev_state + dt*k3, t + dt, params)
        new_state = prev_state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        if new_state > 1:
            return 2 - new_state
        elif new_state < 0:
            return 0 - new_state
        else:
            return new_state
