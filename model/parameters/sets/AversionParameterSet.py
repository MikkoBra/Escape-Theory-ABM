class AversionParameterSet():
    """
    Record class for aversive internal state (A) parameters
    """
    def __init__(
            self,
            feedback=0,
            carrying_capacity=0,
            S_weight=0,
            T_weight=0,
            X_weight=0,
            I_weight=0,
            F_weight=0,
            B_weight=0,
        ):
        """
        Initializes aversive internal state parameters.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            aversive internal state
        carrying_capacity: float
            Carrying capacity of the logistic growth equation
            part of the update equation for aversive internal
            state
        S_weight: float
            Weight of stress from previous timestep on aversive
            internal state
        T_weight: float
            Weight of suicidal thoughts from previous timestep
            on aversive internal state
        X_weight: float
            Weight of escape behavior from previous timestep on
            aversive internal state
        I_weight: float
            Weight of internal escape strategies from previous
            timestep on aversive internal state
        F_weight: float
            Weight of social influence from previous timestep
            on aversive internal state
        """
        self.feedback = feedback
        self.carrying_capacity = carrying_capacity
        self.S_weight = S_weight
        self.T_weight = T_weight
        self.X_weight = X_weight
        self.I_weight = I_weight
        self.F_weight = F_weight
        self.B_weight = B_weight