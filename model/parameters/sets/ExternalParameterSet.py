class ExternalParameterSet():
    """
    Record class for external escape strategy (E) parameters
    """
    def __init__(
            self,
            feedback=0,
            carrying_capacity=0,
            A_weight=0,
            U_weight=0,
        ):
        """
        Initializes external escape strategy parameters.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            external escape strategy
        carrying_capacity: float
            Carrying capacity of the logistic growth equation
            part of the update equation for external escape
            strategy
        A_weight: float
            Weight of aversive internal state from previous 
            timestep on external escape strategy
        U_weight: float
            Weight of urge to escape from previous timestep
            on external escape strategy
        """
        self.feedback = feedback
        self.carrying_capacity = carrying_capacity
        self.A_weight = A_weight
        self.U_weight = U_weight