class UrgeToEscapeParameterSet():
    """
    Record class for urge to escape (U) parameters
    """
    def __init__(
            self,
            feedback=0,
            A_weight=0,
        ):
        """
        Initializes urge to escape parameters.

        Parameters
        ----------
        feedback: float
            Strength of feedback from previous state of
            urge to escape
        A_weight: float
            Weight of aversive internal state from previous
            timestep on urge to escape
        """
        self.feedback = feedback
        self.A_weight = A_weight