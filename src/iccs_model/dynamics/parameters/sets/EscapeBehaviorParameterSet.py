class EscapeBehaviorParameterSet():
    """
    Record class for escape behavior (X) parameters
    """
    def __init__(
            self,
            feedback=0,
            sig_middle=0,
            sig_steepness=0,
        ):
        """
        Initializes escape behavior parameters.

        Parameters
        ----------
        feedback: float
            Weight of suicidal thought based on current
            parameters compared to feedback of escape 
            behavior from previous timestep
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of escape behavior
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of escape behavior
        """
        self.feedback = feedback
        self.sig_middle = sig_middle
        self.sig_steepness = sig_steepness
