class EscapeBehaviorParameterSet():
    """
    Record class for escape behavior (X) parameters
    """
    def __init__(
            self,
            weight_new=0,
            sig_middle=0,
            sig_steepness=0,
        ):
        """
        Initializes escape behavior parameters.

        Parameters
        ----------
        weight_new: float
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
        self.weight_new = weight_new
        self.sig_middle = sig_middle
        self.sig_steepness = sig_steepness
