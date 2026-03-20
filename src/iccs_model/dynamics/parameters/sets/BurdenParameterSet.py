class BurdenParameterSet():
    """
    Record class for burdensomeness (B) parameters
    """
    def __init__(
            self,
            neighbors = [],
            neighbor_ws = [],
            feedback=0,
            A_weight=0,
            I_weight=0,
            B_lonely=0,
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
        self.neighbors = neighbors
        self.neighbor_ws = neighbor_ws
        self.feedback = feedback
        self.A_weight = A_weight
        self.I_weight = I_weight
        self.B_lonely = B_lonely