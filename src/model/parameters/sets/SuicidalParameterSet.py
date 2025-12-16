class SuicidalParameterSet():
    """
    Record class for suicidal thought (T) parameters
    """
    def __init__(
            self,
            weight_new=0,
            sig_middle=0,
            sig_steepness=0,
        ):
        """
        Initializes suicidal thought parameters.

        Parameters
        ----------
        weight_new: float
            Weight of suicidal thought based on current
            parameters compared to feedback of suicidal
            thoughts from previous timestep
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of suicidal thoughts
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of suicidal thoughts
        """
        self.weight_new = weight_new
        self.sig_middle = sig_middle
        self.sig_steepness = sig_steepness