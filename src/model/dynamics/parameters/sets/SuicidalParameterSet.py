class SuicidalParameterSet():
    """
    Record class for suicidal thought (T) parameters
    """
    def __init__(
            self,
            feedback=0,
            sig_middle=0,
            sig_steepness=0,
        ):
        """
        Initializes suicidal thought parameters.

        Parameters
        ----------
        feedback: float
            Decay of past suicidal thought
        sig_middle: float
            Center of the sigmoidal curve representing
            the onset of suicidal thoughts
        sig_steepness: float
            Steepness of the sigmoidal curve representing
            the onset of suicidal thoughts
        """
        self.feedback = feedback
        self.sig_middle = sig_middle
        self.sig_steepness = sig_steepness