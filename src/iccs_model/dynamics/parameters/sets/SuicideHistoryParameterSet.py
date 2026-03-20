class SuicideHistoryParameterSet():
    """
    Record class for suicide history (M) parameters
    """
    def __init__(
            self,
            decay=0,
        ):
        """
        Initializes suicide history parameters.

        Parameters
        ----------
        decay: float
            Rate at which memory of suicidal thought decays
        """
        self.decay = decay