class StressParameterSet():
    """
    Record class for stress (S) parameters
    """
    def __init__(
            self,
            mean=0,
            sigma=0,
            reversion=0,
            E_weight=0
        ):
        """
        Initializes stress parameters.

        Parameters
        ----------
        mean: float
            Long-term mean of the stress movement.
        sigma: float
            Standard deviation of the stochastic jitter of the
            stress movement.
        reversion: float
            Strength of pull towards the mean of the stress
            movement.
        E_weight: float
            Weight of external escape strategies on stress.
        """
        self.mean = mean
        self.sigma = sigma
        self.reversion = reversion
        self.E_weight = E_weight