class StressParameterSet():
    """
    Record class for stress (S) parameters
    """
    def __init__(
            self,
            baseline=0,
            decay=0,
            impulse_rate=0,
            impulse_strength=0,
            morning_impulse=0,
            alpha=0,
            beta=0,
            gamma=0,
            sigma=0,
        ):
        """
        Initializes stress parameters.

        Parameters
        ----------
        baseline: float
            Baseline stress value.
        decay: float
            Decay rate of experienced stress.
        impulse_rate: float
            Rate of impulses in impulse per time unit.
        alpha: float
            Effect of external strategy on baseline stress.
        beta: float
            Effect of external strategy on decay rate.
        gamma: float
            Effect of external strategy on impulse strength.
        """
        self.baseline = baseline
        self.decay = decay
        self.impulse_rate = impulse_rate
        self.impulse_strength = impulse_strength
        self.morning_impulse=morning_impulse
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma