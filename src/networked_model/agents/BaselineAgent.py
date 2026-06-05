from networked_model.agents.DefaultAgent import DefaultAgent


class BaselineAgent(DefaultAgent):
    """
    Agent with dynamics based on the Wang et al. formal model of the General Escape Theory.
    Defining features:
    - No news signal influence
    - No memory of suicidal thought influence
    - No social effects
    """

    def initialize_coefficients(self):
        """
        Returns parameter settings for a baseline agent.
        """
        return {
            # Note: not used (see initialize_characteristics)
            'news_signal': {
                'decay': 0.2,
                'levy_alpha': 2,
                'share_rate': 0.1,
            },

            # Coefficients related to stress computation
            'stress': {
                'baseline': 0.2,            # Lower bound of stress
                'decay': 3,                 # Decay rate of stress spikes
                'impulse_rate': 4,          # How often spikes occur
                'impulse_strength': 0.1,    # Magnitude of stress spikes
                'morning_impulse': 0,       # Magnitude of impulse due to poor sleep
                                            # (should be set dynamically)
                'alpha': 0.1,               # Effect of external strategy on baseline stress
                'beta': 0.1,                # Effect of external strategy on stress decay
                'gamma': 0.1,               # Effect of external strategy on magnitude of spikes
                'sigma': 0.12,              # Strength of random noise signal
            },

            # Coefficients related to aversive internal state computation
            'aversive_internal_state': {
                'feedback': 4,              # How strongly past affects current
                'carrying_capacity': 0.1,   # Upper bound of logistic growth component
                'S_weight': 2,              # Strength of effect of stress
                'T_weight': 0.1,            # Strength of effect of suicidal thought
                'X_weight': 1,              # Strength of effect of escape behavior
                'I_weight': 0.5,            # Strength of effect of internal strategy
                'B_weight': 0,              # Strength of effect of burdensomeness (Note: no social effects so 0)
                'c_weight': 0,              # Strength of effect of clustering (Note: no social effects so 0)
            },

            # Coefficients related to urge to escape computation
            'urge_to_escape': {
                'feedback': 2.5,            # How strongly past affects current
                'A_weight': 1.75,           # Strength of effect of aversive internal state
                'M_weight': 0,              # Strength of effect of memory of suicidal thought (Note: not in Wang et al. so 0)
                'C_weight': 0,              # Strength of effect of connectedness (Note: no social effects so 0)
            },

            # Coefficients related to memory of suicidal thought computation
            'suicide_history': {
                'decay': 0.7,               # Decay rate of memory
            },

            # Coefficients related to suicidal thought computation
            'suicidal_thought': {
                'feedback': 3,              # How strongly past affects current
                'sig_middle': 0.35,         # Threshold at which suicidal thought arises (middle of sigmoid)
                'sig_steepness': 100,       # How abruptly suicidal thought increases at sig_middle
            },

            # Coefficients related to escape behavior computation
            'escape_behavior': {
                'feedback': 3,              # How strongly past affects current
                'sig_middle': 0.3,          # Threshold at which escape behavior arises (middle of sigmoid)
                'sig_steepness': 50,        # How abruptly escape behavior increases at sig_middle
            },

            # Coefficients related to external strategy computation
            'external_strat': {
                'feedback': 0.5,            # How strongly past affects current
                'carrying_capacity': 0.1,   # Upper bound of logistic growth component
                'A_weight': 0.41,           # Strength of effect of aversive internal state
                'U_weight': 0.45,           # Strength of effect of urge to escape
            },

            # Coefficients related to internal strategy computation
            'internal_strat': {
                'feedback': 3,              # How strongly past affects current
                'carrying_capacity': 0.05,  # Upper bound of logistic growth component
                'A_weight': 0.65,           # Strength of effect of aversive internal state
                'U_weight': 0.75,           # Strength of effect of urge to escape
            },

            # Note: not relevant (strength of effect is 0 on all parameters)
            'burdensomeness': {
                'feedback': 0.35,
                'A_weight': 0.25,
                'I_weight': 0.1,
                'B_lonely': 1,
            }
        }


    def initialize_characteristics(
            variable_sleep=False,
            variable_commute=False
            ):
        """
        Sets agent attribute values used in agent updates
        """
        # No variable sleep
        characteristics = super().initialize_characteristics(
            variable_sleep=variable_sleep,
            variable_commute=variable_commute
        )
        # Not affected by news signal
        characteristics['consumes_news'] = False
        
        return characteristics
            