import mesa
import numpy as np
import pandas as pd
from pathlib import Path
from Constants import Constants

SOCIAL_WEIGHT_IDX = 1


class BrightenAgent():

    def initialize_coefficients(self):
        """
        Agent informed by the Brighten study's data.
        Mean sleep and sleep variance is modified wrt
        a default agent, and social events are less likely
        for all agents.
        """
        return {
            'news_signal': {
                'decay': 0.2,
                'levy_alpha': 2,
                'share_rate': 0.1,
            },

            'stress': {
                'baseline': 0.2,
                'decay': 3,
                'impulse_rate': 4,
                'impulse_strength': 0.1,
                'morning_impulse': 0,
                'alpha': 0.1,
                'beta': 0.1,
                'gamma': 0.1,
                'sigma': 0.12,
            },

            'aversive_internal_state': {
                'feedback': 4,
                'carrying_capacity': 0.1,
                'S_weight': 2,
                'T_weight': 0.1,
                'X_weight': 1,
                'I_weight': 0.5,
                'B_weight': 2.5,
                'c_weight': 0.7,
            },

            'urge_to_escape': {
                'feedback': 2.5,
                'A_weight': 1.75,
                'M_weight': 0.1,
                'C_weight': 0.1,
            },

            'suicide_history': {
                'decay': 0.7,
            },

            'suicidal_thought': {
                'feedback': 3,
                'sig_middle': 0.37,
                'sig_steepness': 100,
            },

            'escape_behavior': {
                'feedback': 3,
                'sig_middle': 0.32,
                'sig_steepness': 50,
            },

            'external_strat': {
                'feedback': 0.5,
                'carrying_capacity': 0.1,
                'A_weight': 0.41,
                'U_weight': 0.45,
            },

            'internal_strat': {
                'feedback': 3,
                'carrying_capacity': 0.05,
                'A_weight': 0.65,
                'U_weight': 0.75,
            },

            'burdensomeness': {
                'feedback': 0.35,
                'A_weight': 0.25,
                'I_weight': 0.1,
                'B_lonely': 1,
            }
        }


    def initialize_characteristics(
            self,
            variable_sleep=True,
            variable_commute=False
            ):
        """
        Sets agent attribute values used in agent updates
        """
        characteristics = {
            # No weekends in schedule
            'weekends': True,
            # News effect is applied
            'consumes_news': True,
            # Young adult
            'age': 25,
            # Somewhat impulsive
            'impulsivity': 0.7,
            'event_lambda': 1,
        }

        if variable_commute:
            characteristics['mean_commute'] = np.log(0.5)
            characteristics['sigma_commute'] = 0.4
        else:
            characteristics['commute'] = 0.5 * (1/24) * Constants.DAY_LENGTH
        
        if variable_sleep:
            characteristics['mean_sleep'] = 5.14
            characteristics['sigma_sleep'] = 2.90
        else:
            characteristics['sleep'] = 8
        
        return characteristics
            