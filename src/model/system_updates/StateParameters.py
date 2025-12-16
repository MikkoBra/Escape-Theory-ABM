import numpy as np
from Constants import Constants

class StateParameters():
    def set_sleep_params(self, mean=7, sigma=2):
        self.mean_sleep = mean
        self.sigma_sleep = sigma
    
    def set_commute(self, mean=np.log(0.5), sigma=0.4):
        # At most 1 and half hour commute
        commute_len = np.random.lognormal(mean=mean, sigma=sigma)
        self.commute = min(commute_len * Constants.DAY_LENGTH * (1/24),
                            1.5 * Constants.DAY_LENGTH * (1/24))
    
    def set_homelife_params(self):
        pass
    