class NewsSignalManager:
    """
    Manages news signals sent to news stations.
    """
    def __init__(self):
        self.signal_sent = False
        self.signal_timestep = None
    
    def send_signal(self, intensity=None, current_time=None, warmup=0):
        """
        Send a signal to news stations.
        
        Parameters:
        -----------
        intensity : float, optional
            Intensity of the news signal
        current_timestep : int, optional
            Current timestep in the simulation
        
        Returns:
        --------
        bool
            True if signal should be sent at this timestep, False otherwise
        """
        # Signal sent at timestep 10, only once
        if current_time - warmup >= 10 and not self.signal_sent:
            print('Sending news signal')
            self.signal_sent = True
            self.signal_timestep = current_time
            return True
        return False