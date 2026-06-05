import numpy as np


class NewsStation:
    """
    Represents a news station that influences agents in the model.
    """
    def __init__(self, station_id, opinion_scalar=None, news_intensity=0.5):
        """
        Initialize a news station.
        
        Parameters:
        -----------
        station_id : int
            Unique identifier for this news station
        opinion_scalar : float, optional
            Opinion position of this station (0 to 1). 
            If None, randomly assigned.
        news_intensity : float
            Base intensity of news effects (default: 0.5)
        """
        self.station_id = station_id
        self.opinion_scalar = opinion_scalar if opinion_scalar is not None else np.random.uniform(0, 1)
        self.agents = []  # IDs of agents that consume this news
        self.news_intensity = news_intensity
        self.signal_manager = None
    
    def add_agent(self, agent_id):
        """Add an agent to this news station's audience."""
        if agent_id not in self.agents:
            self.agents.append(agent_id)
    
    def apply_news_effect(self, agent, model):
        """
        Apply news effects to an agent's parameters.
        
        The effect modulates based on:
        - news_intensity: Higher intensity = stronger effect
        - age: Lower age = stronger effect (youth more susceptible)
        - celebrity_agreement: Higher agreement = stronger effect
        - vulnerability: Higher vulnerability = stronger effect
        
        Parameters:
        -----------
        agent : StandardAgent
            The agent to modify
        model : NetworkedModel
            The model containing the agent
        """
        # Calculate effect multiplier based on agent characteristics
        # Normalize age (assuming age range 0-100, inverse relationship)
        age_factor = 1.0 - (agent.age / 100.0) if agent.age <= 100 else 0.5
        
        # Celebrity agreement and vulnerability have direct relationships
        agreement_factor = agent.celebrity_agreement
        vulnerability_factor = agent.vulnerability
        
        # Combined effect multiplier
        base_effect = self.news_intensity
        effect_multiplier = base_effect * (1 + age_factor + agreement_factor + vulnerability_factor) / 4.0
        
        # Increase carrying capacity of aversive internal state
        # Higher carrying capacity allows more aversive feelings to accumulate
        current_capacity = agent.parameters.aversion.carrying_capacity
        capacity_increase = 0 * effect_multiplier
        new_capacity = min(1.0, current_capacity + capacity_increase)
        # agent.parameters.set_aversion_coefficients(carrying_capacity=new_capacity)
        
        # Decrease stress decay (stress persists longer)
        # Lower decay means stress takes longer to dissipate
        current_decay = agent.parameters.stress.decay
        decay_decrease = 0 * effect_multiplier  # Decrease by up to 0.5, modulated by factors
        new_decay = max(0.1, current_decay - decay_decrease)  # Don't go below 0.1
        # agent.parameters.set_stress_coefficients(decay=new_decay)
    
    def __repr__(self):
        return f"NewsStation(id={self.station_id}, opinion={self.opinion_scalar:.2f}, agents={len(self.agents)})"