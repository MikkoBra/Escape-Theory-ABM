import mesa
from model.agents.AgentFactory import AgentFactory
from model.system_updates.state_registry import register_all_states
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


class DefaultModel(mesa.Model):
    """
    Agent-based model of suicidality in a small world network.
    """
    def __init__(self, dt, n=15, seed=None, parameters={}, verbose=False, collect_all=False, warmup=20):
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.warmup=warmup
        self.verbose = verbose
        self.dt = dt
        self.num_agents = n
        self.time = 0
        if collect_all:
            self.datacollector = mesa.DataCollector(
                agent_reporters={
                    "Stress": "stress",
                    "Aversive Internal State": "aversive_internal_state",
                    "Urge to Escape": "urge_to_escape",
                    "Suicidal Thought": "suicidal_thought",
                    "Escape Behavior": "escape_behavior",
                    "External-Focused Change": "external_strat",
                    "Internal-Focused Change": "internal_strat",
                    "Social Burden": "burdensomeness",
                    "Memory of Suicidal Thought": "suicide_history",
                    "Time": "total_time",
                    "State": lambda agent: agent.state_manager.state.to_string(),
                }
            )
        else:
            self.datacollector = mesa.DataCollector(
                agent_reporters={
                    "Suicidal Thought": "suicidal_thought",
                    "Aversion": "aversive_internal_state",
                    "Time": "total_time",
                }
            )
        register_all_states()
        
        AgentFactory.create_agents(type="standard", model=self, n=n, stress_gen=True, default_params=parameters)
        mean_degree = self.num_agents - 1
        self.assign_connections(k=mean_degree, seed=seed)
        self.agents.do(lambda agent: agent.extract_neighbors())
        AgentFactory.create_agents(type="wang", model=self, n=1, stress_gen=True, default_params=parameters)

    def assign_connections(self, k=4, p=0.01, seed=None):
        """
        Generates a Newman Watts Strogatz network model with mean degree
        k and rewiring probability p as a reference for
        agents' social connections. Each agent is assigned a node in the
        network, and edges are given weights according to each agent's
        predefined number of close and medium connections. By design,
        it will first exhaust close connections, then medium, then distant,
        to ensure that each agent (if chosen to have at least one) has
        close connections.
        """
        agents = list(self.agents)
        N = len(agents)
        self.network = nx.newman_watts_strogatz_graph(n=N, k=k, p=p, seed=seed)
        nx.draw_circular(self.network, with_labels=True, node_color="steelblue", edge_color="gray", node_size=300)
        plt.show()

        # Spring layout (force-directed) — shows clustering but non-deterministic
        nx.draw_spring(self.network)
        plt.show()

        # Full manual control over layout + style
        pos = nx.circular_layout(self.network)
        nx.draw_networkx(self.network, pos=pos,
            node_color="steelblue",
            edge_color="gray",
            node_size=300,
            font_size=8,
            with_labels=True,
        )
        plt.show()
        if self.verbose:
            print(f"Number of edges:{self.network.number_of_edges()}")

            # Initialize edge weights
            print("Assigning edge weights.")
        for u, v in self.network.edges():
            agent_1 = self.agents[u]
            agent_2 = self.agents[v]
            if agent_1.close_connections != 0 and agent_2.close_connections != 0:
                self.network.edges[u, v]["strength"] = np.random.uniform(0.99, 1.0)
                agent_1.close_connections -= 1
                agent_2.close_connections -= 1
            elif agent_1.medium_connections != 0 and agent_2.medium_connections != 0:
                self.network.edges[u, v]["strength"] = np.random.uniform(0.4, 0.6)
                agent_1.medium_connections -= 1
                agent_2.medium_connections -= 1
            else:
                self.network.edges[u, v]["strength"] = np.random.uniform(0.05, 0.1)
        
        # Initialize agent clustering coefficients
        if self.verbose:
            print("Initializing clustering coefficients.")
        for agent in agents:
            agent.network_id = agent.unique_id - 1
            neighbors = list(self.network.neighbors(agent.network_id))
            if neighbors:
                avg_weight = np.mean([self.network.edges[agent.network_id, neighbor]["strength"] for neighbor in neighbors])
                clustering = nx.clustering(self.network, nodes=agent.network_id, weight="strength")
                weighted_clustering = clustering * avg_weight
            else:
                weighted_clustering = 0.0
            agent.clustering_coefficient = weighted_clustering

    def step(self):
        """
        Performs one timestep of the model.
        """
        if self.time >= self.warmup:
            self.datacollector.collect(self)
        self.agents.do(lambda agent: agent.update_agent(self.dt))
        self.time += self.dt
