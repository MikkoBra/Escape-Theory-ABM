from numba import njit, prange
import numpy as np

@njit(fastmath=True, cache=True, parallel=True)
def compute_connectedness_vectorized(neighbor_data, neighbor_counts, neighbor_offsets):
    """
    Compute connectedness for all agents based on neighbor weight entropy.
    """
    n = len(neighbor_counts)
    connectedness = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        n_neighbors = neighbor_counts[i]
        
        if n_neighbors == 0:
            connectedness[i] = 0.0
        else:
            offset = neighbor_offsets[i]
            
            # Extract weights
            weights = np.empty(n_neighbors, dtype=np.float32)
            for j in range(n_neighbors):
                weights[j] = neighbor_data[offset + j, 1]
            
            # Normalize to probability distribution
            weight_sum = np.sum(weights)
            p = weights / weight_sum
            
            # Calculate entropy
            entropy = 0.0
            for j in range(n_neighbors):
                if p[j] > 1e-8:
                    entropy -= p[j] * np.log(p[j])
            
            max_entropy = np.log(float(n_neighbors))
            connectedness[i] = 1.0 - (entropy / max_entropy)
    
    return connectedness