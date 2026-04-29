import numpy as np
from numba import njit, prange
import numba


@njit(fastmath=True, cache=True)
def update_news_signal_vectorized(
    prev_state,
    dt,
    decay,
    diffusion_rate,
    neighbor_data,
    neighbor_counts,
    neighbor_offsets
):
    n = len(prev_state)
    new_signal = np.empty(n, dtype=np.float32)

    for i in prange(n):
        # --- Decay term ---
        val = prev_state[i] + dt * (-decay * prev_state[i])

        # --- Diffusion term ---
        n_neighbors = neighbor_counts[i]

        if n_neighbors > 0:
            offset = neighbor_offsets[i]
            spillover = 0.0

            for j in range(n_neighbors):
                idx = offset + j
                neighbor_id = int(neighbor_data[idx, 0])
                weight = neighbor_data[idx, 1]

                spillover += weight * prev_state[neighbor_id]

            # Normalize (keeps scale stable)
            spillover /= n_neighbors

            val += dt * diffusion_rate * spillover

        # Clamp to [0, 1]
        new_signal[i] = max(0.0, min(1.0, val))

    return new_signal

@njit(fastmath=True, cache=True)
def update_stress_vectorized(
    prev_state,
    dt,
    external_strat,
    baseline,
    decay,
    impulse_rate,
    impulse_strength,
    morning_impulse,
    alpha,
    beta,
    gamma,
    sigma,
    random_state
):
    """
    Vectorized stress update for all agents simultaneously.
    """
    n = len(prev_state)
    new_stress = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        E = external_strat[i]
        B_t = baseline + alpha * E
        lambda_t = decay + beta * E
        
        if morning_impulse[i] > 0:
            I_t = morning_impulse[i]
        else:
            # Poisson event
            poisson_val = np.random.poisson(impulse_rate * dt)
            I_t = poisson_val * impulse_strength * (1 - gamma * E)
        
        stress = B_t + np.exp(-lambda_t * dt) * (prev_state[i] - B_t) + I_t
        
        # Add noise
        dW = np.random.normal(0, np.sqrt(dt))
        stress += dW * sigma
        
        # Clamp
        new_stress[i] = max(0.0, min(1.0, stress))
    
    return new_stress


@njit(fastmath=True, cache=True, parallel=True)
def update_aversive_internal_state_vectorized(
    prev_state,
    dt,
    stress,
    suicidal_thought,
    escape_behavior,
    internal_strat,
    burdensomeness,
    clustering_coefficient,
    feedback,
    carrying_capacity,
    S_weight,
    T_weight,
    X_weight,
    I_weight,
    B_weight,
    c_weight
):
    """
    Vectorized aversive internal state update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        val = prev_state[i] + dt * (
            feedback * prev_state[i] * (carrying_capacity - prev_state[i])
            + S_weight * stress[i]
            - T_weight * suicidal_thought[i]
            - X_weight * escape_behavior[i]
            - I_weight * internal_strat[i]
            + B_weight * burdensomeness[i]
            - c_weight * clustering_coefficient[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_suicide_history_vectorized(
    prev_state,
    dt,
    suicidal_thought,
    decay
):
    """
    Vectorized suicide history update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        if suicidal_thought[i] < prev_state[i]:
            val = prev_state[i] + dt * (-decay * prev_state[i])
        else:
            val = suicidal_thought[i]
        
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_urge_to_escape_vectorized(
    prev_state,
    dt,
    aversive_internal_state,
    suicide_history,
    connectedness,
    feedback,
    A_weight,
    M_weight,
    C_weight
):
    """
    Vectorized urge to escape update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        val = prev_state[i] + dt * (
            -feedback * prev_state[i]
            + A_weight * aversive_internal_state[i]
            + M_weight * suicide_history[i]
            - C_weight * connectedness[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_sigmoid_vectorized(
    prev_state,
    dt,
    urge_to_escape,
    feedback,
    sig_middle,
    sig_steepness,
    news_signal
):
    """
    Vectorized sigmoid update (for suicidal thought and escape behavior).
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        if news_signal is not None:
            threshold = sig_middle - news_signal[i] * 0.05
        else:
            threshold = sig_middle
        sigmoid = 1.0 / (1.0 + np.exp(-sig_steepness * (urge_to_escape[i] - threshold)))
        val = prev_state[i] + dt * (-feedback * prev_state[i] + sigmoid)
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_strategy_for_escape_vectorized(
    prev_state,
    dt,
    aversive_internal_state,
    urge_to_escape,
    feedback,
    carrying_capacity,
    A_weight,
    U_weight
):
    """
    Vectorized strategy for escape update (external or internal).
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        val = prev_state[i] + dt * (
            feedback * prev_state[i] * (carrying_capacity - prev_state[i])
            + A_weight * aversive_internal_state[i]
            - U_weight * urge_to_escape[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True)
def update_burdensomeness_vectorized(
    prev_state,
    dt,
    internal_strat,
    neighbor_data,
    neighbor_counts,
    neighbor_offsets,
    aversive_states,
    feedback,
    A_weight,
    I_weight,
    B_lonely
):
    """
    Vectorized burdensomeness update.
    
    neighbor_data: flattened array of (neighbor_id, weight) pairs
    neighbor_counts: number of neighbors per agent
    neighbor_offsets: cumulative offset into neighbor_data
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        n_neighbors = neighbor_counts[i]
        
        if n_neighbors == 0:
            new_state[i] = B_lonely
        else:
            offset = neighbor_offsets[i]
            
            # Calculate weighted mean of neighbor aversive states
            weighted_sum = 0.0
            for j in range(n_neighbors):
                idx = offset + j
                neighbor_id = int(neighbor_data[idx, 0])
                weight = neighbor_data[idx, 1]
                weighted_sum += aversive_states[neighbor_id] * weight
            
            mean_effect = weighted_sum / n_neighbors
            
            val = prev_state[i] + dt * (
                A_weight * mean_effect
                - I_weight * internal_strat[i]
                - feedback * prev_state[i]
            )
            new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


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


class NumbaAgentUpdater:
    """
    Numba-optimized agent updater that processes all agents simultaneously.
    """
    
    def __init__(self):
        pass
    
    def update_all_agents(self, agent_states, params, dt, neighbor_data, 
                         neighbor_counts, neighbor_offsets):
        """
        Update all agents in a single vectorized call.
        
        agent_states: dict with keys matching tracked variables
        params: dict with parameter arrays
        """
        n = len(agent_states['stress'])

        # Update news signal
        agent_states['news_signal'] = update_news_signal_vectorized(
            agent_states['news_signal'],
            dt,
            params['news_signal']['decay'],
            params['news_signal']['diffusion_rate'],
            neighbor_data,
            neighbor_counts,
            neighbor_offsets,
        )
        
        # Update stress
        agent_states['stress'] = update_stress_vectorized(
            agent_states['stress'],
            dt,
            agent_states['external_strat'],
            params['stress']['baseline'],
            params['stress']['decay'],
            params['stress']['impulse_rate'],
            params['stress']['impulse_strength'],
            params['stress']['morning_impulse'],
            params['stress']['alpha'],
            params['stress']['beta'],
            params['stress']['gamma'],
            params['stress']['sigma'],
            None  # random_state placeholder
        )
        
        # Update aversive internal state
        agent_states['aversive_internal_state'] = update_aversive_internal_state_vectorized(
            agent_states['aversive_internal_state'],
            dt,
            agent_states['stress'],
            agent_states['suicidal_thought'],
            agent_states['escape_behavior'],
            agent_states['internal_strat'],
            agent_states['burdensomeness'],
            agent_states['clustering_coefficient'],
            params['aversion']['feedback'],
            params['aversion']['carrying_capacity'],
            params['aversion']['S_weight'],
            params['aversion']['T_weight'],
            params['aversion']['X_weight'],
            params['aversion']['I_weight'],
            params['aversion']['B_weight'],
            params['aversion']['c_weight']
        )
        
        # Update urge to escape
        agent_states['urge_to_escape'] = update_urge_to_escape_vectorized(
            agent_states['urge_to_escape'],
            dt,
            agent_states['aversive_internal_state'],
            agent_states['suicide_history'],
            agent_states['connectedness'],
            params['urge']['feedback'],
            params['urge']['A_weight'],
            params['urge']['M_weight'],
            params['urge']['C_weight']
        )
        
        # Update suicide history
        agent_states['suicide_history'] = update_suicide_history_vectorized(
            agent_states['suicide_history'],
            dt,
            agent_states['suicidal_thought'],
            params['suicide_history']['decay']
        )
        
        # Update suicidal thought
        agent_states['suicidal_thought'] = update_sigmoid_vectorized(
            agent_states['suicidal_thought'],
            dt,
            agent_states['urge_to_escape'],
            params['suicidal_thought']['feedback'],
            params['suicidal_thought']['sig_middle'],
            params['suicidal_thought']['sig_steepness'],
            agent_states['news_signal'],
        )
        
        # Update escape behavior
        agent_states['escape_behavior'] = update_sigmoid_vectorized(
            agent_states['escape_behavior'],
            dt,
            agent_states['urge_to_escape'],
            params['escape_behavior']['feedback'],
            params['escape_behavior']['sig_middle'],
            params['escape_behavior']['sig_steepness'],
            None
        )
        
        # Update external strategy
        agent_states['external_strat'] = update_strategy_for_escape_vectorized(
            agent_states['external_strat'],
            dt,
            agent_states['aversive_internal_state'],
            agent_states['urge_to_escape'],
            params['external']['feedback'],
            params['external']['carrying_capacity'],
            params['external']['A_weight'],
            params['external']['U_weight']
        )
        
        # Update internal strategy
        agent_states['internal_strat'] = update_strategy_for_escape_vectorized(
            agent_states['internal_strat'],
            dt,
            agent_states['aversive_internal_state'],
            agent_states['urge_to_escape'],
            params['internal']['feedback'],
            params['internal']['carrying_capacity'],
            params['internal']['A_weight'],
            params['internal']['U_weight']
        )
        
        # Update burdensomeness
        agent_states['burdensomeness'] = update_burdensomeness_vectorized(
            agent_states['burdensomeness'],
            dt,
            agent_states['internal_strat'],
            neighbor_data,
            neighbor_counts,
            neighbor_offsets,
            agent_states['aversive_internal_state'],
            params['burden']['feedback'],
            params['burden']['A_weight'],
            params['burden']['I_weight'],
            params['burden']['B_lonely']
        )
        
        return agent_states