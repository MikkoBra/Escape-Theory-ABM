import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def propagate_news_signal(
    news_signal,
    prev_news_signal,
    neighbor_data,
    neighbor_counts,
    neighbor_offsets,
    opinion_scalars,
    dt,
    base_rate,
    sigma_opinion
):
    n = len(news_signal)

    new_signal = np.copy(news_signal)

    for i in range(n):
        if prev_news_signal[i] <= 0.0:
            continue

        n_neighbors = neighbor_counts[i]
        offset = neighbor_offsets[i]

        for j in range(n_neighbors):
            idx = offset + j

            neighbor = int(neighbor_data[idx, 0])
            weight = neighbor_data[idx, 1]

            # opinion similarity
            diff = opinion_scalars[i] - opinion_scalars[neighbor]
            sim = np.exp(-(diff * diff) / (2 * sigma_opinion * sigma_opinion))

            # Approximation of poisson process
            rate = base_rate * prev_news_signal[i] * weight * sim

            p = 1.0 - np.exp(-rate * dt)

            if np.random.random() < p:
                # transmission (choose model)
                new_signal[neighbor] = max(new_signal[neighbor], prev_news_signal[i] * weight)

    return new_signal

@njit(fastmath=True, cache=True)
def update_news_signal(
    prev_state,
    dt,
    decay,
):
    n = len(prev_state)
    new_signal = np.empty(n, dtype=np.float32)
    for i in prange(n):
        # Decay term
        if prev_state[i] != 0:
            val = prev_state[i] + dt * (-decay[i] * prev_state[i])

            new_signal[i] = max(0.0, min(1.0, val))

    return new_signal


@njit(fastmath=True, cache=True)
def update_stress(
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
):
    """
    Vectorized stress update for all agents simultaneously.

    Parameters
    ----------
    prev_state: float
        Stress value from previous timestep
    dt: float
        Size of timestep
    external_strat: float
        External strategy value from previous timestep
    baseline: float
        Base (minimum possible) stress value
    decay: float
        Decay rate of stress signal
    impulse_rate: float
        Rate of stress impulse occurrence
    impulse_strength: float
        Magnitude of stress impulses
    morning_impulse: float
        Magnitude of morning impulse (0 at all times except when it's administered)
    alpha: float
        Size of effect of external strategy on baseline stress increase
    beta: float
        Size of effect of external strategy on stress decay
    gamma: float
        Size of effect of external strategy on stress spike magnitude reduction
    sigma: float
        Strength of random noise signal
    """
    n = len(prev_state)
    new_stress = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        E = external_strat[i]
        B_t = baseline[i] + alpha[i] * E
        lambda_t = decay[i] + beta[i] * E
        
        if morning_impulse[i] > 0:
            I_t = morning_impulse[i]
            morning_impulse[i] = 0
        else:
            # Poisson event
            poisson_val = np.random.poisson(impulse_rate[i] * dt)
            I_t = poisson_val * impulse_strength[i] * (1 - gamma[i] * E)
        
        stress = B_t + np.exp(-lambda_t * dt) * (prev_state[i] - B_t) + I_t
        
        # Add noise
        dW = np.random.normal(0, np.sqrt(dt))
        stress += dW * sigma[i]
        
        # Clamp
        new_stress[i] = max(0.0, min(1.0, stress))
    
    return new_stress


@njit(fastmath=True, cache=True, parallel=True)
def update_aversive_internal_state(
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
    c_weight,
):
    """
    Vectorized aversive internal state update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        val = prev_state[i] + dt * (
            feedback[i] * prev_state[i] * (carrying_capacity[i] - prev_state[i])
            + S_weight[i] * stress[i]
            - T_weight[i] * suicidal_thought[i]
            - X_weight[i] * escape_behavior[i]
            - I_weight[i] * internal_strat[i]
            + B_weight[i] * burdensomeness[i]
            - c_weight[i] * clustering_coefficient[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_suicide_history(
    prev_state,
    dt,
    suicidal_thought,
    decay,
):
    """
    Vectorized suicide history update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        if suicidal_thought[i] < prev_state[i]:
            val = prev_state[i] + dt * (-decay[i] * prev_state[i])
        else:
            val = suicidal_thought[i]
        
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_urge_to_escape(
    prev_state,
    dt,
    aversive_internal_state,
    suicide_history,
    connectedness,
    feedback,
    A_weight,
    M_weight,
    C_weight,
):
    """
    Vectorized urge to escape update.
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        val = prev_state[i] + dt * (
            -feedback[i] * prev_state[i]
            + A_weight[i] * aversive_internal_state[i]
            + M_weight[i] * suicide_history[i]
            - C_weight[i] * connectedness[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_sigmoid(
    prev_state,
    dt,
    urge_to_escape,
    news_signal,
    sig_middle,
    sig_steepness,
    feedback,
):
    """
    Vectorized sigmoid update (for suicidal thought and escape behavior).
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        if news_signal is not None:
            threshold = sig_middle[i] - news_signal[i] * 0.05
        else:
            threshold = sig_middle[i]
        sigmoid = 1.0 / (1.0 + np.exp(-sig_steepness[i] * (urge_to_escape[i] - threshold)))
        val = prev_state[i] + dt * (-feedback[i] * prev_state[i] + sigmoid)
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True, parallel=True)
def update_strategy_for_escape(
    prev_state,
    dt,
    aversive_internal_state,
    urge_to_escape,
    feedback,
    carrying_capacity,
    A_weight,
    U_weight,
):
    """
    Vectorized strategy for escape update (external or internal).
    """
    n = len(prev_state)
    new_state = np.empty(n, dtype=np.float32)
    
    for i in prange(n):

        val = prev_state[i] + dt * (
            feedback[i] * prev_state[i] * (carrying_capacity[i] - prev_state[i])
            + A_weight[i] * aversive_internal_state[i]
            - U_weight[i] * urge_to_escape[i]
        )
        new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


@njit(fastmath=True, cache=True)
def update_burdensomeness(
    prev_state,
    dt,
    internal_strat,
    aversive_states,
    neighbor_data,
    neighbor_counts,
    neighbor_offsets,
    feedback,
    A_weight,
    I_weight,
    B_lonely,
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
            new_state[i] = B_lonely[i]
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
                A_weight[i] * mean_effect
                - I_weight[i] * internal_strat[i]
                - feedback[i] * prev_state[i]
            )
            new_state[i] = max(0.0, min(1.0, val))
    
    return new_state


class AgentUpdater:
    """
    Numba-optimized agent updater that processes all agents simultaneously.
    """
    
    def __init__(self, seed=None):
        np.random.seed(seed)
    
    def update_all_agents(self, agent_states, constants, dt, neighbor_data, 
                         neighbor_counts, neighbor_offsets, read_stress=False):
        """
        Update all agents in a single vectorized call.
        
        agent_states: dict with keys matching tracked variables
        params: dict with parameter arrays
        """
        coefficients = constants['coeff_arrays']
        opinion_scalars = constants['opinion_scalar']
        if 'news_signal' in agent_states:
            agent_states['news_signal'] = propagate_news_signal(
                agent_states['news_signal'],
                prev_news_signal=agent_states['news_signal'].copy(),
                neighbor_data=neighbor_data,
                neighbor_counts=neighbor_counts,
                neighbor_offsets=neighbor_offsets,
                opinion_scalars=opinion_scalars,
                dt=dt,
                base_rate=0.5,
                sigma_opinion=0.2
            )
            agent_states['news_signal'] = update_news_signal(
                agent_states['news_signal'], dt,
                coefficients['news_signal_decay'],
            )
 
        # ── stress ────────────────────────────────────────────────────────────
        if not read_stress and 'stress' in agent_states:
            agent_states['stress'] = update_stress(
                agent_states['stress'], dt,
                agent_states['external_strat'],
                coefficients['stress_baseline'],
                coefficients['stress_decay'],
                coefficients['stress_impulse_rate'],
                coefficients['stress_impulse_strength'],
                agent_states['morning_impulse'],
                coefficients['stress_alpha'],
                coefficients['stress_beta'],
                coefficients['stress_gamma'],
                coefficients['stress_sigma'],
            )
 
        # ── suicide history ───────────────────────────────────────────────────
        if "suicide_history" in agent_states:
            agent_states['suicide_history'] = update_suicide_history(
                agent_states['suicide_history'], dt,
                agent_states['suicidal_thought'],
                coefficients['suicide_history_decay']
            )
 
        # ── suicidal thought ──────────────────────────────────────────────────
        if "suicidal_thought" in agent_states:
            news_signal = agent_states.get('news_signal')
            agent_states['suicidal_thought'] = update_sigmoid(
                agent_states['suicidal_thought'], dt,
                agent_states['urge_to_escape'],
                news_signal,
                coefficients['suicidal_thought_sig_middle'],
                coefficients['suicidal_thought_sig_steepness'],
                coefficients['suicidal_thought_feedback'],
            )
 
        # ── escape behavior ───────────────────────────────────────────────────
        if "escape_behavior" in agent_states:
            agent_states['escape_behavior'] = update_sigmoid(
                agent_states['escape_behavior'], dt,
                agent_states['urge_to_escape'],
                None,
                coefficients['escape_behavior_sig_middle'],
                coefficients['escape_behavior_sig_steepness'],
                coefficients['escape_behavior_feedback'],
            )
 
        # ── external strategy ─────────────────────────────────────────────────
        if "external_strat" in agent_states:
            agent_states['external_strat'] = update_strategy_for_escape(
                agent_states['external_strat'], dt,
                agent_states['aversive_internal_state'],
                agent_states['urge_to_escape'],
                coefficients['external_strat_feedback'],
                coefficients['external_strat_carrying_capacity'],
                coefficients['external_strat_A_weight'],
                coefficients['external_strat_U_weight'],
            )
 
        # ── internal strategy ─────────────────────────────────────────────────
        if "internal_strat" in agent_states:
            agent_states['internal_strat'] = update_strategy_for_escape(
                agent_states['internal_strat'], dt,
                agent_states['aversive_internal_state'],
                agent_states['urge_to_escape'],
                coefficients['internal_strat_feedback'],
                coefficients['internal_strat_carrying_capacity'],
                coefficients['internal_strat_A_weight'],
                coefficients['internal_strat_U_weight'],
            )
 
        # ── urge to escape ────────────────────────────────────────────────────
        if 'urge_to_escape' in agent_states:
            suicide_history = agent_states.get("suicide_history", np.zeros_like(agent_states['urge_to_escape']))
            agent_states['urge_to_escape'] = update_urge_to_escape(
                agent_states['urge_to_escape'], dt,
                agent_states['aversive_internal_state'],
                suicide_history,
                constants['connectedness'],
                coefficients['urge_to_escape_feedback'],
                coefficients['urge_to_escape_A_weight'],
                coefficients['urge_to_escape_M_weight'],
                coefficients['urge_to_escape_C_weight'],
            )
 
        # ── aversive internal state ───────────────────────────────────────────
        if 'aversive_internal_state' in agent_states:
            burdensomeness = agent_states.get('burdensomeness', np.zeros_like(agent_states['aversive_internal_state']))
            agent_states['aversive_internal_state'] = \
                update_aversive_internal_state(
                    agent_states['aversive_internal_state'], dt,
                    agent_states['stress'],
                    agent_states['suicidal_thought'],
                    agent_states['escape_behavior'],
                    agent_states['internal_strat'],
                    burdensomeness,
                    constants['clustering_coefficient'],
                    coefficients['aversive_internal_state_feedback'],
                    coefficients['aversive_internal_state_carrying_capacity'],
                    coefficients['aversive_internal_state_S_weight'],
                    coefficients['aversive_internal_state_T_weight'],
                    coefficients['aversive_internal_state_X_weight'],
                    coefficients['aversive_internal_state_I_weight'],
                    coefficients['aversive_internal_state_B_weight'],
                    coefficients['aversive_internal_state_c_weight'],
                )
 
        # ── burdensomeness ────────────────────────────────────────────────────
        if "burdensomeness" in agent_states:
            agent_states['burdensomeness'] = update_burdensomeness(
                agent_states['burdensomeness'], dt,
                agent_states['internal_strat'],
                agent_states['aversive_internal_state'],
                neighbor_data, neighbor_counts, neighbor_offsets,
                coefficients['burdensomeness_feedback'],
                coefficients['burdensomeness_A_weight'],
                coefficients['burdensomeness_I_weight'],
                coefficients['burdensomeness_B_lonely'],
            )
 
        return agent_states