import numpy as np
from numba import njit, prange
from Constants import Constants


SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4
SOCIAL_EVENT = 5


STATE_DICT = {
    SLEEP:   "sleep",
    MORNING: "morning",
    COMMUTE: "commute",
    WORK:    "work",
    HOME:    "home",
    SOCIAL_EVENT: "social_event"
}


@njit(fastmath=True, cache=True)
def _new_state(state, time_of_day, day_length, is_weekend):
    """
    Return the next schedule state (integer) given the current one.

    Parameters
    ----------
    state       : int   — current state (SLEEP / MORNING / COMMUTE / WORK / HOME)
    time_of_day : float — current time modulo day_length
    day_length  : float — length of one full day in model time units
    is_weekend  : bool  — True if today is a weekend day for this agent
    """
    if state == SLEEP:
        return MORNING
    elif state == COMMUTE and time_of_day < day_length / 2:
        return WORK
    elif (state == MORNING and is_weekend) or state == COMMUTE:
        return HOME
    elif state == MORNING or state == WORK:
        return COMMUTE
    elif state == HOME:
        return SLEEP
    elif state == SOCIAL_EVENT:
        return HOME
    else:
        return SLEEP


@njit(fastmath=True, cache=True)
def _generate_sleep_time(time, wake_time, day_length, time_of_day):
    """Duration until the agent wakes up."""
    if time != 0.0:
        if time_of_day < wake_time:
            wakeup_time = (time - time_of_day) + wake_time
        else:
            wakeup_time = (time + (day_length - time_of_day)) + wake_time
        end_time = wakeup_time
    else:
        end_time = wake_time
    return end_time - time


@njit(fastmath=True, cache=True)
def _generate_morning_time(agent_id, time, day_length, work_time, commute, is_weekend):
    """Duration of the morning routine before commute or home."""
    last_midnight = time - (time % day_length)
    if is_weekend:
        morning_end = work_time          # sleep in, morning ends at leisure time
    else:
        morning_end = work_time - commute[agent_id]   # leave early enough to commute
    return last_midnight + morning_end - time


@njit(fastmath=True, cache=True)
def _generate_home_time(agent_id, time, day_length, wake_time,
                        sleep, mean_sleep_arr, sigma_sleep_arr):
    """Duration of the at home period before sleep."""
    time_of_day  = time % day_length
    wake_time_abs = (time + (day_length - time_of_day)) + wake_time

    if sleep[agent_id] != 0.0:
        sleep_hours = sleep[agent_id]
    else:
        mean_s  = mean_sleep_arr[agent_id]
        sigma_s = sigma_sleep_arr[agent_id]
        sleep_hours = max(1.0, np.random.normal(mean_s, sigma_s))

    sleep_hours_scaled = sleep_hours * day_length * (1.0 / 24.0)
    return wake_time_abs - sleep_hours_scaled - time


@njit(fastmath=True, cache=True)
def _generate_state_time(agent_id, next_state, time, time_of_day, day_length,
                         work_time, workday_length, wake_time,
                         sleep, mean_sleep, sigma_sleep, commute, is_weekend,
                         social_event_duration):
    if next_state == SLEEP:
        return _generate_sleep_time(time, wake_time, day_length, time_of_day)
    elif next_state == MORNING:
        return _generate_morning_time(agent_id, time, day_length,
                                      work_time, commute, is_weekend)
    elif next_state == COMMUTE:
        return commute[agent_id]
    elif next_state == WORK:
        return workday_length
    elif next_state == HOME:
        return _generate_home_time(agent_id, time, day_length, wake_time,
                                   sleep, mean_sleep, sigma_sleep)
    elif next_state == SOCIAL_EVENT:
        return social_event_duration
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Main vectorised update — called every model step
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True, parallel=True)
def update_time_left(
    time_left,
    states,
    time,
    dt,
    day_length,
    work_time,
    workday_length,
    wake_time,
    sleep,
    mean_sleep,
    sigma_sleep,
    commute,
    weekends,
    social_event_duration
):
    """
    Advance all agents' schedule states by one timestep *dt*.

    Parameters
    ----------
    time_left     : float32[n]  — remaining time in current state per agent
    states        : int32[n]    — current state integer per agent
    time          : float       — current model time
    dt            : float       — timestep size
    day_length    : float       — model time units per day  (scalar)
    work_time     : float       — time-of-day when work / leisure starts (scalar)
    workday_length: float       — duration of a work period (scalar)
    wake_time     : float       — time-of-day the agent wakes (scalar)
    sleep         : float32[n]  — fixed sleep duration per agent (0 = draw from distribution)
    mean_sleep    : float32[n]  — mean of sleep distribution per agent
    sigma_sleep   : float32[n]  — std of sleep distribution per agent
    commute       : float32[n]  — commute duration per agent
    weekends      : bool[n]     — whether agents experience weekends

    Returns
    -------
    new_states : int32[n]
    new_times  : float32[n]
    """
    n = len(time_left)
    new_times  = np.empty(n, dtype=np.float32)
    new_states = np.empty(n, dtype=np.int32)
    time_of_day = time % day_length

    for i in prange(n):
        if time_left[i] - dt <= 0.0:
            is_weekend = weekends[i] and (int(time / day_length) % 7 >= 5)
            next_st    = _new_state(states[i], time_of_day, day_length, is_weekend)
            new_times[i] = _generate_state_time(
                i, next_st, time, time_of_day, day_length,
                work_time, workday_length, wake_time,
                sleep, mean_sleep, sigma_sleep, commute, is_weekend,
                social_event_duration,
            )
            new_states[i] = next_st
        else:
            new_times[i]  = time_left[i] - dt
            new_states[i] = states[i]

    return new_states, new_times


# ---------------------------------------------------------------------------
# ScheduleManager — thin Python wrapper used by NetworkedModel
# ---------------------------------------------------------------------------

class ScheduleManager:

    SOCIAL_EVENT_DURATION = (2/24) * Constants.DAY_LENGTH

    def __init__(self, constants, social_events=False, only_sleep=False):
        self.prev_states = None
        self.default_coeffs = {
            k: v.copy()
            for k, v in constants["coeff_arrays"].items()
        }
        self.social_events = social_events
        self.only_sleep=only_sleep

    def init_schedule(self, num_agents, wake_time):
        """
        Initialise every agent in the SLEEP state, with time_left set so they
        all wake up at *wake_time* on day 0.

        Parameters
        ----------
        num_agents : int
        wake_time  : float  — pass Constants.WAKE_TIME
        """
        return {
            'state':     np.full(num_agents, SLEEP,     dtype=np.int32),
            'time_left': np.full(num_agents, wake_time, dtype=np.float32),
        }

    def update_states(self, dt, time, states, time_left, constants):
        """
        Drive the vectorised Numba kernel.

        Expects ``constants`` to contain:
            'char_arrays'        : dict with float32[n] arrays:
                                   'sleep', 'mean_sleep', 'sigma_sleep',
                                   'commute', 'weekends'
            'day_length'         : float scalar
            'work_time'          : float scalar
            'workday_length'     : float scalar
            'wake_time'          : float scalar

        Returns
        -------
        new_states : int32[n]
        new_times  : float32[n]
        """

        self.prev_states = states.copy()

        char   = constants['char_arrays']
        n      = len(states)
        zeros  = np.zeros(n, dtype=np.float32)

        sleep       = char.get('sleep',       zeros)
        mean_sleep  = char.get('mean_sleep',  zeros)
        sigma_sleep = char.get('sigma_sleep', zeros)
        commute     = char.get('commute',     zeros)
        weekends    = char.get('weekends',    np.zeros(n, dtype=np.bool_))
        day_length = float(Constants.DAY_LENGTH)
        work_time = float(Constants.WORK_TIME)
        workday_length = float(Constants.WORKDAY_LENGTH)
        wake_time = float(Constants.WAKE_TIME)

        new_states, new_times = update_time_left(
            time_left,
            states,
            float(time),
            float(dt),
            day_length,
            work_time,
            workday_length,
            wake_time,
            sleep,
            mean_sleep,
            sigma_sleep,
            commute,
            weekends,
            float(self.SOCIAL_EVENT_DURATION),
        )

        return new_states, new_times
    
    def _draw_social_event(self, connectedness, lambda_max=5):
        """
        For each agent entering HOME:
          - If social_events_mask[i] is False → never attend.
          - If social_event_today[i] is True  → already attended today, skip.
          - Otherwise draw from Poisson: λ = connectedness * LAMBDA_MAX.
            Attendance = (Poisson draw >= 1), i.e. 1 - exp(-λ).
            connectedness=0 → λ=0 → P=0.
            connectedness=1 → λ=LAMBDA_MAX → P≈1 when LAMBDA_MAX is large (e.g. 5).
        """
        lam = np.clip(connectedness, 0.0, 1.0) * lambda_max
        lam = np.nan_to_num(lam, nan=0.0)
        draws = np.random.poisson(lam)
        return draws >= 1

    def _enter_social_event(self, schedule_states, schedule_times, agent_states, constants, candidate_mask):
        """
        Inject SOCIAL_EVENT for candidates only if the remaining home time
        exceeds the event duration (so the event fits within the home period).
        """
        fits = schedule_times > float(self.SOCIAL_EVENT_DURATION)
        mask = candidate_mask & fits

        if not np.any(mask):
            return

        schedule_states[mask] = SOCIAL_EVENT
        agent_states['home_time_remaining'][mask] = schedule_times[mask] - float(self.SOCIAL_EVENT_DURATION)
        schedule_times[mask]  = float(self.SOCIAL_EVENT_DURATION)
        agent_states['social_event_today'][mask] = 1.0

        constants['coeff_arrays']['urge_to_escape_C_weight'][mask] = np.minimum(
            constants['coeff_arrays']['urge_to_escape_C_weight'][mask] + 0.3, 2.0
        )
        constants['coeff_arrays']['aversive_internal_state_B_weight'][mask] = np.maximum(
            constants['coeff_arrays']['aversive_internal_state_B_weight'][mask] - 2, 0.0
        )

    def _exit_social_event(self, prev, schedule_states, schedule_times, agent_states, constants):
        social_to_home = (prev == SOCIAL_EVENT) & (schedule_states == HOME)
        if not np.any(social_to_home):
            return

        schedule_times[social_to_home] = agent_states['home_time_remaining'][social_to_home]
        constants['coeff_arrays']['urge_to_escape_C_weight'][social_to_home] = \
            self.default_coeffs['urge_to_escape_C_weight'][social_to_home]
        constants['coeff_arrays']['aversive_internal_state_B_weight'][social_to_home] = \
            self.default_coeffs['aversive_internal_state_B_weight'][social_to_home]


    def _transition_to_sleep(self, schedule_times, agent_states, constants, mask):
        # Record sleep length for next morning's stress impulse
        agent_states['prev_sleep'][mask] = schedule_times[mask]
        # No stress spikes during sleep, it reduces to 0
        constants['coeff_arrays']['stress_impulse_rate'][mask] = 0
        # constants['coeff_arrays']['stress_sigma'][mask] = 0
        # constants['coeff_arrays']['stress_baseline'][mask] = 0


    def _transition_to_morning(self, prev, schedule_states, agent_states, constants):
        sleep_to_morning = (prev == SLEEP) & (schedule_states == MORNING)

        # Reset stress parameters to default
        constants['coeff_arrays']['stress_impulse_rate'][sleep_to_morning] = self.default_coeffs['stress_impulse_rate'][sleep_to_morning]
        # constants['coeff_arrays']['stress_sigma'][sleep_to_morning] = self.default_coeffs['stress_sigma'][sleep_to_morning]
        # constants['coeff_arrays']['stress_baseline'][sleep_to_morning] = self.default_coeffs['stress_baseline'][sleep_to_morning]

        # Calculate sleep deficit from sleep state
        sleep_duration = agent_states['prev_sleep'][sleep_to_morning]
        sleep_deficit = np.zeros_like(agent_states['prev_sleep'], dtype=np.float32)
        if sleep_duration.size > 0:
            deficit_vals = (Constants.HEALTHY_SLEEP - sleep_duration)
            deficit_vals = np.clip(deficit_vals, 0.0, None)
            sleep_deficit[sleep_to_morning] = deficit_vals

        # Compute morning impulse
        if np.any(sleep_to_morning):
            base_strength = constants['coeff_arrays']['stress_impulse_strength'][sleep_to_morning]

            morning_impulse = base_strength * (
                1.0 + np.exp(
                    sleep_deficit[sleep_to_morning] / Constants.HEALTHY_SLEEP
                )
            )

            agent_states['morning_impulse'][sleep_to_morning] = morning_impulse
        
        if not self.only_sleep and np.any(sleep_to_morning):
            # Decrease suicidal thought threshold
            old_threshold = constants['coeff_arrays']['suicidal_thought_sig_middle'][sleep_to_morning]

            new_T_threshold = (
                old_threshold
                - 0.1 * sleep_deficit[sleep_to_morning] / Constants.HEALTHY_SLEEP
            )

            constants['coeff_arrays']['suicidal_thought_sig_middle'][sleep_to_morning] = (
                new_T_threshold
            )

        agent_states['social_event_today'][sleep_to_morning] = 0.0
    
    def _transition_to_commute(self, constants, mask):
        # Escape behavior isn't possible while commuting
        constants['coeff_arrays']['escape_behavior_sig_middle'][mask] = 1.1
        
        # Increase baseline stress
        new_stress_baseline = np.minimum(constants['coeff_arrays']['stress_baseline'][mask] + 0.2, 1)
        constants['coeff_arrays']['stress_baseline'][mask] = new_stress_baseline

    def _transition_to_work(self, constants, mask):
        
        # Increase weight of urge to escape on internal and external strat
        new_external_U_weight = constants['coeff_arrays']['external_strat_U_weight'][mask] + 0.3
        constants['coeff_arrays']['external_strat_U_weight'][mask] = new_external_U_weight
        new_internal_U_weight = constants['coeff_arrays']['internal_strat_U_weight'][mask] + 0.3
        constants['coeff_arrays']['internal_strat_U_weight'][mask] = new_internal_U_weight
    
    def _transition_to_home(self, constants, mask):
        # Escape behavior is easier
        old_threshold = constants['coeff_arrays']['escape_behavior_sig_middle'][mask]
        new_threshold = np.maximum(old_threshold - 0.02, 0)
        constants['coeff_arrays']['escape_behavior_sig_middle'][mask] = new_threshold

        # Suicidal thought decays less quickly
        old_feedback = constants['coeff_arrays']['suicidal_thought_feedback'][mask]
        new_feedback = np.maximum(old_feedback - 0.1, 0)
        constants['coeff_arrays']['suicidal_thought_feedback'][mask] = new_feedback

    def _transition_from_home(self, prev, schedule_states, schedule_times, agent_states, constants):
        # ─────────────────────────────────────────────
        # HOME -> SLEEP
        # ─────────────────────────────────────────────
        home_to_sleep = (prev == HOME) & (schedule_states == SLEEP)
        
        # Reset decreased escape behavior threshold and increased suicidal thought feedback
        # to default
        constants['coeff_arrays']['escape_behavior_sig_middle'][home_to_sleep] =\
              self.default_coeffs['escape_behavior_sig_middle'][home_to_sleep]
        constants['coeff_arrays']['suicidal_thought_feedback'][home_to_sleep] =\
              self.default_coeffs['suicidal_thought_feedback'][home_to_sleep]

        self._transition_to_sleep(
            schedule_times=schedule_times,
            agent_states=agent_states,
            constants=constants,
            mask=home_to_sleep
            )


    def _transition_from_morning(self, prev, schedule_states, constants):
        # ─────────────────────────────────────────────
        # MORNING -> COMMUTE
        # ─────────────────────────────────────────────
        morning_to_commute = (prev == MORNING) & (schedule_states == COMMUTE)

        # Reset decreased suicidal thought threshold from morning to default
        constants['coeff_arrays']['suicidal_thought_sig_middle'][morning_to_commute] =\
              self.default_coeffs['suicidal_thought_sig_middle'][morning_to_commute]
        
        self._transition_to_commute(constants=constants, mask=morning_to_commute)

        # ─────────────────────────────────────────────
        # MORNING -> HOME
        # ─────────────────────────────────────────────
        morning_to_home = (prev == MORNING) & (schedule_states == HOME)

        # Reset decreased suicidal thought threshold from morning to default
        constants['coeff_arrays']['suicidal_thought_sig_middle'][morning_to_home] =\
              self.default_coeffs['suicidal_thought_sig_middle'][morning_to_home]

        self._transition_to_home(constants=constants, mask=morning_to_home)
    
    def _transition_from_commute(self, prev, schedule_states, constants):
        # ─────────────────────────────────────────────
        # COMMUTE -> WORK
        # ─────────────────────────────────────────────

        commute_to_work = (prev == COMMUTE) & (schedule_states == WORK)
        
        # Reset increased stress baseline and increased escape behavior threshold
        # from commute to default
        constants['coeff_arrays']['stress_baseline'][commute_to_work] =\
              self.default_coeffs['stress_baseline'][commute_to_work]
        constants['coeff_arrays']['escape_behavior_sig_middle'][commute_to_work] =\
              self.default_coeffs['escape_behavior_sig_middle'][commute_to_work]
        
        self._transition_to_work(constants=constants, mask=commute_to_work)

        # ─────────────────────────────────────────────
        # COMMUTE -> HOME
        # ─────────────────────────────────────────────

        commute_to_home = (prev == COMMUTE) & (schedule_states == HOME)
        
        # Reset increased stress baseline and increased escape behavior threshold
        # from commute to default
        constants['coeff_arrays']['stress_baseline'][commute_to_home] =\
              self.default_coeffs['stress_baseline'][commute_to_home]
        constants['coeff_arrays']['escape_behavior_sig_middle'][commute_to_home] =\
              self.default_coeffs['escape_behavior_sig_middle'][commute_to_home]
        
        self._transition_to_home(constants=constants, mask=commute_to_home)
    
    def _transition_from_work(self, prev, schedule_states, constants):
        # ─────────────────────────────────────────────
        # WORK -> COMMUTE
        # ─────────────────────────────────────────────
        work_to_commute = (prev == WORK) & (schedule_states == COMMUTE)

        constants['coeff_arrays']['internal_strat_U_weight'][work_to_commute] =\
            self.default_coeffs['internal_strat_U_weight'][work_to_commute]
        constants['coeff_arrays']['external_strat_U_weight'][work_to_commute] =\
            self.default_coeffs['external_strat_U_weight'][work_to_commute]

        self._transition_to_commute(constants, work_to_commute)


    def apply_transition_effects(self, schedule_states, schedule_times, agent_states, constants):
        prev = self.prev_states

        # ─────────────────────────────────────────────
        # SOCIAL_EVENT -> HOME
        # ─────────────────────────────────────────────
        if not self.only_sleep and self.social_events:
            self._exit_social_event(prev, schedule_states, schedule_times, agent_states, constants)

        # ─────────────────────────────────────────────
        # HOME -> SLEEP
        # ─────────────────────────────────────────────
        self._transition_from_home(prev=prev, schedule_states=schedule_states, schedule_times=schedule_times,
                                  agent_states=agent_states, constants=constants)

        # ─────────────────────────────────────────────
        # SLEEP -> MORNING
        # ─────────────────────────────────────────────
        self._transition_to_morning(prev=prev, schedule_states=schedule_states,
                                    agent_states=agent_states, constants=constants)
        
        if not self.only_sleep:
            # ─────────────────────────────────────────────
            # MORNING -> COMMUTE OR MORNING -> HOME
            # ─────────────────────────────────────────────
            self._transition_from_morning(prev=prev, schedule_states=schedule_states, constants=constants)
            
            # ─────────────────────────────────────────────
            # COMMUTE -> WORK OR COMMUTE -> HOME
            # ─────────────────────────────────────────────
            self._transition_from_commute(prev=prev, schedule_states=schedule_states, constants=constants)

            # ─────────────────────────────────────────────
            # WORK -> COMMUTE
            # ─────────────────────────────────────────────
            self._transition_from_work(prev=prev, schedule_states=schedule_states, constants=constants)

            if self.social_events:
                # Agents that just entered HOME this step
                just_entered_home = (prev != HOME) & (schedule_states == HOME)

                if np.any(just_entered_home):
                    # Only consider agents whose type enables social events
                    social_events_mask = constants.get('social_events_mask',
                                                    np.ones(len(schedule_states), dtype=bool))
                    # Once-per-day guard
                    already_attended = agent_states['social_event_today'].astype(bool)

                    eligible = just_entered_home & social_events_mask & ~already_attended

                    if np.any(eligible):
                        connectedness = constants['connectedness']
                        event_lambda = constants["char_arrays"].get("event_lambda", 5)
                        attends = self._draw_social_event(connectedness[eligible], lambda_max=event_lambda)

                        attend_mask = np.zeros(len(schedule_states), dtype=bool)
                        eligible_indices = np.where(eligible)[0]
                        attend_mask[eligible_indices[attends]] = True

                        if np.any(attend_mask):
                            self._enter_social_event(
                                schedule_states, schedule_times,
                                agent_states, constants, attend_mask
                            )