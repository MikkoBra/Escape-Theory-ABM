import numpy as np
from numba import njit, prange
from Constants import Constants


SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4


STATE_DICT = {
    SLEEP:   "sleep",
    MORNING: "morning",
    COMMUTE: "commute",
    WORK:    "work",
    HOME:    "home",
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
                         sleep, mean_sleep, sigma_sleep, commute, is_weekend):
    """Selector for state duration generators."""
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
    else:
        return 0.0   # safe fallback


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
        )

        return new_states, new_times