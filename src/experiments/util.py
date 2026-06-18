"""
Example usage of the optimized Numba-based networked model.

This demonstrates how to:
1. Set up and run the optimized model
2. Compare performance with original implementation
3. Validate results
4. Access and analyze output data
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from Constants import MINUTE_LENGTH
from scipy.stats import ttest_ind, mannwhitneyu
from itertools import islice
from matplotlib import rcParams
default_colors = rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib.lines import Line2D

SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4
SOCIAL_EVENT = 5

VAR_DICT = {
    "stress": "Stress",
    "suicidal_thought": "Suicidal Thought",
    "aversive_internal_state": "Aversive Internal State",
    "suicide_history": "Memory of Suicidal Thought",
    "urge_to_escape": "Urge to Escape",
    "escape_behavior": "Escape Behavior",
    "external_strat": "External Strategy",
    "internal_strat": "Internal Strategy",
    "burdensomeness": "Burdensomeness",
}


def run_basic_simulation(model, batch_save=False, run_name=None, batch_size=1000):
    print("\nRunning simulation...")
    start_time = time.time()

    if batch_save:
        from pathlib import Path

        save_dir = Path("src/output") / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        memmaps = {}

        for var in model.tracked_vars:
            if var == "labels":
                memmaps[var] = np.memmap(
                    save_dir / f"{var}.dat",
                    dtype=str,
                    mode="w+",
                    shape=(model.num_steps, model.num_agents)
                )
            else:
                memmaps[var] = np.memmap(
                    save_dir / f"{var}.dat",
                    dtype=np.float32,
                    mode="w+",
                    shape=(model.num_steps, model.num_agents)
                )

    for step in range(model.num_steps):
        model.step()

        if batch_save:
            for var in model.tracked_vars:
                if var == "labels":
                    memmaps[var][step, :] = model.constants[var]
                elif var == "state":
                    memmaps[var][step, :] = model.schedule['state']
                else:
                    memmaps[var][step, :] = model.states[var]

        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"Step {step + 1}/{model.num_steps} ({rate:.1f} steps/sec)")

            if batch_save and (step + 1) % batch_size == 0:
                for m in memmaps.values():
                    m.flush()

    elapsed_time = time.time() - start_time

    print(f"\n✓ Simulation complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/model.num_steps*1000:.2f} ms/step")
    print(f"Throughput: {model.num_steps/elapsed_time:.1f} steps/sec")

    if batch_save:
        print("✓ Data saved via memmap files.")


def load_simulation_data(run_name, vars_list, num_steps, num_agents):

    folder = Path("src/output") / run_name

    data = {}

    for var in vars_list:
        if var != "labels":
            data[var] = np.memmap(
                folder / f"{var}.dat",
                dtype=np.float32,
                mode="r",
                shape=(num_steps, num_agents)
            )
        else:
            data[var] = np.memmap(
                folder / f"{var}.dat",
                dtype=str,
                mode="r",
                shape=(num_steps, num_agents)
            )

        # convert to normal ndarray if needed
        data[var] = np.array(data[var])

    return data


def export_results(model, filename='None'):
    """Export simulation results to a file."""
    
    # Prepare data dictionary
    data_dict = {}
    for var in model.tracked_vars:
        data_dict[var] = model.data[var]
    
    # Save
    if filename is not None:
        np.savez_compressed(filename, **data_dict)
        print(f"✓ Results saved!")
    else:
        return data_dict


def plot_single_agent_dynamics(data=None, filename=None, agent_type='baseline', img_name='dynamics',
                               warmup=0,
                               variables=[
                                    'stress', 'aversive_internal_state', 'urge_to_escape',
                                    'suicidal_thought', 'escape_behavior', 'external_strat',
                                    'internal_strat'
                                ]):
    """Plot all dynamics for a single agent with state backgrounds and x-axis in days."""

    if data is None:
        print(f"\nLoading results from {filename}...")
        data = np.load(filename)
    data = clean_zero_entries(data)

    print(f"Plotting combined dynamics for random {agent_type} agent")
    agent_idx = 0

    FONT_SIZE = 16
    STATE_COLORS = {
        SLEEP:   '#cce5ff',  # pale blue
        MORNING: '#fff3cd',  # pale yellow
        COMMUTE: '#fde2e2',  # pale red
        WORK:    '#d4edda',  # pale green
        HOME:    '#e2d9f3',  # pale purple
        SOCIAL_EVENT: "#c6c0ff",
    }
    STATE_LABELS = {
        SLEEP: 'Sleep', MORNING: 'Morning', COMMUTE: 'Commute',
        WORK: 'Work', HOME: 'Home', SOCIAL_EVENT: "Social Event",
    }

    line_colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 6))

    # ── build time axis ───────────────────────────────────────────────────────
    # Use the first variable to determine series length
    ref_series = data[variables[0]][:, agent_idx]
    time_steps = np.arange(len(ref_series)) * MINUTE_LENGTH

    if 'state' in data:
        state_series = np.asarray(data['state'][:, agent_idx], dtype=int)
        n = len(state_series)

        # Walk through contiguous runs of the same state and shade each span
        seen_states = set()
        i = 0
        while i < n:
            current_state = state_series[i]
            j = i + 1
            while j < n and state_series[j] == current_state:
                j += 1
            # span covers [time_steps[i], time_steps[j-1] + MINUTE_LENGTH)
            t_start = time_steps[i]
            t_end   = time_steps[j - 1] + MINUTE_LENGTH
            color   = STATE_COLORS.get(current_state, '#eeeeee')
            label   = STATE_LABELS.get(current_state, str(current_state)) \
                      if current_state not in seen_states else None
            ax.axvspan(t_start, t_end, facecolor=color, alpha=1.0,
                       label=label, zorder=0)
            seen_states.add(current_state)
            i = j


    # ── dynamics lines ────────────────────────────────────────────────────────
    for i, var in enumerate(variables):
        var_data = data[var]
        if var_data.ndim != 2:
            raise ValueError(f"{var} is not 2D (time x agents)")
        if agent_idx >= var_data.shape[1]:
            raise IndexError(f"Agent index {agent_idx} out of bounds for {var}")

        agent_series = var_data[:, agent_idx]
        ax.plot(
            time_steps,
            agent_series,
            label=VAR_DICT[var],
            color=line_colors[i % len(line_colors)],
            linewidth=2,
            zorder=2,
        )

    ax.set_xlabel("Time (days)", fontsize=FONT_SIZE)
    ax.set_ylabel("Value", fontsize=FONT_SIZE)
    ax.set_title(f"Agent type: {agent_type} — Dynamics Across All Variables",
                 fontsize=FONT_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(warmup, time_steps[-1])

    # Two-column legend: state bands first, then variable lines
    ax.legend(loc='upper right', fontsize=FONT_SIZE-1, ncol=2)
    ax.grid(True, alpha=0.3, zorder=1)
    fig.tight_layout()

    output_path = Path("src/output") / f"{img_name}_single.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved to {output_path}")

    plt.show()


def generate_stress_signal(model, filename="stress_signal.npy"):
    run_basic_simulation(model)
    data = export_results(model, filename=None)
    stress_signal = data["stress"][:, 0]

    output_dir = Path("src/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / filename, stress_signal)


def plot_single_dynamics_in_comparison(ax, time_data, model_data, warmup, agent_idx, lw, title, fontsize, vars, sim_length):
    for var in vars:

        ax.plot(
            time_data,
            model_data[var][:, agent_idx],
            linewidth=lw,
            alpha=1
        )
 
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Days)", fontsize=fontsize)
    ax.set_ylabel("Value", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim(warmup, sim_length)
    ax.set_ylim(0, 1)
    ax.grid(True)


def _find_zeros(data):
    keys = [key for key in data.keys() if key not in ["total_time", "labels"]]

    mask = np.ones(data[keys[0]].shape[0], dtype=bool)

    for key in keys:
        mask &= np.all(data[key] == 0, axis=1)

    mask[0] = False

    return np.where(mask)[0]


def clean_zero_entries(data):
    zeros = _find_zeros(data)

    for key in data:
        if key != "labels":
            data[key] = np.delete(data[key], zeros, axis=0)

    return data


def auc_per_agent(data, dt=MINUTE_LENGTH):
    """
    Compute AUC for each agent (column-wise trapezoidal integration).

    Parameters
    ----------
    data : np.ndarray
        Shape (timesteps, agents)
    dt : float
        Time step size

    Returns
    -------
    np.ndarray
        Shape (agents,), AUC per agent
    """
    data = np.asarray(data)

    # integrate along time axis (axis=0 integrates each column)
    auc = np.trapezoid(data, dx=dt, axis=0)

    return auc


COLORS = ["#E69F00", "#56B4E9", "#009E73"]

def plot_auc_distribution(datasets, labels, title, xlabel, bins=30):
    """
    Plot overlaid histograms of AUC distributions.

    Parameters
    ----------
    datasets : list of np.ndarray
        Each entry is shape (agents,) e.g. AUC per agent
    labels : list of str
        Dataset labels
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fontsize = 16

    datasets = [np.asarray(d) for d in datasets]

    # global binning (like your reference design)
    all_data = np.concatenate(datasets)
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    legend_lines = []
    legend_texts = []

    for data, label, color in zip(datasets, labels, COLORS[:len(datasets)]):
        mu = np.mean(data)
        print(f"{label} mean:{mu}")
        sigma = np.std(data, ddof=1)
        n = len(data)

        # histogram
        ax.hist(
            data,
            bins=bin_edges,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.45
        )

        # mean line
        ax.axvline(mu, linestyle="--", color=color, linewidth=2)

        # gaussian overlay (scaled to histogram counts)
        x = np.linspace(bin_edges[0], bin_edges[-1], 500)

        # avoid division by zero if constant data
        if sigma > 0:
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )
            line, = ax.plot(x, pdf * n * bin_width, color=color, linewidth=2)
        else:
            line, = ax.plot([], [], color=color, linewidth=2)

        legend_lines.append(line)
        legend_texts.append(f"{label} (μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_title(title, fontsize=fontsize, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)

    fig.legend(
        legend_lines,
        legend_texts,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
        fontsize=fontsize - 2,
        frameon=True,
    )

    fig.savefig(f"src/output/{title.replace(' ', '_')}.svg", format="svg")
    plt.show()


def perform_t_test(data_1, name_1, data_2, name_2, mann_whitney=True, ttest=False):

    print("\n=== Means tests (independent samples) ===\n")
    if mann_whitney:
        u_stat, p_val_mw = mannwhitneyu(data_1, data_2, alternative="two-sided")
        if p_val_mw < 1e-300:
            p_text_mw = f"< 1e-300"
        else:
            p_text_mw = f"= {p_val_mw:.4e}"

        print(f"{name_1} vs {name_2}")
        print("  --- Mann Whitney Means Test ---")
        print(f"    mean({name_1}) = {np.mean(data_1):.4f}")
        print(f"    mean({name_2}) = {np.mean(data_2):.4f}")
        print(f"    u = {u_stat:.4f},  p " + p_text_mw)
    if ttest:
        t_stat, p_val_t = ttest_ind(data_1, data_2, equal_var=False)
        if p_val_t < 1e-300:
            p_text_t = f"< 1e-300"
        else:
            p_text_t = f"= {p_val_t:.4e}"

        print("  --- t-Test ---")
        print(f"    mean({name_1}) = {np.mean(data_1):.4f}")
        print(f"    mean({name_2}) = {np.mean(data_2):.4f}")
        print(f"    t = {t_stat:.4f},  p " + p_text_t)

    print()


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def plot_comparison(
    model_1_data,
    model_2_data,
    title_1="Agent 1",
    title_2="Agent 2",
    agent_idx=3,
    main_var="suicidal_thought",
    secondary_var="aversive_internal_state",
    fig_name="memory_v_baseline",
    vars_1=[
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat',
    ],
    vars_2=[
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat'
    ],
    warmup=10,
    sim_length=50,
):
    """
    Plot comparison using the same agent ID from two different simulation runs.
    This ensures identical stress trajectories when using the same random seed.
    
    Parameters:
    -----------
    df1 : DataFrame
        Data from first simulation (e.g., StandardAgent type)
    df2 : DataFrame  
        Data from second simulation (e.g., BaselineAgent type)
    agent_id : int
        The agent ID to compare from both simulations (usually 1)
    title1, title2 : str
        Titles for the two agent types
    main_var, secondary_var : str
        Variables to highlight in the comparison
    fig_name : str
        Output filename (without extension)
    """
    fontsize = 20
    lw = 2
 
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), constrained_layout=False)
    
    # ============================================================
    # PANEL 1 — First agent type (e.g., Baseline)
    # ============================================================
    model_1_data = clean_zero_entries(model_1_data)
    plot_single_dynamics_in_comparison(
        ax=axes[0],
        time_data=model_1_data["total_time"][:, agent_idx],
        model_data=model_1_data,
        warmup=warmup,
        agent_idx=agent_idx,
        lw=lw,
        title=title_1,
        fontsize=fontsize,
        vars=vars_1,
        sim_length=sim_length
        )
 
    # ============================================================
    # PANEL 2 — Second agent type (e.g., Memory)
    # ============================================================
    model_2_data = clean_zero_entries(model_2_data)
    plot_single_dynamics_in_comparison(
        ax=axes[1],
        time_data=model_2_data["total_time"][:, agent_idx],
        model_data=model_2_data,
        warmup=warmup,
        agent_idx=agent_idx,
        lw=lw,
        title=title_2,
        fontsize=fontsize,
        vars=vars_2,
        sim_length=sim_length
        )
 
    # ============================================================
    # PANEL 3 — Direct Comparison
    # ============================================================
    comparison_vars = [main_var, secondary_var]
    default_map = {
        main_var: default_colors[3],
        secondary_var: default_colors[2]
    }
    contrasting_map = {
        main_var: {
            "-": default_colors[3],
            "--": "#0072B2",
        },
        secondary_var: {
            "-": default_colors[4],
            "--": "#003366",
        }
    }
    legend_handles = []
 
    for var in comparison_vars:
        # Calculate statistics
        agent_data_1 = model_1_data[var][:, agent_idx]
        m1, s1 = agent_data_1.mean(), agent_data_1.std()
        agent_data_2 = model_2_data[var][:, agent_idx]
        m2, s2 = agent_data_2.mean(), agent_data_2.std()
 
        # First simulation — solid line
        axes[2].plot(
            model_2_data["total_time"][:, agent_idx],
            agent_data_1,
            color=contrasting_map[var]['-'],
            linestyle="-",
            linewidth=3,
            label=f"{VAR_DICT[var]} ({title_1})"
        )
 
        # Second simulation — dashed line
        axes[2].plot(
            model_2_data["total_time"][:, agent_idx],
            agent_data_2,
            color=contrasting_map[var]['--'],
            linestyle="--",
            linewidth=3,
            label=f"{VAR_DICT[var]} ({title_2})"
        )
 
        legend_handles.extend([
            Line2D([0], [0],
                   color=contrasting_map[var]['-'],
                   linestyle="-",
                   linewidth=3,
                   label=f"{VAR_DICT[var]} ({title_1}, μ={m1:.2f}, σ={s1:.2f})"),
            Line2D([0], [0],
                   color=contrasting_map[var]['--'],
                   linestyle="--",
                   linewidth=3,
                   label=f"{VAR_DICT[var]} ({title_2}, μ={m2:.2f}, σ={s2:.2f})")
        ])
 
    axes[2].set_title(f"Target Variable Comparison between\n{title_1} and {title_2}", fontsize=fontsize,
                  fontweight="bold",
                  pad=15)
    axes[2].set_xlabel("Time (Days)", fontsize=fontsize)
    axes[2].tick_params(labelsize=fontsize)
    axes[2].set_xlim(warmup, sim_length)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True)
 
    # ============================================================
    # LEGENDS
    # ============================================================
    bottom_handles = [
        Line2D([0], [0],
               color=default_colors[i % len(default_colors)],
               linestyle="-",
               linewidth=2,
               label=VAR_DICT[var])
        # Ensure that vars_2 is the set with more variables
        for i, var in enumerate(vars_2)
    ]
 
    # Get axis positions
    bbox0 = axes[0].get_position()
    bbox1 = axes[1].get_position()
    bbox2 = axes[2].get_position()
 
    x_center_left = (bbox0.x0 + bbox1.x1) / 2
    x_center_right = (bbox2.x0 + bbox2.x1) / 2
    y_bottom = min(bbox0.y0, bbox1.y0, bbox2.y0)
 
    vertical_offset = -0.15
    left_legend_to_left = 0.08
    right_legend_to_right = -0.03
 
    # Legend for all parameters
    fig.legend(
        handles=bottom_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_left - left_legend_to_left - 0.02,
                        y_bottom - vertical_offset),
        ncol=min(len(vars_2), 2),
        fontsize=fontsize,
        frameon=False
    )
 
    # Legend for comparison statistics
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(x_center_right + right_legend_to_right,
                        y_bottom - vertical_offset),
        ncol=1,
        fontsize=fontsize,
        frameon=False
    )
 
    fig.subplots_adjust(bottom=0.32)
 
    fig.savefig(
        f"src/output/{fig_name}.svg",
        format="svg",
        bbox_inches="tight"
    )
 
    plt.show()
    print(f"✓ Saved plot to {fig_name}.svg\n")


def suicidal_prevalence_timeseries(data, warmup=10, threshold=0.02):
    """
    Compute time series of proportion of agents with suicidal_thought > threshold.

    Parameters
    ----------
    data : dict
        Must contain 'suicidal_thought' of shape (time, agents)
    warmup : int
        Number of initial timesteps to ignore
    threshold : float
        Threshold for "active suicidal thought"

    Returns
    -------
    np.ndarray
        shape (T - warmup,), proportion per timestep
    """
    arr = np.asarray(data["suicidal_thought"])

    if arr.ndim != 2:
        raise ValueError("suicidal_thought must be (time, agents)")

    arr = arr[warmup:, :]
    return (arr > threshold).mean(axis=1)


def plot_suicidal_prevalence_comparison(
    data_1,
    data_2,
    label_1="Dataset 1",
    label_2="Dataset 2",
    warmup=0,
    sim_length=50,
    threshold=0.02,
    dt=MINUTE_LENGTH,
    save_name="suicidal_prevalence_comparison"
):
    """
    Compare prevalence of suicidal thought over time between two datasets.
    """
    FONTSIZE = 16

    ts_1 = suicidal_prevalence_timeseries(data_1, warmup=warmup, threshold=threshold)
    ts_2 = suicidal_prevalence_timeseries(data_2, warmup=warmup, threshold=threshold)

    t = np.arange(len(ts_1)) * dt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, ts_1, linewidth=2, label=label_1)
    ax.plot(t, ts_2, linewidth=2, label=label_2)

    # vertical reference line: warmup + 10 days
    ax.axvline((warmup + 10), linestyle=":", color="black", linewidth=2)

    ax.set_xlabel("Time (days)", fontsize=FONTSIZE)
    ax.set_ylabel(f"Proportion (suicidal_thought > {threshold})", fontsize=FONTSIZE)
    ax.set_title("Suicidal Thought Prevalence Over Time", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE)

    fig.tight_layout()

    out_path = f"src/output/{save_name}.svg"
    fig.savefig(out_path, format="svg")
    plt.show()

    print(f"✓ Saved to {out_path}")