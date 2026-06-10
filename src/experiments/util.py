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
import json
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.NetworkedModel import NetworkedModel
from networked_model.networks.AbstractNetwork import Network
from Constants import MINUTE_LENGTH
from scipy.stats import ttest_ind, mannwhitneyu

SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4


def run_basic_simulation(model, batch_save=False, run_name=None):
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
                else:
                    memmaps[var][step, :] = model.states[var]

        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"Step {step + 1}/{model.num_steps} ({rate:.1f} steps/sec)")

            if batch_save:
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


def plot_single_agent_dynamics(data=None, filename='experiment_results.npz', agent_type='baseline', img_name='dynamics',
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

    print(f"Plotting combined dynamics for random {agent_type} agent")
    agent_idx = 0

    FONT_SIZE = 16

    line_colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(16, 6))

    # ── build time axis ───────────────────────────────────────────────────────
    # Use the first variable to determine series length
    ref_series = data[variables[0]][:, agent_idx]
    time_steps = np.arange(len(ref_series)) * MINUTE_LENGTH

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
            label=var.replace('_', ' ').title(),
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
    ax.legend(loc='upper right', fontsize=10, ncol=2)
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
    fig, ax = plt.subplots()
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