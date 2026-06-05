import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from networked_model.models.BaselineModel import BaselineModel
from Constants import MINUTE_LENGTH
SLEEP   = 0
MORNING = 1
COMMUTE = 2
WORK    = 3
HOME    = 4


def init_empty_model(num_agents=1000, sim_length=20, warmup=0):
    parameters = {
        'num_steps': int(sim_length/MINUTE_LENGTH),
    }
    
    # Create model
    print("Creating model...")
    return BaselineModel(
        dt=MINUTE_LENGTH,
        seed=42,
        parameters=parameters,
        verbose=True,
        num_agents=num_agents,
        warmup=warmup
    )

def run_basic_simulation(num_agents=1000):
    """Run a basic simulation with the optimized model."""
    
    # Define parameters
    model = init_empty_model(num_agents)
    
    print(f"Model initialized with {model.num_agents} agents")
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    
    for step in range(model.num_steps):
        model.step()
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"Step {step + 1}/{model.num_steps} "
                  f"({rate:.1f} steps/sec)")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Simulation complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/model.num_steps*1000:.2f} ms/step")
    print(f"Throughput: {model.num_steps/elapsed_time:.1f} steps/sec")
    
    return model


def export_results(model, filename='simulation_results.npz'):
    """Export simulation results to a file."""
    
    # Prepare data dictionary
    data_dict = {}
    for var in model.tracked_vars:
        data_dict[var] = model.data[var]
    
    # Add metadata
    data_dict['num_agents'] = model.num_agents
    data_dict['num_steps'] = model.num_steps
    data_dict['dt'] = model.dt
    
    # Save
    if filename is not None:
        np.savez_compressed(filename, **data_dict)
        print(f"✓ Results saved!")
    else:
        return data_dict


def plot_single_agent_dynamics(data=None, filename='simulation_results.npz', agent_idx=0, img_name='dynamics'):
    """Plot all dynamics for a single agent with state backgrounds and x-axis in days."""

    if data is None:
        print(f"\nLoading results from {filename}...")
        data = np.load(filename)

    print(f"Plotting combined dynamics for agent {agent_idx}")

    variables = [
        'stress', 'aversive_internal_state', 'urge_to_escape',
        'suicidal_thought', 'escape_behavior', 'external_strat',
        'internal_strat',
    ]

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
    ax.set_title(f"Agent {agent_idx} — Dynamics Across All Variables",
                 fontsize=FONT_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(time_steps[0], time_steps[-1])

    # Two-column legend: state bands first, then variable lines
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, zorder=1)
    fig.tight_layout()

    output_path = Path("src/output") / f"{img_name}_single.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved to {output_path}")

    plt.show()


def run_and_plot_dynamics(img_name):
    model = run_basic_simulation()
    data = export_results(model, filename=None)
    plot_single_agent_dynamics(data=data, img_name=img_name)
    return model


if __name__ == "__main__":
    run_and_plot_dynamics("single_baseline_agent_dynamics")