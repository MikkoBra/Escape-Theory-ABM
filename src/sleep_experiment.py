import json
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import islice
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

from model.sleep_deprivation.models.SleepModel import SleepModel

N_RUNS = 1000
DT = 0.1
T_MAX = 40
BATCH_SIZE = 50
FILENAME = "sleep_experiment_results.json"


def compute_auc(time, values):
    return np.trapezoid(values, time)


def run_single(simulation_id):

    model = SleepModel(dt=DT, seed=simulation_id, collect_all=False)

    while model.time < T_MAX:
        model.step()

    df = model.datacollector.get_agent_vars_dataframe().reset_index()

    agent_results = []

    for agent_id in df["AgentID"].unique():
        agent_df = df[df["AgentID"] == agent_id]

        t = agent_df["Time"].values
        T_vals = agent_df["Suicidal Thought"].values
        A_vals = agent_df["Aversion"].values

        auc_T = compute_auc(t, T_vals)
        auc_A = compute_auc(t, A_vals)

        agent_results.append({
            "agent_id": int(agent_id),
            "auc_T": float(auc_T),
            "auc_A": float(auc_A)
        })

    return {
        "simulation_id": int(simulation_id),
        "agents": agent_results
    }


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def main():

    all_results = []

    with Pool(cpu_count()) as pool:
        sim_ids = range(N_RUNS)

        for batch in batched(pool.imap_unordered(run_single, sim_ids), BATCH_SIZE):

            all_results.extend(batch)

            # Incremental write
            with open(FILENAME, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"Saved {len(all_results)} simulations")


def load_results():
    with open(FILENAME, "r") as f:
        sim_results = json.load(f)

    sleep_T, bad_T, baseline_T = [], [], []
    sleep_A, bad_A, baseline_A = [], [], []

    for sim in sim_results:
        for agent in sim["agents"]:
            aid = agent["agent_id"]

            if aid == 1:
                sleep_T.append(agent["auc_T"])
                sleep_A.append(agent["auc_A"])

            elif aid == 2:
                bad_T.append(agent["auc_T"])
                bad_A.append(agent["auc_A"])

            elif aid == 3:
                baseline_T.append(agent["auc_T"])
                baseline_A.append(agent["auc_A"])
    return sleep_T, sleep_A, bad_T, bad_A, baseline_T, baseline_A

AGENTS = [
    ("Good Sleep", "sleep"),
    ("Bad Sleep",  "bad"),
    ("Baseline",       "baseline"),
]

# Okabe-Ito colorblind-friendly palette
COLORS = ["#E69F00", "#56B4E9", "#009E73"]


def plot_overlaid_distribution(datasets, labels, title, xlabel, bins=30):
    fig, ax = plt.subplots()
    fontsize = 16

    all_data = np.concatenate([np.asarray(d) for d in datasets])
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    legend_lines = []
    legend_texts = []

    for data, label, color in zip(datasets, labels, COLORS):
        if label == "Baseline":
            label = "Baseline"
        data  = np.asarray(data)
        mu    = np.mean(data)
        sigma = np.std(data, ddof=1)
        n     = len(data)

        ax.hist(data, bins=bin_edges, color=color, edgecolor="black", linewidth=0.7, alpha=0.45)
        ax.axvline(mu, linestyle="--", color=color, linewidth=2)

        x   = np.linspace(bin_edges[0], bin_edges[-1], 500)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        line, = ax.plot(x, pdf * n * bin_width, color=color, linewidth=2)

        legend_lines.append(line)
        legend_texts.append(f"{label}  (μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_title(title,        fontsize=fontsize, fontweight="bold")
    ax.set_xlabel(xlabel,      fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)  # make room below axes

    fig.legend(
        legend_lines,
        legend_texts,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
        fontsize=fontsize - 2,
        frameon=True,
    )

    fig.savefig(f"results_{title}.svg", format="svg")
    plt.show()


def plot_results():
    sleep_T, sleep_A, bad_T, bad_A, baseline_T, baseline_A = load_results()

    labels   = [name for name, _ in AGENTS]
    T_arrays = [sleep_T, bad_T, baseline_T]
    A_arrays = [sleep_A, bad_A, baseline_A]

    plot_overlaid_distribution(
        T_arrays, labels,
        title="AUC of Suicidal Thought (T) – All Agents",
        xlabel="AUC(T)",
    )

    plot_overlaid_distribution(
        A_arrays, labels,
        title="AUC of Aversive State (A) – All Agents",
        xlabel="AUC(A)",
    )


def perform_t_test():

    sleep_T, sleep_A, bad_T, bad_A, baseline_T, baseline_A = load_results()

    comparisons = [
        ("Good Sleep", "Bad Sleep", sleep_T, bad_T, sleep_A, bad_A),
        ("Good Sleep", "Baseline",      sleep_T, baseline_T, sleep_A, baseline_A),
        ("Bad Sleep",  "Baseline",      bad_T,   baseline_T, bad_A,   baseline_A),
    ]

    print("\n=== Means tests (independent samples) ===\n")

    for name1, name2, T1, T2, A1, A2 in comparisons:

        u_stat_T, p_val_T = mannwhitneyu(T1, T2, alternative="two-sided")
        if p_val_T < 1e-300:
            p_text_T = f"< 1e-300"
        else:
            p_text_T = f"= {p_val_T:.4e}"
        t_stat_A, p_val_A = ttest_ind(A1, A2, equal_var=False)
        if p_val_A < 1e-300:
            p_text_A = f"< 1e-300"
        else:
            p_text_A = f"= {p_val_A:.4e}"

        print(f"{name1} vs {name2}")
        print("  --- Suicidal Thought (T) ---")
        print(f"    mean({name1}) = {np.mean(T1):.4f}")
        print(f"    mean({name2}) = {np.mean(T2):.4f}")
        print(f"    u = {u_stat_T:.4f},  p " + p_text_T)

        print("  --- Aversive State (A) ---")
        print(f"    mean({name1}) = {np.mean(A1):.4f}")
        print(f"    mean({name2}) = {np.mean(A2):.4f}")
        print(f"    t = {t_stat_A:.4f},  p " + p_text_A)

        print()


if __name__ == "__main__":
    # main()
    # plot_results()
    perform_t_test()