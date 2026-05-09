# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

# Local imports
from core.utils import load_pickle

def get_experimental_results(
    test_mode: int,
    num_times: int,
    max_budget: int,
    gamma: float,
    initial_frontier_size: int,
    interpolation_eps: float,
    std_idx: int = 3,
    rng_seed: int = 42
) -> dict:
    stds = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    std = None
    if test_mode == 0:
        exp_type = "lomax"
    elif test_mode == 1:
        exp_type = "uniform"
    elif test_mode == 2:
        exp_type = "lomax"
    elif test_mode == 3:
        exp_type = "uniform"
    elif test_mode == 4:
        exp_type = "ICPSR_sim"
        std = stds[std_idx]
    elif test_mode == 5:
        exp_type = "ICPSR_real"
        std = stds[std_idx]
    else:
        raise NotImplementedError
    param_string = f"{num_times}_{max_budget}_{gamma}_{initial_frontier_size}_{interpolation_eps}_{std}_{rng_seed}"
    result_fname = f"results/{exp_type}/full_{param_string}.pkl"
    all_trajectories = load_pickle(result_fname)
    return all_trajectories

def plot_main_figure(num_runs: int, max_budget: int, gamma: float, n: int) -> None:
    plt.rcParams['font.size'] = 20
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))
    legend_handles = dict()

    # Setup label, color, line styles
    policy_labels = {
        ("ConstantPolicy", 2): "Const(2)",
        ("ConstantPolicy", 3): "Const(3)",
        ("ConstantPolicy", 5): "Const(5)",
        ("ConstantPolicy", 10): "Const(10)",
        ("OurPolicy", 0.0): r"$\pi^{\mathrm{our}} (\varepsilon = 0.0)$",
        ("OurPolicy", 0.2): r"$\pi^{\mathrm{our}} (\varepsilon = 0.2)$",
        ("OurPolicy", 0.4): r"$\pi^{\mathrm{our}} (\varepsilon = 0.4)$"
    }
    color_dict = {
        ("ConstantPolicy", 2): cmap(1),
        ("ConstantPolicy", 3): cmap(2),
        ("ConstantPolicy", 5): cmap(3),
        ("ConstantPolicy", 10): cmap(4),
        ("OurPolicy", 0.0): cmap(6),
        ("OurPolicy", 0.2): cmap(7),
        ("OurPolicy", 0.4): cmap(8)
    }
    line_style = {
        ("ConstantPolicy", 2): ":",
        ("ConstantPolicy", 3): ":",
        ("ConstantPolicy", 5): ":",
        ("ConstantPolicy", 10): ":",
        ("OurPolicy", 0.0): "-",
        ("OurPolicy", 0.2): "-",
        ("OurPolicy", 0.4): "-"
    }

    # Collect results
    collated_all_trajs = dict()
    for exp_idx in [0, 1, 2, 3, 4, 5]:
        all_eps = [0.0, 0.2, 0.4] if exp_idx == 2 or exp_idx == 3 else [0.0]
        for eps in all_eps:
            key = (exp_idx, eps)
            all_trajs = get_experimental_results(exp_idx, num_runs, max_budget, gamma, n, eps)
            collated_all_trajs[key] = all_trajs

        # Plot
        row_idx = exp_idx % 2
        col_idx = exp_idx // 2
        ax = axes[row_idx, col_idx]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.set_title(rf"Experiment {exp_idx + 1}")
        overall_max_timestep = 0
        for eps in all_eps:
            key = (exp_idx, eps)
            all_trajs = collated_all_trajs[key]
            for policy, trajs in all_trajs.items():
                if policy[0] == "OurPolicy":
                    policy = ("OurPolicy", eps)
                if policy not in policy_labels.keys():
                    continue
                else:
                    overall_max_timestep = max(overall_max_timestep, max([len(traj) for traj in trajs]))
        for eps in all_eps:
            key = (exp_idx, eps)
            all_trajs = collated_all_trajs[key]
            for policy, trajs in all_trajs.items():
                if policy[0] == "OurPolicy":
                    policy = ("OurPolicy", eps)
                if policy not in policy_labels.keys():
                    continue
                max_timestep = max([len(traj) for traj in trajs])
                acc_disc_rewards = []
                for traj in trajs:
                    padded = np.full(max_timestep, traj[-1], dtype=int)
                    padded[:len(traj)] = traj
                    acc_disc_reward = [0]
                    for t in range(1, len(padded)):
                        val = acc_disc_reward[-1] + pow(gamma, t-1) * (padded[t] - padded[t-1])
                        acc_disc_reward.append(val)
                    acc_disc_rewards.append(acc_disc_reward)
                acc_disc_rewards = np.array(acc_disc_rewards)
                mean_traj = np.mean(acc_disc_rewards, axis=0)
                std_err_traj = np.std(acc_disc_rewards, axis=0) / np.sqrt(num_runs)

                policy_key = policy
                marker_style = "x" if policy[0] == "OurPolicy" else "o"
                X = np.arange(overall_max_timestep)
                line_handle, = ax.plot(
                    X[:len(mean_traj)],
                    mean_traj,
                    color=color_dict[policy_key],
                    ls=line_style[policy_key],
                    label=policy_labels[policy_key],
                    marker=marker_style,
                    markevery=[len(mean_traj)-1],
                    markersize=7
                )
                ax.plot(X[len(mean_traj)-1:], [mean_traj[-1]] * (overall_max_timestep-len(mean_traj)+1), color=color_dict[policy_key], ls=line_style[policy_key])
                base_color = to_rgb(line_handle.get_color())
                fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
                ax.fill_between(
                    X[:len(mean_traj)],
                    mean_traj - std_err_traj,
                    mean_traj + std_err_traj,
                    color=fill_color,
                    alpha=0.25
                )
                legend_handles.setdefault(policy_key, line_handle)

    # Build legend
    legend_order = [
        ("OurPolicy", 0.0),
        ("OurPolicy", 0.2),
        ("OurPolicy", 0.4),
        ("ConstantPolicy", 2),
        ("ConstantPolicy", 3),
        ("ConstantPolicy", 5),
        ("ConstantPolicy", 10)
    ]
    print(legend_handles)
    handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    labels  = [policy_labels[k] for k in legend_order if k in legend_handles]
    fig.legend(handles, labels, loc='lower center', ncol=8, bbox_to_anchor=(0.5, 0.98))

    # Annotate overall figure
    fig.supxlabel("Time step", y=0.03)
    fig.supylabel("Accumulated discounted reward", x=0.08)
    fig.text(0.5, 0.95, rf"Experiments 1 to 6 over {num_runs} runs with maximum budget $b$ = {max_budget}, discount factor $\gamma$ = {gamma}, and initial frontier size $n$ = {n}", ha="center", va='center', fontsize=25)

    # Save plot
    plot_fname = f"./combined_b={max_budget}_gamma={gamma}_n={n}.png"
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

def plot_appendix_figures(num_runs: int, max_budget: int, all_gamma: list, all_n: list) -> None:
    plt.rcParams['font.size'] = 20
    cmap = plt.get_cmap("tab10")

    # Setup label, color, line styles
    policy_labels = {
        ("ConstantPolicy", 2): "Const(2)",
        ("ConstantPolicy", 3): "Const(3)",
        ("ConstantPolicy", 5): "Const(5)",
        ("ConstantPolicy", 10): "Const(10)",
        ("OurPolicy", 0.0): r"$\pi^{\mathrm{our}} (\varepsilon = 0.0)$",
        ("OurPolicy", 0.2): r"$\pi^{\mathrm{our}} (\varepsilon = 0.2)$",
        ("OurPolicy", 0.4): r"$\pi^{\mathrm{our}} (\varepsilon = 0.4)$"
    }
    color_dict = {
        ("ConstantPolicy", 2): cmap(1),
        ("ConstantPolicy", 3): cmap(2),
        ("ConstantPolicy", 5): cmap(3),
        ("ConstantPolicy", 10): cmap(4),
        ("OurPolicy", 0.0): cmap(6),
        ("OurPolicy", 0.2): cmap(7),
        ("OurPolicy", 0.4): cmap(8)
    }
    line_style = {
        ("ConstantPolicy", 2): ":",
        ("ConstantPolicy", 3): ":",
        ("ConstantPolicy", 5): ":",
        ("ConstantPolicy", 10): ":",
        ("OurPolicy", 0.0): "-",
        ("OurPolicy", 0.2): "-",
        ("OurPolicy", 0.4): "-"
    }

    # Collect results
    collated_all_trajs = dict()
    for exp_idx in [0, 1, 2, 3, 4, 5]:
        all_eps = [0.0, 0.2, 0.4] if exp_idx == 2 or exp_idx == 3 else [0.0]
        for eps in all_eps:
            for gamma in all_gamma:
                for n in all_n:
                    key = (exp_idx, gamma, n, eps)
                    all_trajs = get_experimental_results(exp_idx, num_runs, max_budget, gamma, n, eps)
                    collated_all_trajs[key] = all_trajs
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 25))
        legend_handles = dict()
        for gamma_idx in range(len(all_gamma)):
            for n_idx in range(len(all_n)):
                gamma = all_gamma[gamma_idx]
                n = all_n[n_idx]
                ax = axes[gamma_idx, n_idx]
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                ax.set_title(rf"Experiment {exp_idx + 1}: $\gamma$ = {gamma}, $n$ = {n}")
                
                # Plot
                overall_max_timestep = 0
                for eps in all_eps:
                    key = (exp_idx, gamma, n, eps)
                    all_trajs = collated_all_trajs[key]
                    for policy, trajs in all_trajs.items():
                        if policy[0] == "OurPolicy":
                            policy = ("OurPolicy", eps)
                        if policy not in policy_labels.keys():
                            continue
                        else:
                            overall_max_timestep = max(overall_max_timestep, max([len(traj) for traj in trajs]))
                for eps in all_eps:
                    key = (exp_idx, gamma, n, eps)
                    all_trajs = collated_all_trajs[key]
                    for policy, trajs in all_trajs.items():
                        if policy[0] == "OurPolicy":
                            policy = ("OurPolicy", eps)
                        if policy not in policy_labels.keys():
                            continue
                        max_timestep = max([len(traj) for traj in trajs])
                        acc_disc_rewards = []
                        for traj in trajs:
                            padded = np.full(max_timestep, traj[-1], dtype=int)
                            padded[:len(traj)] = traj
                            acc_disc_reward = [0]
                            for t in range(1, len(padded)):
                                val = acc_disc_reward[-1] + pow(gamma, t-1) * (padded[t] - padded[t-1])
                                acc_disc_reward.append(val)
                            acc_disc_rewards.append(acc_disc_reward)
                        acc_disc_rewards = np.array(acc_disc_rewards)
                        mean_traj = np.mean(acc_disc_rewards, axis=0)
                        std_err_traj = np.std(acc_disc_rewards, axis=0) / np.sqrt(num_runs)

                        policy_key = policy
                        marker_style = "x" if policy[0] == "OurPolicy" else "o"
                        X = np.arange(overall_max_timestep)
                        line_handle, = ax.plot(
                            X[:len(mean_traj)],
                            mean_traj,
                            color=color_dict[policy_key],
                            ls=line_style[policy_key],
                            label=policy_labels[policy_key],
                            marker=marker_style,
                            markevery=[len(mean_traj)-1],
                            markersize=7
                        )
                        ax.plot(X[len(mean_traj)-1:], [mean_traj[-1]] * (overall_max_timestep-len(mean_traj)+1), color=color_dict[policy_key], ls=line_style[policy_key])
                        base_color = to_rgb(line_handle.get_color())
                        fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
                        ax.fill_between(
                            X[:len(mean_traj)],
                            mean_traj - std_err_traj,
                            mean_traj + std_err_traj,
                            color=fill_color,
                            alpha=0.25
                        )
                        legend_handles.setdefault(policy_key, line_handle)

        # Build legend
        legend_order = [
            ("OurPolicy", 0.0),
            ("OurPolicy", 0.2),
            ("OurPolicy", 0.4),
            ("ConstantPolicy", 2),
            ("ConstantPolicy", 3),
            ("ConstantPolicy", 5),
            ("ConstantPolicy", 10)
        ]
        handles = [legend_handles[k] for k in legend_order if k in legend_handles]
        labels  = [policy_labels[k] for k in legend_order if k in legend_handles]
        fig.legend(handles, labels, loc='lower center', ncol=8, bbox_to_anchor=(0.5, 0.92))

        # Annotate overall figure
        fig.supxlabel("Time step", y=0.07)
        fig.supylabel("Accumulated discounted reward", x=0.08)
        fig.text(0.5, 0.91, rf"{num_runs} runs with $b$ = {max_budget} with discount factors $\gamma \in$ {all_gamma} and initial frontier size $n \in$ {all_n}", ha="center", va='center', fontsize=25)

        # Save plot
        plot_fname = f"./appendix_exp{exp_idx + 1}_b={max_budget}.png"
        os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

if __name__ == "__main__":
    num_runs = 30
    plot_main_figure(num_runs, max_budget=200, gamma=0.9, n=5)

    all_b = [100, 150, 200]
    all_gamma = [0.5, 0.7, 0.9]
    all_n = [5, 10, 15]
    for b in all_b:
        plot_appendix_figures(num_runs, b, all_gamma, all_n)
