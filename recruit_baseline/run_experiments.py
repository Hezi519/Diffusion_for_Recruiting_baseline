# Standard library imports
import os
import sys

from copy import deepcopy
from multiprocessing import Pool

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# Local imports
from core.ICPSR22140_processor import ICPSR22140Processor
from core.empirical_arrival_distribution import EmpiricalArrivalDistribution
from core.empirical_population import EmpiricalPopulation
from core.interpolated_arrival_distribution import InterpolatedArrivalDistribution
from core.lomax_population import LomaxPopulation
from core.noisy_uniform_population import NoisyUniformPopulation
from core.utils import load_pickle, save_pickle
from policies.constant_policy import ConstantPolicy
from policies.our_policy import OurPolicy, precompute_surrogate

def fit_tree_empirical_predictor(X, y, min_fraction) -> tuple[np.ndarray, dict]:
    n = X.shape[0]
    min_leaf = max(1, int(np.floor(min_fraction * n)))
    clf = DecisionTreeClassifier(
        min_samples_leaf=min_leaf,
        max_depth=None,
    )
    clf.fit(X, y)

    # Compute leaf ID for each sample and store node_type and type_to_node dicts
    leaf_ids = np.asarray(clf.apply(X), dtype=int)

    # Build empirical distributions for each leaf
    empirical_distributions = dict()
    for leaf in np.unique(leaf_ids):
        y_leaf = y[leaf_ids == leaf]
        n_leaf = len(y_leaf)
        empirical_distributions[leaf] = {int(c): (y_leaf == c).sum() / n_leaf for c in clf.classes_}

    return leaf_ids, empirical_distributions

def preprocess_ICPSR(std_name: str, threshold: float) -> tuple[nx.Graph, dict, dict, dict, dict]:
    tsv_file1 = "data/ICPSR_22140/DS0001/22140-0001-Data.tsv"
    tsv_file2 = "data/ICPSR_22140/DS0002/22140-0002-Data.tsv"
    tsv_file3 = "data/ICPSR_22140/DS0003/22140-0003-Data.tsv"
    pickle_filename = "ICPSR_22140.pkl"
    processor = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename)
    covariates, statuses, G, _, _, _ = processor.merged_datasets[std_name]

    node_ids = []
    X_train = []
    y_train = []
    for idx, cov in covariates.items():
        node_ids.append(idx)
        X_train.append(cov + [statuses[idx]])
        y_train.append(G.degree[idx])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    leaf_ids, empirical_distributions = fit_tree_empirical_predictor(X_train, y_train, min_fraction=threshold)

    node_type = {node_ids[i]: int(leaf_ids[i]) for i in range(len(node_ids))}
    type_to_nodes = dict()
    for i, leaf in enumerate(leaf_ids):
        leaf = int(leaf)
        node = node_ids[i]
        type_to_nodes.setdefault(leaf, []).append(node)
    
    denom = sum(len(lst) for lst in type_to_nodes.values())
    type_proportion = {key: len(type_to_nodes[key])/denom for key in type_to_nodes.keys()}

    assert all(u in node_type for u in G.nodes())
    return G, node_type, type_to_nodes, type_proportion, empirical_distributions

def has_budget_trajectory(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        obj = load_pickle(path)
    except Exception:
        return False
    if isinstance(obj, tuple) and len(obj) == 2:
        trajectory = obj[1]
        return isinstance(trajectory, dict) and "system_size" in trajectory and "budget_spent" in trajectory
    if isinstance(obj, dict):
        return all(
            isinstance(trajectory, dict) and "system_size" in trajectory and "budget_spent" in trajectory
            for trajectories in obj.values()
            for trajectory in trajectories
        )
    return False

def run_real_job(job_parameters) -> tuple[tuple[str, int], np.ndarray]:
    fname = job_parameters['fname']
    gamma = job_parameters['gamma']
    max_budget = job_parameters['max_budget']
    population = job_parameters['population']
    initial_frontier_size = job_parameters['initial_frontier_size']
    PolicyClass = job_parameters['policy']
    const_val = job_parameters['const_val']
    rng_seed = job_parameters['rng_seed']
    verbose = job_parameters['verbose']
    surrogate_samples = job_parameters['surrogate_samples']

    G = job_parameters['G']
    node_type = job_parameters['node_type']
    empirical_distributions = job_parameters['empirical_distributions']

    if not has_budget_trajectory(fname):
        # Run simulation
        if str(PolicyClass) == "ConstantPolicy":
            policy = PolicyClass(rng_seed, gamma, population, const_val)
        elif str(PolicyClass) == "OurPolicy":
            surrogate_object = precompute_surrogate(max_budget, gamma, population, surrogate_samples)
            policy = PolicyClass(rng_seed, gamma, population, surrogate_object)
        else:
            policy = PolicyClass(rng_seed, gamma, population)

        rng = np.random.default_rng(rng_seed)
        system_size_over_time = [initial_frontier_size]
        budget_spent_over_time = [0]
        frontier = [int(x) for x in rng.choice(list(G.nodes()), initial_frontier_size, replace=False)]
        in_system = set(frontier)
        initial_frontier = deepcopy(frontier)
        available_budget = max_budget
        timestep = 1
        while len(frontier) > 0 and available_budget > 0:
            # Define distributions based on types
            distributions = []
            rng_seeds = [int(rng.integers(int(1e6))) for _ in frontier]
            for i in range(len(frontier)):
                idx = frontier[i]
                if idx in initial_frontier:
                    # Keep original degree distribution
                    empirical_distribution = empirical_distributions[node_type[idx]]
                else:
                    # Reduce degree by 1 since at least one neighbor is already in the system
                    empirical_distribution = dict()
                    for k, v in empirical_distributions[node_type[idx]].items():
                        assert isinstance(k, int) and k >= 1
                        empirical_distribution[k-1] = v

                # Add shifted empirical distribution as D_i
                ead = EmpiricalArrivalDistribution(rng_seeds[i], empirical_distribution)
                distributions.append(ead)

            if verbose:
                print(f"\nTimestep {timestep}: {len(distributions)} distributions, {available_budget} remaining budget")

            # Compute assignments using policy
            if str(PolicyClass) == "RealizationOraclePolicy":
                available_free_neighbors = [len(set(G.neighbors(idx)).difference(in_system)) for idx in frontier]
                assignments = policy.generate_assignments(available_budget, available_free_neighbors)

                # Deduct only the total number of free neighbors from budget
                available_budget -= len(set([G.neighbors(idx) for idx in frontier]))
            else:
                assignments = policy.generate_assignments(available_budget, deepcopy(distributions))

                # Update budget
                available_budget -= sum(assignments)
            budget_spent_over_time.append(max_budget - available_budget)

            # Next batch of arrivals
            candidates = [set(G.neighbors(idx)) - in_system for idx in frontier]
            available = set().union(*candidates)
            new_frontier = []
            for i in rng.permutation(len(frontier)):
                if assignments[i] <= 0:
                    continue
                feasible = list(candidates[i] & available)
                k = min(assignments[i], len(feasible))
                if k > 0:
                    recruited = rng.choice(feasible, size=k, replace=False).tolist()
                    new_frontier.extend(recruited)
                    available.difference_update(recruited)
            frontier = list(set(new_frontier))
            in_system.update(frontier)
            
            if verbose:
                print(f"{sum(assignments)} budget assigned to {len(assignments)} people in round {timestep}: {assignments}")
                print(f"Realizations: {len(frontier)}")
                print(f"Reward of round {timestep}: {len(frontier)}")

            # Record number of people in the system
            system_size_over_time.append(system_size_over_time[-1] + len(frontier))

            # Update time step
            timestep += 1

        if verbose:
            if available_budget == 0:
                print("\nOut of budget!")
            if len(distributions) == 0:
                print("\nNo more arrivals!")

        # Save results to pickle
        res = (str(policy), const_val), {
            "system_size": np.array(system_size_over_time),
            "budget_spent": np.array(budget_spent_over_time),
        }
        save_pickle(res, fname)

    # Load results from pickle
    res = load_pickle(fname)

    return res

def run_job(job_parameters: dict) -> tuple[tuple[str, int], np.ndarray]:
    fname = job_parameters['fname']
    test_mode = job_parameters['test_mode']
    gamma = job_parameters['gamma']
    max_budget = job_parameters['max_budget']
    population = job_parameters['population']
    observed_population = job_parameters['observed_population']
    initial_frontier_size = job_parameters['initial_frontier_size']
    interpolation_eps = job_parameters['interpolation_eps']
    PolicyClass = job_parameters['policy']
    const_val = job_parameters['const_val']
    rng_seed = job_parameters['rng_seed']
    verbose = job_parameters['verbose']
    surrogate_samples = job_parameters['surrogate_samples']
    
    if not has_budget_trajectory(fname):
        if str(PolicyClass) == "ConstantPolicy":
            policy = PolicyClass(rng_seed, gamma, population, const_val)
        elif str(PolicyClass) == "OurPolicy":
            surrogate_object = precompute_surrogate(max_budget, gamma, observed_population, surrogate_samples)
            policy = PolicyClass(rng_seed, gamma, observed_population, surrogate_object)
        else:
            policy = PolicyClass(rng_seed, gamma, population)
        
        system_size_over_time = [initial_frontier_size]
        budget_spent_over_time = [0]
        arriving_size = initial_frontier_size
        available_budget = max_budget
        timestep = 1
        while arriving_size > 0 and available_budget > 0:
            # Simulate distributions of arrivals
            distributions = population.sample_arrival_distributions(arriving_size)

            if test_mode == 2 and interpolation_eps > 0:
                # Generate noisy arrival distributions, interpolating with Uniform([0,1,...,20])
                ub = 20
                observed_distributions = [
                    InterpolatedArrivalDistribution(
                        dist.params['rng_seed'],
                        interpolation_eps,
                        dist,
                        EmpiricalArrivalDistribution(dist.params['rng_seed'], {i: 1.0/(ub+1) for i in range(ub+1)})
                    )
                    for dist in distributions
                ]
            else:
                observed_distributions = distributions

            if verbose:
                print(f"\nTimestep {timestep}: {len(distributions)} distributions, {available_budget} remaining budget")

            # Simulate realizations
            realizations = [distributions[i].sample() for i in range(len(distributions))]

            # Compute assignments using policy
            if str(PolicyClass) == "RealizationOraclePolicy":
                assignments = policy.generate_assignments(available_budget, realizations)
            else:
                assignments = policy.generate_assignments(available_budget, deepcopy(observed_distributions))

            # Update budget
            available_budget -= sum(assignments)
            budget_spent_over_time.append(max_budget - available_budget)

            # Next batch of arrivals
            arriving_size = sum([min(realizations[i], assignments[i]) for i in range(len(distributions))])
            
            if verbose:
                print(f"{sum(assignments)} budget assigned to {len(assignments)} people in round {timestep}: {assignments}")
                print(f"Realizations: {realizations}")
                print(f"Reward of round {timestep}: {arriving_size}")

            # Record number of people in the system
            system_size_over_time.append(system_size_over_time[-1] + arriving_size)

            # Update time step
            timestep += 1

        if verbose:
            if available_budget == 0:
                print("\nOut of budget!")
            if len(distributions) == 0:
                print("\nNo more arrivals!")

        # Save results to pickle
        res = (str(policy), const_val), {
            "system_size": np.array(system_size_over_time),
            "budget_spent": np.array(budget_spent_over_time),
        }
        save_pickle(res, fname)

    # Load results from pickle
    res = load_pickle(fname)

    return res

def run_experiment(
    test_mode: int,
    num_times: int,
    max_budget: int,
    gamma: float,
    initial_frontier_size: int,
    eps: float,
    std_idx: int = 3,
    rng_seed: int = 42,
    multithread: bool = True,
    surrogate_samples: int = 1000
) -> None:
    stds = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    std = None
    ub = 20
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
        G, node_type, _, type_proportion, empirical_distributions = preprocess_ICPSR(std, 0.01)
    elif test_mode == 5:
        exp_type = "ICPSR_real"
        std = stds[std_idx]
        G, node_type, _, type_proportion, empirical_distributions = preprocess_ICPSR(std, 0.01)
    else:
        raise NotImplementedError
    surrogate_suffix = "" if surrogate_samples == 1000 else f"_surr{surrogate_samples}"
    param_string = f"{num_times}_{max_budget}_{gamma}_{initial_frontier_size}_{eps}_{std}_{rng_seed}{surrogate_suffix}"
    result_fname = f"results/{exp_type}/full_{param_string}.pkl"

    if not has_budget_trajectory(result_fname):
        all_policy_pairs = [
            (OurPolicy, None),
            (ConstantPolicy, 2),
            (ConstantPolicy, 3),
            (ConstantPolicy, 5),
            (ConstantPolicy, 10)
        ]
        all_trajectories = dict()
        jobs = []
        for policy, const_val in all_policy_pairs:
            all_trajectories[(str(policy), const_val)] = []
            rng = np.random.default_rng(rng_seed)
            if test_mode == 0 or test_mode == 2:
                populations = [LomaxPopulation(int(rng.integers(int(1e6)))) for _ in range(num_times)]
                observed_populations = populations
            elif test_mode == 1:
                populations = [NoisyUniformPopulation(int(rng.integers(int(1e6))), ub, 0.0) for _ in range(num_times)]
                observed_populations = populations
            elif test_mode == 3:
                populations = [NoisyUniformPopulation(int(rng.integers(int(1e6))), ub, 0.0) for _ in range(num_times)]
                observed_populations = [NoisyUniformPopulation(int(rng.integers(int(1e6))), ub, eps) for _ in range(num_times)]
            elif test_mode == 4 or test_mode == 5:
                populations = [EmpiricalPopulation(int(rng.integers(int(1e6))), type_proportion, empirical_distributions) for _ in range(num_times)]
                observed_populations = populations
            else:
                raise NotImplementedError

            for t in range(num_times):
                job_parameters = dict()
                job_parameters['fname'] = f"results/{exp_type}/{policy}_{const_val}_{param_string}_{t}.pkl"
                job_parameters['test_mode'] = test_mode
                job_parameters['max_budget'] = max_budget
                job_parameters['gamma'] = gamma
                job_parameters['initial_frontier_size'] = initial_frontier_size
                job_parameters['interpolation_eps'] = eps
                job_parameters['population'] = populations[t]
                job_parameters['observed_population'] = observed_populations[t]
                job_parameters['policy'] = policy
                job_parameters['const_val'] = const_val
                job_parameters['rng_seed'] = int(rng.integers(int(1e6)))
                job_parameters['verbose'] = False
                job_parameters['surrogate_samples'] = surrogate_samples
                if test_mode == 4 or test_mode == 5:
                    job_parameters['G'] = G
                    job_parameters['node_type'] = node_type
                    job_parameters['empirical_distributions'] = empirical_distributions
                jobs.append(job_parameters)

        if multithread:
            with Pool() as pool:
                if test_mode == 5:
                    for policy_key, trajectory in tqdm(
                        pool.imap_unordered(run_real_job, jobs),
                        total=len(jobs),
                        desc=f"Running {exp_type} experiments (multithread)"
                    ):
                        all_trajectories[policy_key].append(trajectory)
                else:
                    for policy_key, trajectory in tqdm(
                        pool.imap_unordered(run_job, jobs),
                        total=len(jobs),
                        desc=f"Running {exp_type} experiments (multithread)"
                    ):
                        all_trajectories[policy_key].append(trajectory)
        else:
            for job in tqdm(jobs, desc=f"Running {exp_type} experiments (single thread)"):
                if test_mode == 5:
                    policy_key, trajectory = run_real_job(job)
                else:
                    policy_key, trajectory = run_job(job)
                all_trajectories[policy_key].append(trajectory)

        # Save results to pickle
        save_pickle(all_trajectories, result_fname)

"""
all_b = [100, 150, 200]
all_gamma = [0.5, 0.7, 0.9]
all_n = [5, 10, 15]
all_eps = [0.0, 0.2, 0.4]
"""
if __name__ == "__main__":
    test_mode = int(sys.argv[1])
    num_times = int(sys.argv[2])
    max_budget = int(sys.argv[3])
    gamma = float(sys.argv[4])
    initial_frontier_size = int(sys.argv[5])
    interpolation_eps = float(sys.argv[6])
    surrogate_samples = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    
    print(f"Mode = {test_mode}, {num_times} runs, b = {max_budget}, gamma = {gamma}, n = {initial_frontier_size}, eps = {interpolation_eps}, surrogate_samples = {surrogate_samples}")
    run_experiment(test_mode, num_times, max_budget, gamma, initial_frontier_size, interpolation_eps, surrogate_samples=surrogate_samples)
