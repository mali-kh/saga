import copy
import logging  # pylint: disable=missing-module-docstring
import pathlib
import random
import tempfile
from typing import List, Optional, Tuple
from itertools import product

import dill as pickle
import numpy as np
from saga.scheduler import Scheduler
from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler,
                             HybridScheduler, BruteForceScheduler, SMTScheduler)
from saga.utils.random_graphs import (add_random_weights, get_chain_dag,
                                      get_network)

from changes import (NetworkChangeEdgeWeight, NetworkChangeNodeWeight,
                     TaskGraphAddDependency, TaskGraphChangeDependencyWeight,
                     TaskGraphChangeTaskWeight, TaskGraphDeleteDependency)
from simulated_annealing import SimulatedAnnealing

thisdir = pathlib.Path(__file__).parent

logging.basicConfig(level=logging.INFO)

# set logging format [time] [level] message
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s')


# algorithms that only work when the compute speed is the same for all nodes
homogenous_comp_algs = {"ETF", "FCP", "FLB"}
# algorithms that only work when the communication speed is the same for all network edges
homogenous_comm_algs = {"BIL", "GDL", "FCP", "FLB"}
rerun_schedulers = ["FLB"]
rerun_base_schedulers = []


def run_experiments(scheduler_pairs: List[Tuple[Tuple[str, Scheduler], Tuple[str, Scheduler]]],
                    max_iterations: int,
                    num_tries: int,
                    max_temp: float,
                    min_temp: float,
                    cooling_rate: float,
                    skip_existing: bool = True,
                    output_path: pathlib.Path = thisdir.joinpath("results"),
                    node_range: Tuple[int, int] = (3, 5),
                    task_range: Tuple[int, int] = (3, 5)) -> None:
    """Run the experiments.
    
    
    Args:
        scheduler_pairs (List[Tuple[Tuple[str, Scheduler], Tuple[str, Scheduler]]]): list of tuples of (scheduler_name, scheduler) pairs.
            The first scheduler is the scheduler to be tested, and the second scheduler is the base scheduler.
        max_iterations (int): maximum number of iterations to run each experiment for.
        num_tries (int): number of times to run each experiment.
        max_temp (float): maximum temperature for simulated annealing.
        min_temp (float): minimum temperature for simulated annealing.
        cooling_rate (float): cooling rate for simulated annealing.
        skip_existing (bool, optional): whether to skip existing experiments. Defaults to True.
        output_path (pathlib.Path, optional): path to save results. Defaults to thisdir.joinpath("results").

    Raises:
        exp: any exception raised during experiment.
    """

    task_graph_changes = [
        TaskGraphAddDependency, TaskGraphDeleteDependency,
        TaskGraphChangeDependencyWeight, TaskGraphChangeTaskWeight
    ]
    get_makespan = SimulatedAnnealing.default_get_makespan

    all_iterations = [max_iterations]*num_tries
    for (scheduler_name, scheduler), (base_scheduler_name, base_scheduler) in scheduler_pairs:
    # for base_scheduler_name, base_scheduler in base_schedulers.items(): # pylint: disable=too-many-nested-blocks
    #     for scheduler_name, scheduler in schedulers.items():
        savepath = output_path / base_scheduler_name / f"{scheduler_name}.pkl"
        if (savepath.exists() and skip_existing
            and not scheduler_name in rerun_schedulers
            and not base_scheduler_name in rerun_base_schedulers):
            logging.info("Skipping experiment for %s/%s", scheduler_name, base_scheduler_name)
            continue

        best_run: Optional[SimulatedAnnealing] = None
        try:
            for max_iter in all_iterations:
                if scheduler_name == base_scheduler_name:
                    continue
                # Brute force is optimal, so no point trying to find a worst-case scenario
                if scheduler_name == "Brute Force":
                    continue

                logging.info(
                    "Running experiment for %s/%s with %s iterations",
                    scheduler_name, base_scheduler_name, max_iter
                )

                network = add_random_weights(get_network(random.randint(*node_range)))
                task_graph = add_random_weights(get_chain_dag(random.randint(*task_range)))

                change_types = copy.deepcopy(task_graph_changes)
                if scheduler_name in homogenous_comp_algs or base_scheduler_name in homogenous_comp_algs:
                    for node in network.nodes:
                        network.nodes[node]["weight"] = 1
                else:
                    change_types.append(NetworkChangeNodeWeight)

                if scheduler_name in homogenous_comm_algs or base_scheduler_name in homogenous_comm_algs:
                    for edge in network.edges:
                        if edge[0] == edge[1]:
                            network.edges[edge]["weight"] = 1e9
                        else:
                            network.edges[edge]["weight"] = 1
                else:
                    change_types.append(NetworkChangeEdgeWeight)


                simulated_annealing = SimulatedAnnealing(
                    task_graph=task_graph,
                    network=network,
                    scheduler=scheduler,
                    base_scheduler=base_scheduler,
                    min_temp=min_temp,
                    cooling_rate=cooling_rate,
                    max_temp=max_temp,
                    max_iterations=max_iter,
                    change_types=change_types,
                    get_makespan=get_makespan
                )

                simulated_annealing.run()

                if best_run is None or (best_run.iterations[-1].best_energy <
                                        simulated_annealing.iterations[-1].best_energy):
                    best_run = simulated_annealing
                    savepath = output_path.joinpath(base_scheduler_name, f"{scheduler_name}.pkl")
                    savepath.parent.mkdir(parents=True, exist_ok=True)
                    savepath.write_bytes(pickle.dumps(simulated_annealing))
        except Exception as exp:
            logging.error(
                "Error running experiment for %s/%s with %s iterations",
                scheduler_name, base_scheduler_name, max_iter
            )
            logging.error(exp)
            raise exp

SCHEDULERS = {
    "CPOP": CpopScheduler(),
    "HEFT": HeftScheduler(),
    "Duplex": DuplexScheduler(),
    "ETF": ETFScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "FCP": FCPScheduler(),
    "GDL": GDLScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MinMin": MinMinScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "OLB": OLBScheduler(),
    "BIL": BILScheduler(),
    "WBA": WBAScheduler(),
    "FLB": FLBScheduler(),
}
def experiment_1(): # pylint: disable=too-many-locals, too-many-statements
    """Run first set of experiments."""
    scheduler_pairs = list(product(SCHEDULERS.items(), SCHEDULERS.items()))
    run_experiments(
        scheduler_pairs=scheduler_pairs,
        max_iterations=1000,
        num_tries=10,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=True,
        output_path=thisdir.joinpath("results")
    )

def experiment_2(): # pylint: disable=too-many-locals, too-many-statements
    """Run second set of experiments."""
    scheduler_pairs = []
    for scheduler_name, scheduler in SCHEDULERS.items():
        all_others = HybridScheduler([s for s in SCHEDULERS.values() if s != scheduler])
        scheduler_pairs.extend([
            ((scheduler_name, scheduler), (f"Not{scheduler_name}", all_others)),
            ((f"Not{scheduler_name}", all_others), (scheduler_name, scheduler)),
        ])

    run_experiments(
        scheduler_pairs=scheduler_pairs,
        max_iterations=1000,
        num_tries=15,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=True,
        output_path=thisdir.joinpath("results")
    )

def ensemble_experiment(): # pylint: disable=too-many-locals, too-many-statements
    """Run ensemble experiment.

    Iteratively construct the best hybrid algorithm.
    1. Start with alg with least max MR (MinMinScheduler)
    2. For each remaining alg:
    4.   Run alg vs. hybrid
    5. Add best alg to hybrid
    6. Repeat from 2 until all algs are in hybrid

    Intuitively, this is adding algorithms that cover problem instances the hybrid doesn't
    currently cover.
    """
    logging.basicConfig(level=logging.WARNING)

    hybrid_algs = ["MinMin"]
    remaining_algs = [s for s in SCHEDULERS.keys() if s not in hybrid_algs]
    round_i = 0
    while remaining_algs:
        round_i += 1
        print(f"Iteration {round_i}")
        print(f"  Hybrid: {hybrid_algs}")
        print(f"  Remaining: {remaining_algs}")

        hybrid_alg = HybridScheduler([SCHEDULERS[s] for s in hybrid_algs])
        pairs = [(("Hybrid", hybrid_alg), (s, SCHEDULERS[s])) for s in remaining_algs]
        # create temp directory to store results
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = pathlib.Path(tempdir)
            run_experiments(
                scheduler_pairs=pairs,
                max_iterations=1000,
                num_tries=5,
                max_temp=10,
                min_temp=0.1,
                cooling_rate=0.99,
                skip_existing=True,
                output_path=tempdir
            )

            # load results
            max_makespan_ratio, best_alg = 0, None
            for scheduler_name in remaining_algs:
                res: SimulatedAnnealing = pickle.loads(
                    tempdir.joinpath(f"{scheduler_name}/Hybrid.pkl").read_bytes()
                )
                makespan_ratio = res.iterations[-1].best_energy
                if makespan_ratio > max_makespan_ratio:
                    max_makespan_ratio = makespan_ratio
                    best_alg = scheduler_name

            print(f"Adding {best_alg} to hybrid (MR={max_makespan_ratio})")
            hybrid_algs.append(best_alg)
            remaining_algs.remove(best_alg)


def ensemble_experiment_2(): # pylint: disable=too-many-locals, too-many-statements
    """Run ensemble experiment 2
    
    Iteratively construct the best hybrid algorithm.
    1. Start with no chosen algs
    2. While there are remaining algs:
    3.   For each alg:
    4.     Run BruteForce vs Hybrid([*chosen algs, alg])
    5.   Choose alg with lowest max MR
    6.   Add alg to chosen algs
    """
    hybrid_algs = []
    remaining_algs = [s for s in SCHEDULERS.keys() if s not in hybrid_algs]
    round_i = 0
    while remaining_algs:
        round_i += 1
        print(f"Iteration {round_i}")
        print(f"  Hybrid: {hybrid_algs}")
        print(f"  Remaining: {remaining_algs}")

        # hybrid_alg = HybridScheduler([SCHEDULERS[s] for s in hybrid_algs])
        hybrid_schedulers = [SCHEDULERS[s] for s in hybrid_algs]
        pairs = [
            ((s, HybridScheduler([*hybrid_schedulers, SCHEDULERS[s]])),
             ("BruteForce", BruteForceScheduler()))
            for s in remaining_algs
        ]
        # create temp directory to store results
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = pathlib.Path(tempdir)
            run_experiments(
                scheduler_pairs=pairs,
                max_iterations=1000,
                num_tries=5,
                max_temp=10,
                min_temp=0.1,
                cooling_rate=0.99,
                skip_existing=True,
                output_path=tempdir,
                node_range=(3, 3),
                task_range=(3, 3)
            )

            # load results
            min_makespan_ratio, best_alg = np.inf, None
            for scheduler_name in remaining_algs:
                res: SimulatedAnnealing = pickle.loads(
                    tempdir.joinpath(f"BruteForce/{scheduler_name}.pkl").read_bytes()
                )
                makespan_ratio = res.iterations[-1].best_energy
                if makespan_ratio < min_makespan_ratio:
                    min_makespan_ratio = makespan_ratio
                    best_alg = scheduler_name

            print(f"Adding {best_alg} to hybrid (MR={min_makespan_ratio})")
            hybrid_algs.append(best_alg)
            remaining_algs.remove(best_alg)

def eval_hybrid_all():
    scheduler = HybridScheduler(SCHEDULERS.values())
    base_scheduler = BruteForceScheduler()

    output_path = thisdir.joinpath("results_hybrid_all")
    output_path.mkdir(exist_ok=True, parents=True)
    run_experiments(
        scheduler_pairs=[(("HybridAll", scheduler), ("BruteForce", base_scheduler))],
        max_iterations=1000,
        num_tries=10,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=True,
        output_path=output_path,
        node_range=(3, 3),
        task_range=(3, 3)
    )

def main():
    """Run the experiments."""
    random.seed(9281995) # for reproducibility
    np.random.seed(9281995) # for reproducibility

    experiment_1()
    # experiment_2()
    # ensemble_experiment()
    # ensemble_experiment_2()
    # eval_hybrid_all()

if __name__ == "__main__":
    main()
