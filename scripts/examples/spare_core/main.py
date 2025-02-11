import random
import numpy as np
from typing import Tuple, Dict, List, Hashable, Optional, Set
import networkx as nx
import pandas as pd
from saga.scheduler import Task, Scheduler
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.bil import BILScheduler
from saga.schedulers.brute_force import BruteForceScheduler
from saga.schedulers.dps import DPSScheduler
from saga.schedulers.minmin import MinMinScheduler
from saga.schedulers.maxmin import MaxMinScheduler
from saga.schedulers.duplex import DuplexScheduler
from saga.schedulers.etf import ETFScheduler
from saga.schedulers.fastest_node import FastestNodeScheduler
from saga.schedulers.fcp import FCPScheduler
from saga.schedulers.flb import FLBScheduler
from saga.schedulers.gdl import GDLScheduler
from saga.schedulers.hbmct import HbmctScheduler
from saga.schedulers.mct import MCTScheduler
from saga.schedulers.met import METScheduler
from saga.schedulers.msbc import MsbcScheduler
from saga.schedulers.olb import OLBScheduler
from saga.schedulers.sufferage import SufferageScheduler
from saga.schedulers.wba import WBAScheduler

from saga.schedulers.dynamic_task_graph.residual.rheft import ResidualHeftScheduler
# from saga.schedulers.dynamic_task_graph.baseline.heft import BaselineHeftScheduler
from saga.schedulers.dynamic_task_graph.residual.rbil import ResidualBILScheduler
from saga.schedulers.dynamic_task_graph.residual.rbrute_force import ResidualBruteForceScheduler
from saga.schedulers.dynamic_task_graph.residual.rcpop import ResidualCpopScheduler
from saga.schedulers.dynamic_task_graph.residual.rdps import ResidualDPSScheduler
from saga.schedulers.dynamic_task_graph.residual.rminmin import ResidualMinMinScheduler
from saga.schedulers.dynamic_task_graph.residual.rmaxmin import ResidualMaxMinScheduler
from saga.schedulers.dynamic_task_graph.residual.rduplex import ResidualDuplexScheduler
from saga.schedulers.dynamic_task_graph.residual.retf import ResidualETFScheduler
from saga.schedulers.dynamic_task_graph.residual.rfastest_node import ResidualFastestNodeScheduler
from saga.schedulers.dynamic_task_graph.residual.rfcp import ResidualFCPScheduler
from saga.schedulers.dynamic_task_graph.residual.rflb import ResidualFLBScheduler
from saga.schedulers.dynamic_task_graph.residual.rgdl import ResidualGDLScheduler
from saga.schedulers.dynamic_task_graph.residual.rhbmct import ResidualHbmctScheduler
from saga.schedulers.dynamic_task_graph.residual.rmct import ResidualMCTScheduler
from saga.schedulers.dynamic_task_graph.residual.rmet import ResidualMETScheduler
from saga.schedulers.dynamic_task_graph.residual.rmsbc import ResidualMsbcScheduler
from saga.schedulers.dynamic_task_graph.residual.rolb import ResidualOLBScheduler
from saga.schedulers.dynamic_task_graph.residual.rsufferage import ResidualSufferageScheduler
from saga.schedulers.dynamic_task_graph.residual.rwba import ResidualWBAScheduler

from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.tools import get_insert_loc
from saga.utils.random_graphs import (
    get_branching_dag, get_chain_dag, get_diamond_dag, get_fork_dag,
    get_network, add_random_weights
)
import pathlib
import plotly.express as px
import matplotlib.pyplot as plt
import logging


from abc import ABC, abstractmethod


logging.basicConfig(level=logging.INFO)



thisdir = pathlib.Path(__file__).parent.absolute()



class ResidualWrapper(ABC):
    def schedule(
        self,
        network: nx.Graph,
        task_graphs: List[Tuple[nx.DiGraph, float]],
        scheduler: Scheduler
    ) -> Dict[str, List[Task]]:
        comp_schedule: Dict[Hashable, List[Task]]

        for idx, task_graph_tupple in enumerate(task_graphs):
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]
            
            if idx <= 0:
                comp_schedule = scheduler.schedule(network, task_graph)
            else:
                comp_schedule = scheduler.schedule(network, task_graph, comp_schedule, task_graph_arrival_time)

        return comp_schedule


# this wrapper is ignorant of the previous task graphs
class HEFTResidualWrapper(ABC):
    def schedule(
        self,
        network: nx.Graph,
        task_graphs: List[Tuple[nx.DiGraph, float]],
        scheduler: Scheduler
    ) -> Dict[str, List[Task]]:
        final_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}

        # create a new list for schedule of each task graph
        comp_schdeules: List[Dict[str, List[Task]]] = []

        for task_graph_tupple in task_graphs:
            task_graph = task_graph_tupple[0]
            # print(f'task_graph: {task_graph}')
            comp_schdeule = scheduler.schedule(network, task_graph)
            comp_schdeules.append(comp_schdeule)
            # print(f'comp_schdeule: {comp_schdeule}')

        # print(f'Length of comp_schdeules: {len(comp_schdeules)}')
        # print(f'comp_schdeules: {comp_schdeules}')

        # combine the schedules
        for idx, task_graph_tupple in enumerate(task_graphs):
            comp_schdeule = comp_schdeules[idx]
            # creat a list of all tasks from comp_schedule values with type List[Task]
            all_tasks = [task for task_list in comp_schdeule.values() for task in task_list]            
            for task in all_tasks:
                # print(f'task: {task}')
                _ , start_time = get_insert_loc(
                        final_schedule[task.node],
                        task.start + task_graph_tupple[1],
                        task.end - task.start
                )
                final_schedule[task.node].append(Task(task.node, task.name, start_time, start_time + task.end - task.start))

        return final_schedule

        

class CumulativeWrapper(ABC):

    def schedule(
        self,
        network: nx.Graph,
        task_graphs: List[Tuple[nx.DiGraph, float]],
        scheduler: Scheduler
    ) -> Dict[str, List[Task]]:
        comp_schedule: Dict[str, List[Task]] = {}
        for idx, task_graph_tupple in enumerate(task_graphs):
            # print(f'idx: {idx}')
            task_graph = task_graph_tupple[0]
            task_graph_arrival_time = task_graph_tupple[1]

            combined_task_graph = nx.DiGraph()

            for i in range(idx + 1):
                combined_task_graph = nx.compose(combined_task_graph, task_graphs[i][0])

            
            # if task_graph_arrival_time > 0:
            #     for task_list in comp_schedule.values():
            #         for task in task_list:
            #             print(f'task {task.name} with start: {task.start}, end: {task.end}')
            #             if task.end > task_graph_arrival_time:
            #                 task_list.remove(task)
            #                 print(f'task {task.name} with start: {task.start}, end: {task.end} needs to be resheduled')

            if task_graph_arrival_time > 0:
                # Flatten all tasks into a single list
                all_tasks = [(key, task) for key, task_list in comp_schedule.items() for task in task_list]

                # Create a new schedule after filtering
                updated_schedule = {key: [] for key in comp_schedule}  # Initialize an empty schedule
                for key, task in all_tasks:
                    # print(f'Task {task.name} with start: {task.start}, end: {task.end}')
                    if task.end <= task_graph_arrival_time:
                        # Retain the task if it ends before the graph arrival time
                        updated_schedule[key].append(task)
                        if task.name in combined_task_graph:
                            combined_task_graph.nodes[task.name]['weight'] = 0
                    else:
                        pass
                        # print(f'Task {task.name} with start: {task.start}, end: {task.end} needs to be rescheduled')

                # Update the original schedule with the filtered tasks
                comp_schedule = updated_schedule

            
            for node, attributes in combined_task_graph.nodes(data=True):
                weight = attributes.get('weight', None)  # Get the 'weight' attribute, default to None if not set
                # print(f"Node: {node}, Weight: {weight}")



            if idx <= 0:
                comp_schedule = scheduler.schedule(network, task_graph)
            else:
                comp_schedule = scheduler.schedule(network, combined_task_graph, comp_schedule, task_graph_arrival_time)

            # print(f'idx: {idx} comp_schedule: {comp_schedule}')
            
        # print(f'final comp_schedule: {comp_schedule}')

        return comp_schedule






def get_random_network() -> nx.Graph:
    # network = get_network(num_nodes=random.randint(3, 8))
    # network = add_random_weights(network)

    network = get_network(num_nodes=3)

    """Adds random weights to the DAG."""
    for node in network.nodes:
        network.nodes[node]["weight"] = 1
    for edge in network.edges:
        if not network.is_directed() and edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9 * 1
        else:
            network.edges[edge]["weight"] = 1

    return network

def get_random_task_graph() -> nx.DiGraph:
    choice = random.choice(["chain", "fork", "diamond", "branching"])
    # choice = random.choice(["branching"])

    if choice == "chain":
        task_graph = get_chain_dag(
            # num_nodes=random.randint(2, 2)
            num_nodes=random.randint(5, 10)
        )
    elif choice == "fork":
        task_graph = get_fork_dag()
    elif choice == "diamond":
        task_graph = get_diamond_dag()
    elif choice == "branching":
        task_graph = get_branching_dag(
            levels=random.randint(2,4),
            branching_factor=random.randint(2,4)
        )

    add_random_weights(task_graph)

    return task_graph

def draw_instance(network: nx.Graph, task_graph: nx.DiGraph):
    logging.basicConfig(level=logging.INFO)
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'task_graph.png'))

    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / 'network.png'))

def draw_my_network(network: nx.Graph, name: str):
    ax: plt.Figure = draw_network(network, draw_colors=False, use_latex=True)
    ax.get_figure().savefig(str(thisdir / f'{name}.png'))

def draw_my_task_graph(task_graph: nx.DiGraph, name: str):
    ax: plt.Axes = draw_task_graph(task_graph, use_latex=True)
    ax.get_figure().savefig(str(thisdir / f'{name}.png'))

def draw_schedule(schedule: Dict[str, List[Task]], name: str, xmax: float = None):
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    ax.get_figure().savefig(str(thisdir / f'{name}.png'))

def main():
    # forth try

    network = get_random_network()
    task_graph = get_random_task_graph()
    draw_my_task_graph(task_graph, f"task_graph")
    draw_my_network(network, "network")


    # non-residual schedulers
    wbaScheduler = WBAScheduler()
    sufferageScheduler = SufferageScheduler()
    olbScheduler = OLBScheduler()
    msbcScheduler = MsbcScheduler()
    metScheduler = METScheduler()
    mctScheduler = MCTScheduler()
    hbmcScheduler = HbmctScheduler()
    gdlscheduler = GDLScheduler()
    flbScheduler = FLBScheduler()
    heftScheduler = HeftScheduler()
    bilScheduler = BILScheduler()
    cpopScheduler = CpopScheduler()
    dpScheduler = DPSScheduler()
    minminScheduler = MinMinScheduler()
    maxminScheduler = MaxMinScheduler()
    duplexScheduler = DuplexScheduler()
    etfScheduler = ETFScheduler()
    fastestNodeScheduler = FastestNodeScheduler()
    fcpScheduler = FCPScheduler()

    # bruteForceScheduler = BruteForceScheduler()



    # todo: implement baseline schedulers for comparison
    # baselineHeftScheduler = BaselineHeftScheduler()





    # non-residual schedulers
    wbaSchedule = wbaScheduler.schedule(network, task_graph)
    sufferageSchedule = sufferageScheduler.schedule(network, task_graph)
    olbSchedule = olbScheduler.schedule(network, task_graph)
    msbcSchedule = msbcScheduler.schedule(network, task_graph)
    metSchedule = metScheduler.schedule(network, task_graph)
    mctSchedule = mctScheduler.schedule(network, task_graph)
    hbmcSchedule = hbmcScheduler.schedule(network, task_graph)
    gdlschedule = gdlscheduler.schedule(network, task_graph)
    bilSchedule = bilScheduler.schedule(network, task_graph)
    heftSchedule = heftScheduler.schedule(network, task_graph)
    cpopSchedule = cpopScheduler.schedule(network, task_graph)
    dpSchedule = dpScheduler.schedule(network, task_graph)
    minminSchedule = minminScheduler.schedule(network, task_graph)
    maxminSchedule = maxminScheduler.schedule(network, task_graph)
    duplexSchedule = duplexScheduler.schedule(network, task_graph)
    etfSchedule = etfScheduler.schedule(network, task_graph)
    fastestNodeSchedule = fastestNodeScheduler.schedule(network, task_graph)
    fcpSchedule = fcpScheduler.schedule(network, task_graph)
    flbSchedule = flbScheduler.schedule(network, task_graph)
    # bruteForceSchedule = bruteForceScheduler.schedule(network, task_graphs[0][0])

    # count the number of idle cores in each schedule
    wbaIdleCores = sum([1 for tasks in wbaSchedule.values() if not tasks])
    sufferageIdleCores = sum([1 for tasks in sufferageSchedule.values() if not tasks])
    olbIdleCores = sum([1 for tasks in olbSchedule.values() if not tasks])
    msbcIdleCores = sum([1 for tasks in msbcSchedule.values() if not tasks])
    metIdleCores = sum([1 for tasks in metSchedule.values() if not tasks])
    mctIdleCores = sum([1 for tasks in mctSchedule.values() if not tasks])
    hbmcIdleCores = sum([1 for tasks in hbmcSchedule.values() if not tasks])
    gdlIdleCores = sum([1 for tasks in gdlschedule.values() if not tasks])
    bilIdleCores = sum([1 for tasks in bilSchedule.values() if not tasks])
    heftIdleCores = sum([1 for tasks in heftSchedule.values() if not tasks])
    cpopIdleCores = sum([1 for tasks in cpopSchedule.values() if not tasks])
    dpIdleCores = sum([1 for tasks in dpSchedule.values() if not tasks])
    minminIdleCores = sum([1 for tasks in minminSchedule.values() if not tasks])
    maxminIdleCores = sum([1 for tasks in maxminSchedule.values() if not tasks])
    duplexIdleCores = sum([1 for tasks in duplexSchedule.values() if not tasks])
    etfIdleCores = sum([1 for tasks in etfSchedule.values() if not tasks])
    fastestNodeIdleCores = sum([1 for tasks in fastestNodeSchedule.values() if not tasks])
    fcpIdleCores = sum([1 for tasks in fcpSchedule.values() if not tasks])
    flbIdleCores = sum([1 for tasks in flbSchedule.values() if not tasks])
    # bruteForceIdleCores = sum([1 for tasks in bruteForceSchedule.values() if not tasks])




    # draw non-residual schedules
    draw_schedule(wbaSchedule, "WBA_schedule")
    draw_schedule(sufferageSchedule, "Sufferage_schedule")
    draw_schedule(olbSchedule, "OLB_schedule")
    draw_schedule(msbcSchedule, "MSBC_schedule")
    draw_schedule(metSchedule, "MET_schedule")
    draw_schedule(mctSchedule, "MCT_schedule")
    draw_schedule(hbmcSchedule, "HBMCT_schedule")
    draw_schedule(gdlschedule, "GDL_schedule")
    draw_schedule(bilSchedule, "BIL_schedule")
    draw_schedule(heftSchedule, "HEFT_schedule")
    draw_schedule(cpopSchedule, "CPOP_schedule")
    draw_schedule(dpSchedule, "DPS_schedule")
    draw_schedule(minminSchedule, "MinMin_schedule")
    draw_schedule(maxminSchedule, "MaxMin_schedule")
    draw_schedule(duplexSchedule, "Duplex_schedule")
    draw_schedule(etfSchedule, "ETF_schedule")
    draw_schedule(fastestNodeSchedule, "FastestNode_schedule")
    draw_schedule(fcpSchedule, "FCP_schedule")
    draw_schedule(flbSchedule, "FLB_schedule")
    # draw_schedule(bruteForceSchedule, "BF_schedule")

    # print makespan for each non-residual scheduler
    print(f'HEFT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in heftSchedule.values()])}, idle cores: {heftIdleCores}')
    print(f'BIL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in bilSchedule.values()])}, idle cores: {bilIdleCores}')
    print(f'CPOP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in cpopSchedule.values()])}, idle cores: {cpopIdleCores}')
    print(f'DPS Makespan: {max([0 if not tasks else tasks[-1].end for tasks in dpSchedule.values()])}, idle cores: {dpIdleCores}')
    print(f'MinMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in minminSchedule.values()])}, idle cores: {minminIdleCores}')
    print(f'MaxMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in maxminSchedule.values()])}, idle cores: {maxminIdleCores}')
    print(f'Duplex Makespan: {max([0 if not tasks else tasks[-1].end for tasks in duplexSchedule.values()])}, idle cores: {duplexIdleCores}')
    print(f'ETF Makespan: {max([0 if not tasks else tasks[-1].end for tasks in etfSchedule.values()])}, idle cores: {etfIdleCores}')
    print(f'FastestNode Makespan: {max([0 if not tasks else tasks[-1].end for tasks in fastestNodeSchedule.values()])}, idle cores: {fastestNodeIdleCores}')
    print(f'FCP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in fcpSchedule.values()])}, idle cores: {fcpIdleCores}')
    print(f'FLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in flbSchedule.values()])}, idle cores: {flbIdleCores}')
    print(f'GDL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in gdlschedule.values()])}, idle cores: {gdlIdleCores}')
    print(f'HBMCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in hbmcSchedule.values()])}, idle cores: {hbmcIdleCores}')
    print(f'MCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in mctSchedule.values()])}, idle cores: {mctIdleCores}')
    print(f'MET Makespan: {max([0 if not tasks else tasks[-1].end for tasks in metSchedule.values()])}, idle cores: {metIdleCores}')
    print(f'MSBC Makespan: {max([0 if not tasks else tasks[-1].end for tasks in msbcSchedule.values()])}, idle cores: {msbcIdleCores}')
    print(f'OLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in olbSchedule.values()])}, idle cores: {olbIdleCores}')
    print(f'Sufferage Makespan: {max([0 if not tasks else tasks[-1].end for tasks in sufferageSchedule.values()])}, idle cores: {sufferageIdleCores}')
    print(f'WBA Makespan: {max([0 if not tasks else tasks[-1].end for tasks in wbaSchedule.values()])}, idle cores: {wbaIdleCores}')

    # baseline makespan
    # print(f'Baseline HEFT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in baselineHeftSchedule.values()])}')

    

if __name__ == '__main__':
    main()

