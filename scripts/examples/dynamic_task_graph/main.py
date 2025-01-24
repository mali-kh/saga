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
    network = get_network(num_nodes=random.randint(3, 8))
    # add equal weights to egdes of the network
    # weight_range = (1, 10)
    # for node in network.nodes:
    #     network.nodes[node]["weight"] = np.random.uniform(*weight_range)
    
    # for edge in network.edges:
    #     if not network.is_directed() and edge[0] == edge[1]:
    #         network.edges[edge]["weight"] = 1e9 * weight_range[1] # very large communication speed
    #     else:
    #         network.edges[edge]["weight"] = 1

    network = add_random_weights(network)
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

    num_task_graphs = 3
    task_graphs = []  

    for i in range(num_task_graphs):
        task_graph = get_random_task_graph()
        new_node_names = {node: f"{node}_{i}" for node in task_graph.nodes}
        nx.relabel_nodes(task_graph, new_node_names, copy=False)
        draw_my_task_graph(task_graph, f"task_graph_{i}")
        task_graphs.append((task_graph, 0.5 * i))  


    draw_my_network(network, "network")


    for task_graph_tupple in task_graphs:
        task_graph = task_graph_tupple[0]
        task_graph_arrival_time = task_graph_tupple[1]
        for node in task_graph.nodes:
            task_graph.nodes[node]["arrival_time"] = task_graph_arrival_time

    # join all task graphs to create a combined task graph
    combined_task_graph = nx.DiGraph()
    for task_graph_tupple in task_graphs:
        combined_task_graph = nx.compose(combined_task_graph, task_graph_tupple[0])

    draw_my_task_graph(combined_task_graph, "combined_task_graph")



    # residual schedulers
    residualWBAScheduler = ResidualWBAScheduler()
    residualSufferageScheduler = ResidualSufferageScheduler()
    residualOLBScheduler = ResidualOLBScheduler()
    residualMsbcScheduler = ResidualMsbcScheduler()
    residualMETScheduler = ResidualMETScheduler()
    residualMCTScheduler = ResidualMCTScheduler()
    residualHbmctScheduler = ResidualHbmctScheduler()
    residualGDLScheduler = ResidualGDLScheduler()
    residualFLBScheduler = ResidualFLBScheduler()
    residualHeftScheduler = ResidualHeftScheduler()
    residualBILScheduler = ResidualBILScheduler()
    residualCpopScheduler = ResidualCpopScheduler()
    residualDPSScheduler = ResidualDPSScheduler()
    residualMinMinScheduler = ResidualMinMinScheduler()
    residualMaxMinScheduler = ResidualMaxMinScheduler()
    residualDuplexScheduler = ResidualDuplexScheduler()
    residualETFScheduler = ResidualETFScheduler()
    residualFastestNodeScheduler = ResidualFastestNodeScheduler()
    residualFCPScheduler = ResidualFCPScheduler()
    # residualBruteForceScheduler = ResidualBruteForceScheduler()


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




    # residual schedulers
    residualWBASchedule = residualWBAScheduler.schedule(network, task_graphs)
    residualSufferageSchedule = residualSufferageScheduler.schedule(network, task_graphs)
    residualOLBSchedule = residualOLBScheduler.schedule(network, task_graphs)
    residualMsbcSchedule = residualMsbcScheduler.schedule(network, task_graphs)
    residualMETSchedule = residualMETScheduler.schedule(network, task_graphs)
    residualMCTSchedule = residualMCTScheduler.schedule(network, task_graphs)
    residualHbmctSchedule = residualHbmctScheduler.schedule(network, task_graphs)
    residualGDLSchedule = residualGDLScheduler.schedule(network, task_graphs)
    residualHeftSchedule = residualHeftScheduler.schedule(network, task_graphs)
    residualBILSchedule = residualBILScheduler.schedule(network, task_graphs)
    residualCpopSchedule = residualCpopScheduler.schedule(network, task_graphs)
    residualDPSSchedule = residualDPSScheduler.schedule(network, task_graphs)
    residualMinMinSchedule = residualMinMinScheduler.schedule(network, task_graphs)
    residualMaxMinSchedule = residualMaxMinScheduler.schedule(network, task_graphs)
    residualDuplexSchedule = residualDuplexScheduler.schedule(network, task_graphs)
    residualETFSchedule = residualETFScheduler.schedule(network, task_graphs)
    residualFastestNodeSchedule = residualFastestNodeScheduler.schedule(network, task_graphs)
    residualFCPSchedule = residualFCPScheduler.schedule(network, task_graphs)
    residualFLBSchedule = residualFLBScheduler.schedule(network, task_graphs)
    # residualBruteForceSchedule = residualBruteForceScheduler.schedule(network, task_graphs)

    # non-residual schedulers
    wbaSchedule = wbaScheduler.schedule(network, task_graphs[0][0])
    sufferageSchedule = sufferageScheduler.schedule(network, task_graphs[0][0])
    olbSchedule = olbScheduler.schedule(network, task_graphs[0][0])
    msbcSchedule = msbcScheduler.schedule(network, task_graphs[0][0])
    metSchedule = metScheduler.schedule(network, task_graphs[0][0])
    mctSchedule = mctScheduler.schedule(network, task_graphs[0][0])
    hbmcSchedule = hbmcScheduler.schedule(network, task_graphs[0][0])
    gdlschedule = gdlscheduler.schedule(network, task_graphs[0][0])
    bilSchedule = bilScheduler.schedule(network, task_graphs[0][0])
    heftSchedule = heftScheduler.schedule(network, task_graphs[0][0])
    cpopSchedule = cpopScheduler.schedule(network, task_graphs[0][0])
    dpSchedule = dpScheduler.schedule(network, task_graphs[0][0])
    minminSchedule = minminScheduler.schedule(network, task_graphs[0][0])
    maxminSchedule = maxminScheduler.schedule(network, task_graphs[0][0])
    duplexSchedule = duplexScheduler.schedule(network, task_graphs[0][0])
    etfSchedule = etfScheduler.schedule(network, task_graphs[0][0])
    fastestNodeSchedule = fastestNodeScheduler.schedule(network, task_graphs[0][0])
    fcpSchedule = fcpScheduler.schedule(network, task_graphs[0][0])
    flbSchedule = flbScheduler.schedule(network, task_graphs[0][0])
    # bruteForceSchedule = bruteForceScheduler.schedule(network, task_graphs[0][0])

    #  baseline schedule
    # baselineHeftSchedule = baselineHeftScheduler.schedule(network, combined_task_graph)


    # draw residual schedules
    draw_schedule(residualWBASchedule, "R_WBA_schedule")
    draw_schedule(residualSufferageSchedule, "R_Sufferage_schedule")
    draw_schedule(residualOLBSchedule, "R_OLB_schedule")
    draw_schedule(residualMsbcSchedule, "R_MSBC_schedule")
    draw_schedule(residualMETSchedule, "R_MET_schedule")
    draw_schedule(residualMCTSchedule, "R_MCT_schedule")
    draw_schedule(residualHbmctSchedule, "R_HBMCT_schedule")
    draw_schedule(residualGDLSchedule, "R_GDL_schedule")
    draw_schedule(residualHeftSchedule, "R_HEFT_schedule")
    draw_schedule(residualBILSchedule, "R_BIL_schedule")
    draw_schedule(residualCpopSchedule, "R_CPOP_schedule")
    draw_schedule(residualDPSSchedule, "R_DPS_schedule")
    draw_schedule(residualMinMinSchedule, "R_MinMin_schedule")
    draw_schedule(residualMaxMinSchedule, "R_MaxMin_schedule")
    draw_schedule(residualDuplexSchedule, "R_Duplex_schedule")
    draw_schedule(residualETFSchedule, "R_ETF_schedule")
    draw_schedule(residualFastestNodeSchedule, "R_FastestNode_schedule")
    draw_schedule(residualFCPSchedule, "R_FCP_schedule")
    draw_schedule(residualFLBSchedule, "R_FLB_schedule")

    # draw_schedule(residualBruteForceSchedule, "R_BF_schedule")

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
    
                  
    # baseline schedule
    # draw_schedule(baselineHeftSchedule, "baseline_HEFT_schedule")

    # print maximum makespan for each scheduler
    print(f'Residual HEFT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualHeftSchedule.values()])}')
    print(f'Residual BIL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualBILSchedule.values()])}')
    print(f'Residual CPOP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualCpopSchedule.values()])}')
    print(f'Residual DPS Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualDPSSchedule.values()])}')
    print(f'Residual MinMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualMinMinSchedule.values()])}')
    print(f'Residual MaxMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualMaxMinSchedule.values()])}')
    print(f'Residual Duplex Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualDuplexSchedule.values()])}')
    print(f'Residual ETF Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualETFSchedule.values()])}')
    print(f'Residual FastestNode Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualFastestNodeSchedule.values()])}')
    print(f'Residual FCP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualFCPSchedule.values()])}')
    print(f'Residual FLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualFLBSchedule.values()])}')
    print(f'Residual GDL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualGDLSchedule.values()])}')
    print(f'Residual HBMCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualHbmctSchedule.values()])}')
    print(f'Residual MCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualMCTSchedule.values()])}')
    print(f'Residual MET Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualMETSchedule.values()])}')
    print(f'Residual MSBC Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualMsbcSchedule.values()])}')
    print(f'Residual OLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualOLBSchedule.values()])}')
    print(f'Residual Sufferage Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualSufferageSchedule.values()])}')
    print(f'Residual WBA Makespan: {max([0 if not tasks else tasks[-1].end for tasks in residualWBASchedule.values()])}')

    # print makespan for each non-residual scheduler
    print(f'HEFT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in heftSchedule.values()])}')
    print(f'BIL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in bilSchedule.values()])}')
    print(f'CPOP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in cpopSchedule.values()])}')
    print(f'DPS Makespan: {max([0 if not tasks else tasks[-1].end for tasks in dpSchedule.values()])}')
    print(f'MinMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in minminSchedule.values()])}')
    print(f'MaxMin Makespan: {max([0 if not tasks else tasks[-1].end for tasks in maxminSchedule.values()])}')
    print(f'Duplex Makespan: {max([0 if not tasks else tasks[-1].end for tasks in duplexSchedule.values()])}')
    print(f'ETF Makespan: {max([0 if not tasks else tasks[-1].end for tasks in etfSchedule.values()])}')
    print(f'FastestNode Makespan: {max([0 if not tasks else tasks[-1].end for tasks in fastestNodeSchedule.values()])}')
    print(f'FCP Makespan: {max([0 if not tasks else tasks[-1].end for tasks in fcpSchedule.values()])}')
    print(f'FLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in flbSchedule.values()])}')
    print(f'GDL Makespan: {max([0 if not tasks else tasks[-1].end for tasks in gdlschedule.values()])}')
    print(f'HBMCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in hbmcSchedule.values()])}')
    print(f'MCT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in mctSchedule.values()])}')
    print(f'MET Makespan: {max([0 if not tasks else tasks[-1].end for tasks in metSchedule.values()])}')
    print(f'MSBC Makespan: {max([0 if not tasks else tasks[-1].end for tasks in msbcSchedule.values()])}')
    print(f'OLB Makespan: {max([0 if not tasks else tasks[-1].end for tasks in olbSchedule.values()])}')
    print(f'Sufferage Makespan: {max([0 if not tasks else tasks[-1].end for tasks in sufferageSchedule.values()])}')
    print(f'WBA Makespan: {max([0 if not tasks else tasks[-1].end for tasks in wbaSchedule.values()])}')

    # baseline makespan
    # print(f'Baseline HEFT Makespan: {max([0 if not tasks else tasks[-1].end for tasks in baselineHeftSchedule.values()])}')

    

if __name__ == '__main__':
    main()

