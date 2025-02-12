from matplotlib import pyplot as plt
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import GreedyInsert, UpwardRanking, ScheduleType

from saga.utils.random_graphs import get_network, get_branching_dag, add_ccr_weights, add_random_weights
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import pathlib
import networkx as nx

from typing import List, Tuple

thisdir = pathlib.Path(__file__).resolve().parent



def residual(network: nx.Graph, task_graphs: nx.DiGraph, arrival_times: List[float]) -> ScheduleType:
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False,
        ),
        cumulative=False
    )

    schedule: ScheduleType = {node: [] for node in network.nodes}
    for i, (arrival_time, task_graph) in enumerate(zip(arrival_times, task_graphs), start=1):
        schedule = scheduler.schedule(
            network=network,
            task_graph=task_graph,
            schedule=schedule,
            min_start_time=arrival_time
        )

        ax_task_graph: plt.Axes = draw_task_graph(task_graph)
        ax_task_graph.get_figure().savefig(thisdir / f"residual_task_graph_{i}.png")
        plt.close(ax_task_graph.get_figure())
        
        ax_schedule: plt.Axes = draw_gantt(schedule)
        ax_schedule.get_figure().savefig(thisdir / f"residual_schedule_{i}.png")
        plt.close(ax_schedule.get_figure())

    return schedule


def cumulative(network: nx.Graph, task_graphs: List[nx.DiGraph], arrival_times: List[float]) -> ScheduleType:
    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=GreedyInsert(
            append_only=False,
            compare="EFT",
            critical_path=False
        ),
        cumulative=True
    )

    schedule: ScheduleType = {node: [] for node in network.nodes}
    for i, (arrival_time, task_graph) in enumerate(zip(arrival_times, task_graphs), start=1):
        # join all previous task graphs with the current task graph
        task_graph = nx.compose_all([task_graphs[j] for j in range(i)])

        schedule = scheduler.schedule(
            network=network,
            task_graph=task_graph,
            schedule=schedule,
            min_start_time=arrival_time
        )

        ax_task_graph: plt.Axes = draw_task_graph(task_graph)
        ax_task_graph.get_figure().savefig(thisdir / f"cumulative_task_graph_{i}.png")
        plt.close(ax_task_graph.get_figure())
        
        ax_schedule: plt.Axes = draw_gantt(schedule)
        ax_schedule.get_figure().savefig(thisdir / f"cumulative_schedule_{i}.png")
        plt.close(ax_schedule.get_figure())

    return schedule











def delayed_task():
    # create a task graph 3 nodes
    network = get_network(num_nodes=2)

    for node in network.nodes:
        network.nodes[node]["weight"] = 1
    for edge in network.edges:
        if not network.is_directed() and edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9 * 1
        else:
            network.edges[edge]["weight"] = 1

    ax_network: plt.Axes = draw_network(network)
    ax_network.get_figure().savefig(thisdir / "network.png")
    plt.close(ax_network.get_figure())

    task_graph_1 = nx.DiGraph()
    task_graph_1.add_nodes_from(["1.A", "1.B", "1.C"])
    task_graph_1.add_edges_from([("1.A", "1.B"), ("1.A", "1.C")])

    task_graph_1.nodes["1.A"]["weight"] = 1.3
    task_graph_1.nodes["1.B"]["weight"] = 2
    task_graph_1.nodes["1.C"]["weight"] = 1

    task_graph_1.edges[("1.A", "1.B")]["weight"] = 1
    task_graph_1.edges[("1.A", "1.C")]["weight"] = 1


    task_graph_2 = nx.DiGraph()
    task_graph_2.add_nodes_from(["2.A", "2.B", "2.C"])
    task_graph_2.add_edges_from([("2.A", "2.B"), ("2.A", "2.C")])

    task_graph_2.nodes["2.A"]["weight"] = 1.3
    task_graph_2.nodes["2.B"]["weight"] = 2
    task_graph_2.nodes["2.C"]["weight"] = 1

    task_graph_2.edges[("2.A", "2.B")]["weight"] = 1
    task_graph_2.edges[("2.A", "2.C")]["weight"] = 1

    task_graph_3 = nx.DiGraph()
    task_graph_3.add_nodes_from(["3.A", "3.B", "3.C"])
    task_graph_3.add_edges_from([("3.A", "3.B"), ("3.A", "3.C")])

    task_graph_3.nodes["3.A"]["weight"] = 1.3
    task_graph_3.nodes["3.B"]["weight"] = 2
    task_graph_3.nodes["3.C"]["weight"] = 1

    task_graph_3.edges[("3.A", "3.B")]["weight"] = 1
    task_graph_3.edges[("3.A", "3.C")]["weight"] = 1

    task_graph_4 = nx.DiGraph()
    task_graph_4.add_nodes_from(["4.A", "4.B", "4.C"])
    task_graph_4.add_edges_from([("4.A", "4.B"), ("4.A", "4.C")])

    task_graph_4.nodes["4.A"]["weight"] = 1.3
    task_graph_4.nodes["4.B"]["weight"] = 2
    task_graph_4.nodes["4.C"]["weight"] = 1

    task_graph_4.edges[("4.A", "4.B")]["weight"] = 1
    task_graph_4.edges[("4.A", "4.C")]["weight"] = 1

    arrival_times = [0, 2, 3, 4]
    task_graphs = [task_graph_1, task_graph_2, task_graph_3, task_graph_4]
    
    cumulative_schedule = cumulative(network, task_graphs, arrival_times)
    residual_schedule = residual(network, task_graphs, arrival_times)

    # print makespan for each schedule
    print("Cumulative makespan:", max(task.end for tasks in cumulative_schedule.values() for task in tasks))
    print("Residual makespan:", max(task.end for tasks in residual_schedule.values() for task in tasks))

    # calculate the sum of the makespan for each task graph
    
    tasks_by_task_graph = {}
    for tasks in cumulative_schedule.values():
        for task in tasks:
            if task.name.split(".")[0] not in tasks_by_task_graph:
                tasks_by_task_graph[task.name.split(".")[0]] = []
            tasks_by_task_graph[task.name.split(".")[0]].append(task)

    cumulative_sum_of_makespans = 0
    for i, tasks in enumerate(dict(sorted(tasks_by_task_graph.items())).values()):
        cumulative_sum_of_makespans += max(task.end for task in tasks) - arrival_times[i]
        print(f'{i+1}: {max(task.end for task in tasks)} - {arrival_times[i]} = {max(task.end for task in tasks) - arrival_times[i]}')

    print(f'cumulative_sum_of_makespans: {cumulative_sum_of_makespans}')

    tasks_by_task_graph = {}
    for tasks in residual_schedule.values():
        for task in tasks:
            if task.name.split(".")[0] not in tasks_by_task_graph:
                tasks_by_task_graph[task.name.split(".")[0]] = []
            tasks_by_task_graph[task.name.split(".")[0]].append(task)

    residual_sum_of_makespans = 0
    for i, tasks in enumerate(dict(sorted(tasks_by_task_graph.items())).values()):
        residual_sum_of_makespans += max(task.end for task in tasks) - arrival_times[i]
        print(f'{i+1}: {max(task.end for task in tasks)} - {arrival_times[i]} = {max(task.end for task in tasks) - arrival_times[i]}')

    print(f'residual_sum_of_makespans: {residual_sum_of_makespans}')


def residual_cumulative_comparison():
    arrival_times = [2, 2.1, 5]
    task_graphs = [
        add_random_weights(get_branching_dag(levels=2, branching_factor=2))
        for _ in arrival_times
    ]

    # Rename nodes with prefixes t1_, t2_, etc.
    for i, _tg in enumerate(task_graphs, start=1):  # start=1 to match t1, t2, etc.
        mapping = {node: f"{i}.{node}" for node in _tg.nodes}
        nx.relabel_nodes(_tg, mapping, copy=False)  # Modify in-place

    network = add_ccr_weights(task_graphs[0], add_random_weights(get_network()), ccr=1.0)

    ax_network: plt.Axes = draw_network(network)
    ax_network.get_figure().savefig(thisdir / "network.png")
    plt.close(ax_network.get_figure())
    
    residual_schedule = residual(network, task_graphs, arrival_times)
    cumulative_schedule = cumulative(network, task_graphs, arrival_times)

    # print makespan for each schedule
    print("Residual makespan:", max(task.end for tasks in residual_schedule.values() for task in tasks))
    print("Cumulative makespan:", max(task.end for tasks in cumulative_schedule.values() for task in tasks))

if __name__ == '__main__':
    # residual_cumulative_comparison()
    delayed_task()