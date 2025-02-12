from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Any, Dict, Hashable, List, Optional, Tuple
import networkx as nx

from saga.scheduler import Scheduler, Task

class IntialPriority(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph) -> List[Hashable]:
        """Return the initial priority of the tasks.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            List[Hashable]: The initial priority of the tasks.
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> "IntialPriority":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            IntialPriority: A new instance of the initial priority.
        """
        pass

ScheduleType = Dict[Hashable, List[Task]]
class InsertTask(ABC):
    @abstractmethod
    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None) -> Task:
        """Insert a task into the schedule.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (ScheduleType): The schedule.
            task (Hashable): The task.  
            node (Optional[Hashable]): The node to insert the task onto. If None, the node is chosen by the algorithm.

        Returns:
            Task: The inserted task
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> "InsertTask":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            InsertTask: A new instance of the initial priority.
        """
        pass

class ParametricScheduler(Scheduler):
    def __init__(self,
                 initial_priority: IntialPriority,
                 insert_task: InsertTask,
                 cumulative: bool = False
                 ) -> None:
            super().__init__()
            self.initial_priority = initial_priority
            self.insert_task = insert_task
            self.cumulative = cumulative
    
    def schedule(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: Optional[ScheduleType] = None,
                 min_start_time: float = 0.0) -> ScheduleType:
        """Schedule the tasks on the network.
        
        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[ScheduleType]): The current schedule.
            min_start_time: float

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        if schedule is not None:
            schedule = {
                node: [
                    Task(task.node, task.name, task.start - min_start_time, task.end - min_start_time)
                    for task in tasks
                ]
                for node, tasks in schedule.items()
            }

        

        queue = self.initial_priority(network, task_graph)

        # print("\n\nschedule before pruning:")
        # for node, tasks in schedule.items():
        #     print(f"Node {node}:")
        #     for task in tasks:
        #         print(f"Task {task.name} starts at {task.start} and ends at {task.end}")
        #     print()

        # print("\nqueue before pruning:")
        # print(queue)
        


        # the code for comulative scheduling (don't need it for the residual scheduling)
        # =============================================
        # the following two loops are for cumulative scheduling (scheduling all graphs together with non-preemptive approach)

        if self.cumulative:
            # remove tasks that have already been scheduled and their start time is less than min_start_time
            for _, tasks in schedule.items():
                for task in tasks:
                    # if task.start < min_start_time:
                    if task.start < 0:
                        queue.remove(task.name)

            # remove tasks that have already been scheduled but their start time is greater than min_start_time from the schedule

            removed_tasks = []
            for _, tasks in schedule.items():
                for task in tasks[:]:  # Iterate over a copy of the list
                    # if task.start > min_start_time:
                    if task.start > 0:
                        tasks.remove(task)
                        removed_tasks.append(task.name)

            for removed in removed_tasks:
                # check if it has any scheduled parents
                for parent in task_graph.predecessors(removed):
                    # check if the parent is in the schedule
                    for _, tasks in schedule.items():
                        for task in tasks:
                            if task.name == parent:
                                task_graph.nodes[parent]["scheduled_task"] = task
        # =============================================

        schedule = {node: [] for node in network.nodes} if schedule is None else deepcopy(schedule)


        # print("\n\nschedule after pruning:")
        # for node, tasks in schedule.items():
        #     print(f"Node {node}:")
        #     for task in tasks:
        #         print(f"Task {task.name} starts at {task.start} and ends at {task.end}")
        #     print()
        # print("\nqueue after pruning:")
        # print(queue)

        while queue:
            # print queue wihout line break
            # print(queue)
            self.insert_task(network, task_graph, schedule, queue.pop(0))

        # print()

        schedule = {
            node: [
                Task(task.node, task.name, task.start + min_start_time, task.end + min_start_time)
                for task in tasks
            ]
            for node, tasks in schedule.items()
        }
        return schedule

    def serialize(self) -> Dict[str, Any]:
        """Return a dictionary representation of the initial priority.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the initial priority.
        """
        return {
            "name": "ParametricScheduler",
            "initial_priority": self.initial_priority.serialize(),
            "insert_task": self.insert_task.serialize(),
            "k_depth": 0
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ParametricScheduler":
        """Return a new instance of the initial priority from the serialized data.
        
        Args:
            data (Dict[str, Any]): The serialized data.

        Returns:
            ParametricScheduler: A new instance of the initial priority.
        """
        return cls(
            initial_priority=IntialPriority.deserialize(data["initial_priority"]),
            insert_task=InsertTask.deserialize(data["insert_task"])
        )
