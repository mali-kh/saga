from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx


@dataclass
class Task:
    """A task."""
    node: str
    name: str
    start: Optional[float]
    end: Optional[float]

class ResidualScheduler(ABC):
    """An abstract class for a residual scheduler."""
    @abstractmethod
    def schedule(self, network: nx.Graph, task_graphs: List[Tuple[nx.DiGraph, float]]) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graphs (List[Tuple[nx.DiGraph, float]]): The task graphs.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        raise NotImplementedError

    @property
    def __name__(self) -> str:
        """Get the name of the scheduler.

        Returns:
            str: The name of the scheduler.
        """
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__



class Scheduler(ABC): # pylint: disable=too-few-public-methods
    """An abstract class for a scheduler."""
    @abstractmethod
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[Hashable, List[Task]]: A dictionary mapping nodes to a list of tasks executed on the node.
        """
        raise NotImplementedError

    @property
    def __name__(self) -> str:
        """Get the name of the scheduler.

        Returns:
            str: The name of the scheduler.
        """
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__
