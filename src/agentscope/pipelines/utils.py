# -*- coding: utf-8 -*-
"""utils for schedulerpipeline"""
from collections import defaultdict
import networkx as nx


def format_dependency(steps: list[tuple]) -> str:
    """
        Formats a list of step tuples into a string that describes the
        dependency relationships among steps.

        Each step tuple is expected to contain:
        - step_number: The identifier for the step.
        - _: (Unused in this function) Can be used to pass additional
            information.
        - agent: The name of the agent responsible for the step.
        - dependency: The identifier(s) of the step(s) this step depends
            on, or "None" if there are no dependencies.

    Parameters:
        steps (list of tuples): A list of tuples where each tuple
            represents a step. Each tuple is formatted as (step_number, _,
            agent, dependency).

    Returns:
        str: A formatted string where each line represents a step and
            its dependencies. Each line is of the format
            "step_number.agent (dependent on dependency)" if there are
            dependencies, or "step_number.agent" if there are no
            dependencies.

    Example:
        Input: [(1, _, "AgentA", "None"), (2, _, "AgentB", "1")]
        Output:
            "1.AgentA
             2.AgentB (dependent on 1)"
    """
    formatted_dependency = []
    for step in steps:
        step_number, _, agent, dependency = step
        if dependency.strip() != "None":
            dependency_str = f"(dependent on {dependency})"
        else:
            dependency_str = ""
        formatted_dependence = f"{step_number}.{agent}{dependency_str}"
        formatted_dependency.append(formatted_dependence)
    return "\n".join(formatted_dependency)


def topological_sort(task_list: list[tuple]) -> tuple[list[tuple], dict]:
    """
    Performs a topological sort on a list of tasks to determine a
        feasible sequence of execution based on dependencies.

    Each task in the task_list is expected to be a tuple containing:
    - step_number: A unique identifier for the task.
    - _: (Unused in this function) Placeholder for additional information.
    - agent: The name of the agent responsible for executing the task.
    - dependencies: A string containing step_numbers this task depends
        on, separated by commas, or "None" if no dependencies.

    Parameters:
        task_list (List[Tuple[str, str, str, str]]): A list of tuples,
            where each tuple represents a task with its associated details.

    Returns:
        Tuple[List[Tuple], Dict[tuple, List[str]]]:
        - A list of tasks sorted in a feasible execution order based on
            the provided dependencies.
        - A dictionary where each key is a tuple consists of ("idx",
            "agent_name"), and the value is a list of agent names that the
            key depends on, sorted in the order of execution.

    Raises:
        ValueError: If there is a circular dependency that prevents
            topological sorting or if a task has undefined dependencies.

    Example:
        Input: [
            ("1", "info", "AgentA", "None"),
            ("2", "info", "AgentB", "1"),
            ("3", "info", "AgentC", "2,1")
        ]
        Output: (
            [("1", "info", "AgentA", "None"), ("2", "info", "AgentB",
            "1"), ("3", "info", "AgentC", "2,1")],
            {("1", "AgentA"): [], ("2", "AgentB"): ["AgentA"], ("3",
            "AgentC"): ["AgentA", "AgentB"]}
        )
    """
    G = nx.DiGraph()

    task_map = {task[0]: task for task in task_list}
    dependencies_by_task = defaultdict(list)

    # Populate the graph and map dependencies with unique identifiers
    for step_number, _, agent, dependencies in task_list:
        G.add_node(step_number)
        if dependencies.strip() != "None":
            for dep in dependencies.split(","):
                dep = dep.strip()
                G.add_edge(dep, step_number)
                dependencies_by_task[(step_number, agent)].append(
                    (dep, task_map[dep][2]),
                )

    # Perform topological sort
    try:
        sorted_task_ids = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible as exc:
        raise ValueError("Circular dependency detected") from exc

    # Create sorted tasks
    sorted_tasks = [task_map[task_id] for task_id in sorted_task_ids]

    # Create the dependency dictionary for each unique task
    dependency_dict = {}
    for task_id in sorted_task_ids:
        agent = task_map[task_id][2]
        key = (task_id, agent)
        if key in dependencies_by_task:
            # Prepare a list of agent names maintaining the task order
            sorted_dependencies = sorted(
                dependencies_by_task[key],
                key=lambda x: sorted_task_ids.index(x[0]),
            )
            dependency_dict[key] = [dep[1] for dep in sorted_dependencies]
        else:
            dependency_dict[key] = []

    return sorted_tasks, dependency_dict
