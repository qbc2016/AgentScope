# -*- coding: utf-8 -*-
""" Functional counterpart for Pipeline """
from typing import (
    Callable,
    Sequence,
    Optional,
    Union,
    Any,
    Mapping,
)
import re

from .utils import format_dependency, topological_sort
from ..agents.operator import Operator
from ..message.msg import Msg
from ..agents import DialogAgent
from ..prompt._prompt_scheduler_pipeline import (
    agent_sys_prompt_with_context,
    agent_sys_prompt_without_context,
    scheduler_sys_prompt,
)

# A single Operator or a Sequence of Operators
Operators = Union[Operator, Sequence[Operator]]


def placeholder(x: dict = None) -> dict:
    r"""A placeholder that do nothing.

    Acts as a placeholder in branches that do not require any operations in
    flow control like if-else/switch
    """
    return x


def sequentialpipeline(
    operators: Sequence[Operator],
    x: Optional[dict] = None,
) -> dict:
    """Functional version of SequentialPipeline.

    Args:
        operators (`Sequence[Operator]`):
            Participating operators.
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `dict`: the output dictionary.
    """
    if len(operators) == 0:
        raise ValueError("No operators provided.")

    msg = operators[0](x)
    for operator in operators[1:]:
        msg = operator(msg)
    return msg


def _operators(operators: Operators, x: Optional[dict] = None) -> dict:
    """Syntactic sugar for executing a single operator or a sequence of
    operators."""
    if isinstance(operators, Sequence):
        return sequentialpipeline(operators, x)
    else:
        return operators(x)


def ifelsepipeline(
    condition_func: Callable,
    if_body_operators: Operators,
    else_body_operators: Operators = placeholder,
    x: Optional[dict] = None,
) -> dict:
    """Functional version of IfElsePipeline.

    Args:
        condition_func (`Callable`):
            A function that determines whether to execute `if_body_operator`
            or `else_body_operator` based on x.
        if_body_operator (`Operators`):
            Operators executed when `condition_func` returns True.
        else_body_operator (`Operators`, defaults to `placeholder`):
            Operators executed when condition_func returns False,
            does nothing and just return the input by default.
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `dict`: the output dictionary.
    """
    if condition_func(x):
        return _operators(if_body_operators, x)
    else:
        return _operators(else_body_operators, x)


def switchpipeline(
    condition_func: Callable[[Any], Any],
    case_operators: Mapping[Any, Operators],
    default_operators: Operators = placeholder,
    x: Optional[dict] = None,
) -> dict:
    """Functional version of SwitchPipeline.


    Args:
        condition_func (`Callable[[Any], Any]`):
            A function that determines which case_operator to execute based
            on the input x.
        case_operators (`Mapping[Any, Operator]`):
            A dictionary containing multiple operators and their
            corresponding trigger conditions.
        default_operators (`Operators`, defaults to `placeholder`):
            Operators that are executed when the actual condition do not
            meet any of the case_operators, does nothing and just return the
            input by default.
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        dict: the output dictionary.
    """
    target_case = condition_func(x)
    if target_case in case_operators:
        return _operators(case_operators[target_case], x)
    else:
        return _operators(default_operators, x)


def forlooppipeline(
    loop_body_operators: Operators,
    max_loop: int,
    break_func: Callable[[dict], bool] = lambda _: False,
    x: Optional[dict] = None,
) -> dict:
    """Functional version of ForLoopPipeline.

    Args:
        loop_body_operators (`Operators`):
            Operators executed as the body of the loop.
        max_loop (`int`):
            maximum number of loop executions.
        break_func (`Callable[[dict], bool]`):
            A function used to determine whether to break out of the loop
            based on the output of the loop_body_operator, defaults to
            `lambda _: False`
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `dict`: The output dictionary.
    """
    for _ in range(max_loop):
        # loop body
        x = _operators(loop_body_operators, x)
        # check condition
        if break_func(x):
            break
    return x  # type: ignore[return-value]


def whilelooppipeline(
    loop_body_operators: Operators,
    condition_func: Callable[[int, Any], bool] = lambda _, __: False,
    x: Optional[dict] = None,
) -> dict:
    """Functional version of WhileLoopPipeline.

    Args:
        loop_body_operators (`Operators`): Operators executed as the body of
            the loop.
        condition_func (`Callable[[int, Any], bool]`, optional): A function
            that determines whether to continue executing the loop body based
            on the current loop number and output of the loop_body_operator,
            defaults to `lambda _,__: False`
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `dict`: the output dictionary.
    """
    i = 0
    while condition_func(i, x):
        # loop body
        x = _operators(loop_body_operators, x)
        # check condition
        i += 1
    return x  # type: ignore[return-value]


def schedulerpipeline(
    model_config_name: str,
    operators: Sequence[Operator],
    desc_list: list[str],
    x: Optional[dict],
) -> dict:
    """Functional version of schedulerpipeline.

    Args:
        model_config_name (`str`): The model config name for
            Scheduler agent.
        operators (`Operators`): Operators executed as the body of
            the pipeline.
        desc_list (`list[str]`): Descriptions corresponding to each operator.
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `Msg`: the output message.
    """
    assert len(operators) == len(
        desc_list,
    ), "The number of operators and descriptions must be the same."
    desc_dict = {}
    candidates = ""
    for i, operator in enumerate(operators):
        desc_dict[operator.name] = operator
        candidates += f"""
### {operator.name}
{desc_list[i]}"""

    candidates = candidates.strip()

    scheduler_agent = DialogAgent(
        name="scheduler",
        sys_prompt=scheduler_sys_prompt.format(candidates=candidates),
        model_config_name=model_config_name,
    )

    scheduler_result = scheduler_agent(x)
    step_pattern = re.compile(
        r"# Step-(\d+):\n<Subtask>: (.*?)\n<Agent>: (.*?)\n<Dependency>: ("
        r".*?)(?=\n# Step|\n$|$)",
        re.DOTALL,
    )
    matches = step_pattern.findall(scheduler_result.content)

    dependence = format_dependency(matches)

    matches, dependent_dict = topological_sort(matches)
    agent_names = [scheduler_agent.name]
    agent_results = [scheduler_result]

    context_dict = {}

    for idx, subtask, agent_name, _ in matches:
        dependencies = dependent_dict[(idx, agent_name)]
        context = "\n".join(
            "\n".join(context_dict[d])
            for d in dependencies
            if d in context_dict
        )
        if context:
            prompt = agent_sys_prompt_with_context.format(
                task=x.content,
                context=context,
                subtask=subtask,
            )
        else:
            prompt = agent_sys_prompt_without_context.format(
                task=x.content,
                subtask=subtask,
            )
        msg = Msg(role="assistant", name=agent_name, content=prompt)
        app_res = desc_dict[agent_name](msg)
        desc_dict[agent_name].memory.clear()
        context_dict.setdefault(agent_name, []).append(app_res.content)
        agent_names.append(agent_name)
        agent_results.append(app_res)
    details = "\n".join(
        f"## {name}\n### Generation Results\n{result}\n"
        for name, result in zip(
            agent_names,
            [agent_result.content for agent_result in agent_results],
        )
    )
    scheduling_progress = f"""# User Goal
{x.content}

# Planning Steps
{dependence}

# Execution Details
{details}

# Execution Results
{agent_results[-1].content}
"""

    # TODO change the following to Msg
    return {
        "role": "assistant",
        "name": agent_results[-1].name,
        "content": agent_results[-1].content,
        "metadata": scheduling_progress,
    }
