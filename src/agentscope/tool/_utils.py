# -*- coding: utf-8 -*-
"""The tool module utils."""
import copy
import inspect
import logging
from typing import Any, Dict, Callable

from docstring_parser import parse
from pydantic import Field, create_model, ConfigDict

logger = logging.getLogger(__name__)


def _remove_title_field(schema: dict) -> dict:
    """Remove the title field from the JSON schema to avoid
    misleading the LLM."""
    # The top level title field
    if "title" in schema:
        schema.pop("title")

    # properties
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                _remove_title_field(prop)

    # items
    if "items" in schema and isinstance(schema["items"], dict):
        _remove_title_field(schema["items"])

    # additionalProperties
    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"],
        dict,
    ):
        _remove_title_field(schema["additionalProperties"])

    # $defs — referenced sub-schemas, e.g. Pydantic models used as parameter
    # types generate "$defs": {"SubModel": {"title": "SubModel", ...}}.
    # These titles are auto-generated noise just like property titles, and
    # should be removed for the same reason.
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        for def_schema in schema["$defs"].values():
            if isinstance(def_schema, dict):
                _remove_title_field(def_schema)

    return schema


def _flatten_json_schema(schema: dict) -> dict:
    """Flatten a JSON schema by resolving all local ``$ref`` references.

    Some LLM providers (e.g. Gemini, GLM-5.x via OpenCode Go) cannot
    process ``$defs`` / ``$ref`` patterns in tool parameter schemas.
    When Pydantic generates schemas for complex nested types it emits a
    ``$defs`` block (or the legacy ``definitions`` block) and refers to
    it with ``{"$ref": "#/$defs/TypeName"}``.

    This function resolves every such reference by substituting the full
    definition inline, then drops the ``$defs`` / ``definitions``
    sections so the output is a flat, self-contained schema.

    Circular references are detected and replaced with a fallback object
    schema to avoid infinite recursion.  Only local references of the
    form ``#/$defs/<name>`` or ``#/definitions/<name>`` are expanded;
    external ``$ref`` URLs are left unchanged.

    Args:
        schema (`dict`):
            The JSON schema that may contain ``$defs`` and ``$ref``
            references.

    Returns:
        `dict`:
            A flattened JSON schema with all references resolved inline.
    """
    schema = copy.deepcopy(schema)
    defs: dict[str, Any] = {}
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema.pop("$defs"))
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema.pop("definitions"))

    if not defs:
        return schema

    def _resolve_ref(obj: Any, visited: frozenset = frozenset()) -> Any:
        if isinstance(obj, list):
            return [_resolve_ref(item, visited) for item in obj]
        if not isinstance(obj, dict):
            return obj
        if "$ref" in obj:
            ref_path = obj["$ref"]
            if isinstance(ref_path, str) and (
                ref_path.startswith("#/$defs/")
                or ref_path.startswith("#/definitions/")
            ):
                def_name = ref_path.split("/")[-1]
                if def_name in visited:
                    logger.warning(
                        "Circular reference detected for '%s' in tool schema",
                        def_name,
                    )
                    return {
                        "type": "object",
                        "description": f"(circular: {def_name})",
                    }
                if def_name in defs:
                    resolved = _resolve_ref(
                        defs[def_name],
                        visited | {def_name},
                    )
                    for key, value in obj.items():
                        if key != "$ref":
                            resolved[key] = _resolve_ref(
                                value,
                                visited | {def_name},
                            )
                    return resolved
            return obj
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if key in ("$defs", "definitions"):
                continue
            result[key] = _resolve_ref(value, visited)
        return result

    return _resolve_ref(schema)


def _extract_func_description(docstring: str) -> str:
    """Extract the function description from the docstring.

    Args:
        docstring (`str`):
            The docstring to extract the function description from.

    Returns:
        `str`:
            The extracted function description.
    """
    parsed_docstring = parse(docstring or "")
    descriptions = []
    if parsed_docstring.short_description is not None:
        descriptions.append(parsed_docstring.short_description)

    if parsed_docstring.long_description is not None:
        descriptions.append(parsed_docstring.long_description)

    return "\n".join(descriptions)


def _extract_input_schema(
    tool_func: Callable,
    include_var_positional: bool = False,
    include_var_keyword: bool = False,
) -> dict:
    """Extract input schema from the tool function's docstring

    Args:
        tool_func (`ToolFunction`):
            The tool function to extract the JSON schema from.
        include_var_positional (`bool`):
            Whether to include variable positional arguments in the JSON
            schema.
        include_var_keyword (`bool`):
            Whether to include variable keyword arguments in the JSON schema.

    Returns:
        `dict`:
            The extracted input JSON schema.
    """
    docstring = parse(tool_func.__doc__ or "")
    params_docstring = {_.arg_name: _.description for _ in docstring.params}

    # Create a dynamic model with the function signature
    fields = {}
    for name, param in inspect.signature(tool_func).parameters.items():
        # Skip the `self` and `cls` parameters
        if name in ["self", "cls"]:
            continue

        # Handle `**kwargs`
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            if not include_var_keyword:
                continue

            fields[name] = (
                Dict[str, Any]
                if param.annotation == inspect.Parameter.empty
                else Dict[str, param.annotation],  # type: ignore
                Field(
                    description=params_docstring.get(
                        f"**{name}",
                        params_docstring.get(name, None),
                    ),
                    default={}
                    if param.default is param.empty
                    else param.default,
                ),
            )

        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not include_var_positional:
                continue

            fields[name] = (
                list[Any]
                if param.annotation == inspect.Parameter.empty
                else list[param.annotation],  # type: ignore
                Field(
                    description=params_docstring.get(
                        f"*{name}",
                        params_docstring.get(name, None),
                    ),
                    default=[]
                    if param.default is param.empty
                    else param.default,
                ),
            )

        else:
            fields[name] = (
                Any
                if param.annotation == inspect.Parameter.empty
                else param.annotation,
                Field(
                    description=params_docstring.get(name, None),
                    default=...
                    if param.default is param.empty
                    else param.default,
                ),
            )

    base_model = create_model(
        "_StructuredOutputDynamicClass",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )
    params_json_schema = base_model.model_json_schema()

    # Remove the title from the json schema
    _remove_title_field(params_json_schema)

    return params_json_schema
