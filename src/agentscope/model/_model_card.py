# -*- coding: utf-8 -*-
"""The model card class."""
import copy
from datetime import datetime
from typing import Any, Literal, Self, Type

import yaml
from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """The model card class."""

    type: Literal["chat_model"] = "chat_model"
    """The model card type."""

    name: str = Field(description="The name of the model")
    """The model name."""

    label: str = Field(description="The model label.")
    """The model label used for frontend rendering."""

    status: Literal["active", "deprecated", "sunset"] = Field(
        title="Status",
        description="The model status",
    )
    """The model status."""

    deprecated_at: datetime | None = Field(
        default=None,
        description="The model deprecation date and time.",
        title="Deprecation date",
    )
    """The model deprecated at."""

    input_types: list[str] = Field(
        description="The supported model input types.",
        title="Input types",
        default=["text/plain"],
    )
    """The model supported input types."""

    output_types: list[str] = Field(
        description="The supported model output types.",
        title="Output types",
        default=["text/plain"],
    )
    """The model supported output types."""

    context_size: int = Field(
        title="Context size",
        description="The context size.",
        gt=0,
    )
    """The model context size."""

    output_size: int = Field(
        title="Max output tokens",
        description="The maximum number of tokens.",
        gt=0,
    )
    """The model max output tokens."""

    parameter_schema: dict
    """The parameters schema, which will be combined with the schema from the
    DashScopeChatParameter class."""

    parameters_overrides: dict[str, dict]
    """The parameter overrides, which will be merged into the parameter schema.
    """

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        parameter_class: Type[BaseModel],
    ) -> Self:
        """Read a model card from a YAML file, and merge the parameter schema
        with the override parameter schema in the yaml file.

        Args:
            yaml_path (`str`):
                Path to the YAML file
            parameter_class (`Type[BaseModel]`):
                The parameter class (e.g., DashScopeChatParameters)

        Returns:
            `list[ModelCard]`:
                ModelCard instance with merged parameter schema
        """

        # Load YAML config
        with open(yaml_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Get base schema from parameter class
        base_schema = parameter_class.model_json_schema()
        defs = base_schema.get("$defs", {})

        def _inline_refs(node: Any) -> Any:
            """Recursively resolve ``$ref`` against ``defs`` and collapse
            ``Optional[X]`` wrappers so the frontend (and yaml overrides)
            see a flat shape.

            Pydantic emits ``T | None`` as
            ``{"anyOf": [<schema>, {"type": "null"}], "default": null}``;
            we flatten that to ``<schema>`` merged with the sibling keys
            (``default``, ``title``, …) so ``audio.properties.voice`` is
            reachable directly instead of via ``audio.anyOf[0].properties``.
            """
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    target = defs.get(ref.split("/", 2)[-1])
                    if isinstance(target, dict):
                        return _inline_refs(copy.deepcopy(target))
                resolved = {k: _inline_refs(v) for k, v in node.items()}
                any_of = resolved.get("anyOf")
                if isinstance(any_of, list) and len(any_of) == 2:
                    non_null = [
                        v
                        for v in any_of
                        if not (
                            isinstance(v, dict) and v.get("type") == "null"
                        )
                    ]
                    if len(non_null) == 1 and isinstance(non_null[0], dict):
                        flat = {
                            k: v for k, v in resolved.items() if k != "anyOf"
                        }
                        # ``anyOf`` variant wins on key collisions so nested
                        # ``properties``/``required`` are preserved.
                        flat.update(non_null[0])
                        return flat
                return resolved
            if isinstance(node, list):
                return [_inline_refs(item) for item in node]
            return node

        properties = _inline_refs(
            copy.deepcopy(base_schema.get("properties", {})),
        )

        # Auto-filter: remove thinking parameters if not supported
        output_types = config.get("output_types", [])
        if "application/x-thinking" not in output_types:
            properties.pop("thinking_enable", None)
            properties.pop("thinking_budget", None)

        # Auto-filter: only omni-style models that declare an ``audio/*``
        # output type should expose the ``audio`` (voice/format) parameter
        # to the frontend popover.
        if not any(
            isinstance(t, str) and t.startswith("audio/") for t in output_types
        ):
            properties.pop("audio", None)

        # Auto-inject: set max_tokens maximum from output_size
        if "max_tokens" in properties and "output_size" in config:
            properties["max_tokens"]["maximum"] = config["output_size"]

        def _deep_merge(base: Any, override: Any) -> Any:
            """Recursive dict merge so a yaml override can target a nested
            JSON-Schema field (e.g. ``audio.properties.voice.enum``) without
            blowing away sibling keys. Lists and scalars are replaced
            wholesale.
            """
            if not isinstance(base, dict) or not isinstance(override, dict):
                return override
            out = dict(base)
            for k, v in override.items():
                out[k] = _deep_merge(out[k], v) if k in out else v
            return out

        # Apply parameter_overrides with deep merge so nested fields survive
        overrides = config.get("parameter_overrides", {})
        for param_name, override in overrides.items():
            if override is None:
                # null means remove
                properties.pop(param_name, None)
                continue

            if isinstance(override, dict):
                # Check for hidden flag
                if override.get("hidden"):
                    properties.pop(param_name, None)
                    continue

                if param_name in properties:
                    properties[param_name] = _deep_merge(
                        properties[param_name],
                        override,
                    )

        # Build final parameter schema
        final_schema = {
            "type": "object",
            "properties": properties,
            "required": base_schema.get("required", []),
        }

        # Create ModelCard instance
        return cls(
            name=config["name"],
            label=config["label"],
            status=config.get("status", "active"),
            deprecated_at=config.get("deprecated_at"),
            input_types=config.get("input_types", ["text/plain"]),
            output_types=config.get("output_types", ["text/plain"]),
            context_size=config["context_size"],
            output_size=config["output_size"],
            parameter_schema=final_schema,
            parameters_overrides=config.get("parameter_overrides", {}),
        )
