# -*- coding: utf-8 -*-
"""The TTS model card class."""
import copy
from datetime import datetime
from typing import Literal, Self, Type

import yaml
from pydantic import BaseModel, Field


class TTSModelCard(BaseModel):
    """The TTS model card class."""

    type: Literal["tts_model"] = "tts_model"
    """The model card type."""

    name: str = Field(description="The name of the TTS model")
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
        description="The supported input types.",
        title="Input types",
        default=["text/plain"],
    )
    """The supported input types."""

    output_types: list[str] = Field(
        description="The supported output types.",
        title="Output types",
        default=["audio/wav"],
    )
    """The supported output types."""

    voices: list[str] = Field(
        description="The supported voices.",
        title="Voices",
    )
    """The supported voice names."""

    parameter_schema: dict
    """The parameters schema for frontend rendering."""

    parameters_overrides: dict[str, dict]
    """The parameter overrides merged into the parameter schema."""

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        parameter_class: Type[BaseModel],
    ) -> Self:
        """Read a TTS model card from a YAML file and merge the parameter
        schema with overrides.

        Args:
            yaml_path (`str`):
                Path to the YAML file.
            parameter_class (`Type[BaseModel]`):
                The parameter class (e.g., ``DashScopeTTSModel.Parameters``).

        Returns:
            `TTSModelCard`:
                A TTSModelCard instance with merged parameter schema.
        """
        with open(yaml_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        base_schema = parameter_class.model_json_schema()
        properties = copy.deepcopy(base_schema.get("properties", {}))

        voices = config.get("voices", [])
        if voices and "voice" in properties:
            properties["voice"] = {
                **properties["voice"],
                "default": voices[0],
                "enum": voices,
            }

        overrides = config.get("parameter_overrides", {})
        for param_name, override in overrides.items():
            if override is None:
                properties.pop(param_name, None)
                continue

            if isinstance(override, dict):
                if override.get("hidden"):
                    properties.pop(param_name, None)
                    continue

                if param_name in properties:
                    properties[param_name] = {
                        **properties[param_name],
                        **override,
                    }

        final_schema = {
            "type": "object",
            "properties": properties,
            "required": base_schema.get("required", []),
        }

        return cls(
            name=config["name"],
            label=config["label"],
            status=config.get("status", "active"),
            deprecated_at=config.get("deprecated_at"),
            input_types=config.get("input_types", ["text/plain"]),
            output_types=config.get("output_types", ["audio/wav"]),
            voices=config.get("voices", []),
            parameter_schema=final_schema,
            parameters_overrides=config.get("parameter_overrides", {}),
        )
