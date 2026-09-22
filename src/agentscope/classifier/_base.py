# -*- coding: utf-8 -*-
"""The base class for classifier models."""
from abc import ABC, abstractmethod
from typing import Any, Mapping, Self

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ._question import ClassifierQuestion
from ._response import ClassifierResponse
from ..credential import CredentialBase


_CLASSIFIER_STATE_ADAPTER = TypeAdapter(
    str,
    config=ConfigDict(
        strict=True,
    ),
)


class ClassifierModelBase(ABC):
    """Base class for models that return typed probabilistic decisions."""

    class Parameters(BaseModel):
        """Provider-specific classifier parameters."""

    def __init__(
        self,
        credential: CredentialBase,
        model: str,
        parameters: BaseModel | None = None,
    ) -> None:
        """Initialize a classifier model.

        Args:
            credential (`CredentialBase`):
                The credential used to authenticate with the provider.
            model (`str`):
                The classifier model name or alias.
            parameters (`BaseModel | None`, defaults to `None`):
                Provider-specific model parameters.
        """
        self.credential = credential
        self.model = model
        self.parameters = parameters or self.Parameters()

    async def __call__(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Evaluate named questions against shared input state.

        Args:
            state (`str`):
                Text shared by all questions.
            questions (`Mapping[str, ClassifierQuestion]`):
                A non-empty mapping of names to typed questions.
            **kwargs (`Any`):
                Provider-specific per-call options passed through unchanged.
                Values supplied here override same-named fields in
                ``parameters``.

        Returns:
            `ClassifierResponse`:
                Typed probabilistic answers to the requested questions.

        Raises:
            `ValidationError`:
                If ``state`` is not a string.
            `ValueError`:
                If no questions are provided.
        """
        if not questions:
            raise ValueError("At least one classifier question is required.")
        validated_state = _CLASSIFIER_STATE_ADAPTER.validate_python(state)
        return await self._call_api(validated_state, questions, **kwargs)

    async def __aenter__(self) -> Self:
        """Enter the classifier model's asynchronous lifecycle."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close resources owned by the classifier model."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close provider resources, if any."""

    @abstractmethod
    async def _call_api(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Call the provider API and normalize its response."""
