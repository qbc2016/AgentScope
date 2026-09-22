# -*- coding: utf-8 -*-
"""Classifier-based chat model routing middleware."""
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Sequence, TYPE_CHECKING

from ._base import MiddlewareBase
from .._logging import logger
from ..classifier import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
)
from ..message import Msg
from ..model import ChatModelBase, ChatResponse

if TYPE_CHECKING:
    from ..agent import Agent


_DEFAULT_INSTRUCTIONS = (
    "Select the most suitable chat model for responding to the user input."
)
_ROUTE_QUESTION_NAME = "chat_model"


@dataclass
class ChatModelCandidate:
    """A named chat model and the criteria for selecting it."""

    name: str
    """The stable name returned by the classifier."""

    model: ChatModelBase
    """The chat model used when this candidate is selected."""

    description: str
    """When this candidate should be selected."""


class ModelRouterMiddleware(MiddlewareBase):
    """Select a chat model by classifying the latest user input.

    A routing decision is made once per reply and reused by later reasoning
    rounds in that reply. Classifier failures fail open to the current chat
    model. Existing Agent fallback-model calls are passed through unchanged.

    The middleware does not own the classifier or candidate model lifecycle.
    Callers remain responsible for closing resources they create.
    """

    def __init__(
        self,
        classifier_model: ClassifierModelBase,
        candidates: Sequence[ChatModelCandidate],
        instructions: str = _DEFAULT_INSTRUCTIONS,
    ) -> None:
        """Initialize the model router.

        Args:
            classifier_model (`ClassifierModelBase`):
                The classifier used to select a candidate.
            candidates (`Sequence[ChatModelCandidate]`):
                At least two uniquely named chat model candidates.
            instructions (`str`):
                Instructions used by the routing choice question.

        Raises:
            `TypeError`:
                If a model has the wrong base type.
            `ValueError`:
                If fewer than two candidates are provided, or candidate
                names are empty, padded, or duplicated.
        """
        if not isinstance(classifier_model, ClassifierModelBase):
            raise TypeError(
                "classifier_model must be a ClassifierModelBase instance.",
            )
        if len(candidates) < 2:
            raise ValueError(
                "At least two chat model candidates are required.",
            )

        candidate_models: dict[str, ChatModelBase] = {}
        criteria: dict[str, str] = {}
        for candidate in candidates:
            if not isinstance(candidate.model, ChatModelBase):
                raise TypeError(
                    f"Candidate {candidate.name!r} must contain a "
                    f"ChatModelBase instance.",
                )
            if not candidate.name or candidate.name != candidate.name.strip():
                raise ValueError(
                    "Candidate names must be non-empty and must not have "
                    "surrounding whitespace.",
                )
            if candidate.name in candidate_models:
                raise ValueError(
                    f"Duplicate chat model candidate: {candidate.name!r}.",
                )
            candidate_models[candidate.name] = candidate.model
            criteria[candidate.name] = candidate.description

        self.classifier_model = classifier_model
        self._candidate_models = candidate_models
        self._routing_question = ChoiceQuestion(
            instructions=instructions,
            criteria=criteria,
        )

    async def on_model_call(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[
            ...,
            Awaitable[ChatResponse | AsyncGenerator[ChatResponse, None]],
        ],
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Route a primary chat model call to the selected candidate."""
        current_model = input_kwargs["current_model"]

        # Preserve the Agent's built-in fallback model behavior.
        if current_model is not agent.model:
            return await next_handler(**input_kwargs)

        selected_model = await self._select_model(
            agent,
            input_kwargs["messages"],
        )
        if selected_model is None:
            return await next_handler(**input_kwargs)

        return await next_handler(
            **{
                **input_kwargs,
                "current_model": selected_model,
            },
        )

    async def _select_model(
        self,
        agent: "Agent",
        messages: list[Msg],
    ) -> ChatModelBase | None:
        """Return the cached or newly selected model for this reply."""
        middleware_key = await self.get_middleware_key()
        cached = agent.state.middle_context.get(middleware_key)
        if isinstance(cached, dict) and cached.get("reply_id") == (
            agent.state.reply_id
        ):
            selected_name = cached.get("selected_model")
            if isinstance(selected_name, str):
                return self._candidate_models.get(selected_name)
            return None

        state = self._get_latest_user_state(messages)
        if state is None:
            self._cache_decision(agent, middleware_key, None)
            return None

        try:
            response = await self.classifier_model(
                state=state,
                questions={
                    _ROUTE_QUESTION_NAME: self._routing_question,
                },
            )
        # pylint: disable-next=broad-exception-caught
        except Exception as error:
            logger.warning(
                "Chat model classifier request failed for agent %s; "
                "using the current model: %s",
                agent.name,
                error,
            )
            self._cache_decision(agent, middleware_key, None)
            return None

        answer = response.answers.get(_ROUTE_QUESTION_NAME)
        if not isinstance(answer, ChoiceAnswer):
            logger.warning(
                "Chat model classifier returned no ChoiceAnswer named "
                "%r for agent %s; using the current model.",
                _ROUTE_QUESTION_NAME,
                agent.name,
            )
            self._cache_decision(agent, middleware_key, None)
            return None

        if answer.choice not in self._candidate_models:
            logger.warning(
                "Chat model classifier selected unknown candidate %r "
                "for agent %s; using the current model.",
                answer.choice,
                agent.name,
            )
            self._cache_decision(agent, middleware_key, None)
            return None

        self._cache_decision(agent, middleware_key, answer.choice)
        logger.debug(
            "Routed agent %s to chat model candidate %s (%s)",
            agent.name,
            answer.choice,
            self._candidate_models[answer.choice].model,
        )
        return self._candidate_models[answer.choice]

    @staticmethod
    def _get_latest_user_state(
        messages: list[Msg],
    ) -> str | None:
        """Extract routing state from the latest user message.

        Assistant messages, including tool results, are ignored. Attachment
        blocks are ignored. If there is no user text, the router returns
        ``None`` and fails open to the current model.
        """
        for message in reversed(messages):
            if message.role != "user":
                continue
            return message.get_text_content()

        return None

    @staticmethod
    def _cache_decision(
        agent: "Agent",
        middleware_key: str,
        selected_model: str | None,
    ) -> None:
        """Store one JSON-compatible routing decision in Agent state."""
        agent.state.middle_context[middleware_key] = {
            "reply_id": agent.state.reply_id,
            "selected_model": selected_model,
        }
