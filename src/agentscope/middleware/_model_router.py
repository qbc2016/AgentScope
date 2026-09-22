# -*- coding: utf-8 -*-
"""Model-based chat model routing middleware."""
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Sequence, TYPE_CHECKING

from ._base import MiddlewareBase
from .._logging import logger
from ..classifier import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
)
from ..event import ReplyStartEvent
from ..message import Msg, SystemMsg, UserMsg
from ..model import ChatModelBase

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
    rounds in that reply. Routing-model failures fail open to the current chat
    model. The selected model remains subject to the Agent's fallback model.

    The middleware does not own the routing or candidate model lifecycle.
    Callers remain responsible for closing the resources they create.
    """

    def __init__(
        self,
        classifier_model: ClassifierModelBase | ChatModelBase,
        candidates: Sequence[ChatModelCandidate],
        instructions: str = _DEFAULT_INSTRUCTIONS,
    ) -> None:
        """Initialize the model router.

        Args:
            classifier_model (`ClassifierModelBase | ChatModelBase`):
                The classifier or chat model used to select a candidate.
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
        if not isinstance(
            classifier_model,
            (ClassifierModelBase, ChatModelBase),
        ):
            raise TypeError(
                "classifier_model must be a ClassifierModelBase or "
                "ChatModelBase instance.",
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
        self._chat_model_prompt = (
            f"{instructions}\n\n"
            f"Select exactly one candidate using these criteria:\n"
            f"{json.dumps(criteria, ensure_ascii=False, indent=2)}"
        )
        self._chat_model_schema = {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": list(candidate_models),
                },
            },
            "required": ["choice"],
            "additionalProperties": False,
        }

    async def on_reply(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        """Use one selected model for the complete reply lifecycle."""
        middleware_key = await self.get_middleware_key()
        original_model = agent.model
        original_reply_id = agent.state.reply_id
        input_messages = self._get_input_messages(input_kwargs.get("inputs"))

        selected_name: str | None = None
        cache_pending = input_messages is not None
        if input_messages is not None:
            selected_name = await self._select_name(agent, input_messages)
            selected_model = (
                self._candidate_models.get(selected_name)
                if selected_name is not None
                else None
            )
        else:
            selected_model = self._get_cached_model(agent, middleware_key)

        if selected_model is not None:
            agent.model = selected_model

        try:
            async for event in next_handler(**input_kwargs):
                if cache_pending and isinstance(event, ReplyStartEvent):
                    self._cache_decision(
                        agent,
                        middleware_key,
                        event.reply_id,
                        selected_name,
                    )
                    cache_pending = False
                yield event
        finally:
            if cache_pending and agent.state.reply_id != original_reply_id:
                self._cache_decision(
                    agent,
                    middleware_key,
                    agent.state.reply_id,
                    selected_name,
                )
            agent.model = original_model

    async def _select_name(
        self,
        agent: "Agent",
        messages: list[Msg],
    ) -> str | None:
        """Classify new reply messages and return a valid candidate name."""
        state = self._get_latest_user_state(messages)
        if state is None:
            return None

        try:
            selected_name = await self._classify(state)
        except Exception as error:
            logger.warning(
                "Chat model routing request failed for agent %s; "
                "using the current model: %s",
                agent.name,
                error,
            )
            return None

        if selected_name not in self._candidate_models:
            logger.warning(
                "Chat model routing selected unknown candidate %r "
                "for agent %s; using the current model.",
                selected_name,
                agent.name,
            )
            return None

        logger.debug(
            "Routed agent %s to chat model candidate %s (%s)",
            agent.name,
            selected_name,
            self._candidate_models[selected_name].model,
        )
        return selected_name

    def _get_cached_model(
        self,
        agent: "Agent",
        middleware_key: str,
    ) -> ChatModelBase | None:
        """Return the selected model cached for the current reply."""
        cached = agent.state.middle_context.get(middleware_key)
        if not isinstance(cached, dict) or cached.get("reply_id") != (
            agent.state.reply_id
        ):
            return None
        selected_name = cached.get("selected_model")
        if not isinstance(selected_name, str):
            return None
        return self._candidate_models.get(selected_name)

    async def _classify(self, state: str) -> str | None:
        """Return the candidate selected by the configured routing model."""
        if isinstance(self.classifier_model, ClassifierModelBase):
            response = await self.classifier_model(
                state=state,
                questions={
                    _ROUTE_QUESTION_NAME: self._routing_question,
                },
            )
            answer = response.answers.get(_ROUTE_QUESTION_NAME)
            return answer.choice if isinstance(answer, ChoiceAnswer) else None

        response = await self.classifier_model.generate_structured_output(
            messages=[
                SystemMsg(
                    name="system",
                    content=self._chat_model_prompt,
                ),
                UserMsg(name="user", content=state),
            ],
            structured_model=self._chat_model_schema,
        )
        choice = response.content.get("choice")
        return choice if isinstance(choice, str) else None

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
    def _get_input_messages(inputs: object) -> list[Msg] | None:
        """Return new reply messages, or ``None`` for a continuation."""
        if isinstance(inputs, Msg):
            return [inputs]
        if isinstance(inputs, list) and all(
            isinstance(message, Msg) for message in inputs
        ):
            return inputs
        return None

    @staticmethod
    def _cache_decision(
        agent: "Agent",
        middleware_key: str,
        reply_id: str,
        selected_model: str | None,
    ) -> None:
        """Store one JSON-compatible routing decision in Agent state."""
        agent.state.middle_context[middleware_key] = {
            "reply_id": reply_id,
            "selected_model": selected_model,
        }
