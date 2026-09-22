# -*- coding: utf-8 -*-
"""Model-based chat model routing middleware."""
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Literal, Sequence, TYPE_CHECKING

from ._base import MiddlewareBase
from .._logging import logger
from ..classifier import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
    ClassifierUsage,
)
from ..event import CustomEvent, ReplyStartEvent
from ..message import Msg, SystemMsg, UserMsg
from ..model import ChatModelBase, ChatUsage

if TYPE_CHECKING:
    from ..agent import Agent


_DEFAULT_INSTRUCTIONS = (
    "Select the most suitable chat model for responding to the user input."
)
_ROUTE_QUESTION_NAME = "chat_model"

_RoutingUsage = dict[str, int | float | None]


@dataclass
class ChatModelCandidate:
    """A named chat model and the criteria for selecting it."""

    name: str
    """The stable name returned by the classifier."""

    model: ChatModelBase
    """The chat model used when this candidate is selected."""

    description: str
    """When this candidate should be selected."""


@dataclass
class _RoutingResult:
    """Normalized result returned by a routing model."""

    model_name: str
    selected_model: str | None
    usage: _RoutingUsage | None


@dataclass
class _RoutingDecision:
    """Validated routing outcome used by the middleware."""

    result: _RoutingResult
    error: str | None = None


class ModelRouterMiddleware(MiddlewareBase):
    """Select a chat model by classifying the latest user input.

    A routing decision is made once per reply and reused by later reasoning
    rounds in that reply. Routing-model failures fail open to the current chat
    model. The selected model remains subject to the Agent's fallback model.

    New routing calls happen after ``ReplyStartEvent`` and before the Agent
    performs token counting or context compression. Each call emits custom
    routing start and end events, including normalized usage when available.

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
        cache_pending = input_messages is not None
        routing_state = (
            self._get_latest_user_state(input_messages)
            if input_messages is not None
            else None
        )
        selected_name: str | None = None
        selected_model = (
            self._get_cached_model(agent, middleware_key)
            if input_messages is None
            else None
        )

        if selected_model is not None:
            agent.model = selected_model

        try:
            async for event in next_handler(**input_kwargs):
                if cache_pending and isinstance(event, ReplyStartEvent):
                    yield event
                    if routing_state is not None:
                        model_name, model_type = self._routing_model_info()
                        yield CustomEvent(
                            name="routing_call_start",
                            value={
                                "reply_id": event.reply_id,
                                "model_name": model_name,
                                "model_type": model_type,
                            },
                        )
                        decision = await self._select_name(
                            agent,
                            routing_state,
                        )
                        selected_name = decision.result.selected_model
                        selected_model = (
                            self._candidate_models.get(selected_name)
                            if selected_name is not None
                            else None
                        )
                        if selected_model is not None:
                            agent.model = selected_model

                    self._cache_decision(
                        agent,
                        middleware_key,
                        event.reply_id,
                        selected_name,
                    )
                    cache_pending = False
                    if routing_state is not None:
                        yield CustomEvent(
                            name="routing_call_end",
                            value={
                                "reply_id": event.reply_id,
                                "model_name": decision.result.model_name,
                                "model_type": model_type,
                                "selected_model": selected_name,
                                "usage": decision.result.usage,
                                "success": decision.error is None,
                                "error": decision.error,
                            },
                        )
                    continue
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
        state: str,
    ) -> _RoutingDecision:
        """Classify routing state and validate the selected candidate."""
        try:
            result = await self._classify(state)
        except Exception as error:
            log_message = (
                f"Chat model routing request failed for agent "
                f"{agent.name}; using the current model: {error}"
            )
            logger.warning(log_message)
            return _RoutingDecision(
                result=_RoutingResult(
                    model_name=self.classifier_model.model,
                    selected_model=None,
                    usage=None,
                ),
                error=f"{type(error).__name__}: {error}",
            )

        if result.selected_model not in self._candidate_models:
            error_message = (
                f"Routing model selected unknown candidate "
                f"{result.selected_model!r}."
            )
            log_message = (
                f"{error_message} Agent {agent.name} will use the current "
                f"model."
            )
            logger.warning(log_message)
            result.selected_model = None
            return _RoutingDecision(result=result, error=error_message)

        log_message = (
            f"Routed agent {agent.name} to chat model candidate "
            f"{result.selected_model} "
            f"({self._candidate_models[result.selected_model].model})"
        )
        logger.debug(log_message)
        return _RoutingDecision(result=result)

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

    async def _classify(self, state: str) -> _RoutingResult:
        """Return the normalized response from the routing model."""
        if isinstance(self.classifier_model, ClassifierModelBase):
            response = await self.classifier_model(
                state=state,
                questions={
                    _ROUTE_QUESTION_NAME: self._routing_question,
                },
            )
            answer = response.content.get(_ROUTE_QUESTION_NAME)
            return _RoutingResult(
                model_name=response.model,
                selected_model=(
                    answer.choice if isinstance(answer, ChoiceAnswer) else None
                ),
                usage=self._normalize_usage(response.usage),
            )

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
        return _RoutingResult(
            model_name=self.classifier_model.model,
            selected_model=choice if isinstance(choice, str) else None,
            usage=self._normalize_usage(response.usage),
        )

    def _routing_model_info(
        self,
    ) -> tuple[str, Literal["classifier", "chat"]]:
        """Return the configured routing model name and interface type."""
        model_type: Literal["classifier", "chat"] = (
            "classifier"
            if isinstance(self.classifier_model, ClassifierModelBase)
            else "chat"
        )
        return self.classifier_model.model, model_type

    @staticmethod
    def _normalize_usage(
        usage: ClassifierUsage | ChatUsage | None,
    ) -> _RoutingUsage | None:
        """Normalize provider-independent classifier and chat usage."""
        if usage is None:
            return None
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "time": usage.time,
            "cache_input_tokens": (
                usage.cache_input_tokens if isinstance(usage, ChatUsage) else 0
            ),
            "cache_creation_input_tokens": (
                usage.cache_creation_input_tokens
                if isinstance(usage, ChatUsage)
                else 0
            ),
        }

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
