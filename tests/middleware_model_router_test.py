# -*- coding: utf-8 -*-
"""Tests for model-based chat model routing middleware."""
from typing import Any, Mapping
from unittest import IsolatedAsyncioTestCase, TestCase

from pydantic import BaseModel

from utils import MockModel

from agentscope.agent import Agent, InjectionConfig, ModelConfig
from agentscope.classifier import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
    ClassifierQuestion,
    ClassifierResponse,
    ClassifierUsage,
)
from agentscope.credential import CredentialBase
from agentscope.event import (
    ModelCallStartEvent,
    ReplyStartEvent,
    RoutingCallEndEvent,
    RoutingCallStartEvent,
)
from agentscope.message import (
    Base64Source,
    DataBlock,
    Msg,
    SystemMsg,
    TextBlock,
    UserMsg,
)
from agentscope.middleware import ChatModelCandidate, ModelRouterMiddleware
from agentscope.model import ChatResponse, ChatUsage, StructuredResponse


class _MockClassifier(ClassifierModelBase):
    """A deterministic classifier for model router tests."""

    def __init__(self, outcomes: list[str | BaseException]) -> None:
        """Initialize the classifier with choices or exceptions."""
        super().__init__(CredentialBase(), "mock-classifier")
        self.outcomes = outcomes
        self.calls: list[
            tuple[
                str,
                Mapping[str, ClassifierQuestion],
            ]
        ] = []

    async def _call_api(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Return the next configured routing choice."""
        del kwargs
        self.calls.append((state, questions))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome

        question = next(iter(questions.values()))
        if not isinstance(question, ChoiceQuestion):
            raise AssertionError("Expected a ChoiceQuestion.")
        probabilities = {
            name: float(name == outcome) for name in question.criteria
        }
        return ClassifierResponse(
            model=self.model,
            answers={
                next(iter(questions)): ChoiceAnswer(
                    choice=outcome,
                    confidence=0.9,
                    probabilities=probabilities,
                ),
            },
            usage=ClassifierUsage(
                time=0.2,
                input_tokens=10,
                output_tokens=1,
            ),
        )


class _MockRoutingChatModel(MockModel):
    """A deterministic chat model for model router tests."""

    def __init__(self, outcomes: list[str | BaseException]) -> None:
        """Initialize the chat model with choices or exceptions."""
        super().__init__(model="routing-chat-model")
        self.outcomes = outcomes
        self.calls: list[
            tuple[
                list[Msg],
                dict,
            ]
        ] = []

    async def generate_structured_output(
        self,
        messages: list[Msg],
        structured_model: type[BaseModel] | dict,
        **kwargs: Any,
    ) -> StructuredResponse:
        """Return the next configured routing choice."""
        del kwargs
        if not isinstance(structured_model, dict):
            raise AssertionError("Expected a JSON schema dictionary.")
        self.calls.append((messages, structured_model))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return StructuredResponse(
            content={"choice": outcome},
            usage=ChatUsage(
                input_tokens=20,
                output_tokens=2,
                time=0.3,
                cache_input_tokens=4,
                cache_creation_input_tokens=3,
            ),
        )


class _CountingMockModel(MockModel):
    """A mock chat model that records token-count requests."""

    def __init__(
        self,
        model: str,
        context_size: int,
    ) -> None:
        """Initialize the model and its token-count counter."""
        super().__init__(model=model, context_size=context_size)
        self.count_tokens_calls = 0

    async def count_tokens(
        self,
        messages: list[Msg],
        tools: list[dict] | None,
    ) -> int:
        """Record the request and return an empty-context estimate."""
        del messages, tools
        self.count_tokens_calls += 1
        return 0


class ModelRouterMiddlewareTest(IsolatedAsyncioTestCase):
    """Test routing behavior at the model-call middleware boundary."""

    def setUp(self) -> None:
        """Create primary and candidate chat models."""
        self.primary = MockModel(model="primary")
        self.fast = MockModel(model="fast-model")
        self.reasoning = MockModel(model="reasoning-model")
        self.fallback = MockModel(model="fallback")
        self.agent = Agent(
            name="router-agent",
            system_prompt="Help the user.",
            model=self.primary,
        )

    def _make_middleware(
        self,
        outcomes: list[str | BaseException],
    ) -> tuple[ModelRouterMiddleware, _MockClassifier]:
        """Create a router and expose its deterministic classifier."""
        classifier = _MockClassifier(outcomes)
        middleware = ModelRouterMiddleware(
            classifier_model=classifier,
            candidates=[
                ChatModelCandidate(
                    name="fast",
                    model=self.fast,
                    description="Short and simple requests.",
                ),
                ChatModelCandidate(
                    name="reasoning",
                    model=self.reasoning,
                    description="Complex reasoning is required.",
                ),
            ],
        )
        return middleware, classifier

    def _make_chat_middleware(
        self,
        outcomes: list[str | BaseException],
    ) -> tuple[ModelRouterMiddleware, _MockRoutingChatModel]:
        """Create a router backed by a deterministic chat model."""
        routing_model = _MockRoutingChatModel(outcomes)
        middleware = ModelRouterMiddleware(
            classifier_model=routing_model,
            candidates=[
                ChatModelCandidate(
                    name="fast",
                    model=self.fast,
                    description="Short and simple requests.",
                ),
                ChatModelCandidate(
                    name="reasoning",
                    model=self.reasoning,
                    description="Complex reasoning is required.",
                ),
            ],
        )
        return middleware, routing_model

    async def _invoke_reply(
        self,
        middleware: ModelRouterMiddleware,
        inputs: Msg | list[Msg] | None,
        reply_id: str,
    ) -> dict:
        """Invoke the reply hook and capture its active model."""
        observed: dict = {}

        async def next_handler(**kwargs: Any) -> Any:
            del kwargs
            if inputs is not None:
                self.agent.state.reply_id = reply_id
                yield ReplyStartEvent(
                    session_id=self.agent.state.session_id,
                    reply_id=reply_id,
                    name=self.agent.name,
                )
            observed["active_model"] = self.agent.model

        events = []
        async for event in middleware.on_reply(
            agent=self.agent,
            input_kwargs={
                "inputs": inputs,
                "structured_schema": None,
            },
            next_handler=next_handler,
        ):
            events.append(event)
        observed["events"] = events
        observed["restored_model"] = self.agent.model
        return observed

    async def test_routes_once_per_reply_and_reroutes_next_reply(
        self,
    ) -> None:
        """One reply should reuse its route and a new reply should reroute."""
        middleware, classifier = self._make_middleware(
            ["reasoning", "fast"],
        )
        first = await self._invoke_reply(
            middleware,
            UserMsg(name="user", content="Prove this theorem."),
            "reply-1",
        )
        cached = await self._invoke_reply(
            middleware,
            None,
            "reply-1",
        )

        self.assertIs(first["active_model"], self.reasoning)
        self.assertIs(cached["active_model"], self.reasoning)
        self.assertIs(first["restored_model"], self.primary)
        self.assertIs(cached["restored_model"], self.primary)
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(classifier.calls[0][0], "Prove this theorem.")
        question = next(iter(classifier.calls[0][1].values()))
        self.assertDictEqual(
            question.model_dump(),
            {
                "type": "choice",
                "criteria": {
                    "fast": "Short and simple requests.",
                    "reasoning": "Complex reasoning is required.",
                },
                "instructions": (
                    "Select the most suitable chat model for responding "
                    "to the user input."
                ),
            },
        )

        rerouted = await self._invoke_reply(
            middleware,
            UserMsg(name="user", content="Say hello."),
            "reply-2",
        )

        self.assertIs(rerouted["active_model"], self.fast)
        self.assertEqual(len(classifier.calls), 2)
        self.assertDictEqual(
            self.agent.state.middle_context["ModelRouterMiddleware"],
            {
                "reply_id": "reply-2",
                "selected_model": "fast",
            },
        )

    async def test_agent_calls_the_selected_model(self) -> None:
        """The Agent middleware chain should invoke the selected model."""
        middleware, classifier = self._make_middleware(["reasoning"])
        self.reasoning.set_responses(
            [
                ChatResponse(
                    content=[TextBlock(text="Routed response")],
                    is_last=True,
                ),
            ],
        )
        agent = Agent(
            name="router-agent",
            system_prompt="Help the user.",
            model=self.primary,
            middlewares=[middleware],
            injection_config=InjectionConfig(inject_runtime_state=False),
        )

        response = await agent.reply(
            UserMsg(name="user", content="Prove this theorem."),
        )

        self.assertEqual(response.get_text_content(), "Routed response")
        self.assertEqual(self.primary.cnt, 0)
        self.assertEqual(self.reasoning.cnt, 1)
        self.assertEqual(len(classifier.calls), 1)

    async def test_selected_model_drives_reply_lifecycle(self) -> None:
        """Token counting and model events should use the selected model."""
        primary = _CountingMockModel(
            model="primary-large-context",
            context_size=1_000_000,
        )
        selected = _CountingMockModel(
            model="selected-small-context",
            context_size=128_000,
        )
        selected.set_responses(
            [
                ChatResponse(
                    content=[TextBlock(text="Routed response")],
                    is_last=True,
                ),
            ],
        )
        classifier = _MockClassifier(["selected"])
        middleware = ModelRouterMiddleware(
            classifier_model=classifier,
            candidates=[
                ChatModelCandidate("primary", primary, "Simple tasks."),
                ChatModelCandidate(
                    "selected",
                    selected,
                    "Complex tasks.",
                ),
            ],
        )
        agent = Agent(
            name="router-agent",
            system_prompt="Help the user.",
            model=primary,
            middlewares=[middleware],
            injection_config=InjectionConfig(inject_runtime_state=False),
        )

        events = [
            event
            async for event in agent.reply_stream(
                UserMsg(name="user", content="Prove this theorem."),
                yield_final_msg=True,
            )
        ]

        model_start_events = [
            event for event in events if isinstance(event, ModelCallStartEvent)
        ]
        reply_start = next(
            event for event in events if isinstance(event, ReplyStartEvent)
        )
        routing_start = next(
            event
            for event in events
            if isinstance(event, RoutingCallStartEvent)
        )
        routing_end = next(
            event for event in events if isinstance(event, RoutingCallEndEvent)
        )
        self.assertEqual(primary.count_tokens_calls, 0)
        self.assertEqual(selected.count_tokens_calls, 1)
        self.assertLess(events.index(reply_start), events.index(routing_start))
        self.assertLess(events.index(routing_start), events.index(routing_end))
        self.assertLess(
            events.index(routing_end),
            events.index(model_start_events[0]),
        )
        self.assertDictEqual(
            routing_start.model_dump(exclude={"id", "created_at"}),
            {
                "metadata": {},
                "type": "ROUTING_CALL_START",
                "reply_id": reply_start.reply_id,
                "model_name": "mock-classifier",
                "model_type": "classifier",
            },
        )
        self.assertDictEqual(
            routing_end.model_dump(exclude={"id", "created_at"}),
            {
                "metadata": {},
                "type": "ROUTING_CALL_END",
                "reply_id": reply_start.reply_id,
                "model_name": "mock-classifier",
                "model_type": "classifier",
                "selected_model": "selected",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "time": 0.2,
                    "cache_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "success": True,
                "error": None,
            },
        )
        self.assertListEqual(
            [event.model_name for event in model_start_events],
            ["selected-small-context"],
        )
        self.assertIs(agent.model, primary)

    async def test_chat_model_routes_with_structured_output(self) -> None:
        """A chat model should route through structured output."""
        middleware, routing_model = self._make_chat_middleware(
            ["reasoning"],
        )
        first = await self._invoke_reply(
            middleware,
            UserMsg(name="user", content="Prove this theorem."),
            "reply-chat-model",
        )
        cached = await self._invoke_reply(
            middleware,
            None,
            "reply-chat-model",
        )

        self.assertIs(first["active_model"], self.reasoning)
        self.assertIs(cached["active_model"], self.reasoning)
        self.assertEqual(len(routing_model.calls), 1)
        routing_end = next(
            event
            for event in first["events"]
            if isinstance(event, RoutingCallEndEvent)
        )
        self.assertDictEqual(
            routing_end.model_dump(exclude={"id", "created_at"}),
            {
                "metadata": {},
                "type": "ROUTING_CALL_END",
                "reply_id": "reply-chat-model",
                "model_name": "routing-chat-model",
                "model_type": "chat",
                "selected_model": "reasoning",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 2,
                    "time": 0.3,
                    "cache_input_tokens": 4,
                    "cache_creation_input_tokens": 3,
                },
                "success": True,
                "error": None,
            },
        )
        messages, schema = routing_model.calls[0]
        self.assertListEqual(
            [
                (
                    message.role,
                    message.name,
                    message.get_text_content(),
                )
                for message in messages
            ],
            [
                (
                    "system",
                    "system",
                    "Select the most suitable chat model for responding "
                    "to the user input.\n\n"
                    "Select exactly one candidate using these criteria:\n"
                    "{\n"
                    '  "fast": "Short and simple requests.",\n'
                    '  "reasoning": "Complex reasoning is required."\n'
                    "}",
                ),
                ("user", "user", "Prove this theorem."),
            ],
        )
        self.assertDictEqual(
            schema,
            {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": ["fast", "reasoning"],
                    },
                },
                "required": ["choice"],
                "additionalProperties": False,
            },
        )

    async def test_chat_model_failures_use_current_model(self) -> None:
        """Chat-model errors and unknown choices should fail open."""
        outcomes: list[str | BaseException] = [
            RuntimeError("unavailable"),
            "unknown",
        ]
        for index, outcome in enumerate(outcomes):
            with self.subTest(outcome=outcome):
                middleware, routing_model = self._make_chat_middleware(
                    [outcome],
                )

                forwarded = await self._invoke_reply(
                    middleware,
                    UserMsg(name="user", content="Hello"),
                    f"reply-chat-{index}",
                )
                cached = await self._invoke_reply(
                    middleware,
                    None,
                    f"reply-chat-{index}",
                )

                self.assertIs(forwarded["active_model"], self.primary)
                self.assertIs(cached["active_model"], self.primary)
                self.assertEqual(len(routing_model.calls), 1)
                routing_end = next(
                    event
                    for event in forwarded["events"]
                    if isinstance(event, RoutingCallEndEvent)
                )
                self.assertFalse(routing_end.success)
                self.assertIsNone(routing_end.selected_model)
                if isinstance(outcome, BaseException):
                    self.assertEqual(
                        routing_end.error,
                        "RuntimeError: unavailable",
                    )
                    self.assertIsNone(routing_end.usage)
                else:
                    self.assertEqual(
                        routing_end.error,
                        "Routing model selected unknown candidate "
                        "'unknown'.",
                    )
                    self.assertIsNotNone(routing_end.usage)

    async def test_routes_only_text_from_message_with_attachment(self) -> None:
        """Routing should ignore attachment content and metadata."""
        middleware, classifier = self._make_middleware(["reasoning"])
        message = UserMsg(
            name="user",
            content=[
                TextBlock(text="Analyze this image."),
                DataBlock(
                    name="private-name.png",
                    source=Base64Source(
                        data="secret-image-data",
                        media_type="image/png",
                    ),
                ),
            ],
        )

        forwarded = await self._invoke_reply(
            middleware,
            message,
            "reply-attachment",
        )

        self.assertIs(forwarded["active_model"], self.reasoning)
        self.assertEqual(classifier.calls[0][0], "Analyze this image.")

    async def test_attachment_only_message_uses_current_model(self) -> None:
        """An attachment-only user message should fail open."""
        middleware, classifier = self._make_middleware(["reasoning"])
        message = UserMsg(
            name="user",
            content=[
                DataBlock(
                    name="private-name.png",
                    source=Base64Source(
                        data="secret-image-data",
                        media_type="image/png",
                    ),
                ),
            ],
        )

        forwarded = await self._invoke_reply(
            middleware,
            message,
            "reply-attachment-only",
        )

        self.assertIs(forwarded["active_model"], self.primary)
        self.assertListEqual(classifier.calls, [])

    async def test_classifier_failures_use_current_model(self) -> None:
        """Classifier errors and unknown choices should fail open."""
        outcomes: list[str | BaseException] = [
            RuntimeError("unavailable"),
            "unknown",
        ]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                middleware, classifier = self._make_middleware([outcome])
                reply_id = f"reply-{type(outcome).__name__}"

                forwarded = await self._invoke_reply(
                    middleware,
                    UserMsg(name="user", content="Hello"),
                    reply_id,
                )
                cached = await self._invoke_reply(
                    middleware,
                    None,
                    reply_id,
                )

                self.assertIs(forwarded["active_model"], self.primary)
                self.assertIs(cached["active_model"], self.primary)
                self.assertEqual(len(classifier.calls), 1)
                routing_end = next(
                    event
                    for event in forwarded["events"]
                    if isinstance(event, RoutingCallEndEvent)
                )
                self.assertFalse(routing_end.success)
                self.assertIsNone(routing_end.selected_model)
                if isinstance(outcome, BaseException):
                    self.assertEqual(
                        routing_end.error,
                        "RuntimeError: unavailable",
                    )
                    self.assertIsNone(routing_end.usage)
                else:
                    self.assertEqual(
                        routing_end.error,
                        "Routing model selected unknown candidate "
                        "'unknown'.",
                    )
                    self.assertIsNotNone(routing_end.usage)

    async def test_fallback_model_is_not_overridden(self) -> None:
        """The selected model should retain the Agent fallback behavior."""
        middleware, classifier = self._make_middleware(["reasoning"])
        self.reasoning.set_responses([RuntimeError("selected failed")])
        self.fallback.set_responses(
            [
                ChatResponse(
                    content=[TextBlock(text="Fallback response")],
                    is_last=True,
                ),
            ],
        )
        agent = Agent(
            name="router-agent",
            system_prompt="Help the user.",
            model=self.primary,
            middlewares=[middleware],
            model_config=ModelConfig(fallback_model=self.fallback),
            injection_config=InjectionConfig(inject_runtime_state=False),
        )

        response = await agent.reply(
            UserMsg(name="user", content="Hello"),
        )

        self.assertEqual(response.get_text_content(), "Fallback response")
        self.assertEqual(self.primary.cnt, 0)
        self.assertEqual(self.reasoning.cnt, 1)
        self.assertEqual(self.fallback.cnt, 1)
        self.assertEqual(len(classifier.calls), 1)
        self.assertIs(agent.model, self.primary)

    async def test_primary_model_is_restored_after_reply_error(self) -> None:
        """An exception in the reply chain should restore the primary model."""
        middleware, classifier = self._make_middleware(["reasoning"])

        async def failing_next_handler(**kwargs: Any) -> Any:
            del kwargs
            self.assertIs(self.agent.model, self.primary)
            yield ReplyStartEvent(
                session_id=self.agent.state.session_id,
                reply_id="reply-error",
                name=self.agent.name,
            )
            self.assertIs(self.agent.model, self.reasoning)
            raise RuntimeError("reply failed")

        with self.assertRaisesRegex(RuntimeError, "reply failed"):
            async for _ in middleware.on_reply(
                agent=self.agent,
                input_kwargs={
                    "inputs": UserMsg(name="user", content="Hello"),
                    "structured_schema": None,
                },
                next_handler=failing_next_handler,
            ):
                pass

        self.assertIs(self.agent.model, self.primary)
        self.assertEqual(len(classifier.calls), 1)

    async def test_no_user_message_uses_current_model(self) -> None:
        """Calls without a user message should bypass classification."""
        middleware, classifier = self._make_middleware(["reasoning"])

        forwarded = await self._invoke_reply(
            middleware,
            SystemMsg(name="system", content="System prompt"),
            "reply-no-user",
        )

        self.assertIs(forwarded["active_model"], self.primary)
        self.assertListEqual(classifier.calls, [])


class ModelRouterMiddlewareValidationTest(TestCase):
    """Test deterministic candidate configuration validation."""

    def setUp(self) -> None:
        """Create reusable model fixtures."""
        self.classifier = _MockClassifier(["a"])
        self.model = MockModel()

    def test_requires_two_candidates(self) -> None:
        """A router requires at least two choices."""
        with self.assertRaisesRegex(ValueError, "At least two"):
            ModelRouterMiddleware(
                self.classifier,
                [ChatModelCandidate("a", self.model, "A")],
            )

    def test_rejects_invalid_routing_model(self) -> None:
        """A routing model must implement a supported model interface."""
        with self.assertRaisesRegex(
            TypeError,
            "ClassifierModelBase or ChatModelBase",
        ):
            ModelRouterMiddleware(
                object(),
                [
                    ChatModelCandidate("a", self.model, "A"),
                    ChatModelCandidate("b", self.model, "B"),
                ],
            )

    def test_rejects_duplicate_and_padded_names(self) -> None:
        """Candidate names must be stable and unambiguous."""
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ModelRouterMiddleware(
                self.classifier,
                [
                    ChatModelCandidate("a", self.model, "A"),
                    ChatModelCandidate("a", self.model, "Again"),
                ],
            )

        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            ModelRouterMiddleware(
                self.classifier,
                [
                    ChatModelCandidate("a", self.model, "A"),
                    ChatModelCandidate(" b ", self.model, "B"),
                ],
            )
