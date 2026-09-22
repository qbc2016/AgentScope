# -*- coding: utf-8 -*-
"""Tests for classifier-based chat model routing middleware."""
from typing import Any, Mapping
from unittest import IsolatedAsyncioTestCase, TestCase

from utils import MockModel

from agentscope.agent import Agent, InjectionConfig
from agentscope.classifier import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
    ClassifierQuestion,
    ClassifierResponse,
)
from agentscope.credential import CredentialBase
from agentscope.message import (
    Base64Source,
    DataBlock,
    SystemMsg,
    TextBlock,
    UserMsg,
)
from agentscope.middleware import ChatModelCandidate, ModelRouterMiddleware
from agentscope.model import ChatModelBase, ChatResponse


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
        )


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

    async def _invoke(
        self,
        middleware: ModelRouterMiddleware,
        messages: list,
        current_model: ChatModelBase | None = None,
    ) -> dict:
        """Invoke the middleware and return arguments forwarded inward."""
        forwarded: dict = {}

        async def next_handler(**kwargs: Any) -> ChatResponse:
            forwarded.update(kwargs)
            return ChatResponse(content=[], is_last=True)

        await middleware.on_model_call(
            agent=self.agent,
            input_kwargs={
                "current_model": current_model or self.primary,
                "messages": messages,
                "tools": [],
                "tool_choice": None,
            },
            next_handler=next_handler,
        )
        return forwarded

    async def test_routes_once_per_reply_and_reroutes_next_reply(
        self,
    ) -> None:
        """One reply should reuse its route and a new reply should reroute."""
        middleware, classifier = self._make_middleware(
            ["reasoning", "fast"],
        )
        self.agent.state.reply_id = "reply-1"

        first = await self._invoke(
            middleware,
            [UserMsg(name="user", content="Prove this theorem.")],
        )
        cached = await self._invoke(
            middleware,
            [UserMsg(name="user", content="This should not reroute.")],
        )

        self.assertIs(first["current_model"], self.reasoning)
        self.assertIs(cached["current_model"], self.reasoning)
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

        self.agent.state.reply_id = "reply-2"
        rerouted = await self._invoke(
            middleware,
            [UserMsg(name="user", content="Say hello.")],
        )

        self.assertIs(rerouted["current_model"], self.fast)
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

        forwarded = await self._invoke(middleware, [message])

        self.assertIs(forwarded["current_model"], self.reasoning)
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

        forwarded = await self._invoke(middleware, [message])

        self.assertIs(forwarded["current_model"], self.primary)
        self.assertListEqual(classifier.calls, [])

    async def test_classifier_failures_use_current_model(self) -> None:
        """Classifier errors and unknown choices should fail open."""
        outcomes: list[str | BaseException] = [
            RuntimeError("unavailable"),
            "unknown",
        ]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                self.agent.state.reply_id = f"reply-{type(outcome).__name__}"
                middleware, classifier = self._make_middleware([outcome])

                forwarded = await self._invoke(
                    middleware,
                    [UserMsg(name="user", content="Hello")],
                )
                cached = await self._invoke(
                    middleware,
                    [UserMsg(name="user", content="Hello again")],
                )

                self.assertIs(forwarded["current_model"], self.primary)
                self.assertIs(cached["current_model"], self.primary)
                self.assertEqual(len(classifier.calls), 1)

    async def test_fallback_model_is_not_overridden(self) -> None:
        """The Agent's fallback-model attempt should bypass routing."""
        middleware, classifier = self._make_middleware(["reasoning"])

        forwarded = await self._invoke(
            middleware,
            [UserMsg(name="user", content="Hello")],
            current_model=self.fallback,
        )

        self.assertIs(forwarded["current_model"], self.fallback)
        self.assertListEqual(classifier.calls, [])

    async def test_no_user_message_uses_current_model(self) -> None:
        """Calls without a user message should bypass classification."""
        middleware, classifier = self._make_middleware(["reasoning"])

        forwarded = await self._invoke(
            middleware,
            [SystemMsg(name="system", content="System prompt")],
        )

        self.assertIs(forwarded["current_model"], self.primary)
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
