# -*- coding: utf-8 -*-
"""Tests for the provider-independent classifier model contract."""
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from unittest import IsolatedAsyncioTestCase, TestCase

from pydantic import ValidationError

from agentscope.classifier import (
    BinaryAnswer,
    BinaryQuestion,
    ChoiceQuestion,
    ClassifierModelBase,
    ClassifierQuestion,
    ClassifierResponse,
    ScoreQuestion,
)
from agentscope.credential import CredentialBase
from agentscope.message import Base64Source, DataBlock, TextBlock


class _MockClassifier(ClassifierModelBase):
    """A minimal classifier used to exercise the base class."""

    def __init__(self) -> None:
        """Initialize the mock classifier."""
        super().__init__(CredentialBase(), "mock-classifier")
        self.closed = False

    async def _call_api(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Return a deterministic response."""
        del state, questions, kwargs
        return ClassifierResponse(
            model=self.model,
            answers={"safe": BinaryAnswer(probability=0.75)},
        )

    async def aclose(self) -> None:
        """Record that the asynchronous lifecycle closed."""
        self.closed = True


class ClassifierQuestionTest(TestCase):
    """Validate the framework-owned question types."""

    def test_question_models(self) -> None:
        """Question models should preserve their complete typed structure."""
        choice = ChoiceQuestion(
            instructions="Select a route.",
            criteria={"billing": None, "support": "Technical support."},
        )
        score = ScoreQuestion(
            instructions="Rate urgency.",
            criteria=["Can wait.", "Handle today."],
        )

        self.assertDictEqual(
            choice.model_dump(),
            {
                "type": "choice",
                "criteria": {
                    "billing": None,
                    "support": "Technical support.",
                },
                "instructions": "Select a route.",
            },
        )
        self.assertDictEqual(
            score.model_dump(),
            {
                "type": "score",
                "criteria": ["Can wait.", "Handle today."],
                "instructions": "Rate urgency.",
            },
        )

    def test_empty_criteria_are_rejected(self) -> None:
        """Choice and score questions require at least one criterion."""
        with self.assertRaises(ValidationError):
            ChoiceQuestion(criteria={})
        with self.assertRaises(ValidationError):
            ScoreQuestion(criteria=[])

    def test_non_string_question_content_is_rejected(self) -> None:
        """Question content must be a string."""
        invalid_content = [
            b"bytes",
            {"message": "hello"},
            ["hello"],
            {"invalid": {1, 2}},
            {"invalid": Path("file.txt")},
            {"invalid": (1, 2)},
            {"invalid": float("nan")},
            TextBlock(text="hello"),
            DataBlock(
                source=Base64Source(
                    data="aGVsbG8=",
                    media_type="image/png",
                ),
            ),
        ]

        for content in invalid_content:
            with self.subTest(content=content):
                with self.assertRaises(ValidationError):
                    BinaryQuestion(instructions=content)


class ClassifierModelBaseTest(IsolatedAsyncioTestCase):
    """Test input validation and lifecycle behavior in the base class."""

    async def test_call_and_lifecycle(self) -> None:
        """A valid call should delegate and the context should close."""
        model = _MockClassifier()
        async with model:
            response = await model(
                state="hello",
                questions={
                    "safe": ChoiceQuestion(criteria={"yes": None}),
                },
            )

        self.assertTrue(model.closed)
        self.assertDictEqual(
            asdict(response),
            {
                "model": "mock-classifier",
                "answers": {
                    "safe": {
                        "probability": 0.75,
                        "type": "binary_answer",
                    },
                },
                "usage": None,
                "id": response.id,
                "created_at": response.created_at,
                "type": "classifier_response",
                "metadata": {},
            },
        )

    async def test_empty_questions_are_rejected(self) -> None:
        """The base class should reject a call with no questions."""
        with self.assertRaisesRegex(ValueError, "At least one"):
            await _MockClassifier()(state="hello", questions={})

    async def test_non_string_state_is_rejected(self) -> None:
        """The base class should reject non-string state."""
        invalid_state = [
            1,
            b"bytes",
            {"message": "hello"},
            ["hello"],
            {"invalid": {1, 2}},
            {"invalid": Path("file.txt")},
            {"invalid": (1, 2)},
            {"invalid": float("nan")},
            TextBlock(text="hello"),
            DataBlock(
                source=Base64Source(
                    data="aGVsbG8=",
                    media_type="image/png",
                ),
            ),
        ]

        for state in invalid_state:
            with self.subTest(state=state):
                with self.assertRaises(ValidationError):
                    await _MockClassifier()(
                        state=state,
                        questions={
                            "route": ChoiceQuestion(criteria={"a": None}),
                        },
                    )
