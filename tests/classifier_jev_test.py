# -*- coding: utf-8 -*-
"""Tests for the TypeSafe Jev classifier adapter."""
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils import AnyValue
from agentscope.classifier import (
    BinaryCriteria,
    BinaryQuestion,
    ChoiceQuestion,
    JevClassifierModel,
    ScoreQuestion,
)
from agentscope.credential import TypeSafeCredential

typesafe_sdk = pytest.importorskip("typesafe_sdk")
SDKChoiceAnswer = typesafe_sdk.ChoiceAnswer
NoulAnswer = typesafe_sdk.NoulAnswer
SDKScoreAnswer = typesafe_sdk.ScoreAnswer
Usage = typesafe_sdk.Usage

A = AnyValue()


class JevClassifierModelTest(IsolatedAsyncioTestCase):
    """Test request and response translation against official SDK types."""

    @patch("typesafe_sdk.AsyncTypeSafeClient")
    async def test_call_translates_all_question_types(
        self,
        client_cls: Any,
    ) -> None:
        """All supported questions and answers should round-trip."""
        client = MagicMock()
        client.system_one = AsyncMock(
            return_value=SimpleNamespace(
                model="jev-1.13.0",
                answers={
                    "urgent": NoulAnswer(type="noul", noul=0.8),
                    "route": SDKChoiceAnswer(
                        type="choice",
                        choice="billing",
                        confidence=0.9,
                        probabilities={"billing": 0.9, "support": 0.1},
                    ),
                    "priority": SDKScoreAnswer(
                        type="score",
                        score=1.7,
                        confidence=0.85,
                        legend={0: "low", 1: "medium", 2: "high"},
                        probabilities={0: 0.05, 1: 0.2, 2: 0.75},
                    ),
                },
                usage=Usage(input_tokens=120, output_tokens=3),
            ),
        )
        client.aclose = AsyncMock()
        client_cls.return_value = client

        model = JevClassifierModel(
            credential=TypeSafeCredential(
                api_key="secret",
                base_url="https://typesafe.example",
            ),
            parameters=JevClassifierModel.Parameters(
                extra_headers={"x-default": "default"},
                extra_body={"trace": True},
            ),
            max_retries=4,
            retry_delay=0.25,
        )
        response = await model(
            state="I was charged twice.",
            questions={
                "urgent": BinaryQuestion(
                    instructions="Is this urgent?",
                    criteria=BinaryCriteria(true="Urgent."),
                ),
                "route": ChoiceQuestion(
                    instructions="Select a route.",
                    criteria={"billing": None, "support": None},
                ),
                "priority": ScoreQuestion(
                    instructions="Rate priority.",
                    criteria=["low", "medium", "high"],
                ),
            },
            extra_headers={"x-request": "request"},
        )

        client_kwargs = client_cls.call_args.kwargs
        self.assertEqual(client_kwargs["api_key"], "secret")
        self.assertEqual(client_kwargs["model"], "jev-latest")
        self.assertEqual(
            client_kwargs["base_url"],
            "https://typesafe.example",
        )
        self.assertEqual(client_kwargs["retry"].max_retries, 4)
        self.assertEqual(client_kwargs["retry"].backoff_initial, 0.25)

        call_kwargs = client.system_one.await_args.kwargs
        self.assertEqual(call_kwargs["state"], "I was charged twice.")
        self.assertEqual(call_kwargs["model"], "jev-latest")
        self.assertDictEqual(
            call_kwargs["extra_headers"],
            {"x-request": "request"},
        )
        self.assertDictEqual(call_kwargs["extra_body"], {"trace": True})
        self.assertDictEqual(
            {
                name: question.model_dump()
                for name, question in call_kwargs["questions"].items()
            },
            {
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": {"true": "Urgent."},
                },
                "route": {
                    "type": "choice",
                    "instructions": "Select a route.",
                    "criteria": {"billing": None, "support": None},
                },
                "priority": {
                    "type": "score",
                    "instructions": "Rate priority.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        )
        self.assertDictEqual(
            asdict(response),
            {
                "model": "jev-1.13.0",
                "answers": {
                    "urgent": {
                        "probability": 0.8,
                        "type": "binary_answer",
                    },
                    "route": {
                        "choice": "billing",
                        "confidence": 0.9,
                        "probabilities": {
                            "billing": 0.9,
                            "support": 0.1,
                        },
                        "type": "choice_answer",
                    },
                    "priority": {
                        "score": 1.7,
                        "confidence": 0.85,
                        "legend": {0: "low", 1: "medium", 2: "high"},
                        "probabilities": {0: 0.05, 1: 0.2, 2: 0.75},
                        "type": "score_answer",
                    },
                },
                "usage": {
                    "time": A,
                    "input_tokens": 120,
                    "output_tokens": 3,
                    "type": "classifier",
                },
                "id": A,
                "created_at": A,
                "type": "classifier_response",
                "metadata": {},
            },
        )

        await model.aclose()
        client.aclose.assert_awaited_once_with()

    @patch("typesafe_sdk.AsyncTypeSafeClient")
    async def test_provider_error_is_not_retried_by_adapter(
        self,
        client_cls: Any,
    ) -> None:
        """The adapter should leave retry ownership to the official SDK."""
        client = MagicMock()
        client.system_one = AsyncMock(side_effect=RuntimeError("failed"))
        client_cls.return_value = client
        model = JevClassifierModel(TypeSafeCredential(api_key="secret"))

        self.assertNotIn("base_url", client_cls.call_args.kwargs)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            await model(
                state="hello",
                questions={
                    "route": ChoiceQuestion(criteria={"a": None}),
                },
            )

        client.system_one.assert_awaited_once()
        self.assertEqual(
            client.system_one.await_args.kwargs["state"],
            "hello",
        )

    async def test_incompatible_sdk_has_clear_error(self) -> None:
        """Missing SDK exports should produce an actionable error."""
        incompatible_sdk = SimpleNamespace(
            AsyncTypeSafeClient=MagicMock(),
            RetryPolicy=MagicMock(),
        )

        with patch.dict("sys.modules", {"typesafe_sdk": incompatible_sdk}):
            with self.assertRaisesRegex(
                ImportError,
                "Unsupported typesafe-sdk version",
            ):
                JevClassifierModel(TypeSafeCredential(api_key="secret"))
