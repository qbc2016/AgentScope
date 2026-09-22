# -*- coding: utf-8 -*-
"""The TypeSafe Jev classifier model implementation."""
from time import perf_counter
from typing import Any, Mapping

from pydantic import ConfigDict, JsonValue

from .._base import ClassifierModelBase
from .._question import (
    BinaryQuestion,
    ChoiceQuestion,
    ClassifierQuestion,
    ScoreQuestion,
)
from .._response import (
    BinaryAnswer,
    ChoiceAnswer,
    ClassifierAnswer,
    ClassifierResponse,
    ScoreAnswer,
)
from .._usage import ClassifierUsage
from ...credential import TypeSafeCredential


class JevClassifierModel(ClassifierModelBase):
    """A classifier backed by TypeSafe's Jev System One API.

    AgentScope's provider-independent binary question maps to TypeSafe's
    semantically equivalent Noul primitive.
    """

    class Parameters(ClassifierModelBase.Parameters):
        """Provider-specific Jev model parameters."""

        model_config = ConfigDict(
            extra="forbid",
            strict=True,
            allow_inf_nan=False,
        )

    def __init__(
        self,
        credential: TypeSafeCredential,
        model: str = "jev-latest",
        parameters: "JevClassifierModel.Parameters | None" = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ) -> None:
        """Initialize the Jev classifier.

        Args:
            credential (`TypeSafeCredential`):
                The TypeSafe API credential.
            model (`str`, defaults to ``"jev-latest"``):
                The Jev model name or alias.
            parameters (`JevClassifierModel.Parameters | None`):
                Provider-specific model parameters.
            extra_headers (`dict[str, str] | None`, defaults to `None`):
                Additional HTTP headers sent with each request.
            extra_body (`dict[str, JsonValue] | None`, defaults to `None`):
                Additional JSON fields sent in each request body.
            timeout (`float`, defaults to `30.0`):
                Per-operation timeout in seconds.
            max_retries (`int`, defaults to `2`):
                Maximum retries after the initial request.
            retry_delay (`float`, defaults to `0.5`):
                Initial exponential backoff delay in seconds.
        """
        try:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        except ImportError as error:
            raise ImportError(
                "JevClassifierModel requires a compatible optional "
                "'typesafe-sdk' dependency. Install it with "
                "`pip install 'agentscope[classifier-jev]'`.",
            ) from error

        super().__init__(credential, model, parameters)
        self.extra_headers = (
            dict(extra_headers) if extra_headers is not None else None
        )
        self.extra_body = dict(extra_body) if extra_body is not None else None
        retry = RetryPolicy(
            max_retries=max_retries,
            backoff_initial=retry_delay,
        )
        client_kwargs: dict[str, Any] = {
            "api_key": credential.api_key.get_secret_value(),
            "model": model,
            "retry": retry,
            "timeout": timeout,
        }
        if credential.base_url is not None:
            client_kwargs["base_url"] = credential.base_url
        self.client = AsyncTypeSafeClient(**client_kwargs)

    async def aclose(self) -> None:
        """Close the underlying TypeSafe asynchronous client."""
        await self.client.aclose()

    async def _call_api(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Call the TypeSafe System One API and normalize its response.

        Args:
            state (`str`):
                Text to classify.
            questions (`Mapping[str, ClassifierQuestion]`):
                Named classifier questions evaluated against ``state``.
            **kwargs (`Any`):
                Per-call TypeSafe options. Values supplied here override
                constructor-level ``extra_headers`` and ``extra_body``.

        Returns:
            `ClassifierResponse`:
                The normalized classifier response.
        """
        sdk_questions = {
            name: self._to_sdk_question(question)
            for name, question in questions.items()
        }
        request_options: dict[str, Any] = {}
        if self.extra_headers is not None:
            request_options["extra_headers"] = dict(self.extra_headers)
        if self.extra_body is not None:
            request_options["extra_body"] = dict(self.extra_body)
        request_options.update(kwargs)
        start_time = perf_counter()
        response = await self.client.system_one(
            state=state,
            questions=sdk_questions,
            model=self.model,
            **request_options,
        )
        elapsed = perf_counter() - start_time
        content = {
            name: self._from_sdk_answer(answer)
            for name, answer in response.answers.items()
        }
        return ClassifierResponse(
            model=response.model,
            content=content,
            usage=ClassifierUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                time=elapsed,
            ),
        )

    def _to_sdk_question(self, question: ClassifierQuestion) -> Any:
        """Translate an AgentScope question to a TypeSafe question."""
        from typesafe_sdk import Choice, Noul, Score

        if isinstance(question, BinaryQuestion):
            criteria = (
                question.criteria.model_dump(exclude_unset=True)
                if question.criteria is not None
                else None
            )
            return Noul(
                instructions=question.instructions,
                criteria=criteria,
            )
        if isinstance(question, ChoiceQuestion):
            return Choice(
                instructions=question.instructions,
                criteria=question.criteria,
            )
        if isinstance(question, ScoreQuestion):
            return Score(
                instructions=question.instructions,
                criteria=question.criteria,
            )
        raise TypeError(
            f"Unsupported classifier question type: "
            f"{type(question).__name__}.",
        )

    def _from_sdk_answer(self, answer: Any) -> ClassifierAnswer:
        """Translate a TypeSafe answer to an AgentScope answer."""
        from typesafe_sdk import ChoiceAnswer as SDKChoiceAnswer
        from typesafe_sdk import NoulAnswer
        from typesafe_sdk import ScoreAnswer as SDKScoreAnswer

        if isinstance(answer, NoulAnswer):
            return BinaryAnswer(probability=answer.noul)
        if isinstance(answer, SDKChoiceAnswer):
            return ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=dict(answer.probabilities),
            )
        if isinstance(answer, SDKScoreAnswer):
            return ScoreAnswer(
                score=answer.score,
                confidence=answer.confidence,
                legend=dict(answer.legend),
                probabilities=dict(answer.probabilities),
            )
        raise TypeError(
            f"Unsupported TypeSafe answer type: {type(answer).__name__}.",
        )
