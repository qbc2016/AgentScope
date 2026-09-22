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
        """Default per-call options for the TypeSafe API.

        Options supplied directly to ``__call__`` override same-named fields.
        """

        model_config = ConfigDict(
            extra="forbid",
            strict=True,
            allow_inf_nan=False,
        )

        extra_headers: dict[str, str] | None = None
        """Additional HTTP headers sent with each request."""

        extra_body: dict[str, JsonValue] | None = None
        """Additional JSON fields sent in each request body."""

    def __init__(
        self,
        credential: TypeSafeCredential,
        model: str = "jev-latest",
        parameters: "JevClassifierModel.Parameters | None" = None,
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
            timeout (`float`, defaults to `30.0`):
                Per-operation timeout in seconds.
            max_retries (`int`, defaults to `2`):
                Maximum retries after the initial request.
            retry_delay (`float`, defaults to `0.5`):
                Initial exponential backoff delay in seconds.
        """
        try:
            import typesafe_sdk
        except ImportError as error:
            raise ImportError(
                "JevClassifierModel requires the optional "
                "'typesafe-sdk' dependency. Install it with "
                "`pip install 'agentscope[classifier-jev]'`.",
            ) from error

        try:
            client_cls = typesafe_sdk.AsyncTypeSafeClient
            retry_policy_cls = typesafe_sdk.RetryPolicy
            self._noul_question_cls = typesafe_sdk.Noul
            self._choice_question_cls = typesafe_sdk.Choice
            self._score_question_cls = typesafe_sdk.Score
            self._noul_answer_cls = typesafe_sdk.NoulAnswer
            self._choice_answer_cls = typesafe_sdk.ChoiceAnswer
            self._score_answer_cls = typesafe_sdk.ScoreAnswer
        except AttributeError as error:
            raise ImportError(
                "Unsupported typesafe-sdk version. Install a compatible "
                "version with `pip install 'typesafe-sdk>=0.7,<0.8'`.",
            ) from error

        super().__init__(credential, model, parameters)
        retry = retry_policy_cls(
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
        self.client = client_cls(**client_kwargs)

    async def aclose(self) -> None:
        """Close the underlying TypeSafe asynchronous client."""
        await self.client.aclose()

    async def _call_api(
        self,
        state: str,
        questions: Mapping[str, ClassifierQuestion],
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Call the TypeSafe System One API."""
        sdk_questions = {
            name: self._to_sdk_question(question)
            for name, question in questions.items()
        }
        request_options = self.parameters.model_dump(exclude_none=True)
        request_options.update(kwargs)
        start_time = perf_counter()
        response = await self.client.system_one(
            state=state,
            questions=sdk_questions,
            model=self.model,
            **request_options,
        )
        elapsed = perf_counter() - start_time
        answers = {
            name: self._from_sdk_answer(answer)
            for name, answer in response.answers.items()
        }
        return ClassifierResponse(
            model=response.model,
            answers=answers,
            usage=ClassifierUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                time=elapsed,
            ),
        )

    def _to_sdk_question(self, question: ClassifierQuestion) -> Any:
        """Translate an AgentScope question to a TypeSafe question."""
        if isinstance(question, BinaryQuestion):
            criteria = (
                question.criteria.model_dump(exclude_unset=True)
                if question.criteria is not None
                else None
            )
            return self._noul_question_cls(
                instructions=question.instructions,
                criteria=criteria,
            )
        if isinstance(question, ChoiceQuestion):
            return self._choice_question_cls(
                instructions=question.instructions,
                criteria=question.criteria,
            )
        if isinstance(question, ScoreQuestion):
            return self._score_question_cls(
                instructions=question.instructions,
                criteria=question.criteria,
            )
        raise TypeError(
            f"Unsupported classifier question type: "
            f"{type(question).__name__}.",
        )

    def _from_sdk_answer(self, answer: Any) -> ClassifierAnswer:
        """Translate a TypeSafe answer to an AgentScope answer."""
        if isinstance(answer, self._noul_answer_cls):
            return BinaryAnswer(probability=answer.noul)
        if isinstance(answer, self._choice_answer_cls):
            return ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=dict(answer.probabilities),
            )
        if isinstance(answer, self._score_answer_cls):
            return ScoreAnswer(
                score=answer.score,
                confidence=answer.confidence,
                legend=dict(answer.legend),
                probabilities=dict(answer.probabilities),
            )
        raise TypeError(
            f"Unsupported TypeSafe answer type: {type(answer).__name__}.",
        )
