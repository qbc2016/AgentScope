# -*- coding: utf-8 -*-
"""Comprehensive formatter unit tests for GeminiChatFormatter and
GeminiMultiAgentFormatter, following the reference test style with exact
ground-truth comparisons.
"""
from unittest import IsolatedAsyncioTestCase

from agentscope.formatter import (
    GeminiChatFormatter,
    GeminiMultiAgentFormatter,
)
from agentscope.message import (
    Msg,
    TextBlock,
    DataBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    Base64Source,
    ThinkingBlock,
    HintBlock,
)


class TestGeminiFormatter(IsolatedAsyncioTestCase):
    """Comprehensive tests for Gemini Chat and MultiAgent formatters."""

    async def asyncSetUp(self) -> None:
        """Set up shared message fixtures and expected ground-truth dicts."""
        image_b64 = "ZmFrZSBpbWFnZSBkYXRh"

        # ---------------------------------------------------------------
        # Message fixtures
        # (Use base64 images: Gemini URL handling downloads from the network)
        # ---------------------------------------------------------------
        self.msgs_system = [
            Msg(
                name="system",
                content="You're a helpful assistant.",
                role="system",
            ),
        ]

        self.msgs_conversation = [
            Msg(
                name="user",
                content=[
                    TextBlock(
                        type="text",
                        text="What is the capital of France?",
                    ),
                    DataBlock(
                        source=Base64Source(
                            type="base64",
                            data=image_b64,
                            media_type="image/png",
                        ),
                    ),
                ],
                role="user",
            ),
            Msg(
                name="assistant",
                content="The capital of France is Paris.",
                role="assistant",
            ),
            Msg(
                name="user",
                content="What is the capital of Germany?",
                role="user",
            ),
            Msg(
                name="assistant",
                content="The capital of Germany is Berlin.",
                role="assistant",
            ),
            Msg(
                name="user",
                content="What is the capital of Japan?",
                role="user",
            ),
        ]

        self.msgs_tools = [
            Msg(
                name="assistant",
                content=[
                    ToolCallBlock(
                        id="call_1",
                        name="get_capital",
                        input='{"country": "Japan"}',
                    ),
                    ToolResultBlock(
                        id="call_1",
                        name="get_capital",
                        output=[
                            TextBlock(
                                type="text",
                                text="The capital of Japan is Tokyo.",
                            ),
                        ],
                        state=ToolResultState.SUCCESS,
                    ),
                    TextBlock(
                        type="text",
                        text="The capital of Japan is Tokyo.",
                    ),
                ],
                role="assistant",
            ),
        ]

        _inline_img = {
            "inline_data": {"data": image_b64, "mime_type": "image/png"},
        }

        # ---------------------------------------------------------------
        # Ground truth: GeminiChatFormatter
        #   - System message becomes role="user" (no special system role).
        #   - Assistant messages become role="model".
        #   - Content is in "parts" (not "content") as a list of dicts.
        #   - ToolCallBlock becomes "function_call" part.
        #   - ToolResultBlock becomes a separate role="user" message with
        #     "function_response" part.
        # ---------------------------------------------------------------
        self.gt_chat = [
            {
                "role": "user",
                "parts": [{"text": "You're a helpful assistant."}],
            },
            {
                "role": "user",
                "parts": [
                    {"text": "What is the capital of France?"},
                    _inline_img,
                ],
            },
            {
                "role": "model",
                "parts": [{"text": "The capital of France is Paris."}],
            },
            {
                "role": "user",
                "parts": [{"text": "What is the capital of Germany?"}],
            },
            {
                "role": "model",
                "parts": [{"text": "The capital of Germany is Berlin."}],
            },
            {
                "role": "user",
                "parts": [{"text": "What is the capital of Japan?"}],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call_1",
                            "name": "get_capital",
                            "args": {"country": "Japan"},
                        },
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call_1",
                            "name": "get_capital",
                            "response": {
                                "output": "The capital of Japan is Tokyo.",
                            },
                        },
                    },
                ],
            },
            {
                "role": "model",
                "parts": [{"text": "The capital of Japan is Tokyo."}],
            },
        ]

        # ---------------------------------------------------------------
        # Ground truth: GeminiMultiAgentFormatter
        #   - System message: role="user" (same as chat formatter).
        #   - Agent messages: collapsed into role="user" with parts list.
        #   - Media blocks interleaved (text flushed before each DataBlock).
        #   - is_first=False still wraps with <history> (no hist_prompt
        #     prefix).
        # ---------------------------------------------------------------
        _hist_prompt = GeminiMultiAgentFormatter().conversation_history_prompt

        self._gt_trailing_asst = {
            "role": "model",
            "parts": [
                {"text": "The capital of Japan is Tokyo."},
            ],
        }

        self._gt_tool_call = {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": "call_1",
                        "name": "get_capital",
                        "args": {"country": "Japan"},
                    },
                },
            ],
        }
        self._gt_tool_result = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "call_1",
                        "name": "get_capital",
                        "response": {
                            "output": "The capital of Japan is Tokyo.",
                        },
                    },
                },
            ],
        }

        self.gt_multiagent = [
            {
                "role": "user",
                "parts": [{"text": "You're a helpful assistant."}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            _hist_prompt + "<history>\n"
                            "user: What is the capital of France?"
                        ),
                    },
                    _inline_img,
                    {
                        "text": (
                            "assistant: The capital of France is Paris.\n"
                            "user: What is the capital of Germany?\n"
                            "assistant: The capital of Germany is Berlin.\n"
                            "user: What is the capital of Japan?\n"
                            "</history>"
                        ),
                    },
                ],
            },
            self._gt_tool_call,
            self._gt_tool_result,
            self._gt_trailing_asst,
        ]

    # -------------------------------------------------------------------
    # GeminiChatFormatter tests
    # -------------------------------------------------------------------

    async def test_chat_formatter(self) -> None:
        """Chat formatter produces exact output for various subsets."""
        fmt = GeminiChatFormatter()
        self.maxDiff = None

        # Full history
        res = await fmt.format(
            [*self.msgs_system, *self.msgs_conversation, *self.msgs_tools],
        )
        self.assertListEqual(self.gt_chat, res)

        # Without system
        res = await fmt.format([*self.msgs_conversation, *self.msgs_tools])
        self.assertListEqual(self.gt_chat[1:], res)

        # Without conversation
        n_tools_gt = len(self.gt_chat) - 1 - len(self.msgs_conversation)
        res = await fmt.format([*self.msgs_system, *self.msgs_tools])
        self.assertListEqual(
            [self.gt_chat[0]] + self.gt_chat[-n_tools_gt:],
            res,
        )

        # Without tools
        res = await fmt.format([*self.msgs_system, *self.msgs_conversation])
        self.assertListEqual(self.gt_chat[:-n_tools_gt], res)

        # Empty
        res = await fmt.format([])
        self.assertListEqual([], res)

    async def test_chat_formatter_thinking_preserved(self) -> None:
        """ThinkingBlock becomes a part with thought=True in Gemini format."""
        fmt = GeminiChatFormatter()
        msgs = [
            Msg(
                name="assistant",
                content=[
                    ThinkingBlock(thinking="inner thoughts"),
                    TextBlock(type="text", text="reply"),
                ],
                role="assistant",
            ),
        ]
        res = await fmt.format(msgs)
        self.assertListEqual(
            res,
            [
                {
                    "role": "model",
                    "parts": [
                        {"thought": True, "text": "inner thoughts"},
                        {"text": "reply"},
                    ],
                },
            ],
        )

    # -------------------------------------------------------------------
    # GeminiMultiAgentFormatter tests
    # -------------------------------------------------------------------

    async def test_multiagent_formatter(self) -> None:
        """MultiAgent formatter produces exact output for various subsets."""
        fmt = GeminiMultiAgentFormatter()
        self.maxDiff = None

        # Full history
        res = await fmt.format(
            [*self.msgs_system, *self.msgs_conversation, *self.msgs_tools],
        )
        self.assertListEqual(self.gt_multiagent, res)

        # Without system
        res = await fmt.format([*self.msgs_conversation, *self.msgs_tools])
        self.assertListEqual(self.gt_multiagent[1:], res)

        # Without tools
        res = await fmt.format([*self.msgs_system, *self.msgs_conversation])
        self.assertListEqual(self.gt_multiagent[:2], res)

        # System only
        res = await fmt.format(self.msgs_system)
        self.assertListEqual([self.gt_multiagent[0]], res)

        # Conversation only
        res = await fmt.format(self.msgs_conversation)
        self.assertListEqual([self.gt_multiagent[1]], res)

        # Tools only
        res = await fmt.format(self.msgs_tools)
        self.assertListEqual(
            [
                self._gt_tool_call,
                self._gt_tool_result,
                self._gt_trailing_asst,
            ],
            res,
        )

        # System + tools
        res = await fmt.format([*self.msgs_system, *self.msgs_tools])
        self.assertListEqual(
            [
                self.gt_multiagent[0],
                self._gt_tool_call,
                self._gt_tool_result,
                self._gt_trailing_asst,
            ],
            res,
        )

        # Empty
        res = await fmt.format([])
        self.assertListEqual([], res)

    async def test_chat_formatter_hint_block(self) -> None:
        """HintBlock flushes preceding content and becomes a user message."""
        fmt = GeminiChatFormatter()
        msgs = [
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    TextBlock(text="Let me think about that."),
                    HintBlock(hint="Remember to be concise."),
                    TextBlock(text="Here is my answer."),
                ],
            ),
        ]
        res = await fmt.format(msgs)
        self.assertListEqual(
            res,
            [
                {
                    "role": "model",
                    "parts": [
                        {"text": "Let me think about that."},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {"text": "Remember to be concise."},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "Here is my answer."},
                    ],
                },
            ],
        )
