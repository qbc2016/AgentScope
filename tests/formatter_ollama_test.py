# -*- coding: utf-8 -*-
"""Comprehensive formatter unit tests for OllamaChatFormatter and
OllamaMultiAgentFormatter, with exact ground-truth comparisons.
"""
from unittest import IsolatedAsyncioTestCase

from agentscope.formatter import OllamaChatFormatter, OllamaMultiAgentFormatter
from agentscope.message import (
    UserMsg,
    AssistantMsg,
    SystemMsg,
    TextBlock,
    DataBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    Base64Source,
    HintBlock,
)


class TestOllamaFormatter(IsolatedAsyncioTestCase):
    """Comprehensive tests for Ollama Chat and MultiAgent formatters."""

    async def asyncSetUp(self) -> None:
        """Set up shared fixtures and ground-truth dicts."""
        _hist_prompt = OllamaMultiAgentFormatter().conversation_history_prompt

        # Base64 image fixture
        self.image_b64 = "ZmFrZSBpbWFnZSBkYXRh"

        self.msgs_system = [
            SystemMsg(
                name="system",
                content="You're a helpful assistant.",
            ),
        ]
        self.msgs_conversation = [
            UserMsg(
                name="user",
                content="What is the capital of France?",
            ),
            AssistantMsg(
                name="assistant",
                content="The capital of France is Paris.",
            ),
            UserMsg(
                name="user",
                content="What is the capital of Germany?",
            ),
            AssistantMsg(
                name="assistant",
                content="The capital of Germany is Berlin.",
            ),
            UserMsg(
                name="user",
                content="What is the capital of Japan?",
            ),
        ]
        self.msgs_tools = [
            AssistantMsg(
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
            ),
        ]

        # --- Chat formatter ground truth ---
        # Ollama content is always a plain string.
        # Tool calls use dict arguments (not JSON string).
        self.gt_chat = [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            {"role": "user", "content": "What is the capital of Germany?"},
            {
                "role": "assistant",
                "content": "The capital of Germany is Berlin.",
            },
            {"role": "user", "content": "What is the capital of Japan?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_capital",
                            "arguments": {"country": "Japan"},
                        },
                    },
                ],
            },
            {"role": "tool", "content": "The capital of Japan is Tokyo."},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        ]

        # --- MultiAgent formatter ground truth ---
        # System content is a plain string.
        # History is a plain string with format "name:\ntext" per message.
        # For is_first=False, there are NO <history> tags (only in
        # is_first=True).
        _conv_text = (
            "user:\nWhat is the capital of France?\n"
            "assistant:\nThe capital of France is Paris.\n"
            "user:\nWhat is the capital of Germany?\n"
            "assistant:\nThe capital of Germany is Berlin.\n"
            "user:\nWhat is the capital of Japan?"
        )
        self._gt_trailing_asst = {
            "role": "assistant",
            "content": "The capital of Japan is Tokyo.",
        }
        self._gt_tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_capital",
                        "arguments": {"country": "Japan"},
                    },
                },
            ],
        }
        self._gt_tool_result = {
            "role": "tool",
            "content": "The capital of Japan is Tokyo.",
        }

        self.gt_multiagent = [
            {"role": "system", "content": "You're a helpful assistant."},
            {
                "role": "user",
                "content": (
                    _hist_prompt + "<history>\n" + _conv_text + "\n</history>"
                ),
            },
            self._gt_tool_call,
            self._gt_tool_result,
            self._gt_trailing_asst,
        ]

    # ------------------------------------------------------------------
    # OllamaChatFormatter tests
    # ------------------------------------------------------------------

    async def test_chat_formatter(self) -> None:
        """Chat formatter produces exact output for various subsets."""
        fmt = OllamaChatFormatter()
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
        self.assertListEqual([], await fmt.format([]))

    async def test_chat_formatter_tool_call_arguments_are_dict(self) -> None:
        """Ollama requires tool call arguments as a dict, not a JSON string."""
        fmt = OllamaChatFormatter()
        tc = ToolCallBlock(id="c1", name="search", input='{"q": "weather"}')
        res = await fmt.format(
            [AssistantMsg(name="assistant", content=[tc])],
        )
        self.assertListEqual(
            res,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": {"q": "weather"},
                            },
                        },
                    ],
                },
            ],
        )

    async def test_chat_formatter_base64_image(self) -> None:
        """Base64 image is placed in the 'images' list as a raw base64
        string."""
        fmt = OllamaChatFormatter()
        msgs = [
            UserMsg(
                name="user",
                content=[
                    TextBlock(type="text", text="What is this?"),
                    DataBlock(
                        source=Base64Source(
                            type="base64",
                            data=self.image_b64,
                            media_type="image/png",
                        ),
                    ),
                ],
            ),
        ]
        res = await fmt.format(msgs)
        self.assertListEqual(
            res,
            [
                {
                    "role": "user",
                    "content": "What is this?",
                    "images": [self.image_b64],
                },
            ],
        )

    # ------------------------------------------------------------------
    # OllamaMultiAgentFormatter tests
    # ------------------------------------------------------------------

    async def test_multiagent_formatter(self) -> None:
        """MultiAgent formatter produces exact output for various subsets."""
        fmt = OllamaMultiAgentFormatter()
        self.maxDiff = None

        # Full
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
        self.assertListEqual([], await fmt.format([]))

    async def test_chat_formatter_hint_block(self) -> None:
        """HintBlock flushes preceding content and becomes a user message."""
        fmt = OllamaChatFormatter()
        msgs = [
            AssistantMsg(
                name="assistant",
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
                    "role": "assistant",
                    "content": "Let me think about that.",
                },
                {
                    "role": "user",
                    "content": "Remember to be concise.",
                },
                {
                    "role": "assistant",
                    "content": "Here is my answer.",
                },
            ],
        )
