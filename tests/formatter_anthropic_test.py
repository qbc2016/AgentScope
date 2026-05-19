# -*- coding: utf-8 -*-
"""Comprehensive formatter unit tests for AnthropicChatFormatter and
AnthropicMultiAgentFormatter, following the reference test style with exact
ground-truth comparisons.
"""
from unittest import IsolatedAsyncioTestCase

from agentscope.formatter import (
    AnthropicChatFormatter,
    AnthropicMultiAgentFormatter,
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


class TestAnthropicFormatter(IsolatedAsyncioTestCase):
    """Comprehensive tests for Anthropic Chat and MultiAgent formatters."""

    async def asyncSetUp(self) -> None:
        """Set up shared message fixtures and expected ground-truth dicts."""
        self.image_b64 = "ZmFrZSBpbWFnZSBkYXRh"

        # ---------------------------------------------------------------
        # Message fixtures
        # (No URL images: Anthropic URL handling downloads from the network)
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
                content="What is the capital of France?",
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

        # ---------------------------------------------------------------
        # Ground truth: AnthropicChatFormatter
        #   - No "name" field.
        #   - Content is always a list of {"type": ..., ...} dicts.
        #   - ToolResultBlock forces role to "user".
        #   - ToolCallBlock "input" is a dict (parsed from JSON string).
        # ---------------------------------------------------------------
        self.gt_chat = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You're a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of France?",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The capital of France is Paris.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of Germany?",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The capital of Germany is Berlin.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of Japan?",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_capital",
                        "input": {"country": "Japan"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "The capital of Japan is Tokyo.",
                            },
                        ],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The capital of Japan is Tokyo.",
                    },
                ],
            },
        ]

        # ---------------------------------------------------------------
        # Ground truth: AnthropicMultiAgentFormatter
        #   - System: {"role": "system", "content": [{"type": "text", ...}]}
        #   - Agent messages (is_first=True): wrapped in hist_prompt +
        #     <history>...</history>.
        #   - Agent messages (is_first=False): no wrapping at all.
        # ---------------------------------------------------------------
        _hist_prompt = (
            AnthropicMultiAgentFormatter().conversation_history_prompt
        )

        _conv_text = (
            "user: What is the capital of France?\n"
            "assistant: The capital of France is Paris.\n"
            "user: What is the capital of Germany?\n"
            "assistant: The capital of Germany is Berlin.\n"
            "user: What is the capital of Japan?"
        )

        self._gt_trailing_asst = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The capital of Japan is Tokyo.",
                },
            ],
        }

        self._gt_tool_call = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_capital",
                    "input": {"country": "Japan"},
                },
            ],
        }
        self._gt_tool_result = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [
                        {
                            "type": "text",
                            "text": "The capital of Japan is Tokyo.",
                        },
                    ],
                },
            ],
        }

        self.gt_multiagent = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You're a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            _hist_prompt
                            + "<history>\n"
                            + _conv_text
                            + "\n</history>"
                        ),
                    },
                ],
            },
            self._gt_tool_call,
            self._gt_tool_result,
            self._gt_trailing_asst,
        ]

    # -------------------------------------------------------------------
    # AnthropicChatFormatter tests
    # -------------------------------------------------------------------

    async def test_chat_formatter(self) -> None:
        """Chat formatter produces exact output for various subsets."""
        fmt = AnthropicChatFormatter()
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

    async def test_chat_formatter_base64_image(self) -> None:
        """Base64-encoded image is formatted as Anthropic image source."""
        fmt = AnthropicChatFormatter()
        msgs = [
            Msg(
                name="user",
                content=[
                    TextBlock(type="text", text="What's in this image?"),
                    DataBlock(
                        source=Base64Source(
                            type="base64",
                            data=self.image_b64,
                            media_type="image/png",
                        ),
                    ),
                ],
                role="user",
            ),
        ]
        res = await fmt.format(msgs)
        self.assertListEqual(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_b64,
                            },
                        },
                    ],
                },
            ],
            res,
        )

    async def test_chat_formatter_thinking_preserved(self) -> None:
        """ThinkingBlock is passed back as a thinking content block."""
        fmt = AnthropicChatFormatter()
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
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "inner thoughts",
                            "signature": "",
                        },
                        {"type": "text", "text": "reply"},
                    ],
                },
            ],
        )

    async def test_chat_formatter_tool_result_role_forced_to_user(
        self,
    ) -> None:
        """Anthropic forces tool_result messages to role='user'."""
        fmt = AnthropicChatFormatter()
        res = await fmt.format(
            [*self.msgs_system, *self.msgs_conversation, *self.msgs_tools],
        )
        tool_result_roles = [
            m["role"]
            for m in res
            if any(
                b.get("type") == "tool_result"
                for b in (m.get("content") or [])
            )
        ]
        self.assertListEqual(tool_result_roles, ["user"])

    async def test_chat_formatter_tool_result_with_image(self) -> None:
        """Tool result containing an image DataBlock inlines the image in the
        tool_result content without crashing on TextBlock system-reminders."""
        fmt = AnthropicChatFormatter()
        msgs = [
            Msg(
                name="assistant",
                content=[
                    ToolCallBlock(
                        id="call_img",
                        name="get_chart",
                        input="{}",
                    ),
                    ToolResultBlock(
                        id="call_img",
                        name="get_chart",
                        output=[
                            TextBlock(type="text", text="Here is the chart."),
                            DataBlock(
                                source=Base64Source(
                                    data=self.image_b64,
                                    media_type="image/png",
                                ),
                            ),
                        ],
                        state=ToolResultState.SUCCESS,
                    ),
                    TextBlock(
                        type="text",
                        text="Here is the chart analysis.",
                    ),
                ],
                role="assistant",
            ),
        ]
        res = await fmt.format(msgs)
        self.assertListEqual(
            res,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_img",
                            "name": "get_chart",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_img",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Here is the chart.",
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": self.image_b64,
                                    },
                                },
                            ],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the chart analysis.",
                        },
                    ],
                },
            ],
        )

    # -------------------------------------------------------------------
    # AnthropicMultiAgentFormatter tests
    # -------------------------------------------------------------------

    async def test_multiagent_formatter(self) -> None:
        """MultiAgent formatter produces exact output for various subsets."""
        fmt = AnthropicMultiAgentFormatter()
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

        # System + tools (no conversation)
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
        fmt = AnthropicChatFormatter()
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
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me think about that."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Remember to be concise."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here is my answer."},
                    ],
                },
            ],
        )
