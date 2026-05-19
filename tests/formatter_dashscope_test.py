# -*- coding: utf-8 -*-
"""Comprehensive formatter unit tests for DashScopeChatFormatter and
DashScopeMultiAgentFormatter, following the reference test style with exact
ground-truth comparisons.
"""
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
)
from agentscope.message import (
    Msg,
    TextBlock,
    DataBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    Base64Source,
    URLSource,
    ThinkingBlock,
)


# A fixed short-uuid used to make promote-to-multimodal tests deterministic.
_FIXED_ID = "TESTID1234567"


class TestDashScopeFormatter(IsolatedAsyncioTestCase):
    """Comprehensive tests for DashScope Chat and MultiAgent formatters."""

    async def asyncSetUp(self) -> None:
        """Set up shared message fixtures and expected ground-truth dicts."""
        # --- URL strings ---
        # Normalise through URLSource so that pydantic URL normalisation is
        # applied consistently to both input and expected values.
        _img_src = URLSource(
            url="https://example.com/image.png",
            media_type="image/png",
        )
        _aud_src = URLSource(
            url="https://example.com/audio.mp3",
            media_type="audio/mpeg",
        )
        self.image_url = str(_img_src.url)
        self.audio_url = str(_aud_src.url)

        # --- Base64 fixtures ---
        self.image_b64 = "ZmFrZSBpbWFnZSBkYXRh"
        self.image_data_uri = f"data:image/png;base64,{self.image_b64}"

        # ---------------------------------------------------------------
        # Message fixtures
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
                        source=URLSource(
                            url=self.image_url,
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
                content=[
                    TextBlock(
                        type="text",
                        text="What is the capital of Germany?",
                    ),
                    DataBlock(
                        source=URLSource(
                            url=self.audio_url,
                            media_type="audio/mpeg",
                        ),
                    ),
                ],
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

        # Messages with ToolResultBlock must use role="assistant" because the
        # system-role validator rejects non-text blocks.
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
        # Ground truth: DashScopeChatFormatter (OpenAI-compatible format)
        #   - Content blocks use {"type": "text", "text": ...} format.
        #   - Images use {"type": "image_url", "image_url": {"url": ...}}.
        #   - Audio uses {"type": "input_audio", "input_audio": {...}}.
        #   - Tool-result content is a plain string.
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
                    {
                        "type": "image_url",
                        "image_url": {"url": self.image_url},
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
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": self.audio_url,
                            "format": "mpeg",
                        },
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
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_capital",
                            "arguments": '{"country": "Japan"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "The capital of Japan is Tokyo.",
                "name": "get_capital",
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
        # Ground truth: DashScopeMultiAgentFormatter
        #   - System content is a plain string (via get_text_content()).
        #   - Conversation history is collapsed into a single user message.
        # ---------------------------------------------------------------
        _hist_prompt = (
            DashScopeMultiAgentFormatter().conversation_history_prompt
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
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_capital",
                        "arguments": '{"country": "Japan"}',
                    },
                },
            ],
        }
        self._gt_tool_result = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "The capital of Japan is Tokyo.",
            "name": "get_capital",
        }

        self.gt_multiagent = [
            {
                "role": "system",
                "content": "You're a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            _hist_prompt + "<history>\n"
                            "user: What is the capital of France?\n"
                            "assistant: The capital of France is Paris.\n"
                            "user: What is the capital of Germany?\n"
                            "assistant: The capital of Germany is Berlin.\n"
                            "user: What is the capital of Japan?\n"
                            "</history>"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.image_url},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": self.audio_url,
                            "format": "mpeg",
                        },
                    },
                ],
            },
            self._gt_tool_call,
            self._gt_tool_result,
            self._gt_trailing_asst,
        ]

    # -------------------------------------------------------------------
    # DashScopeChatFormatter tests
    # -------------------------------------------------------------------

    async def test_chat_formatter(self) -> None:
        """Chat formatter produces exact output for various subsets."""
        fmt = DashScopeChatFormatter()

        # Full history
        res = await fmt.format(
            [*self.msgs_system, *self.msgs_conversation, *self.msgs_tools],
        )
        self.maxDiff = None
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
        """Base64-encoded image is inlined as a data URI."""
        fmt = DashScopeChatFormatter()
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
                            "type": "image_url",
                            "image_url": {"url": self.image_data_uri},
                        },
                    ],
                },
            ],
            res,
        )

    @patch(
        "agentscope.formatter._formatter_base.shortuuid.uuid",
        return_value=_FIXED_ID,
    )
    async def test_chat_formatter_url_image_in_tool_result(
        self,
        _mock_uuid: object,
    ) -> None:
        """URL images in tool results are promoted to a follow-up user message.

        The textual part of the tool result contains a system-reminder with a
        unique identifier; the identifier is mocked to be deterministic.
        """
        fmt = DashScopeChatFormatter()
        msgs = [
            Msg(
                name="assistant",
                content=[
                    ToolCallBlock(
                        id="call_img",
                        name="get_map",
                        input='{"city": "Tokyo"}',
                    ),
                    ToolResultBlock(
                        id="call_img",
                        name="get_map",
                        output=[
                            TextBlock(
                                type="text",
                                text="Here is the map.",
                            ),
                            DataBlock(
                                source=URLSource(
                                    url=self.image_url,
                                    media_type="image/png",
                                ),
                            ),
                        ],
                        state=ToolResultState.SUCCESS,
                    ),
                    TextBlock(
                        type="text",
                        text="Here is the map of Tokyo.",
                    ),
                ],
                role="assistant",
            ),
        ]
        res = await fmt.format(msgs)

        expected_tool_content = (
            "Here is the map.\n"
            f"<system-reminder>A(n) image file is returned "
            f"and will be presented to you with the identifier "
            f"[{_FIXED_ID}].</system-reminder>"
        )
        self.assertListEqual(
            res,
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_img",
                            "type": "function",
                            "function": {
                                "name": "get_map",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_img",
                    "content": expected_tool_content,
                    "name": "get_map",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>The multimodal data "
                                "and their identifiers are listed as "
                                "follows:"
                            ),
                        },
                        {
                            "type": "text",
                            "text": f"- {_FIXED_ID} (image file): ",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_url},
                        },
                        {
                            "type": "text",
                            "text": "</system-reminder>",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the map of Tokyo.",
                        },
                    ],
                },
            ],
        )

    async def test_chat_formatter_thinking_dropped_without_flag(self) -> None:
        """ThinkingBlock is silently dropped when application/x-thinking is
        absent from input_types."""
        fmt = DashScopeChatFormatter()
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
                    "content": [{"type": "text", "text": "reply"}],
                },
            ],
        )

    async def test_chat_formatter_thinking_becomes_reasoning_content(
        self,
    ) -> None:
        """ThinkingBlock becomes reasoning_content when application/x-thinking
        is in input_types."""
        fmt = DashScopeChatFormatter(
            input_types=["text/plain", "application/x-thinking"],
        )
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
                    "content": [{"type": "text", "text": "reply"}],
                    "reasoning_content": "inner thoughts",
                },
            ],
        )

    async def test_chat_formatter_multiple_thinking_blocks_joined(
        self,
    ) -> None:
        """Multiple ThinkingBlocks are joined with a newline into a single
        reasoning_content field."""
        fmt = DashScopeChatFormatter(
            input_types=["text/plain", "application/x-thinking"],
        )
        msgs = [
            Msg(
                name="assistant",
                content=[
                    ThinkingBlock(thinking="part one"),
                    ThinkingBlock(thinking="part two"),
                    TextBlock(type="text", text="answer"),
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
                    "content": [{"type": "text", "text": "answer"}],
                    "reasoning_content": "part one\npart two",
                },
            ],
        )

    # -------------------------------------------------------------------
    # DashScopeMultiAgentFormatter tests
    # -------------------------------------------------------------------

    async def test_multiagent_formatter(self) -> None:
        """MultiAgent formatter produces exact output for various subsets."""
        fmt = DashScopeMultiAgentFormatter()
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

        # Tools only — no prior agent_message group, so the trailing assistant
        # is formatted with is_first=True (includes the full history prompt).
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

    async def test_multiagent_formatter_thinking_in_tool_sequence(
        self,
    ) -> None:
        """ThinkingBlocks inside a tool sequence are forwarded as
        reasoning_content when application/x-thinking is in input_types."""
        fmt = DashScopeMultiAgentFormatter(
            input_types=["text/plain", "application/x-thinking"],
        )
        tc = ToolCallBlock(
            id="call_1",
            name="get_capital",
            input='{"country": "Japan"}',
        )
        tr = ToolResultBlock(
            id="call_1",
            name="get_capital",
            output=[TextBlock(type="text", text="Tokyo")],
            state=ToolResultState.SUCCESS,
        )
        msgs = [
            Msg(
                name="assistant",
                content=[ThinkingBlock(thinking="Need to check"), tc, tr],
                role="assistant",
            ),
        ]
        res = await fmt.format(msgs)
        asst_msgs = [m for m in res if m.get("role") == "assistant"]
        self.assertListEqual(
            [m["reasoning_content"] for m in asst_msgs],
            ["Need to check"],
        )
