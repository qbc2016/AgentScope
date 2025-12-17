# -*- coding: utf-8 -*-
"""The formatter module."""

from abc import abstractmethod
from typing import Any, List, Tuple, Sequence

from .._utils._common import _save_base64_data
from ..message import Msg, AudioBlock, ImageBlock, TextBlock, VideoBlock


class FormatterBase:
    """The base class for formatters."""

    @abstractmethod
    async def format(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Format the Msg objects to a list of dictionaries that satisfy the
        API requirements."""

    @staticmethod
    def assert_list_of_msgs(msgs: list[Msg]) -> None:
        """Assert that the input is a list of Msg objects.

        Args:
            msgs (`list[Msg]`):
                A list of Msg objects to be validated.
        """
        if not isinstance(msgs, list):
            raise TypeError("Input must be a list of Msg objects.")

        for msg in msgs:
            if not isinstance(msg, Msg):
                raise TypeError(
                    f"Expected Msg object, got {type(msg)} instead.",
                )

    @staticmethod
    def convert_tool_result_to_string(
        output: str | List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
    ) -> tuple[
        str,
        Sequence[
            Tuple[
                str,
                ImageBlock | AudioBlock | TextBlock | VideoBlock,
            ]
        ],
    ]:
        """Turn the tool result list into a textual output to be compatible
        with the LLM API that doesn't support multimodal data in the tool
        result.

        For URL-based images, the URL is included in the list. For
        base64-encoded images, the local file path where the image is saved
        is included in the returned list.

        Args:
            output (`str | List[TextBlock | ImageBlock | AudioBlock | \
            VideoBlock]`):
                The output of the tool response, including text and multimodal
                data like images and audio.

        Returns:
            `tuple[str, list[Tuple[str, ImageBlock | AudioBlock | VideoBlock \
            TextBlock]]]`:
                A tuple containing the textual representation of the tool
                result and a list of tuples. The first element of each tuple
                is the local file path or URL of the multimodal data, and the
                second element is the corresponding block.
        """

        if isinstance(output, str):
            return output, []

        textual_output = []
        multimodal_data = []
        for block in output:
            assert isinstance(block, dict) and "type" in block, (
                f"Invalid block: {block}, a TextBlock, ImageBlock, "
                f"AudioBlock, or VideoBlock is expected."
            )
            if block["type"] == "text":
                textual_output.append(block["text"])

            elif block["type"] in ["image", "audio", "video"]:
                assert "source" in block, (
                    f"Invalid {block['type']} block: {block}, 'source' key "
                    "is required."
                )
                source = block["source"]
                # Save the image locally and return the file path
                if source["type"] == "url":
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {source['url']}",
                    )

                    path_multimodal_file = source["url"]

                elif source["type"] == "base64":
                    path_multimodal_file = _save_base64_data(
                        source["media_type"],
                        source["data"],
                    )
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {path_multimodal_file}",
                    )

                else:
                    raise ValueError(
                        f"Invalid image source: {block['source']}, "
                        "expected 'url' or 'base64'.",
                    )

                multimodal_data.append(
                    (path_multimodal_file, block),
                )

            else:
                raise ValueError(
                    f"Unsupported block type: {block['type']}, "
                    "expected 'text', 'image', 'audio', or 'video'.",
                )

        if len(textual_output) == 1:
            return textual_output[0], multimodal_data

        else:
            return "\n".join("- " + _ for _ in textual_output), multimodal_data

    @staticmethod
    def reorder_tool_messages(msgs: list[Msg]) -> list[Msg]:
        """Reorder messages to ensure tool_result immediately follows its
        corresponding tool_use.

        Args:
            msgs (`list[Msg]`):
                The input messages to be reordered.

        Returns:
            `list[Msg]`:
                The reordered messages with tool_result following tool_use.
        """
        # Build a mapping from tool_id to message index
        tool_result_map: dict[str, int] = {}
        for i, msg in enumerate(msgs):
            for block in msg.get_content_blocks():
                if block.get("type") == "tool_result":
                    tool_id = block["id"]
                    if tool_id:
                        tool_result_map[tool_id] = i

        result = []
        processed: set[int] = set()

        for i, msg in enumerate(msgs):
            if i in processed:
                continue

            result.append(msg)
            processed.add(i)

            # Find corresponding tool_result message indices for all
            # tool_use ids
            result_indices: set[int] = set()
            for block in msg.get_content_blocks():
                if block.get("type") == "tool_use":
                    tool_id = block["id"]
                    if tool_id and tool_id in tool_result_map:
                        result_indices.add(tool_result_map[tool_id])

            # Append corresponding tool_result messages in original order
            for j in sorted(result_indices):
                if j not in processed:
                    result.append(msgs[j])
                    processed.add(j)

        return result
