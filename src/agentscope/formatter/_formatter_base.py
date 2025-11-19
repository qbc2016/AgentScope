# -*- coding: utf-8 -*-
"""The formatter module."""

from abc import abstractmethod
from typing import Any, List

from .._utils._common import _save_base64_data
from ..message import Msg, AudioBlock, ImageBlock, TextBlock


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
        output: str | List[TextBlock | ImageBlock | AudioBlock],
    ) -> tuple[str, list]:
        """Turn the tool result list into a textual output to be compatible
        with the LLM API that doesn't support multimodal data.

        Args:
            output (`str | List[TextBlock | ImageBlock | AudioBlock]`):
                The output of the tool response, including text and multimodal
                data like images and audio.

        Returns:
            `str`:
                A string representation of the tool result, with text blocks
                concatenated and multimodal data represented by file paths
                or URLs.
        """

        if isinstance(output, str):
            return output, []

        textual_output = []
        image_paths = []
        for block in output:
            assert isinstance(block, dict) and "type" in block, (
                f"Invalid block: {block}, a TextBlock, ImageBlock, or "
                f"AudioBlock is expected."
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
                    if block["type"] == "image":
                        image_paths.append(source["url"])

                elif source["type"] == "base64":
                    path_temp_file = _save_base64_data(
                        source["media_type"],
                        source["data"],
                    )
                    textual_output.append(
                        f"The returned {block['type']} can be found "
                        f"at: {path_temp_file}",
                    )
                    if block["type"] == "image":
                        image_paths.append(path_temp_file)
                else:
                    raise ValueError(
                        f"Invalid image source: {block['source']}, "
                        "expected 'url' or 'base64'.",
                    )

            else:
                raise ValueError(
                    f"Unsupported block type: {block['type']}, "
                    "expected 'text', 'image', 'audio', or 'video'.",
                )

        if len(textual_output) == 1:
            return textual_output[0], image_paths

        else:
            return "\n".join("- " + _ for _ in textual_output), image_paths

    @staticmethod
    def _extract_image_blocks_from_tool_result(
        output: str | List[Any],
    ) -> List[ImageBlock]:
        """Extract image blocks from tool result output.

        Args:
            output (`str | List[Any]`):
                The output of the tool result, which can be a string or a list
                of content blocks.

        Returns:
            `List[ImageBlock]`:
                A list of image blocks extracted from the tool result output.
                Returns an empty list if no images are found or if output is
                a string.
        """
        image_blocks: List[ImageBlock] = []
        if isinstance(output, list):
            for block in output:
                if isinstance(block, dict) and block.get("type") == "image":
                    image_blocks.append(block)  # type: ignore[arg-type]
        return image_blocks
