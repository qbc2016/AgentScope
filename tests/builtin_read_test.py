# -*- coding: utf-8 -*-
"""Read tool test case."""
import base64
import os
import tempfile
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.tool import ToolChunk, Read
from agentscope.permission import (
    PermissionContext,
    PermissionBehavior,
    PermissionRule,
)
from agentscope.message import TextBlock, DataBlock, Base64Source


# pylint: disable=too-many-public-methods
class ReadToolTest(IsolatedAsyncioTestCase):
    """The read tool test case."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.read_tool = Read()
        # Create a temporary file for testing
        self.temp_file = (
            tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                mode="w",
                delete=False,
                suffix=".txt",
            )
        )
        # Write multiple lines
        for i in range(1, 11):
            self.temp_file.write(f"Line {i}\n")
        self.temp_file.close()

    async def asyncTearDown(self) -> None:
        """Clean up temporary files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    async def test_tool_properties(self) -> None:
        """Test read tool properties."""
        self.assertEqual(self.read_tool.name, "Read")
        self.assertIsInstance(self.read_tool.description, str)
        self.assertEqual(
            set(self.read_tool.input_schema["properties"]),
            {"file_path", "offset", "limit", "pages"},
        )
        self.assertEqual(
            self.read_tool.input_schema["required"],
            ["file_path"],
        )
        self.assertIn("image/png, image/jpeg", self.read_tool.description)
        self.assertFalse(self.read_tool.is_mcp)
        self.assertTrue(self.read_tool.is_read_only)
        self.assertTrue(self.read_tool.is_concurrency_safe)

    async def test_check_permissions(self) -> None:
        """Test read tool permission checking."""
        context = PermissionContext()
        tool_input = {"file_path": "/tmp/test.txt"}
        decision = await self.read_tool.check_permissions(tool_input, context)

        # Read/Glob/Grep are read-only, return PASSTHROUGH
        self.assertEqual(decision.behavior, PermissionBehavior.PASSTHROUGH)

    async def test_simple_read(self) -> None:
        """Test simple file reading."""
        chunk = await self.read_tool(file_path=self.temp_file.name)

        self.assertIsInstance(chunk, ToolChunk)
        self.assertEqual(chunk.state, "running")
        self.assertEqual(len(chunk.content), 1)
        self.assertIsInstance(chunk.content[0], TextBlock)

        content = chunk.content[0].text
        # Should contain all lines with line numbers
        self.assertIn("Line 1", content)
        self.assertIn("Line 10", content)

    async def test_read_with_offset(self) -> None:
        """Test reading with offset."""
        chunk = await self.read_tool(
            file_path=self.temp_file.name,
            offset=5,
        )

        self.assertEqual(chunk.state, "running")
        content = chunk.content[0].text

        # Should start from line 5
        self.assertIn("Line 5", content)
        # Line 1 should not appear (but Line 10 contains "1",
        # so check more specifically)
        lines = content.split("\n")
        line_numbers = [
            int(line.split("\t")[0].strip()) for line in lines if line.strip()
        ]
        self.assertNotIn(1, line_numbers)
        self.assertIn(5, line_numbers)

    async def test_read_with_limit(self) -> None:
        """Test reading with limit."""
        chunk = await self.read_tool(
            file_path=self.temp_file.name,
            offset=1,
            limit=3,
        )

        self.assertEqual(chunk.state, "running")
        content = chunk.content[0].text

        # Should only read 3 lines
        self.assertIn("Line 1", content)
        self.assertIn("Line 2", content)
        self.assertIn("Line 3", content)
        self.assertNotIn("Line 4", content)

    async def test_read_nonexistent_file(self) -> None:
        """Test reading a non-existent file."""
        chunk = await self.read_tool(file_path="/nonexistent/file.txt")

        self.assertEqual(chunk.state, "error")
        self.assertIn("does not exist", chunk.content[0].text)

    async def test_read_directory(self) -> None:
        """Test reading a directory (should fail)."""
        temp_dir = tempfile.mkdtemp()
        try:
            chunk = await self.read_tool(file_path=temp_dir)

            self.assertEqual(chunk.state, "error")
            self.assertIn("directory", chunk.content[0].text.lower())
        finally:
            os.rmdir(temp_dir)

    async def test_match_rule_glob_pattern(self) -> None:
        """Test match_rule with glob patterns."""
        # Test exact match
        self.assertTrue(
            await self.read_tool.match_rule(
                "test.py",
                {"file_path": "test.py"},
            ),
        )

        # Test wildcard pattern
        self.assertTrue(
            await self.read_tool.match_rule(
                "*.py",
                {"file_path": "test.py"},
            ),
        )

        # Test directory pattern
        self.assertTrue(
            await self.read_tool.match_rule(
                "/tmp/**",
                {"file_path": "/tmp/test.py"},
            ),
        )

        # Test non-matching pattern
        self.assertFalse(
            await self.read_tool.match_rule(
                "*.txt",
                {"file_path": "test.py"},
            ),
        )

        # Test empty file_path
        self.assertFalse(
            await self.read_tool.match_rule(
                "*.py",
                {"file_path": ""},
            ),
        )

    async def test_generate_suggestions(self) -> None:
        """Test generate_suggestions for file operations."""

        # Test suggestion for file in subdirectory
        suggestions = await self.read_tool.generate_suggestions(
            {"file_path": "/tmp/project/src/main.py"},
        )

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        self.assertIsInstance(suggestions[0], PermissionRule)

        # Should suggest parent directory pattern
        suggestion_contents = [s.rule_content for s in suggestions]
        self.assertIn("/tmp/project/src/**", suggestion_contents)

        # Test suggestion for file in root
        suggestions = await self.read_tool.generate_suggestions(
            {"file_path": "/test.py"},
        )
        self.assertGreater(len(suggestions), 0)

    async def test_read_image_file_returns_data_block(self) -> None:
        """Test reading an image file returns DataBlock."""
        img_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as f:
            f.write(img_data)
            img_path = f.name

        try:
            chunk = await self.read_tool(file_path=img_path)

            self.assertIsInstance(chunk, ToolChunk)
            self.assertEqual(chunk.state, "running")
            self.assertEqual(len(chunk.content), 1)
            self.assertIsInstance(chunk.content[0], DataBlock)

            block = chunk.content[0]
            self.assertIsInstance(block.source, Base64Source)
            self.assertEqual(block.source.media_type, "image/png")
            self.assertEqual(block.name, os.path.basename(img_path))

            decoded = base64.b64decode(block.source.data)
            self.assertEqual(decoded, img_data)
        finally:
            os.unlink(img_path)

    async def test_read_jpeg_file_returns_data_block(self) -> None:
        """Test reading a JPEG file returns DataBlock."""
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as f:
            f.write(jpg_data)
            jpg_path = f.name

        try:
            chunk = await self.read_tool(file_path=jpg_path)

            self.assertEqual(chunk.state, "running")
            self.assertIsInstance(chunk.content[0], DataBlock)
            self.assertEqual(
                chunk.content[0].source.media_type,
                "image/jpeg",
            )
        finally:
            os.unlink(jpg_path)

    def _write_pdf(self, n_pages: int) -> str:
        """Write a blank PDF with ``n_pages`` pages and return its path."""
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _ in range(n_pages):
            writer.add_blank_page(width=612, height=792)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as f:
            writer.write(f)
        self.addCleanup(os.unlink, f.name)
        return f.name

    async def test_read_pdf_file(self) -> None:
        """Test reading a PDF file extracts text."""
        pdf_path = self._write_pdf(1)
        chunk = await self.read_tool(file_path=pdf_path)

        self.assertIsInstance(chunk, ToolChunk)
        self.assertEqual(chunk.state, "running")
        self.assertEqual(len(chunk.content), 1)
        self.assertIsInstance(chunk.content[0], TextBlock)
        self.assertIn("--- Page 1/1 ---", chunk.content[0].text)

    async def test_read_pdf_with_pages_param(self) -> None:
        """Test reading a page range and a single page from a PDF."""
        pdf_path = self._write_pdf(5)

        chunk = await self.read_tool(file_path=pdf_path, pages="2-3")
        self.assertEqual(chunk.state, "running")
        text = chunk.content[0].text
        self.assertNotIn("--- Page 1/5 ---", text)
        self.assertIn("--- Page 2/5 ---", text)
        self.assertIn("--- Page 3/5 ---", text)
        self.assertNotIn("--- Page 4/5 ---", text)

        chunk = await self.read_tool(file_path=pdf_path, pages="4")
        self.assertEqual(chunk.state, "running")
        text = chunk.content[0].text
        self.assertIn("--- Page 4/5 ---", text)
        self.assertNotIn("--- Page 3/5 ---", text)
        self.assertNotIn("--- Page 5/5 ---", text)

        # A range past the end is clipped to the last page.
        chunk = await self.read_tool(file_path=pdf_path, pages="4-99")
        self.assertEqual(chunk.state, "running")
        self.assertIn("--- Page 5/5 ---", chunk.content[0].text)

    async def test_read_pdf_invalid_pages(self) -> None:
        """Test malformed or out-of-range pages return an error."""
        pdf_path = self._write_pdf(2)

        for pages in ["abc", "0", "3", "2-1", "1-2-3", ""]:
            chunk = await self.read_tool(file_path=pdf_path, pages=pages)
            self.assertEqual(chunk.state, "error", pages)
            self.assertIn("Invalid pages", chunk.content[0].text)

    async def test_read_large_pdf_requires_pages(self) -> None:
        """Test PDFs over 10 pages must be read with the pages parameter."""
        pdf_path = self._write_pdf(11)

        chunk = await self.read_tool(file_path=pdf_path)
        self.assertEqual(chunk.state, "error")
        self.assertIn("11 pages", chunk.content[0].text)
        self.assertIn("pages parameter", chunk.content[0].text)

        chunk = await self.read_tool(file_path=pdf_path, pages="10-11")
        self.assertEqual(chunk.state, "running")
        self.assertIn("--- Page 11/11 ---", chunk.content[0].text)

    async def test_read_pdf_max_pages_per_request(self) -> None:
        """Test a single read returns at most 20 pages."""
        pdf_path = self._write_pdf(25)

        chunk = await self.read_tool(file_path=pdf_path, pages="1-21")
        self.assertEqual(chunk.state, "error")
        self.assertIn("at most 20 pages", chunk.content[0].text)

        chunk = await self.read_tool(file_path=pdf_path, pages="1-20")
        self.assertEqual(chunk.state, "running")
        self.assertIn("--- Page 20/25 ---", chunk.content[0].text)
        self.assertNotIn("--- Page 21/25 ---", chunk.content[0].text)

    async def test_read_unknown_extension_as_text(self) -> None:
        """Test that unknown extensions are read as text."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".xyz",
        ) as f:
            f.write("hello world\n")
            path = f.name

        try:
            chunk = await self.read_tool(file_path=path)

            self.assertEqual(chunk.state, "running")
            self.assertIsInstance(chunk.content[0], TextBlock)
            self.assertIn("hello world", chunk.content[0].text)
        finally:
            os.unlink(path)

    async def test_read_image_unsupported_type(self) -> None:
        """Test images outside ``input_types`` return an error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as f:
            f.write(b"BM" + b"\x00" * 100)
        self.addCleanup(os.unlink, f.name)

        # image/bmp is not in the default image types.
        chunk = await self.read_tool(file_path=f.name)
        self.assertEqual(chunk.state, "error")
        self.assertIn(
            "Unsupported image type image/bmp",
            chunk.content[0].text,
        )
        self.assertIn("image/png", chunk.content[0].text)

        # Model card style input_types, non-image entries ignored.
        tool = Read(input_types=["text/plain", "image/bmp"])
        chunk = await tool(file_path=f.name)
        self.assertEqual(chunk.state, "running")
        self.assertEqual(chunk.content[0].source.media_type, "image/bmp")
        self.assertIn("image/bmp", tool.description)
        self.assertNotIn("text/plain", tool.description)

        # Glob patterns are accepted.
        tool = Read(input_types=["image/*"])
        chunk = await tool(file_path=f.name)
        self.assertEqual(chunk.state, "running")

        # A supported image is rejected once it's excluded.
        tool = Read(input_types=["image/png"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
        self.addCleanup(os.unlink, f.name)
        chunk = await tool(file_path=f.name)
        self.assertEqual(chunk.state, "error")
        self.assertIn("only image/png are supported", chunk.content[0].text)
