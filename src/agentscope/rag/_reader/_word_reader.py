# -*- coding: utf-8 -*-
"""The Word reader to read and chunk Word documents."""
import hashlib
from typing import Literal

from ._reader_base import ReaderBase
from ._text_reader import TextReader
from .._document import Document


class WordReader(ReaderBase):
    """The Word reader that splits text into chunks by a fixed chunk size."""

    def __init__(
        self,
        chunk_size: int = 512,
        split_by: Literal["char", "sentence", "paragraph"] = "sentence",
    ) -> None:
        """Initialize the Word reader.

        Args:
            chunk_size (`int`, default to 512):
                The size of each chunk, in number of characters.
            split_by (`Literal["char", "sentence", "paragraph"]`, default to \
            "sentence"):
                The unit to split the text, can be "char", "sentence", or
                "paragraph". The "sentence" option is implemented using the
                "nltk" library, which only supports English text.
        """
        if chunk_size <= 0:
            raise ValueError(
                f"The chunk_size must be positive, got {chunk_size}",
            )

        if split_by not in ["char", "sentence", "paragraph"]:
            raise ValueError(
                "The split_by must be one of 'char', 'sentence' or "
                f"'paragraph', got {split_by}",
            )

        self.chunk_size = chunk_size
        self.split_by = split_by

        # To avoid code duplication, we use TextReader to do the chunking.
        self._text_reader = TextReader(
            self.chunk_size,
            self.split_by,
        )

    async def __call__(
        self,
        word_path: str,
    ) -> list[Document]:
        """Read a Word document, split it into chunks, and return a list of
        Document objects.

        Args:
            word_path (`str`):
                The input Word document file path (.docx file).
        """
        try:
            from docx import Document as DocxDocument
        except ImportError as e:
            raise ImportError(
                "Please install python-docx to use the Word reader. "
                "You can install it by `pip install python-docx`.",
            ) from e

        # Load the Word document
        doc = DocxDocument(word_path)

        # Extract text from all paragraphs
        gather_texts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                gather_texts.append(text)

        # Extract text from tables if any
        for table in doc.tables:
            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    gather_texts.append(" | ".join(row_texts))

        # Join all text with double newlines to separate paragraphs
        full_text = "\n\n".join(gather_texts)

        if not full_text.strip():
            # If no text content found, return empty list
            return []

        # Generate document ID
        doc_id = self.get_doc_id(word_path)

        # Use TextReader to split the text into chunks
        docs = await self._text_reader(full_text)

        # Update document IDs to match the Word file
        for doc in docs:
            doc.id = doc_id

        return docs

    def get_doc_id(self, word_path: str) -> str:
        """Get the document ID. This function can be used to check if the
        doc_id already exists in the knowledge base."""
        return hashlib.sha256(word_path.encode("utf-8")).hexdigest()
