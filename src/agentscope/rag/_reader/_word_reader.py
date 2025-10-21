# -*- coding: utf-8 -*-
"""The Word reader to read and chunk Word documents."""
import hashlib
from typing import Any, Literal

from ._reader_base import ReaderBase
from ._text_reader import TextReader
from .._document import Document


class WordReader(ReaderBase):
    """The Word reader that splits text into chunks by a fixed chunk size."""

    def __init__(
        self,
        chunk_size: int = 512,
        split_by: Literal["char", "sentence", "paragraph"] = "sentence",
        include_image: bool = False,
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
            include_image (`bool`, default to False):
                Whether to include image content in the document. If True,
                images will be extracted and included as text descriptions.
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
        self.include_image = include_image

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

        # Extract content in order (paragraphs, tables, and images)
        content_parts = []

        # Process all elements in document order using a more reliable approach
        # Create mappings for efficient lookup
        paragraph_elements = {p._element: p for p in doc.paragraphs}
        table_elements = {t._element: t for t in doc.tables}

        # Create a set of all drawing elements that are part of paragraphs
        processed_drawing_elements = set()
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                # Check if this run contains drawing elements
                for elem in run._element.iter():
                    if elem.tag.split("}")[-1] == "drawing":
                        processed_drawing_elements.add(elem)

        # Process elements in document order
        for element in doc.element.body:
            tag_name = self._get_tag_name(element)
            self._process_element_by_type(
                element,
                tag_name,
                paragraph_elements,
                table_elements,
                processed_drawing_elements,
                content_parts,
            )

        # Join all content with double newlines to separate different elements
        full_text = "\n\n".join(content_parts)

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

    def _extract_image_info(self, drawing_element: Any) -> str | None:
        """Extract image information from a drawing element.

        Args:
            drawing_element: The drawing element from the Word document.

        Returns:
            A string describing the image, or None if no image found.
        """
        try:
            # Look for image references in the drawing element
            for elem in drawing_element.iter():
                if elem.tag.endswith("blip"):
                    # Found an image reference
                    r_embed = elem.get(
                        "{http://schemas.openxmlformats.org/officeDocument/"
                        "2006/relationships}embed",
                    )
                    if r_embed:
                        return (
                            f"Image embedded in document (relationship "
                            f"ID: {r_embed})"
                        )
                elif elem.tag.endswith("pic"):
                    # Found a picture element
                    return "Image found in document"
                elif elem.tag.endswith("drawing"):
                    # Found a drawing element
                    return "Drawing element found in document"

            # If no specific image elements found, return a generic description
            return "Image/drawing element found in document"

        except Exception:
            # If we can't extract image info, return a generic description
            return "Image element found (details unavailable)"

    def _extract_inline_image_info(self, run_element: Any) -> str | None:
        """Extract detailed information from inline images in a run element.

        Args:
            run_element: The run element that may contain images.

        Returns:
            A string describing the image with detailed information,
            or None if no image found.
        """
        try:
            image_data = self._collect_image_data(run_element)
            return self._build_image_description(image_data)
        except Exception:
            return "Image element found (details unavailable)"

    def _collect_image_data(self, run_element: Any) -> dict:
        """Collect image data from run element."""
        relationship_ids = set()
        sizes = set()
        has_picture = False
        has_drawing = False
        image_names = set()
        alt_texts = set()

        for elem in run_element.iter():
            if "blip" in elem.tag:
                self._extract_blip_data(elem, relationship_ids)
            elif "pic" in elem.tag:
                has_picture = True
                self._extract_pic_data(elem, sizes, image_names, alt_texts)
            elif "drawing" in elem.tag:
                has_drawing = True
                self._extract_drawing_data(elem, sizes, image_names, alt_texts)
            else:
                self._extract_generic_data(elem, image_names, alt_texts)

        return {
            "relationship_ids": relationship_ids,
            "sizes": sizes,
            "has_picture": has_picture,
            "has_drawing": has_drawing,
            "image_names": image_names,
            "alt_texts": alt_texts,
        }

    def _extract_blip_data(self, elem: Any, relationship_ids: set) -> None:
        """Extract data from blip element."""
        r_embed = elem.get(
            "{http://schemas.openxmlformats.org/officeDocument"
            "/2006/relationships}embed",
        )
        if r_embed:
            relationship_ids.add(r_embed)

    def _extract_pic_data(
        self,
        elem: Any,
        sizes: set,
        image_names: set,
        alt_texts: set,
    ) -> None:
        """Extract data from picture element."""
        for child in elem.iter():
            if "ext" in child.tag:
                self._extract_size_data(child, sizes)
            elif "name" in child.tag and child.text:
                image_names.add(child.text)
            elif "alt" in child.tag and child.text:
                alt_texts.add(child.text)

    def _extract_drawing_data(
        self,
        elem: Any,
        sizes: set,
        image_names: set,
        alt_texts: set,
    ) -> None:
        """Extract data from drawing element."""
        for child in elem.iter():
            if "ext" in child.tag:
                self._extract_size_data(child, sizes)
            elif "name" in child.tag and child.text:
                image_names.add(child.text)
            elif "alt" in child.tag and child.text:
                alt_texts.add(child.text)

    def _extract_generic_data(
        self,
        elem: Any,
        image_names: set,
        alt_texts: set,
    ) -> None:
        """Extract generic data from element."""
        if "alt" in elem.tag and elem.text:
            alt_texts.add(elem.text)
        elif "name" in elem.tag and elem.text:
            image_names.add(elem.text)

    def _extract_size_data(self, child: Any, sizes: set) -> None:
        """Extract size data from child element."""
        cx = child.get("cx")
        cy = child.get("cy")
        if cx and cy:
            try:
                cx_int = int(cx) if cx else 0
                cy_int = int(cy) if cy else 0

                if cx_int > 0 and cy_int > 0:
                    width_px = int(cx_int / 914400 * 96)
                    height_px = int(cy_int / 914400 * 96)
                    sizes.add(f"{width_px}x{height_px}px")
            except (ValueError, TypeError):
                pass

    def _build_image_description(self, image_data: dict) -> str | None:
        """Build image description from collected data."""
        # Check if we have any meaningful data
        # (not just False booleans or empty sets)
        has_meaningful_data = (
            image_data["relationship_ids"]
            or image_data["sizes"]
            or image_data["image_names"]
            or image_data["alt_texts"]
            or image_data["has_picture"]
            or image_data["has_drawing"]
        )

        if not has_meaningful_data:
            return None

        parts = []
        if image_data["relationship_ids"]:
            parts.append(f"ID: {', '.join(image_data['relationship_ids'])}")
        if image_data["sizes"]:
            parts.append(f"Size: {', '.join(image_data['sizes'])}")
        if image_data["image_names"]:
            parts.append(f"Name: {', '.join(image_data['image_names'])}")
        if image_data["alt_texts"]:
            parts.append(f"Alt: {', '.join(image_data['alt_texts'])}")
        if image_data["has_picture"]:
            parts.append("Type: Picture")
        elif image_data["has_drawing"]:
            parts.append("Type: Drawing")

        return f"Inline image ({', '.join(parts)})"

    def _get_tag_name(self, element: Any) -> str:
        """Get tag name from element."""
        return (
            element.tag.split("}")[-1] if "}" in element.tag else element.tag
        )

    def _process_element_by_type(
        self,
        element: Any,
        tag_name: str,
        paragraph_elements: dict,
        table_elements: dict,
        processed_drawing_elements: set,
        content_parts: list,
    ) -> None:
        """Process element based on its type."""
        if tag_name == "p":
            self._process_paragraph_element(
                element,
                paragraph_elements,
                content_parts,
            )
        elif tag_name == "tbl":
            self._process_table_element(element, table_elements, content_parts)
        elif tag_name == "drawing" and self.include_image:
            self._process_drawing_element(
                element,
                processed_drawing_elements,
                content_parts,
            )

    def _process_paragraph_element(
        self,
        element: Any,
        paragraph_elements: dict,
        content_parts: list,
    ) -> None:
        """Process paragraph element."""
        paragraph = paragraph_elements.get(element)
        if not paragraph:
            return

        text = paragraph.text.strip()
        if text:
            content_parts.append(text)

        if self.include_image:
            self._process_paragraph_images(paragraph, content_parts)

    def _process_paragraph_images(
        self,
        paragraph: Any,
        content_parts: list,
    ) -> None:
        """Process images in paragraph."""
        paragraph_images = []
        for run in paragraph.runs:
            # pylint: disable=protected-access
            image_info = self._extract_inline_image_info(run._element)
            if image_info:
                paragraph_images.append(image_info)

        if paragraph_images:
            unique_images = list(dict.fromkeys(paragraph_images))
            for img in unique_images:
                content_parts.append(f"IMAGE: {img}")

    def _process_table_element(
        self,
        element: Any,
        table_elements: dict,
        content_parts: list,
    ) -> None:
        """Process table element."""
        table = table_elements.get(element)
        if not table:
            return

        table_texts = []
        for row in table.rows:
            row_texts = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_texts.append(cell_text)
            if row_texts:
                table_texts.append(" | ".join(row_texts))

        if table_texts:
            table_content = "TABLE:\n" + "\n".join(table_texts)
            content_parts.append(table_content)

    def _process_drawing_element(
        self,
        element: Any,
        processed_drawing_elements: set,
        content_parts: list,
    ) -> None:
        """Process standalone drawing element."""
        if element not in processed_drawing_elements:
            image_info = self._extract_image_info(element)
            if image_info:
                content_parts.append(f"IMAGE: {image_info}")

    def get_doc_id(self, word_path: str) -> str:
        """Get the document ID. This function can be used to check if the
        doc_id already exists in the knowledge base."""
        return hashlib.sha256(word_path.encode("utf-8")).hexdigest()
