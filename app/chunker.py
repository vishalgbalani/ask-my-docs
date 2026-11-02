"""Text extraction and chunking logic."""

import io
from typing import List, Optional
from pydantic import BaseModel
import PyPDF2
import docx


class Chunk(BaseModel):
    chunk_id: int
    text: str
    page_number: Optional[int] = None
    position: str  # "beginning" / "middle" / "end"


def extract_text_from_pdf(file_bytes: bytes) -> tuple[list[str], int]:
    """Extract text from PDF, returning (list of page texts, page count)."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages, len(reader.pages)


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if newline_pos != -1:
                end = newline_pos + 2
            else:
                # Look for sentence break
                period_pos = text.rfind(". ", start + chunk_size // 2, end)
                if period_pos != -1:
                    end = period_pos + 2

        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0

    # Remove empty chunks
    return [c for c in chunks if c.strip()]


def chunk_pdf(file_bytes: bytes) -> tuple[List[Chunk], int]:
    """Extract and chunk a PDF, preserving page numbers."""
    pages, page_count = extract_text_from_pdf(file_bytes)

    if not any(p.strip() for p in pages):
        raise ValueError("no_extractable_text")

    # Build chunks with page tracking
    all_chunks: List[Chunk] = []
    chunk_id = 0

    for page_num, page_text in enumerate(pages, 1):
        if not page_text.strip():
            continue
        page_chunks = chunk_text(page_text, chunk_size=2000, overlap=400)
        for text in page_chunks:
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text,
                page_number=page_num,
                position="beginning"  # will be set after
            ))
            chunk_id += 1

    # Set positions
    total = len(all_chunks)
    for i, chunk in enumerate(all_chunks):
        if total <= 1:
            chunk.position = "beginning"
        elif i < total * 0.2:
            chunk.position = "beginning"
        elif i > total * 0.8:
            chunk.position = "end"
        else:
            chunk.position = "middle"

    return all_chunks, page_count


def extract_text_from_docx(file_bytes: bytes) -> list[str]:
    """Extract text from a .docx file, including tables."""
    doc = docx.Document(io.BytesIO(file_bytes))
    content: list[str] = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]  # strip namespace

        if tag == "p":
            # Paragraph
            text = element.text or ""
            # Also grab text from runs inside the paragraph
            for node in element.iter():
                if node.tag.endswith("}t") and node.text:
                    pass  # already captured by element.text
            # Use the python-docx Paragraph wrapper for clean text
            for para in doc.paragraphs:
                if para._element is element:
                    text = para.text
                    break
            if text.strip():
                content.append(text)

        elif tag == "tbl":
            # Table — convert rows to readable text
            for table in doc.tables:
                if table._tbl is element:
                    rows_text = []
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        rows_text.append(" | ".join(cells))
                    if rows_text:
                        content.append("\n".join(rows_text))
                    break

    return content


def chunk_docx(file_bytes: bytes) -> List[Chunk]:
    """Extract and chunk a .docx file."""
    paragraphs = extract_text_from_docx(file_bytes)
    if not paragraphs:
        raise ValueError("no_extractable_text")

    full_text = "\n\n".join(paragraphs)
    raw_chunks = chunk_text(full_text, chunk_size=2000, overlap=400)

    chunks: List[Chunk] = []
    total = len(raw_chunks)
    for i, t in enumerate(raw_chunks):
        if total <= 1:
            pos = "beginning"
        elif i < total * 0.2:
            pos = "beginning"
        elif i > total * 0.8:
            pos = "end"
        else:
            pos = "middle"
        chunks.append(Chunk(chunk_id=i, text=t, page_number=None, position=pos))

    return chunks


def chunk_plain_text(text: str, title: str = "Pasted Document") -> List[Chunk]:
    """Chunk plain text input."""
    if len(text.strip()) < 100:
        raise ValueError("text_too_short")

    raw_chunks = chunk_text(text, chunk_size=2000, overlap=400)
    chunks: List[Chunk] = []

    total = len(raw_chunks)
    for i, t in enumerate(raw_chunks):
        if total <= 1:
            pos = "beginning"
        elif i < total * 0.2:
            pos = "beginning"
        elif i > total * 0.8:
            pos = "end"
        else:
            pos = "middle"

        chunks.append(Chunk(
            chunk_id=i,
            text=t,
            page_number=None,
            position=pos
        ))

    return chunks
