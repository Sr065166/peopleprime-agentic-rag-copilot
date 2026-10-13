import hashlib
import io
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from docx import Document as DocxReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class DocumentChunk:
    def __init__(self, text: str, metadata: dict[str, Any], chunk_id: str | None = None):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id or str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to avoid directory traversal or invalid characters."""
    name = Path(filename).name
    # Remove control characters and problematic path symbols
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    return sanitized if sanitized else f"document_{uuid.uuid4().hex[:8]}"


def compute_document_id(filename: str, content_bytes: bytes) -> str:
    """Generate deterministic document ID based on filename and hash of content."""
    sha256 = hashlib.sha256(content_bytes).hexdigest()[:16]
    clean_name = Path(filename).stem
    return f"{clean_name}_{sha256}"


def parse_txt_markdown(content_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse UTF-8 or Latin-1 text/markdown content."""
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("latin-1", errors="replace")

    text = text.strip()
    if not text:
        return []
    return [{"text": text, "page_number": None}]


def parse_pdf(content_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse PDF file bytes extracting text per page."""
    reader = PdfReader(io.BytesIO(content_bytes))
    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append({"text": page_text, "page_number": idx})
    return pages


def parse_docx(content_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse DOCX file bytes extracting text from paragraphs."""
    doc = DocxReader(io.BytesIO(content_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs).strip()
    if not full_text:
        return []
    return [{"text": full_text, "page_number": None}]


def extract_document_pages(filename: str, content_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract raw text blocks and page references based on file extension.
    Raises ValueError for unsupported formats or empty document.
    """
    if not content_bytes or len(content_bytes.strip()) == 0:
        raise ValueError(f"Document '{filename}' is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    try:
        if suffix in {".txt", ".md"}:
            sections = parse_txt_markdown(content_bytes, filename)
        elif suffix == ".pdf":
            sections = parse_pdf(content_bytes, filename)
        elif suffix == ".docx":
            sections = parse_docx(content_bytes, filename)
        else:
            raise ValueError(f"Unsupported file format '{suffix}'")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Malformed or unreadable document '{filename}': {str(exc)}") from exc

    if not sections:
        raise ValueError(f"No extractable text found in '{filename}'.")

    return sections


def chunk_document(
    filename: str,
    content_bytes: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Extract, split, and generate deterministic chunk objects with rich metadata.
    """
    clean_filename = sanitize_filename(filename)
    doc_id = compute_document_id(clean_filename, content_bytes)
    source_type = Path(clean_filename).suffix.lower().lstrip(".")
    ingestion_time = datetime.now(timezone.utc).isoformat()

    sections = extract_document_pages(clean_filename, content_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    global_chunk_idx = 0

    for section in sections:
        page_num = section.get("page_number")
        text_splits = splitter.split_text(section["text"])
        for split in text_splits:
            split_text = split.strip()
            if not split_text:
                continue

            metadata = {
                "document_name": clean_filename,
                "document_id": doc_id,
                "chunk_index": global_chunk_idx,
                "page_number": page_num if page_num is not None else 1,
                "source_type": source_type,
                "ingestion_timestamp": ingestion_time,
                "text": split_text,  # Pinecone metadata stores chunk text for retrieval
            }

            # Deterministic chunk ID: {doc_id}_chunk_{global_chunk_idx}
            chunk_id = f"{doc_id}_chunk_{global_chunk_idx}"
            chunks.append(DocumentChunk(text=split_text, metadata=metadata, chunk_id=chunk_id))
            global_chunk_idx += 1

    return chunks
