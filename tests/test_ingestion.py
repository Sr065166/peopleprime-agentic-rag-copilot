import io
import pytest
from docx import Document as DocxWriter
from pypdf import PdfWriter
from app.services.ingestion import (
    chunk_document,
    extract_document_pages,
    sanitize_filename,
    compute_document_id,
)


def create_sample_pdf_bytes(pages: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_sample_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxWriter()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_sanitize_filename():
    assert sanitize_filename("safe_file.pdf") == "safe_file.pdf"
    assert sanitize_filename("../../secret/path.docx") == "path.docx"
    assert sanitize_filename("weird name!@#$%.txt") == "weird_name_____.txt"


def test_compute_document_id_deterministic():
    data1 = b"Sample text for PeoplePrime Technologies"
    data2 = b"Sample text for PeoplePrime Technologies"
    id1 = compute_document_id("policy.txt", data1)
    id2 = compute_document_id("policy.txt", data2)
    assert id1 == id2
    assert id1.startswith("policy_")


def test_txt_chunking_and_metadata():
    text = (
        "PeoplePrime Technologies Annual Leave Policy.\n\n"
        "All full-time employees receive 25 days of paid annual leave per calendar year. "
        "Leave must be requested at least two weeks in advance through the HR Portal.\n\n"
        "Unused leave of up to 5 days can be carried forward to the following calendar year."
    )
    chunks = chunk_document(
        filename="leave_policy.txt",
        content_bytes=text.encode("utf-8"),
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_id.startswith("leave_policy_")
        assert c.metadata["document_name"] == "leave_policy.txt"
        assert c.metadata["source_type"] == "txt"
        assert c.metadata["chunk_index"] == idx
        text_lower = c.text.lower()
        assert "peopleprime" in text_lower or "leave" in text_lower or "calendar" in text_lower or "portal" in text_lower


def test_docx_parsing_and_chunking():
    docx_bytes = create_sample_docx_bytes([
        "PeoplePrime Technologies Remote Work Guidelines",
        "Employees may work remotely up to 3 days per week with manager approval.",
        "A secure VPN connection is mandatory when handling company data.",
    ])
    chunks = chunk_document(
        filename="remote_work.docx",
        content_bytes=docx_bytes,
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["source_type"] == "docx"
    assert "Remote Work Guidelines" in chunk.text
    assert "secure VPN" in chunk.text


def test_pdf_parsing_mocked(monkeypatch):
    # Mock parse_pdf directly to verify page number preservation and chunking flow
    def mock_parse_pdf(content_bytes: bytes, filename: str):
        return [
            {"text": "Page 1: PeoplePrime Code of Conduct and Anti-Harassment rules.", "page_number": 1},
            {"text": "Page 2: Disciplinary escalation procedures and reporting lines.", "page_number": 2},
        ]

    monkeypatch.setattr("app.services.ingestion.parse_pdf", mock_parse_pdf)

    chunks = chunk_document(
        filename="conduct.pdf",
        content_bytes=b"%PDF-1.4 dummy",
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == 1
    assert "Code of Conduct" in chunks[0].text
    assert chunks[1].metadata["page_number"] == 2
    assert "Disciplinary escalation" in chunks[1].text


def test_unsupported_file_type():
    with pytest.raises(ValueError, match="Unsupported file format"):
        chunk_document(
            filename="malicious.exe",
            content_bytes=b"MZ123456",
        )


def test_empty_document():
    with pytest.raises(ValueError, match="empty"):
        chunk_document(
            filename="empty.txt",
            content_bytes=b"   \n   ",
        )
