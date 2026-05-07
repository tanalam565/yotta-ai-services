import logging
from types import SimpleNamespace

import config
from services.document_intelligence_service import DocumentIntelligenceService


async def test_extract_text_plain_text_success_with_paging(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_PAGES", 5)

    service = DocumentIntelligenceService.__new__(DocumentIntelligenceService)
    service.logger = logging.getLogger("test-doc-intelligence")

    content = ("A" * 2000 + "B" * 100).encode("utf-8")
    result = await service.extract_text(content, "notes.txt")

    assert result["success"] is True
    assert result["page_count"] == 2
    assert len(result["page_texts"]) == 2


async def test_extract_text_plain_text_invalid_utf8_returns_error():
    service = DocumentIntelligenceService.__new__(DocumentIntelligenceService)
    service.logger = logging.getLogger("test-doc-intelligence")

    result = await service.extract_text(b"\xff\xfe\xfa", "notes.txt")

    assert result["success"] is False
    assert "UTF-8" in result["error"]


def test_extract_sync_handles_none_lines_from_document_intelligence(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_PAGES", 5)

    service = DocumentIntelligenceService.__new__(DocumentIntelligenceService)
    service.logger = logging.getLogger("test-doc-intelligence")

    mock_result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, lines=None)],
        content=None,
    )

    class MockPoller:
        def result(self):
            return mock_result

    class MockClient:
        def begin_analyze_document(self, model_id, analyze_request):
            return MockPoller()

    service.client = MockClient()

    result = service._extract_sync(b"dummy", "sample.docx")

    assert result["success"] is True
    assert result["page_count"] == 1
    assert result["page_texts"][0]["text"] == ""
    assert result["text"] == ""


def test_extract_sync_falls_back_to_full_text_when_page_texts_are_empty(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_PAGES", 5)

    service = DocumentIntelligenceService.__new__(DocumentIntelligenceService)
    service.logger = logging.getLogger("test-doc-intelligence")

    mock_result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, lines=None)],
        paragraphs=None,
        content="Recovered full text from docx",
    )

    class MockPoller:
        def result(self):
            return mock_result

    class MockClient:
        def begin_analyze_document(self, model_id, analyze_request):
            return MockPoller()

    service.client = MockClient()

    result = service._extract_sync(b"dummy", "sample.docx")

    assert result["success"] is True
    assert result["page_count"] == 1
    assert result["page_texts"][0]["text"] == "Recovered full text from docx"


def test_extract_sync_uses_paragraphs_when_lines_are_missing(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_PAGES", 5)

    service = DocumentIntelligenceService.__new__(DocumentIntelligenceService)
    service.logger = logging.getLogger("test-doc-intelligence")

    paragraph = SimpleNamespace(
        content="Paragraph-level text",
        bounding_regions=[SimpleNamespace(page_number=2)],
    )
    mock_result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, lines=None)],
        paragraphs=[paragraph],
        content="",
    )

    class MockPoller:
        def result(self):
            return mock_result

    class MockClient:
        def begin_analyze_document(self, model_id, analyze_request):
            return MockPoller()

    service.client = MockClient()

    result = service._extract_sync(b"dummy", "sample.docx")

    assert result["success"] is True
    assert result["page_count"] == 1
    assert result["page_texts"][0]["page_number"] == 2
    assert result["page_texts"][0]["text"] == "Paragraph-level text"
