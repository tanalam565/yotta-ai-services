from services.llm_service import LLMService
from openai import BadRequestError
import httpx


def test_extract_citations_deduplicates_same_filename():
    service = LLMService.__new__(LLMService)

    response = "According to policy [2 → Page 4], and also [3 → Page 5]."
    doc_mapping = {
        2: {"filename": "Handbook.pdf", "type": "company", "download_url": None, "pages": {4}},
        3: {"filename": "Handbook.pdf", "type": "company", "download_url": None, "pages": {5}},
    }

    updated_text, sources = service._extract_citations_and_renumber(response, doc_mapping)

    assert "[1 → Page 4]" in updated_text
    assert "[1 → Page 5]" in updated_text
    assert len(sources) == 1
    assert "Handbook.pdf" in sources[0]["filename"]


def test_clean_response_removes_markdown_bold():
    service = LLMService.__new__(LLMService)
    assert service._clean_response("**Hello** world") == "Hello world"


def test_sanitize_history_for_prompt_removes_inline_citations():
    service = LLMService.__new__(LLMService)
    history = [
        {
            "query": "What is in the upload?",
            "response": "It shows instructor role admin [1 → Page 1]."
        }
    ]

    sanitized = service._sanitize_history_for_prompt(history)

    assert sanitized[0]["query"] == "What is in the upload?"
    assert "[1 → Page 1]" not in sanitized[0]["response"]
    assert "instructor role admin" in sanitized[0]["response"]


def test_sanitize_history_for_prompt_skips_non_dict_entries():
    service = LLMService.__new__(LLMService)
    history = [
        "invalid-entry",
        {"query": "Q", "response": "Text [2]"}
    ]

    sanitized = service._sanitize_history_for_prompt(history)

    assert len(sanitized) == 1
    assert sanitized[0]["query"] == "Q"
    assert "[2]" not in sanitized[0]["response"]


def test_extract_citations_normalizes_placeholder_template():
    service = LLMService.__new__(LLMService)

    response = "Uploaded document summary [N → Page X]."
    doc_mapping = {
        1: {"filename": "PME-NA DIRT_2025_4-pager_D4.docx", "type": "uploaded", "download_url": None, "pages": {1}}
    }

    updated_text, sources = service._extract_citations_and_renumber(response, doc_mapping)

    assert "[1 → Page 1]" in updated_text
    assert len(sources) == 1
    assert "PME-NA DIRT_2025_4-pager_D4.docx" in sources[0]["filename"]


def test_extract_citations_expands_grouped_citation_blocks():
    service = LLMService.__new__(LLMService)

    response = "Policies are defined [1 → Page 51; 2 → Page 1; 3 → Page 4]."
    doc_mapping = {
        1: {"filename": "Handbook.pdf", "type": "company", "download_url": None, "pages": {51}},
        2: {"filename": "CustomerService.pdf", "type": "company", "download_url": None, "pages": {1}},
        3: {"filename": "CommsGuide.pdf", "type": "company", "download_url": None, "pages": {4}},
    }

    updated_text, sources = service._extract_citations_and_renumber(response, doc_mapping)

    assert "[1 → Page 51]" in updated_text
    assert "[2 → Page 1]" in updated_text
    assert "[3 → Page 4]" in updated_text
    assert len(sources) == 3


def test_call_openai_sync_falls_back_to_max_completion_tokens():
    service = LLMService.__new__(LLMService)
    service.model = "chatbot-gpt-5.4"

    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "max_tokens" in kwargs:
                request = httpx.Request("POST", "https://example.test/openai/chat/completions")
                response = httpx.Response(status_code=400, request=request)
                raise BadRequestError(
                    message="Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                    response=response,
                    body={
                        "error": {
                            "message": "Unsupported parameter",
                            "param": "max_tokens",
                            "code": "unsupported_parameter",
                        }
                    },
                )
            return _Response("ok")

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    class _Logger:
        def info(self, *_args, **_kwargs):
            return None

    service.client = _Client()
    service.logger = _Logger()

    result = service._call_openai_sync([{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert len(service.client.chat.completions.calls) == 2
    assert "max_tokens" in service.client.chat.completions.calls[0]
    assert "max_completion_tokens" in service.client.chat.completions.calls[1]
