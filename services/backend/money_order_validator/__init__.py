from __future__ import annotations

from money_order_validator.schemas import (
    BatchContext,
    Instrument,
    TokenUsage,
    ValidationResult,
)
from money_order_validator.pipeline import DocumentProcessor, document_processor

__version__ = "2.0.0"


async def process_file(file_name: str, content: bytes) -> ValidationResult:
    """Extract and validate a single PDF batch."""
    return await document_processor.process_file(file_name, content)


async def process_batch(file_payloads: list[tuple[str, bytes]]) -> list[ValidationResult]:
    """Extract and validate multiple PDF payloads concurrently."""
    return await document_processor.process_batch(file_payloads)


__all__ = [
    "process_file",
    "process_batch",
    "document_processor",
    "DocumentProcessor",
    "ValidationResult",
    "BatchContext",
    "Instrument",
    "TokenUsage",
    "__version__",
]
