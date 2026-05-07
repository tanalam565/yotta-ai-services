# backend/scripts/generate_embeddings_incremental.py
"""
Incremental embedding script for new or unindexed documents in blob storage.

Unlike the full regeneration script, this script only processes documents that
are not yet present in the Azure Cognitive Search index. This saves both time
and API cost by skipping documents that have already been embedded.

Usage:
    python generate_embeddings_incremental.py

How it works:
1. Lists all PDF files in blob storage
2. Lists all document titles already present in the search index
3. Compares the two sets (normalized to handle URL encoding differences)
4. Only processes and embeds documents missing from the index
5. Uploads new chunks to Azure Cognitive Search
"""

import asyncio
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import urllib.parse
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from services.embedding_service import EmbeddingService


# CHUNKING CONFIGURATION
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks


def normalize_filename(name: str) -> str:
    """
    Normalize a filename for comparison by URL-decoding and lowercasing.

    Handles both %XX percent-encoding and + as space so that blob names
    and index titles with different encoding styles compare as equal.

    Args:
        name: Raw filename string, possibly URL-encoded.

    Returns:
        str: Decoded, lowercased, stripped filename for comparison.
    """
    if not name:
        return ""
    return urllib.parse.unquote_plus(name).lower().strip()


def chunk_text_with_pages(page_texts: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split text into overlapping chunks while tracking page numbers.

    Concatenates text from all pages and splits it into chunks of specified size with overlap.
    Each chunk is assigned the page number from its middle character position.

    Args:
        page_texts (list): List of dicts with {"page_number": int, "text": str}.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to CHUNK_SIZE.
        overlap (int, optional): Overlap between chunks in characters. Defaults to CHUNK_OVERLAP.

    Returns:
        list: List of dicts with {"text": str, "page_number": int, "chunk_number": int}.
    """
    chunks = []
    chunk_number = 0

    # Concatenate all page texts with page markers
    full_text = ""
    char_to_page = []  # Maps character index to page number

    for page_info in page_texts:
        page_num = page_info["page_number"]
        page_text = page_info["text"]

        # Track which characters belong to which page
        for _ in range(len(page_text)):
            char_to_page.append(page_num)

        full_text += page_text + " "  # Add space between pages
        char_to_page.append(page_num)  # For the space

    # Chunk the full text and determine page for each chunk
    start = 0

    while start < len(full_text):
        end = start + chunk_size

        # If not the last chunk, try to break at sentence/word boundary
        if end < len(full_text):
            # Look for sentence end (. ! ?)
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if i < len(full_text) and full_text[i] in '.!?\n':
                    end = i + 1
                    break
            else:
                # No sentence boundary, look for space
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if i < len(full_text) and full_text[i] == ' ':
                        end = i
                        break

        chunk_text = full_text[start:end].strip()

        if chunk_text:
            # Determine which page this chunk is primarily from
            chunk_middle = start + ((end - start) // 2)
            chunk_middle = min(chunk_middle, len(char_to_page) - 1)
            primary_page = char_to_page[chunk_middle]

            chunks.append({
                "text": chunk_text,
                "page_number": primary_page,
                "chunk_number": chunk_number
            })
            chunk_number += 1

        start = end - overlap if end < len(full_text) else end

    return chunks


def generate_chunk_id(parent_id: str, chunk_number: int) -> str:
    """
    Generate a unique chunk ID from parent ID and chunk number.

    Creates a base64-encoded unique identifier for each chunk.

    Args:
        parent_id (str): The parent document identifier.
        chunk_number (int): The chunk number within the document.

    Returns:
        str: A unique base64-encoded chunk ID.
    """
    import base64
    combined = f"{parent_id}_chunk_{chunk_number}"
    return base64.b64encode(combined.encode()).decode()


def get_indexed_document_titles(search_client: SearchClient) -> set:
    """
    Retrieve the set of normalized document titles already present in the index.

    Queries the search index for all document titles and returns them as a
    normalized set for fast membership testing during incremental comparison.

    Args:
        search_client: Azure SearchClient instance.

    Returns:
        set: Normalized (lowercased, URL-decoded) titles of indexed documents.
    """
    logger = logging.getLogger(__name__)
    results = search_client.search(
        search_text="*",
        select=["title"],
        top=10000
    )

    indexed = set()
    for r in results:
        title = dict(r).get("title")
        if title:
            indexed.add(normalize_filename(title))

    logger.info("Found %d already-indexed document titles in search index", len(indexed))
    return indexed


async def extract_text_from_blob(blob_client, filename: str, doc_intelligence_client) -> dict:
    """
    Download a blob and extract text with page numbers using Document Intelligence.

    Downloads the blob content, processes it with Azure Document Intelligence
    to extract text page by page, and returns structured page data.

    Args:
        blob_client: Azure Blob client for the specific blob.
        filename (str): Name of the file being processed.
        doc_intelligence_client: Azure Document Intelligence client.

    Returns:
        dict: Dictionary containing 'page_texts', 'page_count', 'success', and optionally 'error'.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("Downloading %s", filename)
        blob_data = blob_client.download_blob().readall()

        logger.info("Extracting text with page tracking (size: %d bytes)", len(blob_data))

        analyze_request = AnalyzeDocumentRequest(bytes_source=blob_data)

        poller = doc_intelligence_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=analyze_request
        )

        result = poller.result()

        # Extract text page by page
        page_texts = []

        if hasattr(result, 'pages'):
            for page in result.pages:
                page_num = page.page_number

                page_content = ""
                if hasattr(page, 'lines'):
                    page_content = " ".join([line.content for line in page.lines])

                page_texts.append({
                    "page_number": page_num,
                    "text": page_content
                })

        page_count = len(page_texts)
        total_chars = sum(len(p["text"]) for p in page_texts)

        logger.info("Extracted %d characters from %d pages", total_chars, page_count)

        return {
            "page_texts": page_texts,
            "page_count": page_count,
            "success": True
        }

    except Exception as e:
        logger.exception("Extraction error for %s: %s", filename, e)
        return {
            "page_texts": [],
            "page_count": 0,
            "success": False,
            "error": str(e)
        }


async def generate_embeddings_incremental():
    """
    Incrementally generate embeddings only for documents not yet in the index.

    Main function that orchestrates the incremental process:
    - Lists all PDFs in blob storage
    - Checks which are already indexed
    - Only processes and embeds new/missing documents
    - Uploads new chunks to Azure Cognitive Search

    This avoids re-processing already indexed documents, saving time and API cost.
    """
    logger = logging.getLogger(__name__)

    logger.info("Starting Incremental Embedding Generation")
    logger.info("Configuration: Chunk size=%d, Overlap=%d, Container=%s", CHUNK_SIZE, CHUNK_OVERLAP, config.AZURE_STORAGE_CONTAINER_NAME)

    # Initialize services
    embedding_service = EmbeddingService()

    search_client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=config.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
    )

    blob_service = BlobServiceClient.from_connection_string(
        config.AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service.get_container_client(
        config.AZURE_STORAGE_CONTAINER_NAME
    )

    doc_intelligence_client = DocumentIntelligenceClient(
        endpoint=config.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(config.AZURE_DOCUMENT_INTELLIGENCE_KEY),
        api_version="2024-11-30"
    )

    try:
        # Get already-indexed document titles
        indexed_titles = get_indexed_document_titles(search_client)

        # List all blobs in container
        logger.info("Listing files in blob storage")
        blobs = list(container_client.list_blobs())
        pdf_blobs = [b for b in blobs if b.name.lower().endswith('.pdf')]
        logger.info("Found %d PDF files in blob storage", len(pdf_blobs))

        # Find which blobs are not yet indexed
        new_blobs = []
        for blob_info in pdf_blobs:
            normalized = normalize_filename(blob_info.name)
            if normalized not in indexed_titles:
                new_blobs.append(blob_info)

        if not new_blobs:
            logger.info("All documents are already indexed. Nothing to do.")
            return

        logger.info("Found %d new document(s) to index:", len(new_blobs))
        for b in new_blobs:
            logger.info("  - %s", b.name)

        # Process only new documents
        total_chunks_created = 0
        documents_processed = 0
        chunks_to_upload = []

        for blob_info in new_blobs:
            blob_name = blob_info.name
            documents_processed += 1

            logger.info("Processing document %d/%d: %s", documents_processed, len(new_blobs), blob_name)

            blob_client = container_client.get_blob_client(blob_name)

            extraction_result = await extract_text_from_blob(
                blob_client,
                blob_name,
                doc_intelligence_client
            )

            if not extraction_result['success'] or not extraction_result['page_texts']:
                logger.warning("Skipping %s: No text extracted", blob_name)
                continue

            page_texts = extraction_result['page_texts']
            page_count = extraction_result['page_count']

            parent_id = f"blob://{config.AZURE_STORAGE_CONTAINER_NAME}/{blob_name}"

            chunks = chunk_text_with_pages(page_texts)
            total_chunks_created += len(chunks)

            total_chars = sum(len(p["text"]) for p in page_texts)
            logger.info("Document stats: %d chars, %d pages, created %d chunks", total_chars, page_count, len(chunks))

            for chunk_info in chunks:
                chunk_content = chunk_info["text"]
                chunk_num = chunk_info["chunk_number"]
                page_num = chunk_info["page_number"]

                embedding = embedding_service.generate_embedding(chunk_content)

                chunk_id = generate_chunk_id(parent_id, chunk_num)

                chunk_doc = {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "chunk_number": chunk_num,
                    "page_number": page_num,
                    "title": blob_name,
                    "content": chunk_content,
                    "merged_content": chunk_content,
                    "filepath": blob_name,
                    "url": f"https://{blob_service.account_name}.blob.core.windows.net/{config.AZURE_STORAGE_CONTAINER_NAME}/{blob_name}",
                    "metadata_storage_name": blob_name,
                    "metadata_storage_path": parent_id,
                    "metadata_storage_content_type": "application/pdf",
                    "content_vector": embedding
                }

                chunks_to_upload.append(chunk_doc)

                # Upload in batches of 50
                if len(chunks_to_upload) >= 50:
                    logger.info("Uploading batch of %d chunks", len(chunks_to_upload))
                    try:
                        search_client.upload_documents(documents=chunks_to_upload)
                        logger.info("Batch uploaded successfully")
                    except Exception as batch_error:
                        logger.error("Batch upload error: %s", batch_error)
                        # Try one by one on failure
                        for single_doc in chunks_to_upload:
                            try:
                                search_client.upload_documents(documents=[single_doc])
                            except Exception as doc_error:
                                logger.error("Failed to upload chunk: %s", doc_error)

                    chunks_to_upload = []

        # Upload any remaining chunks
        if chunks_to_upload:
            logger.info("Uploading final batch of %d chunks", len(chunks_to_upload))
            try:
                search_client.upload_documents(documents=chunks_to_upload)
                logger.info("Final batch uploaded successfully")
            except Exception as batch_error:
                logger.error("Final batch upload error: %s", batch_error)

        # Summary
        logger.info(
            "Incremental embedding complete: %d new document(s) processed, %d chunks created",
            documents_processed,
            total_chunks_created
        )

    except Exception as e:
        logger.exception("Error in incremental embedding generation: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(generate_embeddings_incremental())