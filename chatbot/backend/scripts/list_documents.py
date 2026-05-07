# backend/scripts/list_documents.py
"""
Script for listing and reconciling documents in Azure Cognitive Search and Blob Storage.

This script provides utilities to:
- List all unique documents in the search index
- List files in blob storage
- List source files referenced in the search index
- Reconcile differences between blob storage and search index
"""

import asyncio
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import urllib.parse
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def normalize_filename(name: str) -> str:
    """
    Normalize a filename for comparison by URL-decoding and lowercasing.

    Handles both %XX percent-encoding and + as space so that blob names
    and index URLs with different encoding styles compare as equal.

    Args:
        name: Raw filename string, possibly URL-encoded.

    Returns:
        str: Decoded, lowercased, stripped filename for comparison.
    """
    if not name:
        return ""
    # Decode both %XX and + as space, then lowercase for consistent comparison
    decoded = urllib.parse.unquote_plus(name)
    return decoded.lower().strip()


async def list_all_documents():
    """
    List all unique documents in the search index.

    Queries the Azure Cognitive Search index to retrieve all documents,
    extracts unique document names from titles or parent IDs, and logs them.
    """
    logger = logging.getLogger(__name__)
    search_client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=config.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
    )

    results = search_client.search(search_text="*", top=1000, select=["title", "parent_id"])

    # Collect unique document names
    documents = set()

    for result in results:
        r = dict(result)
        title = r.get("title")

        if title and title.strip():
            documents.add(title)
        else:
            # Extract from parent_id if no title
            parent_id = r.get("parent_id")
            if parent_id:
                try:
                    parsed = urllib.parse.urlparse(parent_id)
                    filename = parsed.path.split('/')[-1]
                    filename = urllib.parse.unquote_plus(filename)
                    if filename:
                        documents.add(filename)
                except:
                    pass

    logger.info("Found %d unique documents", len(documents))
    for i, doc in enumerate(sorted(documents), 1):
        logger.info("%d. %s", i, doc)


def list_blob_files():
    """
    List all files in the blob storage container.

    Retrieves all blob names from the configured Azure Blob Storage container,
    excluding directory markers, and returns them as a dict mapping normalized
    filename to display filename.

    Returns:
        dict: Mapping of normalized filename (for comparison) to decoded display name.
    """
    blob_service = BlobServiceClient.from_connection_string(
        config.AZURE_STORAGE_CONNECTION_STRING
    )
    container = blob_service.get_container_client(config.AZURE_STORAGE_CONTAINER_NAME)

    files = {}
    for blob in container.list_blobs():
        if blob.name.endswith("/"):
            continue
        filename = blob.name.split("/")[-1]
        # Decode + and %XX for display
        decoded = urllib.parse.unquote_plus(filename)
        # Normalize for comparison
        normalized = normalize_filename(filename)
        files[normalized] = decoded
    return files


def list_index_files():
    """
    List all source files referenced in the search index.

    Queries the Azure Cognitive Search index to retrieve all document titles
    and URLs, extracts filenames, and returns them as a dict mapping normalized
    filename to display filename.

    Returns:
        dict: Mapping of normalized filename (for comparison) to decoded display name.
    """
    client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=config.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.AZURE_SEARCH_KEY),
    )

    results = client.search(search_text="*", top=1000, select=["url", "title"])
    files = {}

    for r in results:
        d = dict(r)
        # Prefer title as it's the most reliable field
        title = d.get("title")
        if title:
            normalized = normalize_filename(title)
            files[normalized] = title
            continue

        url = d.get("url")
        if not url:
            continue
        parsed = urllib.parse.urlparse(url)
        # Decode both %XX and + when extracting filename from URL
        name = urllib.parse.unquote_plus(parsed.path.split("/")[-1])
        if name:
            normalized = normalize_filename(name)
            files[normalized] = name

    return files


def reconcile_blob_vs_index():
    """
    Reconcile blob storage files with search index files.

    Compares the files in Azure Blob Storage with those referenced in the
    Azure Cognitive Search index using normalized filenames to avoid false
    mismatches caused by URL encoding differences (e.g. + vs %2B vs space).

    Logs any discrepancies:
    - Files present in blob storage but missing from index
    - Files present in index but missing from blob storage
    """
    logger = logging.getLogger(__name__)
    blob_files = list_blob_files()
    index_files = list_index_files()

    # Compare using normalized keys to avoid encoding mismatches
    missing_keys = sorted(set(blob_files.keys()) - set(index_files.keys()))
    extra_keys = sorted(set(index_files.keys()) - set(blob_files.keys()))

    logger.info("Blob files: %d, Index source files: %d", len(blob_files), len(index_files))
    logger.info("Documents currently present in blob container:")
    for i, (key, name) in enumerate(sorted(blob_files.items(), key=lambda x: x[1]), 1):
        logger.info("%d. %s", i, name)

    logger.info("Missing from INDEX (present in blob, not indexed): %d", len(missing_keys))
    for i, key in enumerate(missing_keys, 1):
        logger.info("%d. %s", i, blob_files[key])

    if extra_keys:
        logger.warning("Present in INDEX but not in blob: %d", len(extra_keys))
        for i, key in enumerate(extra_keys, 1):
            logger.warning("%d. %s", i, index_files[key])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Suppress verbose Azure HTTP logging
    logging.getLogger("azure").setLevel(logging.WARNING)

    # Run the reconcile function by default
    reconcile_blob_vs_index()